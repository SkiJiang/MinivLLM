"""Batch scheduler for prefill and decode work."""

from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    """Choose which sequences run next under token, sequence, and cache limits."""

    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # The scheduler decides *when* a sequence runs; the block manager decides
        # *where* the sequence's KV cache lives.
        self.block_manager = BlockManager(max_cached_blocks, block_size)

        # max_num_batched_tokens caps total prompt tokens for prefill, or number
        # of one-token decode steps in decode mode.
        self.max_num_batched_tokens = max_num_batched_tokens
        # max_num_sequences caps the batch dimension passed to ModelRunner.
        self.max_num_sequences = max_num_sequences

        # waiting holds prompts without allocated blocks.  running holds active
        # sequences with block tables and cache ownership.
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # eos is compared after sampling unless the sequence opts into ignore_eos.
        self.eos = eos


    def is_finished(self):
        """Return True once no queued or active sequence remains."""
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        """Submit a new request to the waiting queue."""
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        """Build the next batch and report whether it is a prefill batch.

        Prefill has priority over decode because a newly admitted prompt may
        consume many tokens and must first reserve all of its KV blocks.  Decode
        only appends one token per scheduled sequence.
        """
        scheduled_sequences = []
        current_scheduled_tokens = 0

        # Phase 1: admit waiting prompts for prefill while both the sequence-count
        # limit and token budget allow it.
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                # Allocation fills seq.block_table and records any prefix-cache
                # hits before the model sees the sequence.
                seq = self.waiting.popleft()
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                # Queues are FIFO.  If the first waiting sequence cannot fit, a
                # later sequence is not allowed to jump ahead.
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # Phase 2: no prefill was scheduled, so choose active sequences for one
        # decode token each.
        while self.running:
            seq = self.running.popleft()

            # can_append checks whether adding the next token would require a new
            # cache block and whether that block is currently available.
            if not self.block_manager.can_append(seq):
                if self.running:
                    # Put the current sequence back, preempt a sequence from the
                    # tail, and free its blocks to make space.
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    # If this is the only running sequence and it cannot append,
                    # preempt it and end this scheduling attempt.
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    # The current sequence is still runnable, but this batch is
                    # full.  Put it back at the front for the next step.
                    self.running.appendleft(seq)
                    break
                # Reserve/update metadata for the token that will be produced by
                # this decode step.
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1

        # Put scheduled decode sequences back at the front in their original
        # order, so future decode steps remain round-robin.
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        """Move a running sequence back to waiting after releasing its blocks."""
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """Attach sampled tokens and release sequences that reached a stop rule."""
        for seq, token_id in zip(seqs, token_ids):
            # ModelRunner returns one sampled token per scheduled sequence.
            seq.append_token(token_id)

            # Stop rules are intentionally checked after append_token so the
            # terminal token is included in completion_token_ids.
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                # Finished sequences no longer need KV cache, and they must be
                # removed from running so is_finished() can eventually become true.
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
