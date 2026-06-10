"""Sequence state tracked by scheduler, block manager, and model runner."""

from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    """Lifecycle state for one request inside the scheduler."""

    # Waiting sequences have not yet reserved KV-cache blocks.
    WAITING = auto()
    # Running sequences own block_table entries and can be batched for compute.
    RUNNING = auto()
    # Finished sequences have released their blocks and can be returned.
    FINISHED = auto()


class Sequence:
    """A single prompt plus its generated completion tokens.

    The sequence object intentionally stores both logical token information and
    physical KV-cache bookkeeping.  The scheduler mutates status, the block
    manager mutates block_table/num_cached_tokens, and model execution reads the
    token lengths to build attention metadata.
    """

    # Monotonic ids let generation outputs be sorted back into request order.
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        # Number of tokens represented by one logical/physical cache block.
        self.block_size = block_size

        # seq_id is stable across scheduling, preemption, and completion.
        self.seq_id = next(Sequence.counter)

        # New sequences enter the waiting queue until the scheduler allocates
        # enough KV-cache blocks for their prompt.
        self.status = SequenceStatus.WAITING

        # Copy the input token list so caller-side mutations cannot change the
        # active request after it has entered the engine.
        self.token_ids = copy(token_ids)

        # last_token is read during decode because each decode step feeds only
        # the newest token, not the entire prefix.
        self.last_token = self.token_ids[-1] if self.token_ids else None

        # num_tokens grows during generation; num_prompt_tokens stays fixed so
        # completion tokens can be sliced without storing a separate list.
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)

        # Prefix-cache hits increase this count.  ModelRunner.prepare_prefill()
        # skips these tokens because their KV values are already present.
        self.num_cached_tokens = 0

        # block_table maps logical block index -> physical KV-cache block id.
        # It is filled by BlockManager.allocate/append and consumed by attention.
        self.block_table = []

        # Copy sampling and stopping parameters onto the sequence so scheduler
        # postprocessing can make local decisions after each sampled token.
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        """Return total tokens currently in the sequence."""
        return self.num_tokens

    def __getitem__(self, idx):
        """Expose token indexing for small helper/test code."""
        return self.token_ids[idx]

    @property
    def is_finished(self):
        """Whether generation has reached a terminal condition."""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """Number of generated tokens, excluding the prompt."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """The immutable prompt prefix."""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """The generated suffix that should be decoded for the user."""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """Number of complete blocks skipped by prefix caching."""
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        """Number of logical blocks required for the current token length."""
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        """Token count in the final logical block.

        The scheduler mostly calls this while the last block is partial.  When
        the sequence length is an exact multiple of block_size, this expression
        returns 0, matching the current block() slicing convention.
        """
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

    def block(self, i):
        """Return token ids for logical block i."""
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            # The final block may be partial.  Negative slicing keeps the last
            # last_block_num_tokens tokens without computing a start index.
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            # Non-final blocks are always exactly block_size tokens.
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        """Append one generated token and refresh derived counters."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        """Serialize only the fields needed by worker processes.

        During prefill, workers need the full prompt token list.  During decode,
        the model input is just the last generated token, so the compact state
        sends only that token plus cache metadata.
        """
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """Rebuild a lightweight Sequence after multiprocessing pickle load."""
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state
        # num_completion_tokens identifies which compact serialization shape was
        # used in __getstate__.
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # Prefill: workers need the entire uncached suffix of the prompt.
            self.token_ids = last_token_or_ids
        else:
            # Decode: workers feed a single token and read the rest from KV cache.
            self.token_ids = [last_token_or_ids]
        # last_token is consumed by prepare_decode().
        self.last_token = self.token_ids[-1] if self.token_ids else None
