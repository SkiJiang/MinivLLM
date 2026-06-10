"""High-level generation engine.

LLMEngine owns request submission, scheduling, model execution, and final text
decoding.  It also starts worker processes when tensor parallelism uses more
than one GPU.
"""

import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Initialize a non-zero-rank ModelRunner and wait for shared-memory calls."""
    # Reopen stdout/stderr in line-buffered mode so worker logs appear promptly
    # while the parent process is still running generation.
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # Non-master ranks live in ModelRunner.loop(); rank 0 sends method calls via
    # shared memory and events.
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    """User-facing wrapper around scheduler, tokenizer, and model runner."""

    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)

        # Spawn is safer with CUDA than fork because each process initializes
        # its own CUDA context and joins the distributed process group cleanly.
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []

        # Rank 0 stays in this process.  Additional ranks run worker_process()
        # and receive method calls through ModelRunner.write_shm().
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()

        # The master runner executes locally and coordinates all workers.
        self.model_runner = ModelRunner(config, rank=0, event=self.events)

        # The tokenizer is kept in the engine layer because the model runner only
        # operates on token ids and tensors.
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # The scheduler is initialized after ModelRunner because allocate_kv_cache
        # may refine config["max_cached_blocks"] based on actual GPU memory.  In
        # multi-rank mode, ModelRunner.__init__ also waits for all ranks to join
        # the process group before the scheduler starts using cache limits.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )

        # Register cleanup so worker processes and distributed state are released
        # even if the caller forgets to call exit() explicitly.
        atexit.register(self.exit)


    def exit(self):
        """Stop model runners and join worker processes."""
        # call("exit") also forwards the exit command to workers when world_size
        # is greater than one.
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    def step(self) -> tuple[list[int], bool]:
        """Run one scheduler/model/postprocess iteration."""
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            # No work could be scheduled, usually because all queues are empty or
            # all active sequences were preempted to wait for cache availability.
            return [], is_prefill

        # ModelRunner returns sampled token ids on rank 0.  Worker ranks execute
        # the same method for synchronization but do not return tokens.
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)

        if outputs is not None:
            # Scheduler.postprocess expects a normal Python list, not a CUDA tensor.
            outputs = outputs.cpu().tolist()

        # Append sampled tokens, check stop conditions, and release finished KV
        # blocks.
        self.scheduler.postprocess(scheduled_sequences, outputs)

        # Only finished sequences are returned to the external caller; unfinished
        # sequences remain in the scheduler for more decode steps.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        # Prefill processes every uncached prompt token, while decode processes
        # exactly one token per scheduled sequence.
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        """Tokenize one prompt and add it to the scheduler."""
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """Generate completions for all prompts and return text plus token ids."""
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}

        # Drive the engine until both waiting and running queues are empty.
        while not self.scheduler.is_finished():
            start_t = time.time()
            outputs, num_processed_tokens, is_prefill = self.step()
            end_t = time.time()
            running_time = end_t - start_t + 1e-10
            if is_prefill:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            else:
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        # Preserve input prompt order even though sequences may finish in a
        # different order because of batching and stop conditions.
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
