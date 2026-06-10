"""高层生成引擎。

LLMEngine 负责请求提交、调度、模型执行以及最终文本解码。当张量并行使用多张
GPU 时，它还负责启动 worker 进程。
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
    """初始化非 0 rank 的 ModelRunner，并等待共享内存调用。"""
    # 将 stdout/stderr 重新打开为行缓冲模式，确保父进程还在生成时 worker 日志能及时输出。
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    # 非 master rank 会停在 ModelRunner.loop() 中；rank 0 通过共享内存和 event
    # 向它们发送方法调用。
    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    """面向用户的封装，组合 scheduler、tokenizer 和 model runner。"""

    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)

        # 对 CUDA 来说 spawn 比 fork 更安全，因为每个进程会独立初始化 CUDA context，
        # 并干净地加入分布式进程组。
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []

        # rank 0 留在当前进程。其他 rank 运行 worker_process()，并通过
        # ModelRunner.write_shm() 接收方法调用。
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()

        # master runner 在本进程执行，并负责协调所有 worker。
        self.model_runner = ModelRunner(config, rank=0, event=self.events)

        # tokenizer 放在 engine 层，因为 model runner 只处理 token id 和 tensor。
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler 在 ModelRunner 之后初始化，因为 allocate_kv_cache 可能会根据真实
        # GPU 显存修正 config["max_cached_blocks"]。多 rank 模式下，
        # ModelRunner.__init__ 也会先等待所有 rank 加入进程组，然后 scheduler
        # 才开始使用 cache 上限。
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )

        # 注册清理函数，即使调用方忘记显式调用 exit()，也能释放 worker 和分布式状态。
        atexit.register(self.exit)


    def exit(self):
        """停止 model runner，并等待 worker 进程结束。"""
        # world_size > 1 时，call("exit") 也会把退出命令转发给 worker。
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    def step(self) -> tuple[list[int], bool]:
        """执行一轮 scheduler -> model -> postprocess。"""
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        if not scheduled_sequences:
            # 没有可调度任务，通常意味着队列为空，或活跃序列被抢占后等待 cache 可用。
            return [], is_prefill

        # ModelRunner 在 rank 0 返回采样 token id。worker rank 会执行同样的方法以保持同步，
        # 但不返回 token。
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)

        if outputs is not None:
            # Scheduler.postprocess 需要普通 Python list，而不是 CUDA tensor。
            outputs = outputs.cpu().tolist()

        # 追加采样 token，检查停止条件，并释放已完成序列的 KV block。
        self.scheduler.postprocess(scheduled_sequences, outputs)

        # 只把已完成序列返回给外部调用方；未完成序列继续留在 scheduler 中等待后续 decode。
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]

        # prefill 会处理所有未命中 cache 的 prompt token；decode 每个被调度序列只处理一个 token。
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        return outputs, num_processed_tokens, is_prefill


    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        """将一个 prompt tokenize 后加入 scheduler。"""
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        """为所有 prompt 生成 completion，并返回文本和 token id。"""
        for prompt in prompts:
            self.add_prompt(prompt, sampling_params)
        generated_tokens = {}

        # 持续驱动 engine，直到 waiting 和 running 队列都为空。
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

        # 即使序列因为 batching 和停止条件而以不同顺序完成，也按输入 prompt 顺序返回。
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        return output
