"""由 scheduler、block manager 和 model runner 共同维护的序列状态。"""

from enum import Enum, auto
import math
from itertools import count 
from myvllm.sampling_parameters import SamplingParams
from copy import copy


class SequenceStatus(Enum):
    """单个请求在调度器中的生命周期状态。"""

    # WAITING 表示序列还没有预留 KV-cache block。
    WAITING = auto()
    # RUNNING 表示序列已经拥有 block_table，可以被打包执行。
    RUNNING = auto()
    # FINISHED 表示序列已经释放 block，可以返回给用户。
    FINISHED = auto()


class Sequence:
    """一个 prompt 以及它生成出的 completion token。

    Sequence 会同时保存逻辑 token 信息和物理 KV-cache 账本信息。scheduler 修改
    status，block manager 修改 block_table/num_cached_tokens，模型执行阶段读取
    token 长度来构造 attention 元数据。
    """

    # 单调递增 id 用于在生成完成后按请求提交顺序重新排序输出。
    counter = count()

    def __init__(self, token_ids: list[int], block_size: int, sampling_params = SamplingParams()):
        # 一个逻辑/物理 cache block 代表多少个 token。
        self.block_size = block_size

        # seq_id 在调度、抢占和完成期间都保持稳定。
        self.seq_id = next(Sequence.counter)

        # 新序列先进入 waiting 队列，直到 scheduler 为 prompt 分配足够的 KV-cache block。
        self.status = SequenceStatus.WAITING

        # 复制输入 token 列表，避免调用方在请求进入 engine 后修改原列表影响正在运行的请求。
        self.token_ids = copy(token_ids)

        # decode 阶段只把最新 token 送入模型，而不是送入整个 prefix，所以需要 last_token。
        self.last_token = self.token_ids[-1] if self.token_ids else None

        # num_tokens 会随生成增长；num_prompt_tokens 保持不变，便于不额外存列表也能切出 completion。
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)

        # prefix cache 命中会增加这个计数。prepare_prefill() 会跳过这些 token，
        # 因为它们的 KV 已经在 cache 中。
        self.num_cached_tokens = 0

        # block_table 映射：逻辑 block 下标 -> 物理 KV-cache block id。
        # 它由 BlockManager.allocate/append 填充，attention kernel 会读取它。
        self.block_table = []

        # 将采样和停止参数复制到序列上，使 scheduler 在每次采样后可以就地判断是否停止。
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.max_model_length = sampling_params.max_model_length

    def __len__(self):
        """返回当前序列中的 token 总数。"""
        return self.num_tokens

    def __getitem__(self, idx):
        """为辅助代码和测试暴露 token 下标访问。"""
        return self.token_ids[idx]

    @property
    def is_finished(self):
        """生成是否已经达到终止条件。"""
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        """已生成 token 数，不包含 prompt。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """不可变的 prompt 前缀。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """应当解码并返回给用户的生成后缀。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """被 prefix cache 跳过的完整 block 数量。"""
        return int(math.ceil(self.num_cached_tokens / self.block_size))

    @property
    def num_blocks(self):
        """当前 token 长度需要的逻辑 block 数量。"""
        return int(math.ceil(self.num_tokens / self.block_size))

    @property
    def last_block_num_tokens(self):
        """最后一个逻辑 block 中的 token 数量。

        scheduler 通常在最后一个 block 还未填满时调用它。当序列长度刚好是
        block_size 的整数倍时，这个表达式返回 0，并与当前 block() 的切片约定保持一致。
        """
        full_blocks = int(math.floor(self.num_tokens / self.block_size))
        return len(self.token_ids[full_blocks * self.block_size : ])

    def block(self, i):
        """返回第 i 个逻辑 block 中的 token id。"""
        assert 0 <= i < self.num_blocks, f"Block index {i} out of range [0, {self.num_blocks})"
        if i == self.num_blocks - 1:
            # 最后一个 block 可能不满；负索引切片可以直接取最后 last_block_num_tokens 个 token。
            return self.token_ids[-self.last_block_num_tokens:]
        else:
            # 非最后一个 block 一定正好包含 block_size 个 token。
            start_idx = i * self.block_size
            end_idx = start_idx + self.block_size
            return self.token_ids[start_idx : end_idx]

    def append_token(self, token_id):
        """追加一个生成 token，并刷新相关计数。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1 

    def __getstate__(self):
        """只序列化 worker 进程需要的字段。

        prefill 阶段 worker 需要完整 prompt token 列表；decode 阶段模型输入只有
        最新生成的 token，因此紧凑状态只发送这个 token 和 cache 元数据。
        """
        return (
            self.num_tokens, 
            self.num_prompt_tokens, 
            self.num_cached_tokens, 
            self.block_table,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token
        )

    def __setstate__(self, state):
        """multiprocessing pickle 反序列化后重建轻量 Sequence。"""
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            last_token_or_ids
        ) = state
        # num_completion_tokens 用于判断 __getstate__ 使用的是哪种紧凑序列化形态。
        num_completion_tokens = self.num_tokens - self.num_prompt_tokens
        if num_completion_tokens == 0:
            # prefill：worker 需要 prompt 中未被 cache 命中的完整后缀。
            self.token_ids = last_token_or_ids
        else:
            # decode：worker 只输入单个 token，其余上下文从 KV cache 中读取。
            self.token_ids = [last_token_or_ids]
        # prepare_decode() 会读取 last_token。
        self.last_token = self.token_ids[-1] if self.token_ids else None
