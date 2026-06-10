"""负责 prefill 和 decode 工作分批的调度器。"""

from collections import deque
from myvllm.engine.sequence import Sequence, SequenceStatus
from myvllm.engine.block_manager import BlockManager


class Scheduler:
    """在 token 数、序列数和 cache 容量限制下选择下一批要运行的序列。"""

    def __init__(self, max_num_sequences: int, max_num_batched_tokens: int, max_cached_blocks: int, block_size: int, eos: int):
        # Scheduler 决定序列何时运行；BlockManager 决定该序列的 KV cache 放在哪里。
        self.block_manager = BlockManager(max_cached_blocks, block_size)

        # max_num_batched_tokens 在 prefill 阶段限制 prompt token 总数；
        # 在 decode 阶段限制本轮最多处理多少个单 token decode step。
        self.max_num_batched_tokens = max_num_batched_tokens
        # max_num_sequences 限制传给 ModelRunner 的 batch 维度大小。
        self.max_num_sequences = max_num_sequences

        # waiting 保存尚未分配 block 的 prompt；running 保存已经拥有 block_table 和
        # cache 所有权的活跃序列。
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # 每次采样后会和 eos 比较，除非该序列设置了 ignore_eos。
        self.eos = eos


    def is_finished(self):
        """当没有等待队列和运行队列中的序列时返回 True。"""
        return len(self.waiting) == 0 and len(self.running) == 0
    
    def add_sequence(self, sequence: Sequence):
        """把一个新请求加入 waiting 队列。"""
        self.waiting.append(sequence)


    def schedule(self) -> tuple[list[Sequence], bool]:
        """构造下一批任务，并返回它是否是 prefill batch。

        prefill 优先于 decode，因为新进入的 prompt 可能消耗大量 token，
        而且必须先一次性预留它需要的 KV block。decode 则每个被调度的序列
        只追加一个 token。
        """
        scheduled_sequences = []
        current_scheduled_tokens = 0

        # 阶段 1：在序列数量和 token budget 都允许的情况下，从 waiting 队列接纳
        # prompt 进入 prefill。
        while self.waiting and len(scheduled_sequences) < self.max_num_sequences:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq) and len(seq) + current_scheduled_tokens <= self.max_num_batched_tokens:
                # allocate 会填充 seq.block_table，并在模型看到序列前记录 prefix cache 命中。
                seq = self.waiting.popleft()
                self.block_manager.allocate(seq)
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += len(seq)
            else:
                # 队列是 FIFO：如果队首序列放不下，后面的序列不能插队。
                break
        if scheduled_sequences:
            return scheduled_sequences, True
        
        # 阶段 2：本轮没有调度 prefill，因此从 running 队列中选择序列，
        # 每个序列执行一个 token 的 decode。
        while self.running:
            seq = self.running.popleft()

            # can_append 检查追加下一个 token 是否需要新 cache block，以及该 block
            # 当前是否可用。
            if not self.block_manager.can_append(seq):
                if self.running:
                    # 把当前序列放回队首，从队尾抢占一个序列并释放它的 block 腾空间。
                    self.running.appendleft(seq)
                    self.preempt(self.running.pop())
                else:
                    # 如果这是唯一的 running 序列且无法追加，就抢占它并结束本轮调度。
                    self.preempt(seq)
                    break
            else:
                if current_scheduled_tokens >= self.max_num_batched_tokens or len(scheduled_sequences) >= self.max_num_sequences:
                    # 当前序列可以运行，但本轮 batch 已满；把它放回队首，留到下一轮。
                    self.running.appendleft(seq)
                    break
                # 为本次 decode 即将生成的 token 预留或更新 block 元数据。
                self.block_manager.append(seq)
                scheduled_sequences.append(seq)
                current_scheduled_tokens += 1

        # 将本轮已调度的 decode 序列按原顺序放回队首，使后续 decode 保持轮转顺序。
        if scheduled_sequences:
            self.running.extendleft(reversed(scheduled_sequences))

        return scheduled_sequences, False


    def preempt(self, seq: Sequence) -> None:
        """释放 running 序列占用的 block，并把它放回 waiting。"""
        self.block_manager.deallocate(seq)
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)        


    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        """写入采样 token，并释放已经触发停止条件的序列。"""
        for seq, token_id in zip(seqs, token_ids):
            # ModelRunner 会为每个被调度的序列返回一个采样 token。
            seq.append_token(token_id)

            # 停止条件故意放在 append_token 后检查，这样终止 token 也会包含在
            # completion_token_ids 中。
            stop_due_to_eos = not seq.ignore_eos and token_id == self.eos
            stop_due_to_max_tokens = seq.num_completion_tokens >= seq.max_tokens
            stop_due_to_max_length = seq.max_model_length is not None and seq.num_tokens >= seq.max_model_length

            if stop_due_to_eos or stop_due_to_max_tokens or stop_due_to_max_length:
                # finished 序列不再需要 KV cache，也必须从 running 中移除，
                # 这样 is_finished() 最终才能变成 True。
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
