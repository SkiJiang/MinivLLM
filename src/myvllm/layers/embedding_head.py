"""张量并行 token embedding 和语言模型输出 head 层。"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


class VocabParallelEmbedding(nn.Module):
    """按词表 id 维度切分的 embedding 表。

    每个 rank 拥有一段连续 token id。forward 时，不属于当前 rank 的 id 会在本地被
    mask 成 0，然后所有 rank 对局部 embedding 求和，得到完整 embedding。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # 保留原始词表大小，用于 logits 截断和 token 范围检查。
        self.num_embeddings = num_embeddings
        # 将全局词表 padding 到能被 tp_size 整除，使每个 rank 拥有相同行数。
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # 当前 rank 分配到的 embedding 行数。
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        # 参数形状是 local_vocab x hidden_size。
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 自定义 loader 知道如何从完整 checkpoint tensor 中只拷贝当前 rank 的分片。
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """从完整 embedding 权重中加载当前 rank 的词表分片。"""
        param_data = param.data

        # padding 后分配给当前 rank 的全局行范围。
        offset = self.tp_rank * self.num_embeddings_per_partition
        shard_size = self.num_embeddings_per_partition

        # padding 后的范围可能有一部分超出真实词表。
        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)

        if actual_size > 0:
            # 只从 checkpoint 拷贝真实存在的行。
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        # padding 行不应贡献 logits 或 embedding。
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """带词表并行 mask 的 embedding 查询。"""
        # mask 标识哪些 token id 属于当前 rank，并且不是 padding 区域。
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # 将全局 token id 转换成当前 rank 的局部行下标。被 mask 的条目会暂时变成 0，
        # 查询后还会再次清零。
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # 如果没有这次 mask，不属于当前 rank 的 token 会错误贡献第 0 行 embedding。
            output = mask.unsqueeze(1) * output
            # 对所有 rank 的局部 embedding 求和；每个真实 token id 只有一个 rank 贡献非零向量。
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

class ParallelLMHead(VocabParallelEmbedding):
    """词表并行输出投影，可选地与 embedding 权重绑定。"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 hidden state 投影成 logits，并在 rank 0 汇聚词表分片。"""
        context = get_context()
        if context.is_prefill:
            # prefill 阶段只需要每个序列最后一个 prompt token 的 logits；
            # 更早位置不会被采样。
            last_token = context.cu_seqlens_q[1:] - 1  # 排除第一个 0 元素
            x = x[last_token].contiguous()

        # 本地 logits 只覆盖当前 rank 的词表分片。
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # 只有 rank 0 需要完整 logits 来进行采样。
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, gather_list=all_logits, dst=0)
            if self.tp_rank == 0:
                # 拼接词表分片，并去掉 padding 行。
                logits = torch.cat(all_logits, dim=-1)
                logits = logits[..., :self.num_embeddings]

        return logits
