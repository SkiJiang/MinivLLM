"""张量并行 Linear 层以及 checkpoint 分片加载器。"""

import torch.nn as nn 
import torch
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    给参数挂载自定义权重加载逻辑的基类。

    模型会直接以张量并行后的形状构造，因此完整 checkpoint tensor 不能总是原样拷贝；
    每个子类通过实现 weight_loader()，抽取属于当前 rank 的本地分片。
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # tp_dim 记录哪个维度被切分：column parallel 切输出行，row parallel 切输入列，
        # replicated 则为 None。
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # weight 以当前 rank 使用的本地形状保存。
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # loader 工具会检查这个属性，并把拷贝逻辑委托给这里。
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            # bias 沿用 weight 行方向的切分方式。
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """将 checkpoint 中相关切片拷贝到 param。"""
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行当前 rank 对应的 linear 操作。"""
        raise NotImplementedError("Subclasses should implement this method.")

"""
这些 loader 支持常见推理流程：
1. 在每张 GPU 上构造随机初始化的张量并行模型。
2. 从磁盘读取完整 checkpoint tensor。
3. 让每个参数自己的 weight_loader 只拷贝当前 GPU 拥有的分片。

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # 完整模型参数 (4096, 4096)
        
        # 检查参数是否有自定义 weight_loader
        if hasattr(param, 'weight_loader'):
            # 调用自定义 weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader 会自动：
            # 1. 抽取当前 GPU 对应的分片
            # 2. 将它拷贝到 param.data
        else:
            # 默认情况：直接拷贝
            param.data.copy_(loaded_weight)
"""

class ReplicatedLinear(LinearBase):
    """在每个张量并行 rank 上完整复制的 Linear 层。"""

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # replicated 权重在每个 rank 上形状相同。
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 rank 都有完整权重，因此不需要通信。
        return nn.functional.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    """沿输出特征维在多个 rank 间切分。

    完整权重形状为 (output_size, input_size)，沿 dim 0 切分。
    每个 rank 只产生输出特征中的一片。
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert output_size % tp_size == 0, "Output size must be divisible by tensor parallel size."
        super().__init__(input_size, output_size//tp_size, bias, tp_dim=0)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """从完整 tensor 中加载当前 rank 的输出特征分片。"""
        param_data = param.data 
        full_data_output_size = loaded_weights.size(0)
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 返回 tensor 只包含当前 rank 的输出特征分片。
        return nn.functional.linear(x, self.weight, self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):
    """将多个投影打包到同一个矩阵中的 column-parallel 层。"""

    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int],
        bias: bool = True,
    ):
        # output_sizes 保存每个被打包片段的完整未切分大小；
        # 例如 SwiGLU MLP 中的 [gate_size, up_size]。
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        加载到
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        )，并且它同样会被 tp_size 切分
        """
        param_data = param.data
        # offset 按切分后的本地 packed 参数计量。
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size

        # 定位到 loaded_weight_id 对应的 packed 片段。
        param_data = param_data.narrow(0, offset, shard_size)

        # 再从完整 checkpoint 片段中拷贝当前 rank 的分片。
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    """融合 QKV 投影，每个 rank 保存自己的 Q/K/V head 分片。"""

    def __init__(
        self,
        input_size: int,
        head_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.tp_size = dist.get_world_size()
        num_kv_heads = num_kv_heads or num_heads
        self.head_size = head_size
        # 保存本地 head 数，因为 Q/K/V 片段按 head 切分。
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # 每个 rank 的输出按 [Q heads, K heads, V heads] 打包。
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # 父类会把完整输出大小除以 tp_size，以分配本地权重。
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        """将 q/k/v 中某个 checkpoint tensor 加载到融合后的本地参数中。"""
        param_data = param.data
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # 融合本地矩阵布局依次是 Q 片段、K 片段、V 片段；
        # 这里为请求的组件计算目标切片。
        if load_weight_id == 'q':
            offset = 0
            shard_size = self.head_size * self.num_heads
        elif load_weight_id == 'k':
            offset = self.head_size * self.num_heads
            shard_size = self.head_size * self.num_kv_heads
        elif load_weight_id == 'v':
            offset = self.head_size * self.num_heads + self.head_size * self.num_kv_heads
            shard_size = self.head_size * self.num_kv_heads
        else:
            raise ValueError(f"Unknown load_weight_id: {load_weight_id}")

        param_data = param_data.narrow(0, offset, shard_size)
        # 从完整 q/k/v tensor 中拷贝当前 rank 的 head 分片。
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    """沿输入特征维切分，并对输出部分和做 all-reduce。"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
    ):
        tp_size = dist.get_world_size()
        assert input_size % tp_size == 0, "Input size must be divisible by tensor parallel size."
        super().__init__(input_size // tp_size, output_size, bias, tp_dim=1)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """从完整 tensor 中加载当前 rank 的输入特征分片。"""
        param_data = param.data 
        full_data_input_size = loaded_weights.size(1)
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 每个 rank 只基于本地输入列计算部分矩阵乘。
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # 对各 rank 的部分结果求和，在每个 rank 上重建完整输出。
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


if __name__ == "__main__":
    # 最小 smoke test：检查进程组初始化和层构造是否可用。
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)
