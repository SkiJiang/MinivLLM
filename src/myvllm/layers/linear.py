"""Tensor-parallel linear layers and checkpoint shard loaders."""

import torch.nn as nn 
import torch
import torch.distributed as dist

class LinearBase(nn.Module):
    """
    Base class that attaches custom weight loading to parameters.

    The model is constructed directly in tensor-parallel shape.  Full checkpoint
    tensors therefore cannot always be copied verbatim; each subclass implements
    weight_loader() to extract the local shard that belongs to this rank.
    """

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True,
        tp_dim: int | None = None
    ):
        super().__init__()
        # tp_dim records which dimension is sharded: output rows for column
        # parallel, input columns for row parallel, or None for replicated.
        self.tp_dim = tp_dim 
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # Weight is stored in the local shape used by this rank.
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # The loader utility checks this attribute and delegates copying here.
        self.weight.weight_loader = self.weight_loader

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
            # Bias follows the same sharding dimension as weight rows.
            self.bias.weight_loader = self.weight_loader 
        else:
            self.register_parameter('bias', None)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """Copy the relevant checkpoint slice into param."""
        raise NotImplementedError("Subclasses should implement this method.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear operation for this rank."""
        raise NotImplementedError("Subclasses should implement this method.")

"""
These loaders support the common inference flow:
1. Build a randomly initialized tensor-parallel model on each GPU.
2. Read a full checkpoint tensor from disk.
3. Let each parameter's weight_loader copy only the shard this GPU owns.

for name, param in model.named_parameters():
    if name in checkpoint:
        loaded_weight = checkpoint[name]  # full model parameter (4096, 4096)
        
        # check if the parameter has a custom weight_loader
        if hasattr(param, 'weight_loader'):
            # call custom weight_loader
            param.weight_loader(param, loaded_weight)
            # weight_loader will automatically:
            # 1. extract the shard corresponding to the current GPU
            # 2. copy it to param.data
        else:
            # default: copy directly
            param.data.copy_(loaded_weight)
"""

class ReplicatedLinear(LinearBase):
    """Linear layer copied identically onto every tensor-parallel rank."""

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        bias: bool = True
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        # Replicated weights have the same shape on every rank.
        param.data.copy_(loaded_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No communication is needed because every rank has the full weight.
        return nn.functional.linear(x, self.weight, self.bias)

class ColumnParallelLinear(LinearBase):
    """Shard output features across ranks.

    A full weight of shape (output_size, input_size) is split along dim 0.  Each
    rank produces a slice of the output features.
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
        """Load this rank's output-feature shard from a full tensor."""
        param_data = param.data 
        full_data_output_size = loaded_weights.size(0)
        shard_size = full_data_output_size // self.tp_size
        assert shard_size == param_data.size(0), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(0, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The returned tensor contains only this rank's output-feature shard.
        return nn.functional.linear(x, self.weight, self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel layer that packs several projections into one matrix."""

    def __init__(
        self, 
        input_size: int, 
        output_sizes: list[int],
        bias: bool = True,
    ):
        # output_sizes stores the full, unsharded size for each packed segment,
        # for example [gate_size, up_size] in a SwiGLU MLP.
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, loaded_weight_id: int):
        """
        checkpoint = {
            'q_proj.weight': torch.randn(4096, 4096),  
            'k_proj.weight': torch.randn(4096, 4096),
            'v_proj.weight': torch.randn(4096, 4096),
        }
        load to 
        merged_layer = Linear(
            input_size=4096,
            output_sizes=sum([4096, 4096, 4096]),  # Q, K, V
        ) which is also sharded by tp_size
        """
        param_data = param.data
        # Offset is measured in the local packed parameter after sharding.
        offset = sum(self.output_sizes[:loaded_weight_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_weight_id] // self.tp_size

        # Narrow to the packed segment that corresponds to loaded_weight_id.
        param_data = param_data.narrow(0, offset, shard_size)

        # Then copy this rank's shard from the full checkpoint segment.
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)
        param_data.copy_(shard_weights)


class QKVColumnParallelLinear(ColumnParallelLinear):
    """Fused QKV projection with per-rank query/key/value shards."""

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
        # Store local head counts because Q/K/V segments are sharded by heads.
        self.num_heads = num_heads // self.tp_size
        self.num_kv_heads = num_kv_heads // self.tp_size
        # Per-rank output is [Q heads, K heads, V heads] packed together.
        self.output_size = head_size * (self.num_heads + 2 * self.num_kv_heads)
        # Parent divides the full output size by tp_size to allocate local weight.
        total_output_size = head_size * (num_heads + 2 * num_kv_heads)
        super().__init__(input_size, total_output_size, bias=bias)

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor, load_weight_id: str):
        """Load one of q/k/v checkpoint tensors into the fused local parameter."""
        param_data = param.data
        assert load_weight_id in ['q', 'k', 'v'], "load_weight_id must be one of 'q', 'k', 'v'"
        # The fused local matrix layout is Q segment, then K segment, then V
        # segment.  Compute the destination slice for the requested component.
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
        # Copy the rank-local head shard from the full q/k/v tensor.
        loaded_weights_start_index = self.tp_rank * shard_size
        shard_weights = loaded_weights.narrow(0, loaded_weights_start_index, shard_size)

        param_data.copy_(shard_weights)


class RowParallelLinear(LinearBase):
    """Shard input features across ranks and all-reduce output sums."""

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
        """Load this rank's input-feature shard from a full tensor."""
        param_data = param.data 
        full_data_input_size = loaded_weights.size(1)
        shard_size = full_data_input_size // self.tp_size
        assert shard_size == param_data.size(1), "Shard size does not match parameter size."
        start_index = self.tp_rank * shard_size
        slided_weight = loaded_weights.narrow(1, start_index, shard_size)
        param_data.copy_(slided_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each rank computes its partial matrix product over local input columns.
        result = nn.functional.linear(x, self.weight, self.bias)
        if self.tp_size > 1:
            # Summing partial products reconstructs the full output on every rank.
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


if __name__ == "__main__":
    # Minimal smoke check for process-group initialization and construction.
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29500",
            rank=0,
            world_size=1,
        )
    layer = LinearBase(input_size=10, output_size=5)
    print("LinearBase layer initialized:", layer)
