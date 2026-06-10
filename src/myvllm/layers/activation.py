"""Activation functions used by transformer feed-forward blocks."""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    SwiGLU-style activation used after a fused gate/up projection.

    The input's last dimension is expected to be twice the intermediate size.
    The first half is the gate branch, the second half is the value/up branch:
    output = silu(gate) * value.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split only on the hidden dimension so all leading dimensions
        # (batch/sequence or varlen tokens) are preserved unchanged.
        x, y = x.chunk(2, -1)
        # SiLU(gate) controls how much of the value branch passes through.
        return F.silu(x) * y

if __name__ == "__main__":
    # Local microbenchmark for the fused activation shape used by large MLPs.
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()
    
    # Warm-up avoids including one-time CUDA kernel setup in timing.
    for _ in range(10):
        _ = layer(input_tensor)

    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(input_tensor)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
