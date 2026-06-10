"""Transformer 前馈网络中使用的激活函数。"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

class SiluAndMul(nn.Module):
    """
    融合 gate/up 投影之后使用的 SwiGLU 风格激活层。

    输入最后一维应当是 intermediate size 的两倍。前半部分是 gate 分支，
    后半部分是 value/up 分支：
    output = silu(gate) * value.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只沿隐藏维拆分，因此 batch/sequence 或 varlen token 等前置维度保持不变。
        x, y = x.chunk(2, -1)
        # SiLU(gate) 控制 value 分支有多少信息通过。
        return F.silu(x) * y

if __name__ == "__main__":
    # 本地微基准：测试大 MLP 中融合激活形状的执行时间。
    layer = SiluAndMul().cuda()
    input_tensor = torch.randn(8, 4000, 8000).cuda()
    
    # 预热可以避免把一次性的 CUDA kernel 初始化开销计入计时。
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
