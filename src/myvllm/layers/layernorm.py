"""带可选残差相加融合路径的 RMSNorm 层。"""

import torch
import time 

class LayerNorm(torch.nn.Module):
    """Llama/Qwen block 中使用的均方根归一化层。"""

    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # 将 gamma 保存为 parameter，使 checkpoint 加载可以直接拷贝；
        # 如果后续加入训练，optimizer 也能看到它。
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        # hidden state 接近全 0 时，eps 可以避免除零。
        self.eps = eps

    @property
    def gamma(self):
        """向后兼容别名：给仍然使用 gamma 的调用方使用。"""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm 会沿最后一维独立归一化每个 token：
        # x / sqrt(mean(x^2) + eps)，再乘以可学习的 weight。
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Transformer block 通常会先加上上一条 residual，再做下一次归一化。
        # 这里同时返回归一化后的 x 和更新后的 residual，方便下一层继续复用。
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # 如果没有传 residual，这个层就退化成普通 RMSNorm。
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # 本地微基准：分别测试普通 RMSNorm 和融合 residual 的 RMSNorm。
    x = torch.randn(8,4000,8000).cuda()
    gamma = torch.full((8000,), 0.5, device="cuda", dtype=x.dtype)
    layer = LayerNorm(gamma=gamma).cuda()
    residual = torch.full_like(x,fill_value=1)

    for _ in range(10):
        _ = layer(x)
    
    times = [] 
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[Without residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")

    times.clear()
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = layer(x,residual)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"[With residuals] Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
    
