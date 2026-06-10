"""RMSNorm layer with optional fused residual add."""

import torch
import time 

class LayerNorm(torch.nn.Module):
    """Root-mean-square normalization used by Llama/Qwen blocks."""

    def __init__(self, gamma: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        # Store gamma as a parameter so checkpoint loading can copy directly into
        # the model and optimizers would see it if training were added later.
        self.weight = torch.nn.Parameter(gamma.detach().clone())
        # eps prevents division by zero when the hidden state is near all zeros.
        self.eps = eps

    @property
    def gamma(self):
        """Backward compatibility alias for callers that still use gamma."""
        return self.weight

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm normalizes each token independently across the last dimension:
        # x / sqrt(mean(x^2) + eps), then scales by the learned weight.
        variance = x.pow(2).mean(dim=-1, keepdim=True) + self.eps
        sqrt_variance = variance.sqrt()
        x_norm = (x / sqrt_variance * self.weight)

        return x_norm

    def residual_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # Transformer blocks commonly add the previous residual before applying
        # the next normalization.  Return both normalized x and updated residual
        # so the caller can reuse the residual in the next block.
        x = x + residual
        return self.rms_forward(x), x

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        # If residual is omitted, this behaves like a plain RMSNorm layer.
        if residual is not None:
            return self.residual_rms_forward(x, residual)
        else:
            return self.rms_forward(x)

if __name__ == "__main__":
    # Local microbenchmark for standalone and residual-fused RMSNorm.
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
    
