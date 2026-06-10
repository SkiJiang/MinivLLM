"""Token 采样工具。"""

import torch 
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    对 logits 的每一行按 temperature 缩放后采样一个 token。

    这里使用指数形式的 Gumbel-max 技巧：
    argmax(p / Exp(1)) 可以在不调用 torch.multinomial 的情况下，
    按类别概率 p 进行采样。
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # temperature 是逐序列的，unsqueeze 后可以广播到词表维度。
        logits/= temperature.unsqueeze(-1)
        # 将缩放后的 logits 转成类别分布。
        probs = torch.softmax(logits, dim=-1)
        # 用指数噪声相除再取 argmax，等价于按 probs 采样，同时更容易被 torch.compile 处理。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
