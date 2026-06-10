"""Token sampling utilities."""

import torch 
import torch.nn as nn


class SamplerLayer(nn.Module):
    """
    Sample one token per row of logits with temperature scaling.

    The implementation uses the Gumbel-max trick in exponential form:
    argmax(p / Exp(1)) samples from categorical probabilities p without calling
    torch.multinomial.
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
        # temperature is per sequence, so unsqueeze makes it broadcast across
        # the vocabulary dimension.
        logits/= temperature.unsqueeze(-1)
        # Convert scaled logits into a categorical distribution.
        probs = torch.softmax(logits, dim=-1)
        # Dividing by exponential noise and taking argmax is equivalent to
        # sampling according to probs, while staying easy for torch.compile.
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
