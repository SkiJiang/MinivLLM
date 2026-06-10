"""Rotary positional embeddings for attention Q/K tensors."""

import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to either varlen or batched attention tensors."""
    # The project uses 3D tensors for concatenated variable-length prefill and
    # 4D tensors for batched shapes.  Both layouts keep head_dim at the end.
    if x.dim() == 3:
        # Varlen mode: (total_tokens, num_heads, head_dim).
        total_tokens, num_heads, head_dim = x.shape
        # Expand cos/sin across the head dimension.
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # RoPE treats the hidden dimension as two halves that form rotation pairs.
        x1, x2 = x.chunk(2, dim=-1)

        # [x1, x2] rotated by angle theta:
        # out1 = x1*cos - x2*sin, out2 = x1*sin + x2*cos.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # Batched mode: (B, seq_len, num_heads, head_dim).
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # Expand cos/sin across batch and head dimensions.
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        x1, x2 = x.chunk(2, dim=-1)

        # The same rotation formula broadcasts over B and num_heads.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    """Precompute and serve cosine/sine RoPE tables."""

    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # the following params are only used in llama3.2
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        # base controls the frequency ladder.  Larger values make rotations vary
        # more slowly across positions and support longer contexts.
        self.base = base
        # Only the first rotary_embedding dimensions of each head are rotated.
        self.rotary_embedding = rotary_embedding
        # The cache must cover the largest position index used by generation.
        self.max_position = max_position

        # inv_freq[j] = 1 / base^(2j / rotary_dim).  Each pair of hidden dims
        # receives a different angular frequency.
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # Llama 3.x rescales low-frequency RoPE components to extend context
            # length while preserving high-frequency behavior.
            import math
            inv_freq = self.inv_freq
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # Hard cutoff: frequencies with long wavelengths are divided by
                # factor, shorter wavelengths remain unchanged.
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                # Smoothly interpolate between unchanged and scaled frequencies
                # across the configured wavelength band.
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        # positions is [0, 1, ..., max_position-1].  freqs[p, j] is the angle
        # for position p and frequency j.
        positions = torch.arange(self.max_position).float()
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # Store cos and sin together so forward() performs one indexed gather.
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # Buffers move with the module across devices but are not trainable.
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    def forward(self, positions, query, key):
        """Rotate query and key tensors at the provided token positions."""
        # positions may be one index per token in varlen prefill or one index per
        # sequence in decode.
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    # Small arithmetic probe for checking the frequency table construction.
    base = 5
    rotary_dim = 16
    max_position = 100
    print(torch.arange(0, rotary_dim, 2))
    print(base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
    print(inv_freq)

    t = torch.arange(max_position).float()

    freqs = torch.einsum("i,j -> ij", t, inv_freq)

    print(freqs.size())

    print(freqs[2])

