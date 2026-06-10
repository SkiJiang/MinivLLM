"""用于 attention Q/K tensor 的旋转位置编码。"""

import torch.nn as nn
import torch 

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """对 varlen 或 batched attention tensor 应用 RoPE 旋转。"""
    # 本项目在拼接变长 prefill 中使用 3D tensor，在 batched shape 中使用 4D tensor。
    # 两种布局都把 head_dim 放在最后一维。
    if x.dim() == 3:
        # varlen 模式：(total_tokens, num_heads, head_dim)。
        total_tokens, num_heads, head_dim = x.shape
        # 将 cos/sin 扩展到 head 维度上广播。
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # RoPE 将隐藏维拆成两半，并把它们组成旋转对。
        x1, x2 = x.chunk(2, dim=-1)

        # [x1, x2] 按角度 theta 旋转：
        # out1 = x1*cos - x2*sin, out2 = x1*sin + x2*cos.
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)
    else:
        # batched 模式：(B, seq_len, num_heads, head_dim)。
        B = x.size(0)
        seq_len = x.size(1)
        num_heads = x.size(2)
        head_dim = x.size(-1)

        # 将 cos/sin 扩展到 batch 和 head 维度上广播。
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        x1, x2 = x.chunk(2, dim=-1)

        # 同一个旋转公式会在 B 和 num_heads 上广播。
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos

        return torch.cat([out1, out2], dim=-1)


class RotaryEmbedding(nn.Module):
    """预计算并提供 RoPE 的 cos/sin 表。"""

    def __init__(
        self, 
        base:int,
        rotary_embedding: int, 
        max_position: int = 2048,
        is_llama3: bool = False,
        # 以下参数只在 Llama 3.2 的 RoPE 缩放中使用。
        llama3_rope_factor: float = 32.0,
        llama3_rope_high_freq_factor: float = 4.0,
        llama3_rope_low_freq_factor: float = 1.0,
        llama3_rope_original_max_position_embeddings: int = 8192,
    ):
        super().__init__()
        # base 控制频率阶梯。数值越大，旋转随位置变化越慢，也更适合长上下文。
        self.base = base
        # 每个 head 只有前 rotary_embedding 个维度参与旋转。
        self.rotary_embedding = rotary_embedding
        # cache 必须覆盖生成过程中可能使用到的最大 position 下标。
        self.max_position = max_position

        # inv_freq[j] = 1 / base^(2j / rotary_dim)。每一对隐藏维使用不同角频率。
        self.inv_freq = 1/(base ** (torch.arange(0, self.rotary_embedding, 2)/self.rotary_embedding))

        if is_llama3:
            # Llama 3.x 会重新缩放低频 RoPE 分量，以扩展上下文长度，
            # 同时尽量保留高频行为。
            import math
            inv_freq = self.inv_freq
            wave_len = 2 * math.pi / inv_freq
            if llama3_rope_low_freq_factor == llama3_rope_high_freq_factor:
                # 硬截断：长波长频率除以 factor，短波长频率保持不变。
                inv_freq = torch.where(
                    wave_len < llama3_rope_original_max_position_embeddings / llama3_rope_high_freq_factor,
                    inv_freq,
                    inv_freq / llama3_rope_factor,
                )
            else:
                # 在配置的波长区间内，在“不缩放”和“缩放”之间做平滑插值。
                delta = llama3_rope_high_freq_factor - llama3_rope_low_freq_factor
                smooth = (llama3_rope_original_max_position_embeddings / wave_len - llama3_rope_low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                factor = (1 - smooth) / llama3_rope_factor + smooth
                inv_freq = factor * inv_freq
            self.inv_freq = inv_freq

        # positions 是 [0, 1, ..., max_position-1]。
        # freqs[p, j] 表示位置 p 在频率 j 上的角度。
        positions = torch.arange(self.max_position).float()
        freqs = torch.einsum("i,j -> ij", positions, self.inv_freq)

        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        # 将 cos 和 sin 存在一起，使 forward() 只需要一次 indexed gather。
        cos_sin_cache = torch.cat([cos, sin], dim=-1)
        # buffer 会跟随 module 跨设备移动，但不会作为可训练参数。
        self.register_buffer("cos_sin_cache", cos_sin_cache)

    @torch.compile
    def forward(self, positions, query, key):
        """在给定 token 位置上旋转 query 和 key tensor。"""
        # positions 在 varlen prefill 中可能是每个 token 一个下标，
        # 在 decode 中则可能是每个序列一个下标。
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return (
            apply_rotary_pos_emb(query, cos, sin),
            apply_rotary_pos_emb(key, cos, sin)
        )


if __name__ == "__main__":
    # 小型算术探针：用于检查频率表构造是否符合预期。
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

