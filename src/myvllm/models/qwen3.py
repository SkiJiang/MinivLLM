"""基于本项目张量并行层实现的 Qwen3 模型组件。"""

from myvllm.layers import *
import torch 
import torch.nn as nn

class Qwen3Attention(nn.Module):
    """包含融合 QKV、可选 Q/K norm、RoPE 和 paged attention 的 Qwen3 attention。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        # total_* 是全局 checkpoint 中的数量；num_* 是当前张量并行 rank 本地的数量。
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        # 融合 QKV 将三个投影合并为一次矩阵乘。自定义 weight loader 会把
        # q_proj/k_proj/v_proj 的 checkpoint tensor 拷贝到正确的 packed 切片。
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # 每个 rank 的 packed 输出拆分大小。
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # 当 QKV projection 没有 bias 时，Qwen3 会对 Q 和 K 使用 RMSNorm。
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        # 标准 Qwen RoPE 表。
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        # Attention 同时处理 prefill flash attention 和 decode paged attention。
        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        # 输出投影消费本地 head，并跨 rank all-reduce 得到完整 hidden state。
        self.o_proj = RowParallelLinear(
            input_size=head_dim * self.total_num_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """运行 Qwen3 attention 子层。"""
        # column-parallel 投影前，x 在各 rank 上是复制的。

        # projection 输出只包含当前 rank 的 head。
        qkv = self.qkv_projection(x)

        # 拆分本地 packed Q/K/V 片段。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # prefill 可以是 2D varlen tensor；decode graph capture 可以使用 batched tensor。
        # 两者在 attention 前都会显式展开 (heads, head_dim)。
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # Q/K normalization 会在 softmax 前稳定点积幅度。
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # 保留这个 import 不影响运行，也便于快速本地诊断。
        import sys

        # RoPE 为 Q/K 加入位置相位信息。
        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)

        # row parallel 输出投影会在每个 rank 上重建 hidden_size。
        o = self.o_proj(o)

        return o

class Qwen3MLP(nn.Module):
    """带 SwiGLU 激活的 Qwen3 前馈网络。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # 融合 gate/up 投影可以减少矩阵乘 launch 次数。
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        # down projection 会跨 rank 汇总部分 intermediate feature。
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 gate/up、激活和 down projection。"""
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
    """一个带 residual 传递 RMSNorm 的 Qwen3 decoder block。"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        block_size: int = 256,
    ):
        super().__init__()
        # gamma tensor 是可以从 checkpoint 加载的 RMSNorm 权重。
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma)
        self.self_attn = Qwen3Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            block_size=block_size,
        )
        self.post_attention_layernorm = LayerNorm(gamma)
        self.mlp = Qwen3MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        """带 residual 状态地执行 attention 和 MLP 子层。"""
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            # 第一层开启 residual 流。
            residual = x
            x = self.input_layernorm(x)

        # packed prefill 要求每个序列的 position 从 0 重新开始；
        # decode 使用每个序列当前长度减一。
        from myvllm.utils import get_context
        context = get_context()
        if context.is_prefill and context.cu_seqlens_q is not None:
            positions = []
            cu_seqlens = context.cu_seqlens_q.cpu().tolist()
            for i in range(len(cu_seqlens) - 1):
                seq_len = cu_seqlens[i+1] - cu_seqlens[i]
                positions.extend(range(seq_len))
            positions = torch.tensor(positions, dtype=torch.long, device=x.device)
        elif context.is_prefill:
            positions = torch.arange(x.size(0), device=x.device)
        else:
            positions = context.context_lens - 1

        x = self.self_attn(x, positions=positions)
        # post_attention_layernorm 内部会把 attention 输出加到 residual 上。
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual

class Qwen3Model(nn.Module):
    """不包含 LM head 的 Qwen3 Transformer 主干。"""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        block_size: int = 256,
    ):
        super().__init__()
        # 词表并行 embedding 在 reduce 后返回复制到各 rank 的 hidden state。
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        # decoder 层堆叠，同一次 forward 中共享相同 context 元数据。
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                scale=scale,
                num_kv_heads=num_kv_heads,
                rms_norm_epsilon=rms_norm_epsilon,
                qkv_bias=qkv_bias,
                base=base,
                max_position=max_position,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """执行 embedding、decoder 层和最终 RMSNorm。"""
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 主干加词表投影，用于得到 next-token logits。"""

    # 这个 mapping 记录 checkpoint 名称如何对应到本地融合模块。
    packed_module_mapping = {
        "q_proj": ('q_proj', 'q'),
        "k_proj": ('k_proj', 'k'),
        "v_proj": ('v_proj', 'v'),
        "gate_up": ('gate_up_proj', '0'),
        "gate_down": ('gate_down_proj', '1'),
    }
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        head_dim: int | None = None,
        scale: float = 1.0,
        num_kv_heads: int | None = None,
        rms_norm_epsilon: float = 1e-5,
        qkv_bias: bool = False,
        base: int = 10000,
        max_position: int = 16384,
        intermediate_size: int = 4 * 1024,
        ffn_bias: bool = True,
        num_layers: int = 12,
        tie_word_embeddings: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        # 如果配置里没有 head_dim，就根据 hidden_size/num_heads 推导。
        head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.model = Qwen3Model(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            rms_norm_epsilon=rms_norm_epsilon,
            qkv_bias=qkv_bias,
            base=base,
            max_position=max_position,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        if tie_word_embeddings:
            # 当模型配置要求共享权重时，将输出投影绑定到 token embedding。
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回 hidden state；logits 会在采样前单独计算。"""
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 hidden state 投影到词表 logits。"""
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    # 轻量构造 smoke test：检查 Qwen3 模块栈是否能实例化。
    model = Qwen3ForCausalLM(
        vocab_size=50257,
        hidden_size=768,
        num_heads=12,
        head_dim=64,
        intermediate_size=3072,
        num_layers=2,
    )
    input_ids = torch.randint(0, 50257, (2, 16)).cuda()
    output = model(input_ids)
