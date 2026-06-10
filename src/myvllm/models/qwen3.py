"""Qwen3 model components built from the local tensor-parallel layers."""

from myvllm.layers import *
import torch 
import torch.nn as nn

class Qwen3Attention(nn.Module):
    """Qwen3 attention with fused QKV, optional Q/K norm, RoPE, and paged attention."""

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

        # total_* values are global checkpoint counts; num_* values are local to
        # the current tensor-parallel rank.
        self.total_num_heads = num_heads
        self.num_heads = num_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.scale = scale

        # Fused QKV keeps three projections in one matrix multiply.  The custom
        # weight loader copies q_proj/k_proj/v_proj checkpoint tensors into the
        # correct packed slices.
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=self.total_num_heads,
            num_kv_heads=self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # Per-rank packed output split sizes.
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        self.qkv_bias = qkv_bias

        # Qwen3 uses RMSNorm on Q and K when QKV projection has no bias.
        self.q_norm = LayerNorm(torch.ones(head_dim))
        self.k_norm = LayerNorm(torch.ones(head_dim))

        # Standard Qwen RoPE table.
        self.rotary_emb = RotaryEmbedding(
            base=base,
            rotary_embedding=head_dim,
            max_position=max_position
        )

        # Attention handles both prefill flash attention and decode paged attention.
        self.attention = Attention(
            self.num_heads,
            head_dim,
            scale,
            self.num_kv_heads,
            block_size
        )

        # Output projection consumes local heads and all-reduces the full hidden
        # state across ranks.
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
        """Run the Qwen3 attention sublayer."""
        # x is replicated before the column-parallel projection.

        # Projection output contains only this rank's heads.
        qkv = self.qkv_projection(x)

        # Split packed local Q/K/V segments.
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Prefill can be a 2D varlen tensor; decode graph capture can use a
        # batched tensor.  Both expose (heads, head_dim) before attention.
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # Q/K normalization stabilizes dot-product magnitudes before softmax.
        if self.qkv_bias is False:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Kept import is harmless and useful for quick local diagnostics.
        import sys

        # RoPE adds positional phase to Q/K.
        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)

        # Row parallel output projection reconstructs hidden_size on every rank.
        o = self.o_proj(o)

        return o

class Qwen3MLP(nn.Module):
    """Qwen3 feed-forward network with SwiGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # Fused gate/up projection halves the number of matmul launches.
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        self.activation = SiluAndMul()
        # Down projection sums partial intermediate features across ranks.
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run gate/up, activation, and down projection."""
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x


class Qwen3DecoderLayer(nn.Module):
    """One Qwen3 decoder block with residual-carry RMSNorm."""

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
        # Gamma tensors are checkpoint-loadable RMSNorm weights.
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
        """Apply attention and MLP sublayers with residual state."""
        if residual is not None:
            x, residual = self.input_layernorm(x, residual)
        else:
            # First layer starts the residual stream.
            residual = x
            x = self.input_layernorm(x)

        # Packed prefill requires positions to restart for every sequence; decode
        # uses each sequence's current length minus one.
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
        # post_attention_layernorm adds attention output to residual internally.
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual

class Qwen3Model(nn.Module):
    """Qwen3 transformer backbone without the LM head."""

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
        # Vocab-parallel embedding returns a replicated hidden state after reduce.
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim = hidden_size
        )
        # Decoder stack with identical per-forward context metadata.
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
        """Run embeddings, decoder layers, and final RMSNorm."""
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 backbone plus vocabulary projection for next-token logits."""

    # Mapping documents how checkpoint names correspond to fused local modules.
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
        # Infer head_dim from hidden_size/num_heads if the config omits it.
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
            # Tie output projection to token embedding when the model config
            # expects shared weights.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hidden states; logits are computed separately for sampling."""
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        logits = self.lm_head(hidden_states)
        return logits

if __name__ == "__main__":
    # Lightweight construction smoke test for the Qwen3 module stack.
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
