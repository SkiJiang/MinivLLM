"""Llama model components built from the local tensor-parallel layers."""

from myvllm.layers import *

from typing import Tuple

import torch 
import torch.nn as nn

class LlamaAttn(nn.Module):
    """Llama self-attention block with fused QKV projection and RoPE."""

    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-5,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        block_size: int = 256,
    ):
        super().__init__()
        self.tp_size = dist.get_world_size()

        # total_* values describe the checkpoint/global model.  num_* values
        # describe this rank's local shard after tensor parallelism.
        self.total_num_heads = num_qo_heads
        self.num_heads = num_qo_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_qo_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_qo_heads

        # Fuses q_proj/k_proj/v_proj into one column-parallel matrix multiply.
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            bias=has_attn_bias,
        )

        # Local split sizes for the fused qkv output.
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        
        # Llama 3.x applies RoPE directly to Q/K and does not use Q/K RMSNorm.
        self.rotary_emb = RotaryEmbedding(
            base=rope_base,
            rotary_embedding=head_dim,
            max_position=max_position_embeddings,
            is_llama3=True
        )
        self.attention = Attention(
            num_heads=num_qo_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
            block_size=block_size,
        )
        # Row-parallel output projection reduces partial head outputs across ranks.
        self.o_proj = RowParallelLinear(
            input_size= head_dim * num_qo_heads,
            output_size=hidden_size,
            bias=False,
        )

    def forward(
        self, 
        x: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one attention sublayer to normalized hidden states."""
        # x is replicated across ranks before the column-parallel projection.

        # Column parallelism shards output heads, so each rank receives only its
        # local Q heads and local KV heads.
        qkv = self.qkv_projection(x)

        # Split the packed local projection into Q, K, and V segments.
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Prefill uses a concatenated 2D varlen tensor; CUDA graph decode can use
        # a batched tensor.  Both are reshaped to expose head dimension.
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # RoPE injects token position into Q/K before attention scores are formed.
        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)

        # RowParallelLinear all-reduces partial projections so every rank receives
        # the replicated hidden_size output.
        o = self.o_proj(o)

        return o
    
class LlamaMLP(nn.Module):
    """Llama feed-forward network using fused gate/up projection."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # gate_up produces two intermediate vectors in one column-parallel matmul.
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        # SiluAndMul implements the SwiGLU nonlinearity over the packed output.
        self.activation = SiluAndMul()
        # down_proj maps the sharded intermediate state back to hidden_size.
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run gate/up projection, activation, and down projection."""
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x

class LlamaDecoderLayer(nn.Module):
    """One pre-norm Llama decoder layer."""

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-05,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        block_size: int = 256,
    ):
        super().__init__()
        # The local LayerNorm class implements RMSNorm.  Initial gamma matches
        # checkpoint shape and is overwritten during weight loading.
        gamma = torch.ones(hidden_size)
        self.input_layernorm = LayerNorm(gamma)
        self.self_attn = LlamaAttn(
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_attn_bias=has_attn_bias,
            rms_norm_epsilon=rms_norm_epsilon,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            block_size=block_size,
        )
        self.post_attention_layernorm = LayerNorm(gamma)
        self.mlp = LlamaMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=ffn_bias
        )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RMSNorm, attention, RMSNorm, and MLP with residual carry."""
        if residual is not None:
            # Add previous residual before normalizing, returning the updated
            # residual for the next sublayer.
            x, residual = self.input_layernorm(x, residual)
        else:
            # First layer has no incoming residual object yet.
            residual = x
            x = self.input_layernorm(x)

        # Position ids must restart at zero for each sequence in a packed prefill
        # batch, but decode uses the current context length minus one.
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
        # The attention output is added to residual inside post_attention_layernorm.
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual
    
class LlamaModel(nn.Module):
    """Backbone transformer without the final LM head."""

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        head_dim: int = 64,
        num_qo_heads: int = 32,
        num_kv_heads: int = 8,
        has_attn_bias: bool = False,
        rms_norm_epsilon: float = 1e-5,
        rope_base: int = 500000,
        max_position_embeddings: int = 131072,
        intermediate_size: int = 8192,
        ffn_bias: bool = False,
        num_layers: int = 16,
        block_size: int = 256,
    ):
        super().__init__()

        # Token embedding is vocab-parallel and returns replicated hidden states.
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        # Decoder stack; every layer shares the same context metadata for the
        # current forward pass.
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                head_dim=head_dim,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                has_attn_bias=has_attn_bias,
                rms_norm_epsilon=rms_norm_epsilon,
                rope_base=rope_base,
                max_position_embeddings=max_position_embeddings,
                intermediate_size=intermediate_size,
                ffn_bias=ffn_bias,
                block_size=block_size,
            ) for _ in range(num_layers)
        ])
        gamma = torch.ones(hidden_size)
        self.norm = LayerNorm(gamma)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Convert token ids into final hidden states."""
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class LlamaForCausalLM(nn.Module):
    """Llama backbone plus vocabulary projection for next-token logits."""

    def __init__(
            self,
            vocab_size: int = 128256,
            hidden_size: int = 2048,
            head_dim: int = 64,
            num_qo_heads: int = 32,
            num_kv_heads: int = 8,
            has_attn_bias: bool = False,
            rms_norm_epsilon: float = 1e-5,
            rope_base: int = 500000,
            max_position_embeddings: int = 131072,
            intermediate_size: int = 8192,
            ffn_bias: bool = False,
            num_layers: int = 16,
            block_size: int = 256,
            tie_word_embeddings: bool = True
        ):
        super().__init__()
        # The backbone produces hidden states.  compute_logits() applies lm_head
        # only when the engine is ready to sample.
        self.model = LlamaModel(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            has_attn_bias=has_attn_bias,
            rms_norm_epsilon=rms_norm_epsilon,
            rope_base=rope_base,
            max_position_embeddings=max_position_embeddings,
            intermediate_size=intermediate_size,
            ffn_bias=ffn_bias,
            num_layers=num_layers,
            block_size=block_size,
        )
        self.lm_head = ParallelLMHead(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        if tie_word_embeddings:
            # Reuse embedding weights for output projection when the checkpoint
            # uses tied input/output embeddings.
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return hidden states, not logits, to let the runner choose sampling rows."""
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states to vocabulary logits."""
        logits = self.lm_head(hidden_states)
        return logits
