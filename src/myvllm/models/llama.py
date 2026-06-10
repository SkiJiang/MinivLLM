"""基于本项目张量并行层实现的 Llama 模型组件。"""

from myvllm.layers import *

from typing import Tuple

import torch 
import torch.nn as nn

class LlamaAttn(nn.Module):
    """带融合 QKV 投影和 RoPE 的 Llama self-attention block。"""

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

        # total_* 表示 checkpoint/全局模型里的数量；num_* 表示张量并行后当前 rank 的本地分片数量。
        self.total_num_heads = num_qo_heads
        self.num_heads = num_qo_heads // self.tp_size

        self.total_num_kv_heads = num_kv_heads if num_kv_heads is not None else num_qo_heads
        self.num_kv_heads = self.total_num_kv_heads // self.tp_size

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_qo_heads

        # 将 q_proj/k_proj/v_proj 融合为一次 column-parallel 矩阵乘。
        self.qkv_projection = QKVColumnParallelLinear(
            input_size=hidden_size,
            head_size=head_dim,
            num_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            bias=has_attn_bias,
        )

        # 融合 qkv 输出在本地的拆分大小。
        self.q_size = head_dim * self.num_heads
        self.kv_size = head_dim * self.num_kv_heads
        
        # Llama 3.x 直接对 Q/K 应用 RoPE，不使用 Q/K RMSNorm。
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
        # row-parallel 输出投影会在多个 rank 间规约部分 head 输出。
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
        """对归一化后的 hidden state 应用一个 attention 子层。"""
        # column-parallel 投影前，x 在各个 rank 上是复制的。

        # column parallel 会切分输出 head，因此每个 rank 只得到自己的本地 Q head 和 KV head。
        qkv = self.qkv_projection(x)

        # 将本地 packed projection 拆成 Q、K、V 三段。
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # prefill 使用拼接后的 2D varlen tensor；CUDA graph decode 可以使用 batched tensor。
        # 两者都会 reshape 出 head 维度。
        if q.dim() == 2:
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
        else:
            B, N = q.size(0), q.size(1)
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_kv_heads, self.head_dim)
            v = v.view(B, N, self.num_kv_heads, self.head_dim)

        # RoPE 在计算 attention score 前将 token 位置信息注入 Q/K。
        q, k = self.rotary_emb(positions, q, k) 

        o = self.attention(q, k, v)

        # RowParallelLinear 会 all-reduce 部分投影结果，使每个 rank 都拿到复制后的 hidden_size 输出。
        o = self.o_proj(o)

        return o
    
class LlamaMLP(nn.Module):
    """使用融合 gate/up 投影的 Llama 前馈网络。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = True,
    ):
        super().__init__()
        # gate_up 用一次 column-parallel matmul 生成两个 intermediate 向量。
        self.gate_up = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
        )
        # SiluAndMul 在 packed 输出上实现 SwiGLU 非线性。
        self.activation = SiluAndMul()
        # down_proj 将切分后的 intermediate state 映射回 hidden_size。
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 gate/up 投影、激活和 down projection。"""
        x = self.down_proj(self.activation(self.gate_up(x)))
        return x

class LlamaDecoderLayer(nn.Module):
    """一个 pre-norm Llama decoder 层。"""

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
        # 本地 LayerNorm 类实际实现 RMSNorm。初始 gamma 与 checkpoint 形状匹配，
        # 会在权重加载时被覆盖。
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
        """带 residual 传递地执行 RMSNorm、attention、RMSNorm 和 MLP。"""
        if residual is not None:
            # 归一化前先加上上一条 residual，并返回更新后的 residual 供下一子层使用。
            x, residual = self.input_layernorm(x, residual)
        else:
            # 第一层还没有传入的 residual 对象。
            residual = x
            x = self.input_layernorm(x)

        # packed prefill batch 中，每个序列的 position id 都必须从 0 重新开始；
        # decode 则使用当前上下文长度减一。
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
        # attention 输出会在 post_attention_layernorm 内部加到 residual 上。
        x, residual = self.post_attention_layernorm(x, residual)
        x = self.mlp(x)
        return x, residual
    
class LlamaModel(nn.Module):
    """不包含最终 LM head 的 Transformer 主干。"""

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

        # token embedding 是词表并行的，最终返回复制到各 rank 的 hidden state。
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
        )
        # decoder 层堆叠；同一次 forward 中每层共享同一份 context 元数据。
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
        """将 token id 转成最终 hidden state。"""
        x = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            x, residual = layer(x, residual)
        x, _ = self.norm(x, residual)
        return x


class LlamaForCausalLM(nn.Module):
    """Llama 主干加词表投影，用于得到 next-token logits。"""

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
        # 主干只产生 hidden state；engine 准备采样时才由 compute_logits() 应用 lm_head。
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
            # 当 checkpoint 使用输入/输出 embedding 绑定时，复用 embedding 权重作为输出投影。
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """返回 hidden state 而非 logits，让 runner 决定哪些行需要采样。"""
        x = self.model(input_ids)
        return x 

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """将 hidden state 投影到词表 logits。"""
        logits = self.lm_head(hidden_states)
        return logits
