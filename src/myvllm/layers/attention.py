"""Attention kernel，以及负责选择 prefill/decode 模式的 nn.Module 封装。"""

import triton 
import triton.language as tl
from myvllm.utils import get_context
import torch
import torch.nn as nn

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr
):
    """
    将 key 和 value 写入 paged KV cache。

    每个 token 都通过 slot_mapping 映射到一个 cache slot。
    Grid 布局：(num_tokens, num_kv_heads)
    Cache 布局：(num_blocks, block_size, num_kv_heads, head_dim)
    """
    # program_id(0) 选择 token 行；program_id(1) 选择 KV head。
    token_idx = tl.program_id(0)
    # slot_mapping 将 token 行映射到扁平化物理 cache slot。
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        # -1 表示 padding 或已经 cache、无需写入的 token。
        return
    
    # 将扁平 slot id 转成 block id 和 block 内 offset。
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    head_idx = tl.program_id(1)
    
    # 每个 lane 对应一个 head 内的标量维度。
    head_offsets = tl.arange(0, head_dim)

    # 输入布局是连续的 (num_tokens, num_kv_heads, head_dim)。
    input_offset = (token_idx * num_kv_heads * head_dim + # 跳过前面的 token
                    head_idx * head_dim + # 跳过前面的 head
                    head_offsets)

    # cache 布局是连续的 (num_blocks, block_size, num_kv_heads, head_dim)。
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # 跳过前面的 block
                   block_offset * num_kv_heads * head_dim + # 跳过 block 内前面的位置
                   head_idx * head_dim + # 跳过前面的 head
                   head_offsets) 
    
    # 读取当前 token 的 key/value 向量，并存入 slot_mapping 选中的物理 cache 页。
    key = tl.load(key_ptr + input_offset)
    value = tl.load(value_ptr + input_offset)
    
    tl.store(k_cache_ptr + cache_offset, key)
    tl.store(v_cache_ptr + cache_offset, value)


def store_kvcache(
    key: torch.Tensor, 
    value: torch.Tensor, 
    k_cache: torch.Tensor, 
    v_cache: torch.Tensor, 
    slot_mapping: torch.Tensor,
    block_size: int
):
    """
    将 key-value 对写入 paged cache。
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - 将每个 token 映射到 cache slot
        block_size: 每个 block 包含的 token 数
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Triton 指针算术假设 tensor 是稠密连续的。
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    # 一个 Triton program 存储一个 token/head 对。
    grid = (num_tokens, num_kv_heads)
    store_kvcache_kernel[grid](
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size
    )


@triton.jit
def flash_attention_varlen_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    面向变长序列的 Flash Attention kernel。

    每个 program 处理某个序列、某个 head 上的一块 query。
    """
    # Grid 轴：query tile、query head、sequence。
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # GQA 会把多个 query head 映射到同一个 KV head。
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # cu_seqlens_q 保存拼接 token tensor 中各序列的累积边界。
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if start_m * BLOCK_M >= seq_len:
        return
    
    # 当前 tile 的 query 行 offset，以及 hidden 维向量 lane。
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Q 布局：(total_tokens, num_heads, head_dim)。
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # 无效行用 0 填充；mask 会阻止它们后续写回。
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # online softmax 状态：m_i 是行最大值，l_i 是行归一化因子，
    # acc 是 sum(exp(score - m_i) * V)。
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # 遍历当前序列的所有 key/value tile。
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        mask_n = offs_n < seq_len
        
        # K 以转置形式 (head_dim, BLOCK_N) 读取，便于 tl.dot(q, k)。
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # 当前 Q tile 与当前 K tile 之间的 attention score。
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # causal mask 防止 query token 关注未来 token。
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # 用新 tile 更新 online softmax，同时在跨 tile 处理时保持数值稳定。
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        # V 布局：(total_tokens, num_kv_heads, head_dim)。
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # 除以最终 softmax 分母。
    acc = acc / l_i[:, None]
    
    # O 布局与 Q 一致：(total_tokens, num_heads, head_dim)。
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """
    prefill 阶段用于变长序列的优化版 Flash Attention。
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: 累积序列长度
        scale: attention 缩放系数
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # kernel 使用显式指针算术，因此要求稠密布局。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # output 形状跟随 q，因为每个 query token/head 都会得到一个向量。
    output = torch.empty_like(q)
    
    # 保守的 block size 让不同 head_dim 下每个 program 的临时存储都保持可控。
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # cu_seqlens 多一个边界元素，因此序列数是 len - 1。
    num_seqs = cu_seqlens.shape[0] - 1
    
    # grid 的 query-tile 维度由 batch 中最长序列决定。
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # 一次 kernel 调用启动所有 sequence/head/query tile。
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_varlen_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    return output


@triton.jit
def paged_attention_decode_kernel(
    output_ptr,
    query_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_size: tl.constexpr,
    max_num_blocks: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    decode 阶段的优化 paged attention kernel。

    以 chunk 为单位遍历 KV cache。
    """
    # 一个 program 计算一个 (batch item, query head) 输出向量。
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # 使用 GQA/MQA 时，将多个 query head 分组映射到更少的 KV head。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # context_len 包含当前 decode token。
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # 读取当前 token 的 query 向量。
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # 单个 query 向量的 online softmax 状态。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # 迭代足够多的 chunk 来覆盖 padding 后的 block table。
    # 无效位置会被 context_len 和 physical_block_idx 检查跳过。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    for chunk_idx in range(max_chunks):
        # 当前 chunk 起点对应的逻辑 token 下标。
        token_start = chunk_idx * BLOCK_N
        
        if token_start < context_len:
            # 序列内逻辑 token 位置的有效 mask。
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            
          
            # 先把无效 score 初始化为很大的负数，再逐个填入有效 token 位置的 score。
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            
            # 通过 block_tables 将每个逻辑 token 位置解析到存储其 key 向量的物理页。
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # block_tables[batch_idx, block_num] 将逻辑 block 编号映射到
                        # 物理 cache block id。
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # K cache 布局：(num_blocks, block_size, num_kv_heads, head_dim)。
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
                            score = tl.sum(q * k_vec) * scale
                            
                            # 将标量 score 写入 qk[i]。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            # softmax 前 mask 掉 chunk 中的 padding 项。
            qk = tl.where(mask_n, qk, -1e10)
            
            # 用当前 chunk 更新 online softmax。
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            # 将之前 chunk 的贡献缩放到新的最大值参考系下。
            acc = acc * alpha
            l_i = l_i * alpha
            
            # 从同一批物理页读取 value 向量，并累加加权和。
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            
                            # 用 one-hot mask 从向量中取出 p[i]。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    # 用 softmax 分母归一化累积加权值。
    output = acc / l_i
    
    # 为当前 batch item 和 query head 写回一个输出向量。
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int
) -> torch.Tensor:
    """
    使用 paged KV cache 计算 decode 模式下的 attention。
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention 缩放系数
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # query 来自投影和 reshape，可能不是连续内存。
    query = query.contiguous()
    
    # 形状与 query 一致：每个 batch item/head 一个输出向量。
    output = torch.empty_like(query)
    
    # head 越宽，使用越小的 chunk 以降低临时存储压力。
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    # 每个 batch/head 输出向量对应一个 program。
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output,
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        scale=scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        max_num_blocks=max_num_blocks,
        BLOCK_N=BLOCK_N,
    )
    
    return output


class Attention(nn.Module):
    """保存 KV cache，并根据上下文分发 prefill/decode kernel 的 attention 封装。"""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        # num_heads 是 query head 数；GQA 下 num_kv_heads 可能更小。
        self.num_heads = num_heads
        self.head_dim = head_dim
        # scale 是模型特定倍率；forward 中还会除以 sqrt(d)。
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        # ModelRunner.allocate_kv_cache 会把这些空 tensor 替换成全局 KV-cache 池中的
        # layer view。
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """使用 ModelRunner 准备的全局 context 运行 attention。"""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # 每次 forward 都会把新计算的 K/V 写入 paged cache。
        # prefill 会写多个 token；decode 每个序列只写一个 token。
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                # batched tensor 会被展平成 varlen cache 写入布局。
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # varlen prefill 已经是 (num_tokens, num_kv_heads, head_dim)。
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # 标准 attention 缩放，再乘上可选的模型特定因子。
        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # prefill 会对未被 prefix cache 跳过的 prompt token 计算完整 causal attention。
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # 输出：(total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # decode 从 paged cache 中读取所有历史 K/V，并为每个序列计算一个输出向量。
            o = paged_attention_decode(
                q, 
                k_cache, 
                v_cache,
                context.block_tables,
                context.context_lens,
                scale,
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                self.block_size
            )
            # o：(batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # 本地计时脚手架：用于手动 attention 层实验。
    layer = Attention(num_heads=8, head_dim=64).cuda()
    B, N, D = 4, 1024, 512
    q = torch.randn(B, N, D).cuda()
    k = torch.randn(B, N, D).cuda()
    v = torch.randn(B, N, D).cuda()
    layer.k_cache = torch.zeros(B, N, D).cuda()
    layer.v_cache = torch.zeros(B, N, D).cuda()
    slot_mapping = torch.arange(N).cuda()

    for _ in range(10):
        _ = layer(q, k, v)

    import time
    times = []
    for _ in range(100):
        torch.cuda.synchronize()
        start_time = time.time()
        output_tensor = layer(q, k, v)
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    print(f"Average inference time over 100 runs: {avg_time * 1000:.4f} ms")
