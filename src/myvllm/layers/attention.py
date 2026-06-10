"""Attention kernels and the nn.Module wrapper that selects prefill/decode mode."""

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
    Store keys and values into paged KV cache.

    Each token is mapped to a slot via slot_mapping.
    Grid layout: (num_tokens, num_kv_heads)
    Cache layout: (num_blocks, block_size, num_kv_heads, head_dim)
    """
    # program_id(0) selects the token row; program_id(1) selects the KV head.
    token_idx = tl.program_id(0)
    # slot_mapping maps token row -> flattened physical cache slot.
    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    
    if slot_idx == -1:
        # -1 marks padding or cached tokens that do not need a write.
        return
    
    # Convert flat slot id into block id and offset inside that block.
    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    
    head_idx = tl.program_id(1)
    
    # Vector lane for every scalar in one head.
    head_offsets = tl.arange(0, head_dim)

    # Input layout is contiguous (num_tokens, num_kv_heads, head_dim).
    input_offset = (token_idx * num_kv_heads * head_dim + # skip previous tokens
                    head_idx * head_dim + # skip previous heads
                    head_offsets)

    # Cache layout is contiguous (num_blocks, block_size, num_kv_heads, head_dim).
    cache_offset = (block_idx * block_size * num_kv_heads * head_dim + # skip previous blocks
                   block_offset * num_kv_heads * head_dim + # skip previous positions in block
                   head_idx * head_dim + # skip previous heads
                   head_offsets) 
    
    # Load the current token's key/value vector and store it into the physical
    # cache page selected by slot_mapping.
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
    Store key-value pairs into paged cache.
    
    Args:
        key: (num_tokens, num_kv_heads, head_dim)
        value: (num_tokens, num_kv_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        slot_mapping: (num_tokens,) - maps each token to a cache slot
        block_size: number of tokens per block
    """
    num_tokens, num_kv_heads, head_dim = key.shape
    
    # Triton pointer arithmetic assumes dense contiguous tensors.
    if not key.is_contiguous():
        key = key.contiguous()
    if not value.is_contiguous():
        value = value.contiguous()
    
    assert k_cache.shape == v_cache.shape, "K and V cache shapes must match"
    assert slot_mapping.numel() == num_tokens, "Slot mapping size must match number of tokens"
    
    # One Triton program stores one token/head pair.
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
    Flash Attention kernel for variable-length sequences.

    Each program processes one block of queries for one head in one sequence.
    """
    # Grid axes: query tile, query head, sequence.
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)

    # Grouped-query attention maps multiple query heads onto one KV head.
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # cu_seqlens_q stores cumulative boundaries in the concatenated token tensor.
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if start_m * BLOCK_M >= seq_len:
        return
    
    # Query row offsets for this tile and hidden-dimension vector lanes.
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    # Q layout: (total_tokens, num_heads, head_dim).
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    
    # Invalid rows are zero-filled; masks prevent them from being written later.
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # Online softmax state: m_i is row max, l_i is row normalizer, acc is
    # sum(exp(score - m_i) * V).
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    # Sweep all key/value tiles in this sequence.
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        
        mask_n = offs_n < seq_len
        
        # K is loaded transposed as (head_dim, BLOCK_N) for tl.dot(q, k).
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        # Attention scores for this Q tile against this K tile.
        qk = tl.dot(q, k)
        qk = qk * scale
        
        # Causal mask prevents each query token from attending to future tokens.
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # Update online softmax with the new tile while preserving numerical
        # stability across all processed tiles.
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        # V layout is (total_tokens, num_kv_heads, head_dim).
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    # Divide by the final softmax denominator.
    acc = acc / l_i[:, None]
    
    # O layout matches Q: (total_tokens, num_heads, head_dim).
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
    Optimized Flash Attention for prefill phase with variable-length sequences.
    
    Args:
        q: (total_tokens, num_heads, head_dim)
        k: (total_tokens, num_kv_heads, head_dim)
        v: (total_tokens, num_kv_heads, head_dim)
        cu_seqlens: cumulative sequence lengths
        scale: attention scale factor
    
    Returns:
        output: (total_tokens, num_heads, head_dim)
    """
    # Kernels use explicit pointer arithmetic, so require dense layouts.
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # Output shape follows q because every query token/head receives one vector.
    output = torch.empty_like(q)
    
    # Conservative block sizes keep per-program temporary storage reasonable for
    # different head dimensions.
    if head_dim <= 64:
        BLOCK_M = 64
        BLOCK_N = 64
    elif head_dim <= 128:
        BLOCK_M = 32
        BLOCK_N = 32
    else:
        BLOCK_M = 16
        BLOCK_N = 16
    
    # cu_seqlens has one extra boundary element, so sequence count is len - 1.
    num_seqs = cu_seqlens.shape[0] - 1
    
    # Grid's query-tile dimension is based on the longest sequence in the batch.
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # Launch all sequence/head/query tiles in one kernel call.
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
    Optimized paged attention kernel for decode phase.

    Processes KV cache in chunks.
    """
    # One program computes one (batch item, query head) output vector.
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # Group query heads onto fewer KV heads when using GQA/MQA.
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # context_len includes the current decode token.
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # Load the current token's query vector.
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Online softmax state for a single query vector.
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # Iterate enough chunks to cover the padded block table.  Invalid positions
    # are skipped by context_len and physical_block_idx checks.
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    for chunk_idx in range(max_chunks):
        # Logical token index at the start of this chunk.
        token_start = chunk_idx * BLOCK_N
        
        if token_start < context_len:
            # Valid mask for logical token positions inside the sequence.
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            
          
            # Initialize invalid scores to a large negative value before filling
            # valid token positions one by one.
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            
            # Resolve each logical token position through block_tables to find
            # the physical page that stores its key vector.
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # block_tables[batch_idx, block_num] maps logical block
                        # number to physical cache block id.
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # K cache layout: (num_blocks, block_size,
                            # num_kv_heads, head_dim).
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
                            score = tl.sum(q * k_vec) * scale
                            
                            # Write the scalar score into qk[i].
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            # Mask padded chunk entries before softmax.
            qk = tl.where(mask_n, qk, -1e10)
            
            # Online softmax update for this chunk.
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            # Rescale previous chunk contributions into the new max frame.
            acc = acc * alpha
            l_i = l_i * alpha
            
            # Load value vectors from the same physical pages and accumulate the
            # weighted sum.
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
                            
                            # Extract p[i] from the vector using a one-hot mask.
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    # Normalize accumulated weighted values by the softmax denominator.
    output = acc / l_i
    
    # Store one output vector for this batch item and query head.
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
    Compute attention in decode mode using paged KV cache.
    
    Args:
        query: (batch_size, num_heads, head_dim)
        k_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        v_cache: (num_blocks, block_size, num_kv_heads, head_dim)
        block_tables: (batch_size, max_num_blocks)
        context_lens: (batch_size,)
        scale: attention scale factor
    
    Returns:
        output: (batch_size, num_heads, head_dim)
    """
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    
    # Query comes from projection/reshape and may not be contiguous.
    query = query.contiguous()
    
    # Shape matches query: one output vector per batch item/head.
    output = torch.empty_like(query)
    
    # Smaller chunks for wider heads reduce temporary storage pressure.
    BLOCK_N = 64 if head_dim <= 128 else 32
    
    # One program per batch/head output vector.
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
    """Attention wrapper that stores KV cache and dispatches prefill/decode kernels."""

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float = 1.0,
        num_kv_heads: int = None,
        block_size: int = 16,
    ):
        super().__init__()
        # num_heads is query heads; num_kv_heads may be smaller for GQA.
        self.num_heads = num_heads
        self.head_dim = head_dim
        # scale is a model-specific multiplier; forward also divides by sqrt(d).
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.block_size = block_size
        # ModelRunner.allocate_kv_cache replaces these empty tensors with layer
        # views into the global KV-cache pool.
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Run attention using global context prepared by ModelRunner."""
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache

        # Every forward stores newly computed K/V into the paged cache.  Prefill
        # writes many tokens; decode writes exactly one token per sequence.
        if k_cache.numel() > 0 and v_cache.numel() > 0 and context.slot_mapping is not None:
            if k.dim() == 4:
                # Batched tensors are flattened to the varlen cache-write layout.
                B, N, num_kv_heads, head_dim = k.shape
                k_to_store = k.reshape(B * N, num_kv_heads, head_dim).contiguous()
                v_to_store = v.reshape(B * N, num_kv_heads, head_dim).contiguous()
            else:
                # Varlen prefill already uses (num_tokens, num_kv_heads, head_dim).
                k_to_store = k.contiguous()
                v_to_store = v.contiguous()
            
            store_kvcache(k_to_store, v_to_store, k_cache, v_cache, context.slot_mapping, self.block_size)

        # Standard attention scaling with an optional model-specific factor.
        scale = self.scale / (self.head_dim ** 0.5)

        if context.is_prefill:
            # Prefill computes full causal attention over the prompt tokens that
            # are not skipped by prefix cache.
            cu_seqlens = context.cu_seqlens_q
            if cu_seqlens is None:
                raise ValueError("cu_seqlens_q must be provided for varlen attention")
            
            o = flash_attention_prefill(q, k, v, cu_seqlens, scale, 
                                        self.num_heads, self.num_kv_heads, self.head_dim)
            # Output: (total_tokens, num_heads, head_dim) -> (total_tokens, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)
        else:
            # Decode reads all previous K/V from the paged cache and computes one
            # output vector per sequence.
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
            # o: (batch_size, num_heads, head_dim) -> (batch_size, num_heads * head_dim)
            return o.reshape(o.shape[0], self.num_heads * self.head_dim)


if __name__ == "__main__":
    # Local timing harness for manual attention-layer experiments.
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
