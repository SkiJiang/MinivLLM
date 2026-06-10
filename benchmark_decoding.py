"""Paged-attention decode benchmark for PyTorch and Triton implementations."""

import torch
import time
import triton 
import triton.language as tl

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
    """Compute one decode attention output per batch item and query head."""
    # Grid axes: batch index and query head index.
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # GQA maps several query heads to one KV head.
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    # Number of valid tokens in this sequence's KV history.
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # Load the current query vector.
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # Online softmax state for the single query.
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # Iterate over all possible cache tokens in fixed-size chunks.
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    for chunk_idx in range(max_chunks):
        # token_start is a logical token position, not a physical cache offset.
        token_start = chunk_idx * BLOCK_N
        
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            
            # Fill attention scores for this chunk by following block_tables.
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # block_tables maps logical blocks to physical cache blocks.
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # Load K from (num_blocks, block_size, num_kv_heads, head_dim).
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            # Invalid chunk entries should not affect softmax.
            qk = tl.where(mask_n, qk, -1e10)
            
            # Numerically stable online softmax update.
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            acc = acc * alpha
            l_i = l_i * alpha
            
            # Accumulate weighted V vectors from the same physical cache blocks.
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # Load V from the paged cache.
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            
                            # Extract scalar p[i] with a one-hot mask.
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    # Store the normalized output vector.
    output = acc / l_i
    output_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    tl.store(output_ptr + output_offset, output)


def paged_attention_decode_triton(
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
    """Launch the Triton paged-attention decode kernel."""
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    # Kernel pointer arithmetic assumes contiguous query storage.
    query = query.contiguous()
    output = torch.empty_like(query)
    
    # Wider heads use smaller chunks to keep register/shared-memory pressure down.
    BLOCK_N = 64 if head_dim <= 128 else 32
    # One program per batch item and query head.
    grid = (batch_size, num_heads)
    
    paged_attention_decode_kernel[grid](
        output, query, k_cache, v_cache, block_tables, context_lens,
        scale=scale, num_heads=num_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size, 
        max_num_blocks=max_num_blocks, BLOCK_N=BLOCK_N,
    )
    return output


def decode_torch_optimized(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """Vectorized PyTorch baseline that first gathers paged cache into dense tensors."""
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    # Dense padded buffers make PyTorch matmul simple but add gather/copy overhead.
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    for i in range(batch_size):
        # Follow the block table and truncate the final block to seq_len.
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        valid_blocks = block_tables[i, :num_blocks_needed]
        valid_blocks = valid_blocks[valid_blocks != -1]
        
        if len(valid_blocks) > 0:
            gathered_k = k_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            gathered_v = v_cache[valid_blocks].reshape(-1, num_kv_heads, head_dim)[:seq_len]
            
            padded_k[i, :seq_len] = gathered_k
            padded_v[i, :seq_len] = gathered_v
    
    if num_kv_heads != num_heads:
        # Expand grouped KV heads so standard attention can use one KV head per Q.
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    q = q.unsqueeze(2)
    padded_k = padded_k.transpose(1, 2)
    padded_v = padded_v.transpose(1, 2)
    
    # q attends to the full padded context for each sequence.
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    # Mask out padding beyond each sequence length.
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)
    
    return output


def naive_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
) -> torch.Tensor:
    """
    Naive decode implementation
    This reconstructs full K, V sequences and uses standard PyTorch attention.
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    # Gather K/V into Python lists first, which is intentionally simple but slow.
    all_k = []
    all_v = []
    
    for i in range(batch_size):
        seq_len = context_lens[i].item()
        num_blocks_needed = (seq_len + block_size - 1) // block_size
        
        seq_k_list = []
        seq_v_list = []
        for block_idx in range(num_blocks_needed):
            block_id = block_tables[i, block_idx].item()
            if block_id == -1:
                break
            seq_k_list.append(k_cache[block_id])
            seq_v_list.append(v_cache[block_id])
        
        if len(seq_k_list) > 0:
            seq_k = torch.cat(seq_k_list, dim=0)[:seq_len]
            seq_v = torch.cat(seq_v_list, dim=0)[:seq_len]
            all_k.append(seq_k)
            all_v.append(seq_v)
    
    # Pad variable-length gathered K/V into dense tensors for batch matmul.
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    
    for i, (k_seq, v_seq) in enumerate(zip(all_k, all_v)):
        seq_len = len(k_seq)
        padded_k[i, :seq_len] = k_seq
        padded_v[i, :seq_len] = v_seq
    
    if num_kv_heads != num_heads:
        # Repeat KV heads to match query-head count for a standard attention call.
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    # Reshape to (B, H, 1, D) x (B, H, D, N).
    q = q.unsqueeze(2)  # (B, H, 1, D)
    padded_k = padded_k.transpose(1, 2)  # (B, H, N, D)
    padded_v = padded_v.transpose(1, 2)  # (B, H, N, D)
    
    # Materializes the full attention score vector for every batch/head.
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)
    
    return output



def setup_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device='cuda'):
    """Setup test data for benchmarking"""
    # Query represents the current decode token for each sequence.
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # Allocate enough physical blocks for every sequence's full context.
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks
    
    # KV cache mimics ModelRunner's paged layout.
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    
    # Consecutive physical block ids make correctness inspection straightforward.
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch_size, max_num_blocks)
    
    # Benchmark uses equal sequence lengths for clearer timing.
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    # Standard attention scale.
    scale = 1.0 / (head_dim ** 0.5)
    
    return q, k_cache, v_cache, block_tables, context_lens, scale


def benchmark(batch_size, seq_len, num_heads=32, num_kv_heads=8, 
                                  head_dim=128, block_size=16, num_iterations=100):
    """Compare all three implementations"""
    
    print(f"\n{'='*70}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*70}")
    
    # Synthetic data isolates attention kernel performance from model overhead.
    q, k_cache, v_cache, block_tables, context_lens, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
    )
    
    results = {}
    
    # 1. Naive implementation.
    print("\n1. Testing Naive PyTorch implementation...")
    for _ in range(10):  # warmup
        _ = naive_decode_attention(q, k_cache, v_cache, block_tables, context_lens,
                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_naive = naive_decode_attention(q, k_cache, v_cache, block_tables, context_lens,
                                           scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start) / num_iterations
    results['Naive PyTorch'] = naive_time
    print(f"   Time: {naive_time*1000:.3f}ms")
    
    # 2. Optimized PyTorch baseline.
    print("\n2. Testing Optimized PyTorch implementation...")
    for _ in range(10):  # warmup
        _ = decode_torch_optimized(q, k_cache, v_cache, block_tables, context_lens,
                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_pytorch = decode_torch_optimized(q, k_cache, v_cache, block_tables, context_lens,
                                            scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    pytorch_time = (time.perf_counter() - start) / num_iterations
    results['Optimized PyTorch'] = pytorch_time
    print(f"   Time: {pytorch_time*1000:.3f}ms")
    
    # 3. Triton paged attention kernel.
    print("\n3. Testing Triton implementation...")
    for _ in range(10):  # warmup
        _ = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, context_lens,
                                          scale, num_heads, num_kv_heads, head_dim, block_size)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iterations):
        out_triton = paged_attention_decode_triton(q, k_cache, v_cache, block_tables, context_lens,
                                                   scale, num_heads, num_kv_heads, head_dim, block_size)
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start) / num_iterations
    results['Triton'] = triton_time
    print(f"   Time: {triton_time*1000:.3f}ms")
    
    return results


if __name__ == "__main__":
    # Run a small sweep over context sizes and batch sizes.
    print("\n" + "="*70)
    print("COMPREHENSIVE PAGED ATTENTION DECODE BENCHMARK")
    print("Comparing: Naive PyTorch | Optimized PyTorch | Triton")
    print("="*70)
    
    benchmark(batch_size=2, seq_len=60, num_iterations=100)
    benchmark(batch_size=1, seq_len=512, num_iterations=100)
    benchmark(batch_size=16, seq_len=256, num_iterations=50)
    benchmark(batch_size=4, seq_len=2048, num_iterations=20)
