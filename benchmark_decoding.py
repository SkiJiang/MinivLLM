"""PyTorch 和 Triton 实现的 paged-attention decode benchmark。"""

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
    """为每个 batch item 和 query head 计算一个 decode attention 输出。"""
    # Grid 轴：batch 下标和 query head 下标。
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # GQA 会把多个 query head 映射到一个 KV head。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    # 当前序列 KV 历史中的有效 token 数。
    context_len = tl.load(context_lens_ptr + batch_idx)
    
    # 读取当前 query 向量。
    offs_d = tl.arange(0, head_dim)
    q_offset = batch_idx * num_heads * head_dim + head_idx * head_dim + offs_d
    q = tl.load(query_ptr + q_offset)
    
    # 单个 query 的 online softmax 状态。
    acc = tl.zeros([head_dim], dtype=tl.float32)
    l_i = 0.0
    m_i = -1e10
    
    # 以固定大小 chunk 遍历所有可能的 cache token。
    max_chunks = tl.cdiv(max_num_blocks * block_size, BLOCK_N)
    
    for chunk_idx in range(max_chunks):
        # token_start 是逻辑 token 位置，不是物理 cache offset。
        token_start = chunk_idx * BLOCK_N
        
        if token_start < context_len:
            offs_n = token_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < context_len
            
            # 根据 block_tables 为当前 chunk 填充 attention score。
            qk = tl.zeros([BLOCK_N], dtype=tl.float32) - 1e10
            
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        # block_tables 将逻辑 block 映射到物理 cache block。
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # 从 (num_blocks, block_size, num_kv_heads, head_dim) 中读取 K。
                            k_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            k_vec = tl.load(k_cache_ptr + k_offset)
                            
                            score = tl.sum(q * k_vec) * scale
                            mask_i = tl.arange(0, BLOCK_N) == i
                            qk = tl.where(mask_i, score, qk)
            
            # 无效 chunk 条目不应影响 softmax。
            qk = tl.where(mask_n, qk, -1e10)
            
            # 数值稳定的 online softmax 更新。
            m_ij = tl.max(qk)
            m_i_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_i_new)
            p = tl.exp(qk - m_i_new)
            
            acc = acc * alpha
            l_i = l_i * alpha
            
            # 从同一批物理 cache block 累加加权后的 V 向量。
            for i in range(BLOCK_N):
                token_idx = token_start + i
                if token_idx < context_len:
                    block_num = token_idx // block_size
                    block_offset = token_idx % block_size
                    
                    if block_num < max_num_blocks:
                        block_table_offset = batch_idx * max_num_blocks + block_num
                        physical_block_idx = tl.load(block_tables_ptr + block_table_offset)
                        
                        if physical_block_idx != -1:
                            # 从 paged cache 中读取 V。
                            v_offset = (physical_block_idx * block_size * num_kv_heads * head_dim +
                                       block_offset * num_kv_heads * head_dim +
                                       kv_head_idx * head_dim + offs_d)
                            v_vec = tl.load(v_cache_ptr + v_offset)
                            
                            # 使用 one-hot mask 提取标量 p[i]。
                            mask_i = tl.arange(0, BLOCK_N) == i
                            weight = tl.sum(tl.where(mask_i, p, 0.0))
                            
                            acc = acc + weight * v_vec
                            l_i = l_i + weight
            
            m_i = m_i_new
    
    # 写回归一化后的输出向量。
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
    """启动 Triton paged-attention decode kernel。"""
    batch_size = query.shape[0]
    max_num_blocks = block_tables.shape[1]
    # kernel 指针算术假设 query 是连续存储。
    query = query.contiguous()
    output = torch.empty_like(query)
    
    # head 越宽，使用越小的 chunk，以降低寄存器/共享内存压力。
    BLOCK_N = 64 if head_dim <= 128 else 32
    # 每个 batch item 和 query head 对应一个 program。
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
    """向量化 PyTorch 基线：先把 paged cache gather 成稠密 tensor。"""
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    # 稠密 padding buffer 让 PyTorch matmul 更简单，但会增加 gather/copy 开销。
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    for i in range(batch_size):
        # 按 block table 读取，并将最后一个 block 截断到 seq_len。
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
        # 扩展 grouped KV head，使标准 attention 能做到每个 Q head 一个 KV head。
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    q = q.unsqueeze(2)
    padded_k = padded_k.transpose(1, 2)
    padded_v = padded_v.transpose(1, 2)
    
    # q 会关注每个序列的完整 padded context。
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    # mask 掉每个序列长度之外的 padding。
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
    朴素 decode 实现。
    它会重建完整 K/V 序列，并使用标准 PyTorch attention。
    """
    batch_size = q.shape[0]
    device = q.device
    dtype = q.dtype
    
    max_context_len = context_lens.max().item()
    
    # 先把 K/V gather 到 Python list 中；这种写法故意简单，但速度较慢。
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
    
    # 将 gather 出来的变长 K/V padding 成稠密 tensor，以便 batch matmul。
    padded_k = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    padded_v = torch.zeros(batch_size, max_context_len, num_kv_heads, head_dim,
                           device=device, dtype=dtype)
    
    for i, (k_seq, v_seq) in enumerate(zip(all_k, all_v)):
        seq_len = len(k_seq)
        padded_k[i, :seq_len] = k_seq
        padded_v[i, :seq_len] = v_seq
    
    if num_kv_heads != num_heads:
        # 重复 KV head，使其匹配标准 attention 调用需要的 query-head 数。
        num_groups = num_heads // num_kv_heads
        padded_k = padded_k.repeat_interleave(num_groups, dim=2)
        padded_v = padded_v.repeat_interleave(num_groups, dim=2)
    
    # reshape 成 (B, H, 1, D) x (B, H, D, N)。
    q = q.unsqueeze(2)  # (B, H, 1, D)
    padded_k = padded_k.transpose(1, 2)  # (B, H, N, D)
    padded_v = padded_v.transpose(1, 2)  # (B, H, N, D)
    
    # 为每个 batch/head 物化完整 attention score 向量。
    attn_scores = torch.matmul(q, padded_k.transpose(-2, -1)) * scale
    
    mask = torch.arange(max_context_len, device=device)[None, :] < context_lens[:, None]
    mask = mask[:, None, None, :]
    attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
    
    attn_probs = torch.softmax(attn_scores, dim=-1)
    output = torch.matmul(attn_probs, padded_v).squeeze(2)
    
    return output



def setup_test_data(batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size, device='cuda'):
    """构造 benchmark 使用的测试数据。"""
    # query 表示每个序列当前 decode token。
    q = torch.randn(batch_size, num_heads, head_dim, device=device, dtype=torch.float16)
    
    # 为每个序列的完整 context 分配足够多的物理 block。
    max_num_blocks = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * max_num_blocks
    
    # KV cache 模拟 ModelRunner 的 paged 布局。
    k_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_cache = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    
    # 使用连续物理 block id，方便检查正确性。
    block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(batch_size, max_num_blocks)
    
    # benchmark 使用相同序列长度，使计时更清楚。
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    # 标准 attention scale。
    scale = 1.0 / (head_dim ** 0.5)
    
    return q, k_cache, v_cache, block_tables, context_lens, scale


def benchmark(batch_size, seq_len, num_heads=32, num_kv_heads=8, 
                                  head_dim=128, block_size=16, num_iterations=100):
    """比较三个实现。"""
    
    print(f"\n{'='*70}")
    print(f"batch_size={batch_size}, seq_len={seq_len}, num_heads={num_heads}")
    print(f"num_kv_heads={num_kv_heads}, head_dim={head_dim}, block_size={block_size}")
    print(f"{'='*70}")
    
    # 合成数据可以隔离 attention kernel 性能，不混入模型其他开销。
    q, k_cache, v_cache, block_tables, context_lens, scale = setup_test_data(
        batch_size, seq_len, num_heads, num_kv_heads, head_dim, block_size
    )
    
    results = {}
    
    # 1. 朴素实现。
    print("\n1. Testing Naive PyTorch implementation...")
    for _ in range(10):  # 预热
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
    
    # 2. 优化版 PyTorch 基线。
    print("\n2. Testing Optimized PyTorch implementation...")
    for _ in range(10):  # 预热
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
    
    # 3. Triton paged attention kernel。
    print("\n3. Testing Triton implementation...")
    for _ in range(10):  # 预热
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
    # 对 context size 和 batch size 做一个小范围 sweep。
    print("\n" + "="*70)
    print("COMPREHENSIVE PAGED ATTENTION DECODE BENCHMARK")
    print("Comparing: Naive PyTorch | Optimized PyTorch | Triton")
    print("="*70)
    
    benchmark(batch_size=2, seq_len=60, num_iterations=100)
    benchmark(batch_size=1, seq_len=512, num_iterations=100)
    benchmark(batch_size=16, seq_len=256, num_iterations=50)
    benchmark(batch_size=4, seq_len=2048, num_iterations=20)
