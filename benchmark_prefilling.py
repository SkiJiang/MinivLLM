"""对比标准 PyTorch、朴素 Triton 和 Flash kernel 的 prefill attention benchmark。"""

import torch
import time
import triton 
import triton.language as tl

# ============================================================================
# 1. PyTorch 标准实现（O(N²) 显存）
# ============================================================================
def pytorch_standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """参考 PyTorch causal attention 实现，会物化 O(N^2) score。"""
    total_tokens = q.shape[0]
    # output 保持与 q 相同的拼接 varlen 布局。
    output = torch.zeros_like(q)
    
    # 将边界移动到 CPU，便于参考实现使用 Python 切片。
    cu_seqlens_cpu = cu_seqlens.cpu().tolist()
    
    for i in range(len(cu_seqlens_cpu) - 1):
        # 从拼接 token tensor 中切出一个序列。
        start = cu_seqlens_cpu[i]
        end = cu_seqlens_cpu[i + 1]
        seq_len = end - start
        
        q_seq = q[start:end].transpose(0, 1)  # (num_heads, seq_len, head_dim)
        k_seq = k[start:end].transpose(0, 1)
        v_seq = v[start:end].transpose(0, 1)
        
        # 扩展 grouped KV head，使标准 attention 能做到每个 query head 对应一个 KV head。
        if num_kv_heads != num_heads:
            num_groups = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(num_groups, dim=0)
            v_seq = v_seq.repeat_interleave(num_groups, dim=0)
        
        # 完整 attention score 矩阵是 O(N^2) 显存的来源。
        attn_scores = torch.matmul(q_seq, k_seq.transpose(1, 2)) * scale
        
        # mask 掉未来 token，以满足自回归生成。
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn_scores.masked_fill_(causal_mask.unsqueeze(0), float('-inf'))
        
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out_seq = torch.matmul(attn_probs, v_seq).transpose(0, 1)
        
        output[start:end] = out_seq
    
    return output


# ============================================================================
# 2. 朴素 Triton（O(N²) 显存，仅适合短序列）
# ============================================================================
@triton.jit  
def naive_triton_attention_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    朴素 Triton：物化完整 O(N²) attention 矩阵。

    显存/共享内存限制：BLOCK_SIZE^2 * 4 bytes < 约 48KB
    BLOCK_SIZE=64 时：64*64*4 = 16KB，可行
    BLOCK_SIZE=128 时：128*128*4 = 64KB，超过限制
    """
    # 一个 program 处理一个完整序列和一个 query head。
    seq_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    
    # GQA 将多个 query head 映射到一个 KV head。
    kv_head_idx = head_idx // (num_heads // num_kv_heads)
    
    # 从累积长度中读取序列边界。
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if seq_len > BLOCK_SIZE:
        return  # 跳过过长序列
    
    # 将整个序列载入一个 program，这也是 BLOCK_SIZE 必须较小的原因。
    offs_m = tl.arange(0, BLOCK_SIZE)
    offs_d = tl.arange(0, head_dim)
    mask = offs_m < seq_len
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
    k_ptrs = K + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
    v_ptrs = V + (seq_start + offs_m[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
    
    q = tl.load(q_ptrs, mask=mask[:, None], other=0.0)
    k = tl.load(k_ptrs, mask=mask[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask[:, None], other=0.0)
    
    # 在当前 program 中物化完整 attention 矩阵。
    qk = tl.dot(q, tl.trans(k)) * scale  # (BLOCK_SIZE, BLOCK_SIZE)
    
    # 只保留有效的 causal 位置。
    causal_mask = offs_m[:, None] >= offs_m[None, :]
    seq_mask = mask[:, None] & mask[None, :]
    qk = tl.where(causal_mask & seq_mask, qk, float('-inf'))
    
    # 对物化 score 做逐行 softmax。
    qk_max = tl.max(qk, axis=1)
    qk_exp = tl.exp(qk - qk_max[:, None])
    qk_sum = tl.sum(tl.where(seq_mask, qk_exp, 0.0), axis=1)
    attn = qk_exp / qk_sum[:, None]
    
    # 将概率与 value 相乘。
    out = tl.dot(attn.to(v.dtype), v)
    
    # 写回拼接后的 output 布局。
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + head_idx * head_dim + offs_d[None, :]
    tl.store(o_ptrs, out, mask=mask[:, None])


def naive_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
) -> torch.Tensor:
    """朴素 Triton wrapper，受完整 attention 矩阵的共享内存需求限制。"""
    # kernel 假设 tensor 连续存储。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.empty_like(q)
    num_seqs = cu_seqlens.shape[0] - 1
    
    # 选择能让完整 score 矩阵放进本地资源的 BLOCK_SIZE。
    if head_dim <= 64:
        BLOCK_SIZE = 128  # 有风险，但可能可行
    else:
        BLOCK_SIZE = 64   # 更安全的选择
    
    # Triton 的 block size 用 2 的幂通常更友好。
    actual_size = 2 ** ((max_seq_len - 1).bit_length())
    actual_size = min(actual_size, BLOCK_SIZE)
    
    if max_seq_len > BLOCK_SIZE:
        print(f"      WARNING: seq_len ({max_seq_len}) > BLOCK_SIZE ({BLOCK_SIZE}), results may be incorrect")
    
    grid = (num_seqs, num_heads)
    
    naive_triton_attention_kernel[grid](
        q, k, v, output,
        cu_seqlens,
        scale,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_SIZE=actual_size,
    )
    
    return output


# ============================================================================
# 3. Flash Attention（O(N) 显存）
# ============================================================================
@triton.jit
def flash_attention_kernel(
    Q, K, V, O,
    cu_seqlens_q_ptr,
    scale,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash Attention kernel：使用 online softmax，避免保存完整 score。"""
    # Grid 轴：query tile、query head、sequence index。
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    seq_idx = tl.program_id(2)
    
    kv_head_idx = off_h // (num_heads // num_kv_heads)
    
    # 在拼接 varlen tensor 中解析序列局部 start/end。
    seq_start = tl.load(cu_seqlens_q_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_q_ptr + seq_idx + 1)
    seq_len = seq_end - seq_start
    
    if start_m * BLOCK_M >= seq_len:
        return
    
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, head_dim)
    
    q_ptrs = Q + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    mask_m = offs_m < seq_len
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # online softmax 状态只保存行最大值、归一化因子和输出累加器。
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)
    
    num_blocks = tl.cdiv(seq_len, BLOCK_N)
    
    for block_n in range(num_blocks):
        start_n = block_n * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        k_ptrs = K + (seq_start + offs_n[None, :]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[:, None]
        k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
        
        qk = tl.dot(q, k) * scale
        
        mask_causal = (offs_m[:, None] + seq_start) >= (offs_n[None, :] + seq_start)
        qk = tl.where(mask_causal & mask_n[None, :], qk, -1e10)
        
        # 使用当前 K/V tile 更新行最大值和分母。
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        p = tl.exp(qk - m_i_new[:, None])
        
        acc = acc * alpha[:, None]
        
        v_ptrs = V + (seq_start + offs_n[:, None]) * num_kv_heads * head_dim + kv_head_idx * head_dim + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        acc = acc + tl.dot(p.to(v.dtype), v)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new
    
    acc = acc / l_i[:, None]
    
    o_ptrs = O + (seq_start + offs_m[:, None]) * num_heads * head_dim + off_h * head_dim + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=mask_m[:, None])


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Flash Attention wrapper，根据 head_dim 选择 tile size。"""
    # 确保内存稠密连续，便于指针算术。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    output = torch.empty_like(q)
    
    # head 越宽，使用越小的 tile 控制寄存器/共享内存压力。
    if head_dim <= 64:
        BLOCK_M, BLOCK_N = 64, 64
    elif head_dim <= 128:
        BLOCK_M, BLOCK_N = 32, 32
    else:
        BLOCK_M, BLOCK_N = 16, 16
    
    # 序列数就是 cu_seqlens 中区间的数量。
    num_seqs = cu_seqlens.shape[0] - 1
    cu_seqlens_cpu = cu_seqlens.cpu()
    max_seq_len = (cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]).max().item()
    
    # 一起启动所有 query tile、head 和 sequence。
    grid = (triton.cdiv(max_seq_len, BLOCK_M), num_heads, num_seqs)
    
    flash_attention_kernel[grid](
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


def find_crossover_point():
    """寻找 Flash attention 开始快于 naive 的序列长度。"""
    
    print("\n" + "="*80)
    print("FINDING CROSSOVER POINT: When does Flash beat Naive?")
    print("="*80)
    
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    num_seqs = 2
    
    results = []
    
    # 短序列可能偏向 naive kernel，因为它启动的 program 更少；
    # 长序列则需要 Flash 的 O(N) 显存行为。
    seq_lengths = [16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512, 1024]
    
    for seq_len in seq_lengths:
        print(f"\nTesting seq_len = {seq_len}...")
        
        q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
        
        # 当完整 attention tile 超过当前 head_dim 的安全长度时，跳过朴素 Triton。
        max_safe_seq = 64 if head_dim > 64 else 128
        if seq_len <= max_safe_seq:
            for _ in range(10):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(50):
                _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
            torch.cuda.synchronize()
            naive_time = (time.perf_counter() - start) / 50
        else:
            naive_time = None
        
        # Flash Attention 应当能覆盖所有测试序列长度。
        for _ in range(10):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(50):
            _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
        torch.cuda.synchronize()
        flash_time = (time.perf_counter() - start) / 50
        
        if naive_time:
            speedup = naive_time / flash_time
            winner = "Naive" if speedup < 1 else "Flash"
            print(f"  Naive: {naive_time*1000:.3f}ms | Flash: {flash_time*1000:.3f}ms | Winner: {winner} ({abs(speedup):.2f}x)")
            results.append((seq_len, naive_time, flash_time, winner))
        else:
            print(f"  Naive: SKIPPED | Flash: {flash_time*1000:.3f}ms | Winner: Flash (by default)")
            results.append((seq_len, None, flash_time, "Flash"))
    
    # 打印紧凑表格，方便肉眼查看 crossover。
    print("\n" + "="*80)
    print("CROSSOVER ANALYSIS")
    print("="*80)
    print(f"{'Seq Len':>10} | {'Naive (ms)':>12} | {'Flash (ms)':>12} | {'Winner':>10} | {'Speedup':>10}")
    print("-" * 80)
    
    crossover = None
    for seq_len, naive_time, flash_time, winner in results:
        if naive_time:
            speedup = naive_time / flash_time if flash_time < naive_time else flash_time / naive_time
            speedup_str = f"{speedup:.2f}x"
            naive_str = f"{naive_time*1000:.3f}"
        else:
            speedup_str = "N/A"
            naive_str = "OOM"
        
        flash_str = f"{flash_time*1000:.3f}"
        print(f"{seq_len:>10} | {naive_str:>12} | {flash_str:>12} | {winner:>10} | {speedup_str:>10}")
        
        # 两个 kernel 都运行时，Flash 第一次胜出的序列长度。
        if crossover is None and winner == "Flash" and naive_time is not None:
            crossover = seq_len
    

def analyze_kernel_launches():
    """展示为什么 naive 的 kernel launch 数更少。"""
    
    print("\n" + "="*80)
    print("KERNEL LAUNCH ANALYSIS")
    print("="*80)
    
    num_seqs = 2
    seq_len = 60
    num_heads = 32
    BLOCK_M = 32
    
    # naive 每个 sequence/head 启动一个 program。
    naive_grid = (num_seqs, num_heads)
    naive_kernels = num_seqs * num_heads
    
    # Flash 会沿 query 维切 tile，因此 program 数更多。
    num_blocks_m = (seq_len + BLOCK_M - 1) // BLOCK_M
    flash_grid = (num_blocks_m, num_heads, num_seqs)
    flash_kernels = num_blocks_m * num_heads * num_seqs
    
    print(f"\nFor {num_seqs} sequences × {seq_len} tokens:")
    print(f"  Naive Triton grid:    {naive_grid}")
    print(f"  Naive total kernels:  {naive_kernels}")
    print(f"\n  Flash Attention grid: {flash_grid}")
    print(f"  Flash total kernels:  {flash_kernels}")
    print(f"\n  Ratio: Flash launches {flash_kernels/naive_kernels:.1f}x more kernels")
    print(f"\n  Each kernel launch has ~5-20μs overhead")
    print(f"  Extra overhead: ~{(flash_kernels - naive_kernels) * 10}μs = {(flash_kernels - naive_kernels) * 0.01:.2f}ms")


# ============================================================================
# Benchmark
# ============================================================================

def setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim):
    """创建合成 varlen Q/K/V tensor 和累积长度。"""
    total_tokens = num_seqs * seq_len
    device = 'cuda'
    
    # 拼接 varlen 布局：所有序列 token 沿 dim 0 堆叠。
    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    
    # 等长序列让 benchmark 中的 cu_seqlens 更容易构造。
    cu_seqlens = torch.tensor([i * seq_len for i in range(num_seqs + 1)], 
                              device=device, dtype=torch.int32)
    
    scale = 1.0 / (head_dim ** 0.5)
    
    return q, k, v, cu_seqlens, scale


def benchmark(num_seqs, seq_len, num_heads=32, num_kv_heads=8, head_dim=128, num_iter=50):
    """为所有可用 prefill attention 实现计时。"""
    print(f"\n{'='*80}")
    print(f"Benchmark: {num_seqs} seqs × {seq_len} tokens (total: {num_seqs*seq_len} tokens)")
    print(f"Heads: {num_heads}/{num_kv_heads}, Dim: {head_dim}")
    print(f"{'='*80}")
    
    q, k, v, cu_seqlens, scale = setup_data(num_seqs, seq_len, num_heads, num_kv_heads, head_dim)
    
    results = {}
    outputs = {}
    
    # 1. PyTorch 参考实现。
    print("\n[1/3] PyTorch Standard (O(N²) memory)...")
    for _ in range(5):
        _ = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        outputs['pytorch'] = pytorch_standard_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) / num_iter
    results['PyTorch (O(N²))'] = t
    print(f"      {t*1000:.3f} ms")
    
    # 2. 朴素 Triton，仅在完整 score 矩阵放得下时运行。
    max_safe_seq = 64 if head_dim > 64 else 128
    if seq_len <= max_safe_seq:
        print(f"\n[2/3] Naive Triton (O(N²), materializes full attention)...")
        for _ in range(5):
            _ = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iter):
            outputs['naive'] = naive_triton_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim, seq_len)
        torch.cuda.synchronize()
        t = (time.perf_counter() - start) / num_iter
        results['Naive Triton (O(N²))'] = t
        print(f"      {t*1000:.3f} ms")
    else:
        print(f"\n[2/3] Naive Triton: SKIPPED (seq_len={seq_len} > {max_safe_seq}, would exceed shared memory)")
    
    # 3. Flash Attention。
    print("\n[3/3] Flash Attention (O(N), online softmax)...")
    for _ in range(5):
        _ = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(num_iter):
        outputs['flash'] = flash_attention(q, k, v, cu_seqlens, scale, num_heads, num_kv_heads, head_dim)
    torch.cuda.synchronize()
    t = (time.perf_counter() - start) / num_iter
    results['Flash Attention (O(N))'] = t
    print(f"      {t*1000:.3f} ms")
    


if __name__ == "__main__":
    # 运行一个小范围 sweep，并做 crossover 与 launch 数分析。
    print("\n" + "="*80)
    print("PREFILL ATTENTION BENCHMARK")
    print("Comparing: PyTorch (O(N²)) | Naive Triton (O(N²)) | Flash (O(N))")
    print("="*80)
    
    benchmark(num_seqs=2, seq_len=60, num_iter=100)
    benchmark(num_seqs=4, seq_len=64, num_iter=100)
    benchmark(num_seqs=2, seq_len=1024, num_iter=30)
    benchmark(num_seqs=1, seq_len=4096, num_iter=10)

    # 接着分析 crossover 点。
    find_crossover_point()

    # 解释背后的 kernel launch 差异。
    analyze_kernel_launches()
