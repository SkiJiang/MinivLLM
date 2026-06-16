# benchmark_decoding.py

## 文件作用

该脚本比较 decode 阶段的 paged attention 实现：

- 朴素 PyTorch：用 Python 循环 gather K/V。
- 优化 PyTorch：向量化 gather 到稠密 padded buffer。
- Triton paged attention：直接通过 block table 从 paged KV cache 读取。

## 数据布局

合成数据由 `setup_test_data()` 构造：

```text
q: (batch_size, num_heads, head_dim)
k_cache/v_cache: (total_blocks, block_size, num_kv_heads, head_dim)
block_tables: (batch_size, max_num_blocks)
context_lens: (batch_size,)
```

示例：

```text
batch_size = 2
seq_len = 60
block_size = 16
max_num_blocks = ceil(60 / 16) = 4
total_blocks = 2 * 4 = 8
block_tables =
  [[0, 1, 2, 3],
   [4, 5, 6, 7]]
```

## 逻辑 token 到物理 cache 的映射

假设 batch 1 的第 35 个历史 token：

```text
block_size = 16
token_idx = 35
logical block = 35 // 16 = 2
block offset = 35 % 16 = 3
physical block = block_tables[1, 2] = 6
```

读取：

```text
k_cache[6, 3, kv_head_idx, :]
v_cache[6, 3, kv_head_idx, :]
```

## 朴素 PyTorch 实现

`naive_decode_attention()`：

1. 对每个 batch item 遍历 block table。
2. `torch.cat()` 拼出该序列完整 K/V。
3. padding 到 batch 内最大长度。
4. 做标准 attention。

这种写法清楚，但 Python 循环和多次 concat 开销较大。

## 优化 PyTorch 实现

`decode_torch_optimized()`：

1. 预分配 `padded_k/padded_v`。
2. 用 block id 一次性索引 `k_cache[valid_blocks]`。
3. reshape 后截断到真实 `seq_len`。
4. 用 batch matmul 计算 attention。

仍然需要把 paged cache gather 成稠密 tensor，因此内存搬运不少。

## Triton 实现

`paged_attention_decode_kernel()` 一个 program 计算一个：

```text
(batch item, query head)
```

它按 `BLOCK_N` chunk 遍历上下文，用 online softmax 累加输出，避免构造稠密 padded K/V。

## attention 计算示例

对单个 query，假设它看到 3 个 key，scale 后 score 为：

```text
[1.0, 2.0, 0.0]
```

softmax：

```text
exp = [e^1, e^2, e^0] = [2.718, 7.389, 1]
sum = 11.107
prob = [0.245, 0.665, 0.090]
```

输出：

```text
out = 0.245 * V0 + 0.665 * V1 + 0.090 * V2
```

Triton kernel 用 chunk 化的 online softmax 得到同样结果。

## benchmark 解释

每个实现都会预热 10 次，再重复 `num_iterations`：

```text
avg_ms = (end - start) / num_iterations * 1000
```

脚本最后返回三种实现的平均时间。

## 注意事项

当前 benchmark 不比较输出正确性。要做严谨测试，应把三种输出用 `assert_close` 比较，并设置合适的 fp16 容差。
