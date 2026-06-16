# benchmark_prefilling.py

## 文件作用

该脚本比较 prefill 阶段三种 causal attention 实现：

- PyTorch 标准实现：物化完整 attention score，显存复杂度 O(N^2)。
- 朴素 Triton：同样物化完整 score，只适合短序列。
- Flash Attention：用 online softmax，显存复杂度 O(N)。

## 数据布局

benchmark 使用变长拼接布局：

```text
q: (total_tokens, num_heads, head_dim)
k: (total_tokens, num_kv_heads, head_dim)
v: (total_tokens, num_kv_heads, head_dim)
cu_seqlens: (num_seqs + 1,)
```

如果有 2 个序列，每个 60 token：

```text
total_tokens = 120
cu_seqlens = [0, 60, 120]
```

## PyTorch 标准 attention

对每个序列和 head 计算：

```text
scores = Q @ K^T * scale
probs = softmax(mask_causal(scores))
out = probs @ V
```

计算示例：

```text
seq_len = 1024
score matrix = 1024 * 1024 = 1,048,576 个元素
float16 约 2 MB / head / seq
32 heads 约 64 MB，仅 score 就很大
```

序列更长时 O(N^2) 会迅速变成瓶颈。

## 朴素 Triton attention

一个 program 处理一个完整 sequence/head，并在本地保存完整 `BLOCK_SIZE x BLOCK_SIZE` score。

示例：

```text
BLOCK_SIZE = 64
score 元素 = 64 * 64 = 4096
float32 临时内存约 16 KB
```

如果 `BLOCK_SIZE=128`：

```text
128 * 128 * 4 bytes = 64 KB
```

可能超过共享内存或寄存器限制，所以脚本会跳过过长序列。

## Flash Attention

Flash kernel 按 query tile 和 key tile 分块，维护：

```text
m_i: 行最大值
l_i: softmax 分母
acc: 加权 V 累加器
```

它不保存完整 score 矩阵。对于 `seq_len=4096`，每个 query tile 逐块扫过 K/V，显存主要和 tile size、head_dim 相关，而不是 `4096^2`。

## crossover 分析

`find_crossover_point()` 测试不同序列长度，寻找 Flash 开始快于 naive 的位置。

短序列时 naive 可能更快，因为：

```text
num_seqs = 2
num_heads = 32
naive programs = 2 * 32 = 64
```

Flash 如果 `seq_len=60`、`BLOCK_M=32`：

```text
query tiles = ceil(60 / 32) = 2
flash programs = 2 * 32 * 2 = 128
```

program 数更多，launch/调度开销更高。长序列时 Flash 的 O(N) 显存和分块计算优势会胜出。

## benchmark 结果解释

`benchmark()` 对每种实现先预热，再重复运行 `num_iter` 次：

```text
average_time = (end - start) / num_iter
```

脚本只计时，不做数值误差校验。如果要验证 kernel 正确性，应加入 `torch.testing.assert_close()`，比较 PyTorch 和 Triton/Flash 输出。
