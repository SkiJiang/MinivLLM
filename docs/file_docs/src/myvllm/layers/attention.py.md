# src/myvllm/layers/attention.py

## 文件作用

该文件实现项目中最核心的 attention 逻辑，包括：

- 将新 K/V 写入 paged KV cache 的 Triton kernel。
- prefill 阶段的变长 Flash Attention。
- decode 阶段的 paged attention。
- `Attention` 模块封装，根据全局 context 自动选择 prefill 或 decode 路径。

## KV cache 写入

`store_kvcache_kernel` 使用 `slot_mapping` 把每个 token 的 K/V 写入物理 cache：

```text
slot_idx = block_idx * block_size + block_offset
```

示例：

```text
block_size = 4
slot_idx = 9
block_idx = 9 // 4 = 2
block_offset = 9 % 4 = 1
```

该 token 的 K/V 会写入：

```text
k_cache[2, 1, head_idx, :]
v_cache[2, 1, head_idx, :]
```

如果 `slot_idx == -1`，表示 padding 或 prefix cache 已命中的 token，不需要写入。

## prefill Flash Attention

`flash_attention_varlen_kernel` 针对拼接后的变长序列：

```text
q: (total_tokens, num_heads, head_dim)
k/v: (total_tokens, num_kv_heads, head_dim)
cu_seqlens: [0, len(seq0), len(seq0)+len(seq1), ...]
```

每个 Triton program 处理：

```text
(query tile, query head, sequence)
```

它不保存完整 attention 矩阵，而是用 online softmax 累加：

```text
m_i = 当前行最大值
l_i = 当前行 exp 分母
acc = sum(exp(score - m_i) * V)
```

每读一个 K/V tile，就更新这三个状态。最后：

```text
output = acc / l_i
```

## Flash Attention 计算示例

假设一个 query 行分两块看到 score：

```text
tile1 scores = [1, 2]
tile2 scores = [3]
```

处理 tile1：

```text
m = 2
l = exp(1-2) + exp(2-2) = 0.368 + 1 = 1.368
```

处理 tile2 时新最大值变成 3：

```text
alpha = exp(old_m - new_m) = exp(2-3) = 0.368
l_new = l_old * alpha + exp(3-3)
      = 1.368 * 0.368 + 1
      ≈ 1.503
```

这等价于一次性对 `[1,2,3]` 做 softmax，但不用保存完整 score 矩阵。

## decode paged attention

decode 阶段每个序列只有一个 query token，但要读该序列全部历史 K/V。历史 K/V 不一定连续，而是通过 `block_tables` 映射：

```text
logical token idx -> logical block -> physical block -> cache offset
```

示例：

```text
block_size = 4
block_tables[batch0] = [5, 2, 8]
token_idx = 6
logical block = 6 // 4 = 1
block offset = 6 % 4 = 2
physical block = block_tables[0, 1] = 2
```

K/V 读取位置：

```text
k_cache[2, 2, kv_head_idx, :]
v_cache[2, 2, kv_head_idx, :]
```

## GQA/MQA head 映射

如果 query head 多于 KV head：

```text
num_heads = 16
num_kv_heads = 8
```

每 2 个 query head 共享一个 KV head：

```text
kv_head_idx = query_head_idx // (num_heads // num_kv_heads)
```

例如 query head 6 和 7 都映射到 KV head 3。

## `Attention.forward()`

1. 从 `get_context()` 读取当前 forward 元数据。
2. 如果有 KV cache 和 slot mapping，先写入新的 K/V。
3. 计算缩放因子：

```text
scale = self.scale / sqrt(head_dim)
```

4. prefill：调用 `flash_attention_prefill()`。
5. decode：调用 `paged_attention_decode()`。
6. 把 `(tokens, heads, head_dim)` reshape 回 `(tokens, heads * head_dim)`。

## 注意事项

`ModelRunner.prepare_prefill()` 已经会为 prefix cache 场景准备 `block_tables` 和 `cu_seqlens_k`，但当前 `Attention.forward()` 的 prefill 分支调用 `flash_attention_prefill(q, k, v, cu_seqlens_q, ...)`，主要消费本轮输入产生的 Q/K/V。也就是说，prefix-cache prefill 的显式 cached-K/V 读取仍是后续扩展点；修改这部分时应同时检查 `ModelRunner.prepare_prefill()`、`Context` 字段和 Flash Attention kernel 参数。
