# src/myvllm/layers/rotary_embedding.py

## 文件作用

该文件实现 RoPE（Rotary Position Embedding，旋转位置编码）。RoPE 通过旋转 query/key 的隐藏维对，让 attention score 携带相对位置信息。

## 核心函数

### `apply_rotary_pos_emb(x, cos, sin)`

支持两种形状：

- varlen prefill：`(total_tokens, num_heads, head_dim)`
- batched decode：`(B, seq_len, num_heads, head_dim)`

最后一维被一分为二：

```text
x = [x1, x2]
out = [x1*cos - x2*sin, x1*sin + x2*cos]
```

## `RotaryEmbedding`

初始化时预计算：

```text
inv_freq[j] = 1 / base^(2j / rotary_dim)
freqs[position, j] = position * inv_freq[j]
cos = cos(freqs)
sin = sin(freqs)
```

`cos_sin_cache` 保存所有位置的 cos/sin，forward 时按 `positions` 索引。

## 计算示例

设：

```text
base = 10000
rotary_dim = 4
position = 2
```

频率下标是 `[0, 2]`：

```text
inv_freq[0] = 1 / 10000^(0/4) = 1
inv_freq[1] = 1 / 10000^(2/4) = 1/100 = 0.01
```

位置 2 的角度：

```text
freqs = [2 * 1, 2 * 0.01] = [2, 0.02]
```

如果某个 head 的向量是：

```text
x = [1, 2, 3, 4]
x1 = [1, 2]
x2 = [3, 4]
```

则旋转后：

```text
out1 = x1 * cos([2, 0.02]) - x2 * sin([2, 0.02])
out2 = x1 * sin([2, 0.02]) + x2 * cos([2, 0.02])
```

同一公式作用在 query 和 key 上，使点积 attention 对位置差敏感。

## Llama 3 RoPE 缩放

当 `is_llama3=True` 时，代码根据波长 `wave_len = 2*pi/inv_freq` 对低频分量做缩放。低频对应更长周期，影响长上下文位置区分；缩放后可以扩展上下文长度，同时尽量保持高频局部行为。

## 注意事项

`max_position` 必须覆盖生成中可能出现的最大 position。如果 decode 长度超过 cache 范围，`self.cos_sin_cache[positions]` 会越界。
