# src/myvllm/models/llama.py

## 文件作用

该文件用项目自定义层实现 Llama causal LM，特别面向 Llama-3.2-1B-Instruct 这类使用 GQA、RoPE 缩放和 RMSNorm 的 decoder-only 模型。

## 组件结构

### `LlamaAttn`

包含：

- 融合 QKV 的 `QKVColumnParallelLinear`。
- Llama 3 风格 RoPE 的 `RotaryEmbedding(is_llama3=True)`。
- `Attention`，统一处理 prefill Flash Attention 和 decode paged attention。
- row-parallel 输出投影 `o_proj`。

### `LlamaMLP`

与 Qwen3 类似：

```text
gate_up -> SiluAndMul -> down_proj
```

### `LlamaDecoderLayer`

一个 pre-norm decoder block，带 residual 流：

```text
input RMSNorm -> self attention -> post attention RMSNorm -> MLP
```

### `LlamaModel`

embedding、decoder layers 和最终 RMSNorm。

### `LlamaForCausalLM`

主干加词表并行 LM head。

## attention shape 示例

Llama-3.2-1B-Instruct 配置中：

```text
hidden_size = 2048
num_qo_heads = 32
num_kv_heads = 8
head_dim = 64
tp_size = 1
```

融合 QKV 输出维度：

```text
Q = 32 * 64 = 2048
K = 8  * 64 = 512
V = 8  * 64 = 512
total = 3072
```

输入：

```text
x:   (tokens, 2048)
qkv: (tokens, 3072)
q:   (tokens, 32, 64)
k/v: (tokens, 8, 64)
```

因为 `num_qo_heads / num_kv_heads = 4`，每 4 个 query head 共享一个 KV head。

## MLP shape 示例

配置：

```text
hidden_size = 2048
intermediate_size = 8192
```

`gate_up` 输出：

```text
(tokens, 16384)
```

激活后：

```text
(tokens, 8192)
```

down projection 回到：

```text
(tokens, 2048)
```

## residual 流示例

第一层输入没有 residual：

```text
residual = x
x = RMSNorm(x)
```

attention 后：

```text
x, residual = post_attention_layernorm(attn_out, residual)
```

`post_attention_layernorm` 内部先做：

```text
combined = attn_out + residual
```

然后对 `combined` 归一化，并把 `combined` 作为新 residual 传给后续 MLP/下一层。

## RoPE position 示例

prefill 中两个序列长度为 2 和 3：

```text
cu_seqlens_q = [0, 2, 5]
positions = [0, 1, 0, 1, 2]
```

decode 中如果上下文长度是：

```text
context_lens = [6, 10]
```

positions：

```text
[5, 9]
```

## 注意事项

`LlamaAttn` 创建 `Attention` 时传入的是全局 `num_qo_heads` 和 `num_kv_heads`，而 Qwen3 传入本地 head 数。单卡时二者等价；多卡时需要重点检查该差异是否符合 `Attention` 内部对本地 tensor shape 的假设。
