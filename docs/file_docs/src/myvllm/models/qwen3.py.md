# src/myvllm/models/qwen3.py

## 文件作用

该文件用项目自定义的张量并行层实现 Qwen3 causal LM。它包含 attention、MLP、decoder layer、Transformer 主干和带 LM head 的完整模型。

## 组件结构

### `Qwen3Attention`

包含：

- `QKVColumnParallelLinear`：融合 Q/K/V 投影。
- `LayerNorm`：当 `qkv_bias=False` 时对 Q/K 做 RMSNorm。
- `RotaryEmbedding`：为 Q/K 注入位置信息。
- `Attention`：prefill 用 Flash Attention，decode 用 paged attention。
- `RowParallelLinear`：输出投影并跨 rank all-reduce。

### `Qwen3MLP`

包含：

- `MergedColumnParallelLinear`：融合 gate/up 投影。
- `SiluAndMul`：SwiGLU 激活。
- `RowParallelLinear`：down projection。

### `Qwen3DecoderLayer`

一个 decoder block：

```text
RMSNorm -> Self Attention -> residual RMSNorm -> MLP
```

### `Qwen3Model`

包含 token embedding、多个 decoder layer 和最终 RMSNorm。

### `Qwen3ForCausalLM`

在主干后加 `ParallelLMHead`，并提供 `compute_logits()`。

## attention shape 示例

以 Qwen3-0.6B 配置为例：

```text
hidden_size = 1024
num_heads = 16
num_kv_heads = 8
head_dim = 128
tp_size = 1
```

QKV 融合投影输出维度：

```text
Q = 16 * 128 = 2048
K = 8  * 128 = 1024
V = 8  * 128 = 1024
total = 4096
```

输入 `x` 如果是 prefill varlen：

```text
x:   (total_tokens, 1024)
qkv: (total_tokens, 4096)
q:   (total_tokens, 16, 128)
k/v: (total_tokens, 8, 128)
```

GQA 下每 2 个 query head 共享一个 KV head。

## MLP 计算示例

配置：

```text
hidden_size = 1024
intermediate_size = 3072
```

`gate_up` 输出：

```text
(tokens, 6144)
```

`SiluAndMul` 拆成两个 `(tokens, 3072)`：

```text
activated = silu(gate) * up
```

`down_proj` 再映射回：

```text
(tokens, 1024)
```

## position 计算

prefill 使用拼接 varlen tensor，因此每个序列的位置要从 0 重新开始。若：

```text
cu_seqlens_q = [0, 3, 7]
```

说明两个序列长度是 3 和 4，positions 为：

```text
[0, 1, 2, 0, 1, 2, 3]
```

decode 中只有一个 token，位置是：

```text
positions = context_lens - 1
```

如果某序列当前上下文长度为 9，当前 token 的位置就是 8。

## 权重绑定

`tie_word_embeddings=True` 时：

```python
self.lm_head.weight = self.model.embed_tokens.weight
```

输入 embedding 和输出词表投影共享同一份参数，符合部分 HF checkpoint 的配置。

## 注意事项

`packed_module_mapping` 目前记录了 checkpoint 名称和本地融合模块的关系，但实际 `loader.py` 使用自己的名称匹配逻辑。修改权重加载时应同步检查这两处设计是否一致。
