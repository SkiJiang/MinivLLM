# main_llama32.py

## 文件作用

这是使用 Llama-3.2-1B-Instruct 运行 Mini vLLM 的示例入口。结构与 `main.py` 类似，但配置换成 Llama 架构字段。

## 主要配置

关键 Llama 字段：

- `model_name_or_path='meta-llama/Llama-3.2-1B-Instruct'`
- `vocab_size=128256`
- `hidden_size=2048`
- `head_dim=64`
- `num_qo_heads=32`
- `num_kv_heads=8`
- `intermediate_size=8192`
- `num_layers=16`
- `rope_base=500000`
- `max_position_embeddings=32768`
- `eos=128009`

相比 Qwen，Llama 配置字段名使用 `num_qo_heads`、`rope_base`、`max_position_embeddings`。

## 执行流程

1. 把 `src` 加入 `sys.path`。
2. 加载 Llama tokenizer。
3. 创建 `LLMEngine`。
4. 设置采样参数。
5. 使用 tokenizer chat template 包装 prompt。
6. 批量生成并打印结果。

## 计算示例

Llama attention 维度：

```text
hidden_size = 2048
num_qo_heads = 32
head_dim = 64
num_kv_heads = 8
```

Q/K/V 维度：

```text
Q = 32 * 64 = 2048
K = 8  * 64 = 512
V = 8  * 64 = 512
QKV packed = 3072
```

GQA 分组：

```text
num_qo_heads / num_kv_heads = 32 / 8 = 4
```

也就是说每 4 个 query head 共享一个 KV head。

## 长度限制示例

采样参数：

```text
max_tokens = 256
max_model_length = 128
```

如果 chat template 后 prompt 长度是 45，则最多生成：

```text
min(256, 128 - 45) = 83 token
```

## 注意事项

Llama 官方 gated checkpoint 可能需要本机 Hugging Face 登录权限。这个脚本只负责本地推理流程，不处理认证问题。
