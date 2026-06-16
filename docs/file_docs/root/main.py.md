# main.py

## 文件作用

这是使用 Qwen3-0.6B 运行 Mini vLLM 的示例入口。它展示从 tokenizer chat template、engine 初始化、prompt 批量提交到 completion 打印的完整流程。

## 主要配置

`config` 分为几组：

- scheduler/KV-cache：`max_num_sequences`、`max_num_batched_tokens`、`max_cached_blocks`、`block_size`。
- 模型身份：`model_name_or_path='Qwen/Qwen3-0.6B'`。
- Qwen3 架构：`vocab_size`、`hidden_size`、`num_heads`、`num_kv_heads`、`num_layers` 等。
- RoPE/RMSNorm/MLP：`base`、`rms_norm_epsilon`、`intermediate_size`。
- 显存和长度：`max_num_batch_tokens`、`max_model_length`、`gpu_memory_utilization`。
- 停止 token：`eos=151645`。

这些字段必须和 checkpoint 的 HF config 匹配，否则权重 shape、head 切分或 RoPE 位置都会出错。

## 执行流程

1. 将仓库 `src` 加入 `sys.path`，支持直接运行源码。
2. 加载 Qwen tokenizer。
3. 创建 `LLMEngine(config=config)`。
4. 创建 `SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)`。
5. 把普通文本 prompt 包装成 chat prompt。
6. 调用 `llm.generate(prompts, sampling_params)`。
7. 打印 prompt 和 completion。

## 计算示例

Qwen3 配置中：

```text
hidden_size = 1024
num_heads = 16
head_dim = 128
num_kv_heads = 8
```

attention Q/K/V 维度：

```text
Q = 16 * 128 = 2048
K = 8  * 128 = 1024
V = 8  * 128 = 1024
QKV packed = 4096
```

如果 prompt token 长度是 30，`max_model_length=128`，最多还能生成：

```text
128 - 30 = 98 token
```

即使 `max_tokens=256`，scheduler 也会在总长度达到 128 时停止。

## chat template 示例

原始 prompt：

```text
introduce yourself
```

会通过：

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": prompt}],
    tokenize=False,
    add_generation_prompt=True,
)
```

变成模型 instruction/chat checkpoint 期望的格式。这样模型知道用户消息结束，接下来应该由 assistant 生成。

## 注意事项

脚本导入了 `AutoModelForCausalLM`、`torch`、`torch.distributed`，但当前主流程没有直接使用它们。它们可以清理，但保留也不影响示例运行。
