# benchmark_tps.py

## 文件作用

该脚本比较三个推理路径的端到端吞吐：

- 本项目 Mini vLLM。
- 上游 vLLM。
- Hugging Face Transformers。

指标包括 latency、生成 token 数和 tokens per second。

## 核心流程

### `run_minivllm(tokenizer)`

1. 创建 `MiniLLM(config=config)`。
2. 构造 Mini `SamplingParams`。
3. 用同一个 tokenizer chat template 处理 prompt。
4. 预热若干轮。
5. 正式计时一次 `llm.generate()`。
6. 统计 completion token 数。

### `run_vllm(tokenizer)`

使用上游 vLLM 的 `LLM` 和 `SamplingParams`，prompt 与采样设置尽量保持一致。

### `run_transformers_test(tokenizer)`

使用 `AutoModelForCausalLM.generate()` 作为基线。

## TPS 计算示例

如果一次正式计时：

```text
latency = 2.5 秒
outputs["token_ids"] 长度分别是 [80, 90, 70]
total_tokens = 240
```

则：

```text
tps = total_tokens / latency = 240 / 2.5 = 96 tokens/sec
```

脚本打印：

```text
latency: 2.5000
tokens: 240.0000
tps: 96.0000
```

## 预热的意义

预热轮次不计入正式计时，主要排除：

- CUDA kernel 首次加载。
- `torch.compile` 首次编译。
- KV-cache 分配。
- vLLM 内部 graph/cache 初始化。
- 模型权重首次触达显存。

## 可比性注意事项

三个路径并非完全同构：

- Mini vLLM 的 `max_model_length=128` 会限制 prompt + completion 总长度。
- vLLM 设置 `max_model_len=256`。
- Transformers 使用 `max_length=OUTPUT_TOKENS`，这里表示总输出序列长度而不一定是新增 token 数。

因此该 benchmark 更适合做粗略参考。如果要严格公平，需要统一“新增 token 数”和“总长度限制”的语义。

## 注意事项

脚本导入了 `numpy` 和 `matplotlib.pyplot`，但当前没有绘图逻辑。可以后续扩展为柱状图，或清理未使用导入。
