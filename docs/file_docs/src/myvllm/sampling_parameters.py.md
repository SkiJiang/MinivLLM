# src/myvllm/sampling_parameters.py

## 文件作用

该文件定义 `SamplingParams`，用于描述一次生成请求的采样行为和停止条件。`LLMEngine.add_prompt()` 会把它复制进每个 `Sequence`，之后 scheduler 在每次采样后根据这些字段判断请求是否完成。

## 主要对象

`SamplingParams` 是一个 dataclass，包含：

- `temperature`：采样温度，越大越随机，越小越接近贪心。
- `max_tokens`：最多生成多少个 completion token，不包含 prompt token。
- `ignore_eos`：是否忽略 EOS token。
- `max_model_length`：prompt + completion 的总长度上限。

`__post_init__()` 会断言 `temperature > 1e-10`。当前 sampler 使用随机采样路径，不支持真正的 greedy temperature=0。

## 数据流

1. 用户创建 `SamplingParams`。
2. `LLMEngine.add_prompt()` 创建 `Sequence(token_ids, block_size, sampling_params)`。
3. `Sequence` 保存 temperature、max_tokens、ignore_eos、max_model_length。
4. `Scheduler.postprocess()` 在每轮采样后检查停止条件。

## 计算示例

假设 prompt 有 20 个 token：

```python
sampling = SamplingParams(
    temperature=0.6,
    max_tokens=5,
    ignore_eos=False,
    max_model_length=23,
)
```

生成过程中的停止判断如下：

- 生成第 1 个 token 后，总长度是 21，completion 长度是 1，继续。
- 生成第 3 个 token 后，总长度是 23，触发 `max_model_length`，停止。
- 即使 `max_tokens=5`，也不会继续到 5 个 completion token。
- 如果第 2 个 token 是 EOS 且 `ignore_eos=False`，会更早停止。

温度缩放会在 `SamplerLayer` 中使用。例如 logits 为 `[2.0, 1.0]`：

- `temperature=1.0` 时 softmax 输入仍是 `[2.0, 1.0]`。
- `temperature=0.5` 时输入变成 `[4.0, 2.0]`，高分 token 概率更集中。
- `temperature=2.0` 时输入变成 `[1.0, 0.5]`，分布更平。

## 注意事项

`max_tokens` 只限制生成部分；`max_model_length` 同时限制 prompt 和 completion。因此长 prompt 可能很快触发总长度停止。
