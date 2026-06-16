# src/myvllm/engine/llm_engine.py

## 文件作用

该文件提供面向用户的高层生成引擎 `LLMEngine`。它组合 tokenizer、scheduler 和 model runner，负责从字符串 prompt 到生成文本的完整流程。

## 主要对象

### `worker_process(config, rank, event)`

多 GPU 张量并行时，非 0 rank 运行该函数。worker 会初始化自己的 `ModelRunner`，然后进入共享内存事件循环，镜像执行 rank 0 发来的方法调用。

### `LLMEngine`

核心职责：

- 启动 worker 进程。
- 创建 rank 0 的 `ModelRunner`。
- 创建 tokenizer。
- 初始化 `Scheduler`。
- 提供 `add_prompt()`、`step()`、`generate()`。
- 退出时清理 worker 和分布式资源。

## 生成数据流

`generate(prompts, sampling_params)` 的流程：

1. 每个 prompt 通过 tokenizer 编码成 token id。
2. 为每个 token 列表创建 `Sequence` 并加入 scheduler。
3. 循环调用 `step()`，直到 waiting 和 running 都为空。
4. `step()` 内部执行：
   - `scheduler.schedule()`
   - `model_runner.call("run", scheduled_sequences, is_prefill)`
   - `scheduler.postprocess()`
5. 收集完成序列的 completion token。
6. 按 `seq_id` 排序，保证输出顺序和输入 prompt 顺序一致。
7. tokenizer decode 成文本。

## 计算示例

假设传入 3 个 prompt，内部 seq_id 分别是 0、1、2。它们可能因为长度和 EOS 不同，以顺序 1、0、2 完成：

```python
generated_tokens = {
    1: [101, 102],
    0: [201],
    2: [301, 302, 303],
}
```

返回前会排序：

```python
[generated_tokens[0], generated_tokens[1], generated_tokens[2]]
```

因此用户收到的输出仍然对应原 prompt 列表顺序。

吞吐日志计算也在这里完成：

```text
running_time = end_t - start_t + 1e-10
tokens_per_second = num_processed_tokens / running_time
```

prefill 阶段的 `num_processed_tokens` 是本轮 prompt token 总数；decode 阶段是本轮序列数，因为每个序列生成一个 token。

## 多进程示例

如果 `world_size=4`：

- 当前进程是 rank 0。
- 额外启动 rank 1、2、3 三个 worker。
- rank 0 的 `ModelRunner.call("run", ...)` 会把方法名和参数写入共享内存。
- worker 被 event 唤醒后读取共享内存并执行同名方法。

## 注意事项

`exit()` 注册在 `atexit` 中，因此正常退出时会清理 worker。但如果进程被强杀，可能留下分布式端口或共享内存残留；`ModelRunner` 初始化中已有清理旧共享内存的尝试。
