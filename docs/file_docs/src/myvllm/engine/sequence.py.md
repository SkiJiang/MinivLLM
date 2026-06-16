# src/myvllm/engine/sequence.py

## 文件作用

该文件定义生成请求在引擎内部的状态表示。一个 `Sequence` 同时保存逻辑 token 列表、采样停止参数、调度状态，以及逻辑 block 到物理 KV-cache block 的映射。

## 主要对象

### `SequenceStatus`

序列生命周期：

- `WAITING`：尚未分配 KV-cache block。
- `RUNNING`：已经拥有 block table，可以被调度执行。
- `FINISHED`：完成生成，cache 已释放。

### `Sequence`

核心字段：

- `seq_id`：单调递增 id，用于输出重新排序。
- `token_ids`：prompt + completion 的完整 token。
- `num_prompt_tokens`：prompt 长度，用来切出 completion。
- `num_cached_tokens`：prefix cache 命中的 token 数。
- `block_table`：逻辑 block 到物理 block id 的映射。
- `temperature/max_tokens/ignore_eos/max_model_length`：从 `SamplingParams` 复制来的停止与采样参数。

## block 计算示例

假设：

```text
block_size = 4
token_ids = [10, 11, 12, 13, 14, 15, 16]
```

则：

```text
num_tokens = 7
num_blocks = ceil(7 / 4) = 2
block(0) = [10, 11, 12, 13]
block(1) = [14, 15, 16]
last_block_num_tokens = 3
```

如果追加 token `17`：

```text
token_ids = [10, 11, 12, 13, 14, 15, 16, 17]
num_blocks = ceil(8 / 4) = 2
last_block_num_tokens = 0
```

这里最后一个 block 刚好填满，`last_block_num_tokens` 返回 0 是为了配合当前切片约定。scheduler 通常在“尾块未满”时使用它。

## completion 计算示例

假设 prompt token 为 `[1, 2, 3]`，生成了 `[8, 9]`：

```text
num_prompt_tokens = 3
token_ids = [1, 2, 3, 8, 9]
prompt_token_ids = [1, 2, 3]
completion_token_ids = [8, 9]
num_completion_tokens = 2
```

`LLMEngine.generate()` 最终只 decode `completion_token_ids`。

## 序列化逻辑

`__getstate__()` 为多进程 worker 传输做了压缩：

- prefill 阶段：completion 还为空，需要传完整 prompt token。
- decode 阶段：模型只输入最新 token，历史来自 KV cache，因此只传 `last_token` 和 cache 元数据。

示例：prompt 长度 5，已生成 2 个 token，decode worker 收到的 token 列表只会是 `[last_token]`，但 `num_tokens=7` 和 `block_table` 仍保留，用于 attention 读取完整上下文。

## 注意事项

`Sequence.counter` 是进程内全局计数器。测试或长生命周期服务中如果需要从 0 重新编号，需要显式重建进程或重置计数器。
