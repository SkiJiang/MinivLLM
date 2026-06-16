# src/myvllm/engine/scheduler.py

## 文件作用

该文件实现推理调度器。它在序列数量、token batch budget 和 KV-cache 容量限制下，决定下一轮运行哪些请求，以及本轮是 prefill 还是 decode。

## 主要对象

`Scheduler` 包含：

- `waiting`：等待 prefill 的序列。
- `running`：已经分配 KV-cache block 的活跃序列。
- `block_manager`：负责物理 cache 分配、追加和释放。
- `max_num_batched_tokens`：一轮最多处理多少 token。
- `max_num_sequences`：一轮最多处理多少个序列。
- `eos`：停止 token id。

## 调度流程

### 1. prefill 优先

只要 waiting 队列中有 prompt 且资源足够，scheduler 会先调度 prefill：

```text
条件:
len(scheduled) < max_num_sequences
seq.num_tokens + current_scheduled_tokens <= max_num_batched_tokens
block_manager.can_allocate(seq)
```

满足后：

1. 从 `waiting` 弹出。
2. 分配 block table。
3. 状态改为 `RUNNING`。
4. 同时加入 `running` 和本轮 `scheduled_sequences`。

### 2. decode 轮转

如果本轮没有 prefill，scheduler 从 `running` 中取序列，每个序列只 decode 一个 token。被调度的序列会重新放回 running 队列，等待下一轮。

## token budget 示例

假设：

```text
max_num_batched_tokens = 6
waiting = [seqA(len=4), seqB(len=3), seqC(len=2)]
```

调度过程：

- seqA 放入 batch，当前 token 数 4。
- seqB 会让总数变成 7，超过 6，因此停止。
- seqC 即使能放下也不能插队，因为 waiting 是 FIFO。

结果：

```text
scheduled = [seqA]
is_prefill = True
waiting = [seqB, seqC]
```

## decode batch 示例

假设：

```text
max_num_batched_tokens = 2
running = [A, B, C]
```

decode 阶段每个序列只消耗 1 个 token budget：

```text
scheduled = [A, B]
running 仍追踪 A、B、C
```

测试文件中专门覆盖了这个场景，防止 C 被 `popleft()` 后因为 batch 满而丢失。

## 抢占示例

如果某个 running 序列准备追加新 token 时需要新 block，但没有空闲 block：

1. 如果还有其他 running 序列，scheduler 会抢占队尾序列，释放它的 cache。
2. 被抢占序列状态回到 `WAITING`。
3. 之后需要重新 prefill，可能借助 prefix cache 跳过已缓存 block。

## 停止条件示例

`postprocess()` 每次接收模型采样出的 token：

```text
seq.append_token(token_id)
```

然后检查：

- token 是否等于 `eos` 且没有 `ignore_eos`。
- completion 长度是否达到 `max_tokens`。
- 总长度是否达到 `max_model_length`。

假设 prompt 长度 10，`max_tokens=3`，依次采样 `[20, 21, 22]`，第三次后：

```text
num_completion_tokens = 3
seq.status = FINISHED
block_manager.deallocate(seq)
```

## 注意事项

prefill 优先有利于让新请求尽快进入系统，但长 prompt 会占用较大 token budget。真实服务中可能需要更复杂的公平性策略。
