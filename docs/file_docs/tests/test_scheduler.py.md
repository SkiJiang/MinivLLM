# tests/test_scheduler.py

## 文件作用

该文件是 scheduler 的回归测试，重点验证调度过程中序列不会丢失，并覆盖 prefill 优先、decode batch 限制和抢占行为。

## 测试辅助函数

### `make_scheduler(...)`

创建一个小型 scheduler，默认：

```text
max_num_batched_tokens = 100
max_num_sequences = 10
max_cached_blocks = 100
block_size = 4
eos = 0
```

这些值让测试可以轻松构造边界场景。

### `inject_running(scheduler, *seqs)`

绕过真实 prefill 分配，直接把序列放进 `running`。测试会用 `MagicMock` 替换 block manager，因此不需要真实 block table。

### `all_tracked(scheduler, scheduled)`

返回当前 scheduler 能追踪到的全部序列：

```text
running ∪ waiting ∪ scheduled
```

用于检查“序列守恒”。

## Bug 2：batch limit break 后丢序列

场景：

```text
running = [A, B, C]
max_num_batched_tokens = 2
can_append 全部 True
```

预期：

```text
scheduled = [A, B]
C 仍在 running
```

曾经的错误模式是 C 被 `popleft()` 后发现 batch 满，然后直接 `break`，但没有放回 running，导致 C 永久消失。

同一类测试还覆盖 `max_num_sequences=2` 的变体。

## Bug 1：can_append False 后丢当前序列

场景：

```text
running = [A, B]
can_append(A) = False
```

预期：A 要么被抢占到 waiting，要么放回 running 等待重试，但不能消失。

错误模式：

```text
A 被 popleft
can_append(A) False
代码抢占 running.pop()，也就是 B
A 没有恢复
```

测试通过 `all_tracked()` 确认 A 和 B 都仍可追踪。

## happy path 测试

### `test_prefill_scheduled_first`

waiting 中的新 prompt 应先被调度为 prefill，并进入 running。

### `test_all_running_seqs_scheduled_when_budget_allows`

预算足够时，所有 running 序列都应进入本轮 decode，且继续留在 running 中等待下一轮。

### `test_preempt_only_seq_when_cant_append_and_running_empty`

如果唯一 running 序列无法 append，应被抢占回 waiting，状态改为 `WAITING`。

## 计算示例

`block_size=4`，序列 `[1,2,3]` 当前长度为 3。下一次 decode 后长度会变成 4，不需要新 block；如果长度已经是 4，下一 token 会开启新 block，此时 `can_append()` 可能因为没有空闲 block 返回 False。

测试通过 mock 控制 `can_append()` 返回值，专注验证 scheduler 队列操作，而不是 block manager 的真实分配。
