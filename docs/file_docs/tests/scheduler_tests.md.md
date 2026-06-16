# tests/scheduler_tests.md

## 文件作用

这是 scheduler 测试的人工说明文档，告诉开发者如何安装依赖、运行测试，以及每个测试类守护的行为。

## 运行命令

全部测试：

```bash
python3 -m pytest tests/test_scheduler.py -v
```

只运行某个测试类：

```bash
python3 -m pytest tests/test_scheduler.py::TestBug2TokenLimitBreak -v
python3 -m pytest tests/test_scheduler.py::TestBug1CanAppendFailure -v
python3 -m pytest tests/test_scheduler.py::TestSchedulerHappyPath -v
```

## 文档内容解释

### `TestBug2TokenLimitBreak`

说明 token budget 或 sequence-count limit 达到时，scheduler 不能把已经弹出的序列弄丢。

示例：

```text
max_num_batched_tokens = 2
running = [A, B, C]
```

一轮只能 decode A 和 B，但 C 必须仍在 `running` 或其他可追踪集合中。

### `TestBug1CanAppendFailure`

说明当 `block_manager.can_append()` 返回 False 时，当前序列不能消失。

示例：

```text
running = [A, B]
can_append(A) = False
```

调度器可能抢占某个序列释放 cache，但 A 和 B 都必须仍能在 `waiting`、`running` 或 `scheduled` 中找到。

### `TestSchedulerHappyPath`

说明正常调度行为：

- waiting prompt 优先 prefill。
- decode budget 足够时，所有 running 序列都进入本轮 batch。
- 唯一 running 序列无法追加时，会回到 waiting。

## 与源码测试的关系

本文件不是 pytest 自动执行的测试，而是 `tests/test_scheduler.py` 的阅读指南。真正的断言在 Python 测试文件中。

## 可维护建议

每次修改 scheduler 队列逻辑时，应同步检查这里的说明是否仍然准确，尤其是“序列守恒”的定义。
