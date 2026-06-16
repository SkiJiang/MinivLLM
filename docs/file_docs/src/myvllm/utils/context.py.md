# src/myvllm/utils/context.py

## 文件作用

该文件定义一次模型 forward 期间共享的执行上下文 `Context`。它避免把 attention 需要的大量元数据逐层传入 Transformer，而是由 `ModelRunner` 在 forward 前设置，由各层通过 `get_context()` 读取。

## 主要对象

`Context` 字段包括：

- `is_prefill`：当前是否是 prompt prefill。
- `cu_seqlens_q` / `cu_seqlens_k`：变长序列拼接后的累积长度。
- `max_seqlen_q` / `max_seqlen_k`：当前 batch 内最大 query/key 长度。
- `slot_mapping`：每个新 token 的 K/V 写入哪个物理 cache slot。
- `context_lens`：decode 阶段每个序列的完整上下文长度。
- `block_tables`：逻辑 block 到物理 KV-cache block 的映射。

模块级变量 `_context` 保存当前上下文。

## 数据流

prefill：

1. `ModelRunner.prepare_prefill()` 拼接 input token。
2. 计算 `cu_seqlens_q`、`slot_mapping`、可选 `block_tables`。
3. 调用 `set_context(is_prefill=True, ...)`。
4. `Attention.forward()` 使用 Flash Attention。
5. `ParallelLMHead.forward()` 只取每个序列最后一个 prompt token 的 logits。
6. `ModelRunner.run()` 调用 `reset_context()`。

decode：

1. `ModelRunner.prepare_decode()` 只准备每个序列的最后一个 token。
2. 设置 `context_lens`、`slot_mapping`、`block_tables`。
3. `Attention.forward()` 使用 paged attention 从 KV cache 中读取历史。

## 计算示例

假设有两个 prompt，长度分别为 3 和 5，且没有 prefix cache：

```text
seq0 token 位置: 0 1 2
seq1 token 位置: 0 1 2 3 4
拼接后长度: 8
cu_seqlens_q = [0, 3, 8]
```

Flash Attention kernel 可以用这个边界知道：

- 第一个序列在拼接 tensor 的 `[0:3]`。
- 第二个序列在拼接 tensor 的 `[3:8]`。

decode 时如果 batch 中两个序列长度分别为 4 和 7：

```text
context_lens = [4, 7]
```

paged attention 会让第一个序列只读前 4 个历史 token，第二个序列读前 7 个。

## 注意事项

`_context` 是进程级全局状态。每次模型运行后必须 `reset_context()`，否则下一批 forward 可能误用旧的 block table 或 slot mapping。
