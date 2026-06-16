# src/myvllm/engine/block_manager.py

## 文件作用

该文件管理物理 KV-cache block。它负责把每个 `Sequence` 的逻辑 block 映射到预分配的 GPU KV cache 物理页，并实现基于 block hash 的 prefix cache 复用。

## 主要对象

### `Block`

单个物理 cache block 的元数据：

- `block_id`：物理页编号。
- `hash`：完整 block 的前缀相关 hash；`-1` 表示不可 cache。
- `ref_count`：当前有多少活跃序列引用该 block。
- `token_ids`：该 block 对应的 token 内容，用于防止 hash 碰撞。

### `BlockManager`

核心结构：

- `blocks`：所有物理 block 元数据。
- `hash_to_block_id`：prefix cache 索引。
- `free_block_ids`：空闲物理 block 队列。
- `used_block_ids`：当前被运行序列持有的 block。

## 分配示例

假设：

```text
block_size = 4
num_blocks = 3
free_block_ids = [0, 1, 2]
seq.token_ids = [10, 11, 12, 13, 14]
```

该序列需要：

```text
num_blocks = ceil(5 / 4) = 2
block(0) = [10, 11, 12, 13]  # 完整 block，可 hash
block(1) = [14]              # 不完整 block，hash=-1
```

`allocate(seq)` 会分配两个物理 block，例如：

```text
seq.block_table = [0, 1]
free_block_ids = [2]
used_block_ids = {0, 1}
```

第一个完整 block 会登记到 `hash_to_block_id`，第二个不完整 block 不进入 prefix cache。

## prefix hash 计算示例

`compute_hash(token_ids, prefix_hash_value)` 把当前完整 block token 和前一个 block 的 hash 一起计算：

```text
seq A blocks:
block0 = [1, 2, 3, 4], prefix=-1 -> hash=h0
block1 = [5, 6, 7, 8], prefix=h0 -> hash=h1

seq B blocks:
block0 = [9, 9, 9, 9], prefix=-1 -> hash=g0
block1 = [5, 6, 7, 8], prefix=g0 -> hash=g1
```

虽然两个序列的第二个 block token 相同，prefix 不同，所以 `h1 != g1`。这避免了不同上下文中相同局部 token 片段误命中 cache。

## decode 追加示例

假设 `block_size=4`，序列长度变化如下：

- 长度从 5 到 6：新 token 落在已有尾部 partial block，不需要新 block。
- 长度从 7 到 8：尾部 block 刚好填满，计算 hash 并加入 prefix cache。
- 长度从 8 到 9：新 token 开启新逻辑 block，需要分配新的物理 block。

这就是 `append(seq)` 中三个分支：

```text
len % block_size == 0 -> finalize 当前 block
len % block_size == 1 -> 分配新 partial block
其他 -> 继续写入当前 partial block
```

## 回收示例

如果两个序列共享物理 block 0：

```text
block0.ref_count = 2
```

释放一个序列后：

```text
block0.ref_count = 1
```

block 仍不能回到 free 队列。只有 ref_count 降到 0 时，`_deallocate_block()` 才把它放回 `free_block_ids`。

## 注意事项

释放时会保留 `hash_to_block_id` 索引，便于未来 prefix cache 命中。但 `token_ids` 会被清空，后续命中时必须再次校验 token 内容；校验失败就当作 cache miss。
