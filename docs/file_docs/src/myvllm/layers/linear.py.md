# src/myvllm/layers/linear.py

## 文件作用

该文件实现张量并行版本的线性层，以及配套的 checkpoint 分片加载逻辑。它是模型多 GPU 推理的基础：不同 rank 只持有完整权重的一部分，通过必要的通信恢复完整输出。

## 基类

### `LinearBase`

保存通用字段：

- `tp_rank`：当前张量并行 rank。
- `tp_size`：总 rank 数。
- `tp_dim`：切分维度。
- `weight.weight_loader`：挂载到参数对象上的自定义加载函数。

子类必须实现 `weight_loader()` 和 `forward()`。

## ReplicatedLinear

每个 rank 保存完整权重，不做切分，也不需要通信。

示例：权重 `(4, 3)`，`tp_size=2`，rank 0 和 rank 1 都有完整 `(4, 3)`。

## ColumnParallelLinear

沿输出维切分完整权重。完整线性层：

```text
y = x @ W.T
W shape = (output_size, input_size)
```

如果：

```text
output_size = 8
input_size = 4
tp_size = 2
```

则每个 rank 保存：

```text
rank0 weight = W[0:4, :]  -> 输出 y 的前 4 维
rank1 weight = W[4:8, :]  -> 输出 y 的后 4 维
```

forward 不做 all-reduce，因为输出本来就是分片。

## RowParallelLinear

沿输入维切分完整权重。假设：

```text
input_size = 8
output_size = 4
tp_size = 2
```

每个 rank 保存：

```text
rank0 weight = W[:, 0:4]
rank1 weight = W[:, 4:8]
```

输入 `x` 也应在最后一维按同样方式切分。两个 rank 各自得到部分结果：

```text
rank0 partial = x0 @ W0.T
rank1 partial = x1 @ W1.T
```

完整结果是二者相加：

```text
y = partial0 + partial1
```

因此 `forward()` 中使用 `dist.all_reduce(SUM)`。

## MergedColumnParallelLinear

用于把多个输出投影打包成一个矩阵。例如 SwiGLU 的 gate/up：

```text
gate_proj: (3072, 1024)
up_proj:   (3072, 1024)
gate_up:   (6144, 1024)
```

在 `tp_size=2` 时，每个 rank 保存 `3072` 行：

```text
rank0: gate 前 1536 行 + up 前 1536 行
rank1: gate 后 1536 行 + up 后 1536 行
```

`weight_loader(param, loaded_weights, loaded_weight_id)` 会把 checkpoint 中的第几个组件拷贝到 packed 参数的正确位置。

## QKVColumnParallelLinear

用于融合 attention 的 Q/K/V 投影。假设：

```text
num_heads = 16
num_kv_heads = 8
head_dim = 128
tp_size = 2
```

全局输出维度：

```text
Q = 16 * 128 = 2048
K = 8  * 128 = 1024
V = 8  * 128 = 1024
total = 4096
```

每个 rank 保存：

```text
Q heads = 8  -> 1024 维
K heads = 4  -> 512 维
V heads = 4  -> 512 维
local total = 2048
```

packed 布局为：

```text
[local Q, local K, local V]
```

## 注意事项

所有并行切分都要求被切分的维度能整除 `tp_size`。例如 `ColumnParallelLinear` 要求 `output_size % tp_size == 0`。
