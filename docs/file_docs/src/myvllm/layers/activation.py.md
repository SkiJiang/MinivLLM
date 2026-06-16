# src/myvllm/layers/activation.py

## 文件作用

该文件实现 Transformer MLP 中常用的 SwiGLU 风格激活层 `SiluAndMul`。它用于 Qwen3 和 Llama 的前馈网络，在融合的 gate/up 投影之后执行：

```text
output = silu(gate) * value
```

## 主要对象

### `SiluAndMul`

输入 tensor 最后一维必须是两个 intermediate 向量拼接后的结果。`forward()` 会沿最后一维二等分：

```python
x, y = x.chunk(2, -1)
return F.silu(x) * y
```

其中：

```text
silu(x) = x * sigmoid(x)
```

## 计算示例

假设某个 token 的 fused MLP 输出为：

```text
[gate0, gate1, value0, value1] = [1.0, -1.0, 3.0, 4.0]
```

拆分后：

```text
gate = [1.0, -1.0]
value = [3.0, 4.0]
```

计算 SiLU：

```text
silu(1.0)  = 1.0 * sigmoid(1.0)  ≈ 0.731
silu(-1.0) = -1.0 * sigmoid(-1.0) ≈ -0.269
```

最终输出：

```text
[0.731 * 3.0, -0.269 * 4.0] ≈ [2.193, -1.076]
```

输出维度从 `2 * intermediate_size` 变回 `intermediate_size`。

## 性能意义

模型中 gate 和 up/value 通常来自两个线性层。项目把它们融合成一个 `MergedColumnParallelLinear`，然后用 `SiluAndMul` 一次性处理。这样可以减少一次矩阵乘调用和一些中间调度开销。

## 注意事项

`forward()` 使用 `torch.compile`，首次运行可能有编译开销；benchmark 或真实计时时需要预热。
