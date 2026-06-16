# src/myvllm/layers/layernorm.py

## 文件作用

该文件实现 RMSNorm，并支持把 residual 相加和 RMSNorm 结合在一个接口中。虽然类名叫 `LayerNorm`，实际计算是 Llama/Qwen 常用的 RMSNorm。

## 主要对象

### `LayerNorm`

初始化参数：

- `gamma`：归一化后的逐维缩放参数，保存为 `weight`。
- `eps`：防止除零的小常数。

核心方法：

- `rms_forward(x)`：普通 RMSNorm。
- `residual_rms_forward(x, residual)`：先 `x + residual`，再 RMSNorm，并返回更新后的 residual。
- `forward(x, residual=None)`：根据是否传入 residual 选择路径。

## RMSNorm 公式

对最后一维进行：

```text
variance = mean(x^2) + eps
y = x / sqrt(variance) * weight
```

它不同于 LayerNorm：RMSNorm 不减均值，只按均方根缩放。

## 计算示例

假设：

```text
x = [3, 4]
weight = [0.5, 0.5]
eps = 0
```

计算：

```text
mean(x^2) = (9 + 16) / 2 = 12.5
sqrt(12.5) ≈ 3.536
x / sqrt = [0.849, 1.131]
y = [0.849, 1.131] * [0.5, 0.5] = [0.4245, 0.5655]
```

带 residual 时，若：

```text
x = [1, 2]
residual = [3, 4]
```

先得到：

```text
x + residual = [4, 6]
```

然后对 `[4, 6]` 做 RMSNorm，同时把 `[4, 6]` 作为新的 residual 返回给下一子层。

## 在模型中的位置

decoder layer 中使用方式：

1. 第一层没有 residual，先把输入保存为 residual，再归一化。
2. attention 输出后，与 residual 相加并归一化。
3. MLP 输出再交给下一层，由下一层继续 residual 流。

## 注意事项

`gamma` 属性只是 `weight` 的兼容别名，方便旧代码访问。
