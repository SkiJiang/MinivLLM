# src/myvllm/layers/sampler.py

## 文件作用

该文件实现 next-token 采样层 `SamplerLayer`。它把模型输出 logits 按 temperature 缩放后转成概率，并用指数噪声形式的 Gumbel-max 技巧采样 token。

## 核心公式

`forward(logits, temperature)`：

1. 温度缩放：

```text
scaled_logits = logits / temperature
```

2. softmax 得到概率：

```text
probs = softmax(scaled_logits)
```

3. 用指数噪声采样：

```text
sample = argmax(probs / Exp(1))
```

这里 `Exp(1)` 是均值为 1 的指数分布。该技巧等价于按 `probs` 分布随机采样，但更容易和 `torch.compile` 配合。

## 温度示例

假设 logits 为：

```text
[2.0, 1.0, 0.0]
```

temperature=1：

```text
softmax([2, 1, 0]) ≈ [0.665, 0.245, 0.090]
```

temperature=0.5：

```text
softmax([4, 2, 0]) ≈ [0.867, 0.117, 0.016]
```

分布更尖锐，更偏向第一个 token。

temperature=2：

```text
softmax([1, 0.5, 0]) ≈ [0.506, 0.307, 0.186]
```

分布更平，更随机。

## 指数噪声采样示例

假设概率：

```text
probs = [0.7, 0.2, 0.1]
noise = [1.4, 0.1, 0.5]
probs / noise = [0.5, 2.0, 0.2]
```

虽然第一个 token 概率最大，本次噪声让第二个 token 胜出。长期重复采样时，各 token 被选中的频率仍服从 `probs`。

## 注意事项

`logits /= temperature.unsqueeze(-1)` 是原地操作，会修改传入的 logits tensor。如果后续还需要未缩放 logits，应在调用前复制。
