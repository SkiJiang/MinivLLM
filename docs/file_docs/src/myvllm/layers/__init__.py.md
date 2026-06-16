# src/myvllm/layers/__init__.py

## 文件作用

这是 `myvllm.layers` 子包的统一导出入口。模型文件可以从 `myvllm.layers` 一次性导入所有基础层，而不需要分别写多个模块路径。

## 导出的对象

显式导出：

- `SiluAndMul`
- `Attention`
- `ParallelLMHead`
- `VocabParallelEmbedding`
- `LayerNorm`
- `RotaryEmbedding`
- `SamplerLayer`

并通过：

```python
from .linear import *
```

导出 `linear.py` 中的张量并行线性层。

## 在项目中的作用

`qwen3.py` 和 `llama.py` 都使用：

```python
from myvllm.layers import *
```

这让模型结构代码更像 Transformer 架构描述，避免被具体文件路径打断。

## 计算示例

该文件本身不执行数值计算。它的价值是把模型层构造中的导入简化为：

```python
self.gate_up = MergedColumnParallelLinear(...)
self.activation = SiluAndMul()
self.rotary_emb = RotaryEmbedding(...)
self.attention = Attention(...)
```

如果没有这个门面，模型文件需要分别从 `activation.py`、`linear.py`、`attention.py` 等模块导入。

## 注意事项

`import *` 会让命名空间更宽，适合这个小型教学/实验项目。大型项目中通常会显式列出 `__all__`，减少意外导出。
