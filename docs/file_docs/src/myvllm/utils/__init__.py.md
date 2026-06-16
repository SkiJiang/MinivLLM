# src/myvllm/utils/__init__.py

## 文件作用

这是 `myvllm.utils` 子包的导出门面。它重新导出 `context.py` 中的三个函数：

- `get_context`
- `reset_context`
- `set_context`

这样其他模块可以写：

```python
from myvllm.utils import get_context, set_context, reset_context
```

而不需要直接依赖 `myvllm.utils.context` 的具体路径。

## 在项目中的位置

模型层和运行器之间需要共享 attention 元数据，例如 `slot_mapping`、`block_tables`、`cu_seqlens_q`。这些信息由 `ModelRunner.prepare_prefill()` 或 `ModelRunner.prepare_decode()` 设置，再由 `Attention`、`ParallelLMHead` 和模型 decoder layer 读取。

## 计算示例

这个文件不做数值计算，但它让下面的数据流更短：

```python
set_context(is_prefill=True, cu_seqlens_q=cu_seqlens, slot_mapping=slots)
context = get_context()
```

如果没有这个门面，所有调用方都要从 `myvllm.utils.context` 导入，后续移动实现文件时也更容易破坏调用方。

## 注意事项

该文件目前没有导出 `loader.py` 的 checkpoint 加载函数。需要加载权重的代码直接在 `ModelRunner` 内部导入：

```python
from myvllm.utils.loader import load_weights_from_checkpoint
```
