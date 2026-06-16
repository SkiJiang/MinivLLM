# src/myvllm/models/__init__.py

## 文件作用

这是 `myvllm.models` 子包的标记文件。具体模型结构分别放在：

- `llama.py`
- `qwen3.py`

当前文件只包含说明性注释，没有重新导出模型类。

## 运行时影响

该文件不会触发任何模型初始化、权重加载、CUDA 初始化或分布式初始化。真正的模型选择逻辑位于 `ModelRunner.__init__()`：

```python
match model_name:
    case "Qwen3-0.6B":
        self.model = Qwen3ForCausalLM(...)
    case "Llama-3.2-1B-Instruct":
        self.model = LlamaForCausalLM(...)
```

## 使用示例

调用方可以直接导入具体模型：

```python
from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
```

## 注意事项

如果未来希望支持 `from myvllm.models import Qwen3ForCausalLM`，可以在此文件中显式导入并设置 `__all__`。
