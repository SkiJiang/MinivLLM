# src/myvllm/__init__.py

## 文件作用

这是 `myvllm` 包的顶层包标记文件。它让 `src/myvllm` 被 Python 识别为可导入包，从而可以使用 `import myvllm`、`from myvllm.engine.llm_engine import LLMEngine` 等导入方式。

## 当前内容

文件只包含一行说明性注释，没有主动导出类或函数。这种设计让包入口保持最小化，避免在用户导入 `myvllm` 时触发模型、CUDA、Triton 或分布式相关的重依赖初始化。

## 运行时影响

该文件本身不参与计算，也不会改变推理流程。真正的功能模块分布在：

- `myvllm.engine`：请求状态、调度、KV cache 管理和模型运行。
- `myvllm.layers`：Transformer 层、attention、张量并行线性层等。
- `myvllm.models`：Qwen3 和 Llama 模型结构。
- `myvllm.utils`：执行上下文和 checkpoint 加载工具。

## 使用示例

仓库采用 `src-layout`，入口脚本会把 `src` 加入 `sys.path`，之后就能导入：

```python
from myvllm.engine.llm_engine import LLMEngine
from myvllm.sampling_parameters import SamplingParams
```

如果项目通过 `pip install -e .` 或 `uv sync` 安装，调用方不需要手动修改 `sys.path`。
