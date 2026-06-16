# pyproject.toml

## 文件作用

这是项目的现代 Python 包配置文件，定义项目元数据、依赖、构建系统和开发依赖。`uv sync`、`pip`、`setuptools` 都会读取其中的部分信息。

## 项目元数据

```toml
[project]
name = "myvllm"
version = "0.1.0"
description = "A custom vLLM project"
readme = "README.md"
requires-python = ">=3.11,<3.12"
```

这表示包名是 `myvllm`，要求 Python 版本至少 3.11 且小于 3.12。

## 运行依赖

```toml
dependencies = [
    "transformers",
    "torch",
    "xxhash",
    "vllm>=0.15.0",
]
```

用途：

- `torch`：模型、张量、CUDA、分布式。
- `transformers`：tokenizer、HF config/model 工具。
- `xxhash`：prefix cache block hash。
- `vllm`：benchmark 中的上游参考实现。

注意：源码还使用 `triton`、`safetensors`、`huggingface_hub`、`matplotlib` 等。部分依赖可能由 torch/vllm/transformers 间接提供，但如果单独运行某些脚本，可能需要确认环境中已安装。

## src-layout 配置

```toml
[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

这告诉 setuptools 包源码位于 `src/`，最终可导入包是 `src/myvllm`。

## 开发依赖

文件同时定义了：

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "isort"]

[dependency-groups]
dev = ["pytest>=7.0", "black>=23.0", "isort>=5.0"]
```

它们都表达开发工具依赖，后一种是 uv 常见的 dependency group 写法。

## 计算示例

Python 版本约束：

```text
>=3.11,<3.12
```

允许：

```text
3.11.0, 3.11.14
```

不允许：

```text
3.10.13, 3.12.0
```

## 注意事项

`setup.py` 中写了 `python_requires=="3.11.14"`，比这里更严格。两者如果同时被工具读取，可能造成环境约束不一致。
