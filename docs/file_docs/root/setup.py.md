# setup.py

## 文件作用

这是传统 setuptools 安装脚本，用于本地安装 `src-layout` 结构的 `myvllm` 包。虽然项目已有 `pyproject.toml`，保留 `setup.py` 可以兼容一些旧工具或手动安装方式。

## 配置内容

```python
setup(
    name="myvllm",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires="==3.11.14",
    install_requires=["torch"],
)
```

含义：

- 包名是 `myvllm`。
- 源码在 `src/` 目录。
- 自动查找 `src` 下的 Python 包。
- Python 版本严格限定为 `3.11.14`。
- 安装依赖只声明了 `torch`。

## 与 pyproject.toml 的关系

`pyproject.toml` 中的运行依赖更多：

```text
transformers, torch, xxhash, vllm>=0.15.0
```

而 `setup.py` 只写了：

```text
torch
```

如果使用只读取 `setup.py` 的安装路径，运行 engine 可能缺少 `transformers`、`xxhash`、`safetensors` 等依赖。

## 计算示例

`find_packages(where="src")` 会扫描：

```text
src/myvllm
src/myvllm/engine
src/myvllm/layers
src/myvllm/models
src/myvllm/utils
```

并把它们注册为可安装包。安装后可以：

```python
import myvllm
from myvllm.engine.scheduler import Scheduler
```

## 注意事项

`python_requires="==3.11.14"` 非常严格，而 `pyproject.toml` 是 `>=3.11,<3.12`。如果没有必须锁死 patch 版本，建议未来统一成同一条约束。
