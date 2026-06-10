# 最小 setuptools 元数据，用于在本地安装 src-layout 结构的包。
from setuptools import setup, find_packages

setup(
    # src/myvllm 对外暴露的导入名称。
    name="myvllm",
    version="0.1.0",
    # 源代码位于 src/ 下，而不是仓库根目录。
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    # 当前项目锁定到本地开发环境使用的 Python 版本。
    python_requires="==3.11.14",
    install_requires=[
        # 核心张量和运行时依赖。脚本、benchmark 使用的其他工具没有在这里固定。
        "torch",
    ],
)
