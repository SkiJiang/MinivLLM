# 文件文档索引

本目录按原仓库路径镜像，为项目中的可维护文件提供逐文件说明。文档重点覆盖 Python 源码、入口脚本、benchmark、测试和包配置；`uv.lock`、图片资源、许可证和已有教程文章属于锁定/静态/已有说明文件，没有再生成逐文件功能说明。

## 核心包

- [src/myvllm/__init__.py](src/myvllm/__init__.py.md)
- [src/myvllm/sampling_parameters.py](src/myvllm/sampling_parameters.py.md)
- [src/myvllm/utils/__init__.py](src/myvllm/utils/__init__.py.md)
- [src/myvllm/utils/context.py](src/myvllm/utils/context.py.md)
- [src/myvllm/utils/loader.py](src/myvllm/utils/loader.py.md)
- [src/myvllm/engine/sequence.py](src/myvllm/engine/sequence.py.md)
- [src/myvllm/engine/block_manager.py](src/myvllm/engine/block_manager.py.md)
- [src/myvllm/engine/scheduler.py](src/myvllm/engine/scheduler.py.md)
- [src/myvllm/engine/model_runner.py](src/myvllm/engine/model_runner.py.md)
- [src/myvllm/engine/llm_engine.py](src/myvllm/engine/llm_engine.py.md)
- [src/myvllm/layers/__init__.py](src/myvllm/layers/__init__.py.md)
- [src/myvllm/layers/activation.py](src/myvllm/layers/activation.py.md)
- [src/myvllm/layers/layernorm.py](src/myvllm/layers/layernorm.py.md)
- [src/myvllm/layers/linear.py](src/myvllm/layers/linear.py.md)
- [src/myvllm/layers/embedding_head.py](src/myvllm/layers/embedding_head.py.md)
- [src/myvllm/layers/rotary_embedding.py](src/myvllm/layers/rotary_embedding.py.md)
- [src/myvllm/layers/sampler.py](src/myvllm/layers/sampler.py.md)
- [src/myvllm/layers/attention.py](src/myvllm/layers/attention.py.md)
- [src/myvllm/models/__init__.py](src/myvllm/models/__init__.py.md)
- [src/myvllm/models/qwen3.py](src/myvllm/models/qwen3.py.md)
- [src/myvllm/models/llama.py](src/myvllm/models/llama.py.md)

## 脚本、测试和配置

- [main.py](root/main.py.md)
- [main_llama32.py](root/main_llama32.py.md)
- [benchmark_tps.py](root/benchmark_tps.py.md)
- [benchmark_prefilling.py](root/benchmark_prefilling.py.md)
- [benchmark_decoding.py](root/benchmark_decoding.py.md)
- [tests/test_scheduler.py](tests/test_scheduler.py.md)
- [tests/scheduler_tests.md](tests/scheduler_tests.md.md)
- [pyproject.toml](root/pyproject.toml.md)
- [setup.py](root/setup.py.md)
