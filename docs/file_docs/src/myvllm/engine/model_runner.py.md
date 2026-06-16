# src/myvllm/engine/model_runner.py

## 文件作用

该文件是模型执行层，负责 CUDA 设备、张量并行进程组、权重加载、KV-cache 分配、prefill/decode 输入准备、CUDA graph 捕获以及 worker 共享内存 RPC。

## 初始化流程

`ModelRunner.__init__()` 主要做以下事情：

1. 初始化 NCCL 进程组并设置当前 CUDA device。
2. 根据 `model_name_or_path` 选择 Qwen3 或 Llama 模型结构。
3. 将模型移动到当前 rank 的 GPU。
4. 从 checkpoint 加载权重。
5. 创建 sampler。
6. 运行 warmup，估计 activation 峰值显存。
7. 分配 paged KV-cache。
8. 可选捕获 decode CUDA graph。
9. 多卡时创建或连接共享内存段。

## KV-cache 容量计算

`allocate_kv_cache()` 的关键公式：

```text
block_bytes =
    block_size
  * 2                      # K 和 V
  * num_layers
  * local_num_kv_heads
  * head_dim
  * dtype_size
```

示例：

```text
block_size = 256
num_layers = 28
num_kv_heads = 8
world_size = 1
head_dim = 128
dtype_size = 4  # float32
```

则每个物理 block 需要：

```text
256 * 2 * 28 * 8 * 128 * 4 = 58,720,256 bytes ≈ 56 MB
```

如果可用于 KV cache 的显存约 10 GB：

```text
num_available_kv_blocks = floor(10 GB / 56 MB) ≈ 182
```

多 rank 模式下，每个 rank 都会算自己的可用 block 数，然后用 `all_reduce(MIN)` 取最小值，保证 scheduler 不会分配超过任何一张卡能承受的 block 数。

## prefill 输入准备示例

假设两个序列：

```text
seq0 token_ids = [1, 2, 3], block_table = [5]
seq1 token_ids = [4, 5],    block_table = [6]
block_size = 4
```

没有 prefix cache 时：

```text
input_ids = [1, 2, 3, 4, 5]
seqlens_q = [3, 2]
cu_seqlens_q = [0, 3, 5]
slot_mapping = [
  5*4+0, 5*4+1, 5*4+2,
  6*4+0, 6*4+1
]
```

attention 用 `cu_seqlens_q` 切分序列，cache 写入用 `slot_mapping` 定位物理位置。

如果 seq0 前 4 个 token 命中 prefix cache，则那些 token 不会放入 `input_ids`，`prepare_prefill()` 也会把 `block_table` 放入 context。需要注意的是，当前 `Attention.forward()` 的 prefill 路径主要把本轮计算出的 Q/K/V 和 `cu_seqlens_q` 传给 Flash Attention；要让 prefill kernel 在同一轮显式读取 cached prefix K/V，还需要继续扩展 prefill attention 对 `block_tables`、`cu_seqlens_k` 的消费逻辑。

## decode 输入准备示例

假设：

```text
seq.last_token = 99
seq.num_tokens = 9
seq.block_table = [3, 7, 8]
block_size = 4
last_block_num_tokens = 1
```

decode 只输入：

```text
input_ids = [99]
context_lens = [9]
slot_mapping = [8 * 4 + 1 - 1] = [32]
```

也就是说当前 token 的 K/V 会写入物理 block 8 的第 0 个位置，而 attention 会通过 `block_tables=[3,7,8]` 读取完整 9 token 上下文。

## CUDA graph 逻辑

prefill 长度变化大，使用 eager。decode 每个序列只处理一个 token，shape 较稳定，因此可以按 batch size 桶捕获 CUDA graph：

```text
batch_sizes = [1, 2, 4, 8, 16, 32, ...]
```

运行时选择能容纳当前 batch 的最小 graph。例如当前 batch size 是 6，会复用 batch size 8 的 graph，只读取前 6 行输出。

## 注意事项

`capture_cudagraph()` 使用 `config['max_num_seqs']`，而其他配置中常见字段是 `max_num_sequences`。如果关闭 `enforce_eager=False`，这里需要确认配置键名一致，否则可能触发 KeyError。
