# src/myvllm/layers/embedding_head.py

## 文件作用

该文件实现词表并行 embedding 和语言模型输出 head。它们都沿词表维度切分，使大词表模型可以在多张 GPU 上分摊 embedding/lm_head 权重。

## `VocabParallelEmbedding`

### 切分逻辑

假设：

```text
num_embeddings = 10
embedding_dim = 4
tp_size = 2
```

每个 rank 拥有 5 个 token id：

```text
rank0: token 0-4
rank1: token 5-9
```

如果词表不能整除 `tp_size`，会 padding 到可整除。例如：

```text
num_embeddings = 11
tp_size = 2
padded_num_embeddings = 12
num_embeddings_per_partition = 6
```

rank1 中最后一行是 padding 行，会被清零。

### forward 示例

输入 token：

```text
x = [2, 7]
```

在 rank0：

```text
mask = [True, False]
local ids = [2, 0]
embedding 后第二项被 mask 清零
```

在 rank1：

```text
mask = [False, True]
local ids = [0, 1]  # token 7 对 rank1 的局部 id 是 7 - 6 = 1
embedding 后第一项被 mask 清零
```

最后 all-reduce 求和，每个 token 只有所属 rank 贡献非零 embedding。

## `ParallelLMHead`

它继承 `VocabParallelEmbedding`，但 forward 逻辑是线性投影：

```text
logits_local = hidden @ weight_local.T
```

多卡时 rank 0 gather 所有 rank 的词表分片：

```text
rank0 logits: vocab[0:6]
rank1 logits: vocab[6:12]
concat -> vocab[0:12]
truncate -> vocab[0:11]
```

## prefill 中只取最后 token

prefill 阶段模型会为 prompt 中每个 token 产生 hidden state，但采样只需要每个序列最后一个 prompt token 的 next-token logits。

示例：

```text
cu_seqlens_q = [0, 3, 8]
```

两个序列的最后 token 在拼接 tensor 中的位置：

```text
last_token = [3, 8] - 1 = [2, 7]
```

`ParallelLMHead.forward()` 会先取 `x[[2, 7]]`，再计算 logits，避免对中间 token 做无用词表投影。

## 注意事项

rank 0 才需要完整 logits 来采样。非 rank 0 在 gather 后可能返回本地 logits 或不参与后续采样，具体由 `ModelRunner.run()` 控制。
