# src/myvllm/utils/loader.py

## 文件作用

该文件负责把 Hugging Face safetensors checkpoint 加载到项目自定义模型结构中。难点在于本项目为了推理效率做了权重融合，例如把 `q_proj/k_proj/v_proj` 合成一个 `qkv_projection`，把 MLP 的 `gate_proj/up_proj` 合成一个 `gate_up`。

## 主要函数

### `default_weight_loader(param, weight)`

默认加载逻辑：要求参数形状完全一致，然后直接 `copy_`。

### `load_weights_from_checkpoint(model, model_name_or_path)`

完整加载流程：

1. 判断 `model_name_or_path` 是本地路径还是 Hugging Face repo id。
2. 如果不是本地路径，调用 `snapshot_download()` 下载 safetensors 和 json。
3. 读取所有 `.safetensors` 文件到 `hf_weights`。
4. 遍历 HF 参数名，按规则映射到本地模型参数。
5. 打印成功加载、跳过和未加载参数的诊断信息。

## 权重融合示例

假设某层 HF checkpoint 中有：

```text
model.layers.0.self_attn.q_proj.weight: (1024, 1024)
model.layers.0.self_attn.k_proj.weight: (512, 1024)
model.layers.0.self_attn.v_proj.weight: (512, 1024)
```

加载器会拼接：

```python
qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
```

得到：

```text
qkv_weight: (2048, 1024)
```

然后复制到：

```text
model.layers.0.self_attn.qkv_projection.weight
```

MLP 的 gate/up 也是同样思路：

```text
gate_proj.weight: (3072, 1024)
up_proj.weight:   (3072, 1024)
gate_up.weight:   (6144, 1024)
```

`SiluAndMul` 要求 gate 在前、up/value 在后，所以拼接顺序不能反。

## 词表 padding 示例

如果 checkpoint embedding 是 `(151936, 1024)`，但张量并行 padding 后本地参数可能包含额外行。形状不完全一致时，代码会复制重叠的前缀：

```python
min_size = min(param.shape[0], hf_weight.shape[0])
param.data[:min_size].copy_(hf_weight[:min_size])
```

多出来的 padding 行由 embedding 层 loader 清零，避免贡献无效 logits。

## 注意事项

当前加载器在融合 QKV 时直接复制完整拼接 tensor。如果运行在多卡张量并行模式下，还需要确保参数形状与本地分片逻辑匹配，否则应优先使用各参数自带的 `weight_loader` 分片加载策略。
