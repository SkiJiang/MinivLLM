"""使用 Qwen3-0.6B 运行 Mini vLLM 的示例入口。"""

import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 允许在不先安装包的情况下，直接从仓库根目录运行该文件。
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # scheduler 和 KV-cache 设置。
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,

    # 模型身份与执行模式。
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,

    # Qwen3 架构参数，必须与 checkpoint config 匹配。
    'vocab_size': 151936,  # HF 模型使用 151936 行 token。
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,  # Qwen3-0.6B 中 hidden_size / num_heads 的结果。
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,  # 该 checkpoint 的 HF rope_theta。
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768, # 必须覆盖 RoPE 可能使用的最大位置。
    'ffn_bias': False,  # HF Qwen3 不使用 MLP bias。

    # ModelRunner 用于 warmup 和 cache size 计算的设置。
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # 应当与 tokenizer.eos_token_id 匹配。
}

def main():
    """对 prompt 做 tokenize，运行生成，并打印 completion。"""
    # 显式 cache 路径用于说明本地模型文件预期放在哪里。
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)
    
    # max_tokens 限制 completion 长度；max_model_length 限制 prompt + completion 总长度，
    # 同时也约束 decode 位置和 KV-cache 使用量。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    # chat checkpoint 需要 prompt template，而不只是原始用户文本。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs["text"] 按原始 prompt 顺序保存解码后的 completion。
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
