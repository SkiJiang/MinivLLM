"""使用 Llama-3.2-1B-Instruct 运行 Mini vLLM 的示例入口。"""

import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 允许在安装包之前直接从仓库根目录执行该文件。
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # scheduler 限制。
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,

    # warmup 和 KV-cache size 计算。
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,

    # cache 分页和分布式执行。
    'block_size': 256,
    'world_size': 1,

    # eager 模式在模型调通阶段避免 CUDA graph capture。
    'enforce_eager': True,

    # Llama 架构参数，必须与 HF checkpoint config 保持一致。
    'model_name_or_path': 'meta-llama/Llama-3.2-1B-Instruct',
    'vocab_size': 128256, 
    'hidden_size': 2048,
    'head_dim': 64, 
    'num_qo_heads': 32,
    'num_kv_heads': 8,
    'has_attn_bias': False,
    'rms_norm_epsilon': 1e-5,
    'rope_base': 500000, 
    'max_position_embeddings': 32768, # 必须覆盖最大的 RoPE position。
    'intermediate_size': 8192,
    'ffn_bias': False,
    'num_layers': 16,
    'tie_word_embeddings': True,
    'eos': 128009,  # 从 checkpoint 的 EOS token 集合中选择。
}

def main():
    """对 chat prompt 做 tokenize，运行 engine，并打印解码后的 completion。"""
    model_name = config.get('model_name_or_path')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(config=config)
    
    # max_tokens 只计算 completion；max_model_length 也包含 prompt token。
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    # Llama instruct checkpoint 需要 chat template 包装。
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs["text"] 保存每个 prompt 对应的生成文本。
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
