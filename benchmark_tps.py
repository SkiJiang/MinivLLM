"""Mini vLLM、vLLM 和 Transformers 的端到端吞吐量对比。"""

import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,AutoModelForCausalLM

# Mini vLLM 导入用于测试本地实现。
from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

# vLLM 导入提供优化后的参考实现。
from vllm import LLM as VLLM
from vllm import SamplingParams as VLLMSamplingParams



config = {
    # Mini vLLM 的 scheduler/cache 设置。
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,

    # Qwen3 模型参数，必须与 MODEL_NAME 匹配。
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,
    'vocab_size': 151936,
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768,
    'ffn_bias': False,

    # warmup 和 cache size 计算。
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,
}

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "introduce yourself" ,
    "list all prime numbers within 100" ,
    "give me your opinion on the impact of artificial intelligence on society" ,
]

WARMUP_STEPS = 2
OUTPUT_TOKENS = 256
device = "cuda" if torch.cuda.is_available() else "cpu"

def cuda_sync():
    """存在 GPU 时同步 CUDA 计时。"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_minivllm(tokenizer):
    """测试本地 engine，并返回 latency/token/TPS 指标。"""
    llm = MiniLLM(config=config)  
    sampling = MiniSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
        max_model_length=128,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # warmup 避免把一次性编译、cache 分配和 kernel 初始化开销计入正式计时。
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    # 这里只统计生成出来的 completion token。
    total_tokens = sum(len(x) for x in outputs["token_ids"])
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_vllm(tokenizer):
    """用可比的 prompt 和采样设置测试上游 vLLM。"""
    llm = VLLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        trust_remote_code=False, 
        gpu_memory_utilization=0.75,  
        max_model_len=256, 
        speculative_config=None, 
    )

    sampling = VLLMSamplingParams(
        temperature=0.6,
        max_tokens=OUTPUT_TOKENS,
    )

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in PROMPTS
    ]

    # 预热 vLLM 内部 kernel、cache manager 以及 graph/capture 路径。
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_transformers_test(tokenizer):
    """测试原生 Transformers generate 路径，作为基线。"""
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # left padding 需要显式 attention_mask，确保 padding token 被忽略。
    attention_mask = inputs["attention_mask"]

    # 预热设备上的模型权重和生成 kernel。
    for _ in range(WARMUP_STEPS):
        with torch.no_grad():
            model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=OUTPUT_TOKENS)
    end = time.perf_counter()

    total_tokens = sum(len(output) for output in outputs)
    latency = end - start

    tps = total_tokens / latency

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": tps,
    }


def main():
    """运行所有 benchmark 变体，并打印紧凑指标表。"""
    # 所有 engine 使用同一个 tokenizer/template，保证 prompt 完全一致。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side='left')

    print("Running minivllm benchmark...")
    mini = run_minivllm(tokenizer)

    print("Running vLLM benchmark...")
    vllm = run_vllm(tokenizer)

    print("Running transformers benchmark...")
    transformers = run_transformers_test(tokenizer)


    results = {
        "minivllm": mini,
        "vLLM": vllm,
        "transformers":transformers
    }

    # 打印 latency、生成 token 数和每秒生成 token 数。
    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv:.4f}")



if __name__ == "__main__":
    main()
