"""End-to-end throughput comparison between Mini vLLM, vLLM, and Transformers."""

import time
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,AutoModelForCausalLM

# Mini vLLM imports exercise the local implementation.
from myvllm.engine.llm_engine import LLMEngine as MiniLLM
from myvllm.sampling_parameters import SamplingParams as MiniSamplingParams

# vLLM imports provide the optimized reference implementation.
from vllm import LLM as VLLM
from vllm import SamplingParams as VLLMSamplingParams



config = {
    # Scheduler/cache settings for Mini vLLM.
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,

    # Qwen3 model parameters.  They must match MODEL_NAME.
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

    # Warmup/cache sizing.
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
    """Synchronize CUDA timers when a GPU is present."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_minivllm(tokenizer):
    """Benchmark the local engine and return latency/token/TPS metrics."""
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

    # Warmup avoids measuring one-time compilation, cache allocation, and kernel
    # initialization costs.
    for _ in range(WARMUP_STEPS):
        llm.generate(prompts, sampling)
        cuda_sync()

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    cuda_sync()
    end = time.perf_counter()

    # Count generated completion tokens only.
    total_tokens = sum(len(x) for x in outputs["token_ids"])
    latency = end - start

    return {
        "latency": latency,
        "tokens": total_tokens,
        "tps": total_tokens / latency,
    }


def run_vllm(tokenizer):
    """Benchmark upstream vLLM with comparable prompt and sampling settings."""
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

    # Warm up vLLM's internal kernels, cache manager, and graph/capture paths.
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
    """Benchmark the plain Transformers generate path as a baseline."""
    inputs = tokenizer(PROMPTS, return_tensors="pt", padding=True, truncation=True).to(device)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # Left padding requires an explicit mask so padded tokens are ignored.
    attention_mask = inputs["attention_mask"]

    # Warm up model weights on device and generation kernels.
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
    """Run all benchmark variants and print a compact metrics table."""
    # Use the same tokenizer/template for every engine to keep prompts identical.
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

    # Print latency, generated-token count, and generated tokens per second.
    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"{k}:")
        for kk, vv in v.items():
            print(f"  {kk}: {vv:.4f}")



if __name__ == "__main__":
    main()
