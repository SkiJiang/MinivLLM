"""Example entry point for running Mini vLLM with Llama-3.2-1B-Instruct."""

import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Allow direct execution from the repository root before package installation.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # Scheduler limits.
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,

    # Warmup and KV-cache sizing.
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,

    # Cache paging and distributed execution.
    'block_size': 256,
    'world_size': 1,

    # Eager mode avoids CUDA graph capture while bringing up the model.
    'enforce_eager': True,

    # Llama architecture parameters.  These must mirror the HF checkpoint config.
    'model_name_or_path': 'meta-llama/Llama-3.2-1B-Instruct',
    'vocab_size': 128256, 
    'hidden_size': 2048,
    'head_dim': 64, 
    'num_qo_heads': 32,
    'num_kv_heads': 8,
    'has_attn_bias': False,
    'rms_norm_epsilon': 1e-5,
    'rope_base': 500000, 
    'max_position_embeddings': 32768, # Must cover the largest RoPE position.
    'intermediate_size': 8192,
    'ffn_bias': False,
    'num_layers': 16,
    'tie_word_embeddings': True,
    'eos': 128009,  # Chosen from the checkpoint's EOS token set.
}

def main():
    """Tokenize chat prompts, run the engine, and print decoded completions."""
    model_name = config.get('model_name_or_path')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = LLM(config=config)
    
    # max_tokens is completion-only; max_model_length includes prompt tokens too.
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    # Llama instruct checkpoints expect the chat template wrapper.
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs["text"] contains generated completion text for each prompt.
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
