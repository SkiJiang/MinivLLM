"""Example entry point for running Mini vLLM with Qwen3-0.6B."""

import sys, os
from pathlib import Path
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Allow running this file directly from the repository root without installing
# the package first.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.engine.llm_engine import LLMEngine as LLM
from myvllm.sampling_parameters import SamplingParams

config = {
    # Scheduler and KV-cache settings.
    'max_num_sequences': 16,
    'max_num_batched_tokens': 1024,
    'max_cached_blocks': 1024,
    'block_size': 256,
    'world_size': 1,

    # Model identity and execution mode.
    'model_name_or_path': 'Qwen/Qwen3-0.6B',
    'enforce_eager': True,

    # Qwen3 architecture parameters.  These must match the checkpoint config.
    'vocab_size': 151936,  # HF model uses 151936 token rows.
    'hidden_size': 1024,
    'num_heads': 16,
    'head_dim': 128,  # hidden_size / num_heads for Qwen3-0.6B.
    'num_kv_heads': 8,
    'intermediate_size': 3072,
    'num_layers': 28,
    'tie_word_embeddings': True,
    'base': 1000000,  # HF rope_theta for this checkpoint.
    'rms_norm_epsilon': 1e-6,
    'qkv_bias': False,
    'scale': 1,
    'max_position': 32768, # Must cover the largest position used by RoPE.
    'ffn_bias': False,  # HF Qwen3 does not use MLP bias.

    # Warmup/cache sizing settings used by ModelRunner.
    'max_num_batch_tokens': 4096,
    'max_model_length': 128,
    'gpu_memory_utilization': 0.9,
    'eos': 151645,  # Should match tokenizer.eos_token_id.
}

def main():
    """Tokenize prompts, run generation, and print completions."""
    # The explicit cache path documents where local model files are expected.
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    model_name = config.get('model_name_or_path', 'Qwen/Qwen3-0.6B')
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=path)
    llm = LLM(config=config)
    
    # max_tokens limits completion length; max_model_length limits prompt plus
    # completion, which also bounds decode positions and KV-cache usage.
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256, max_model_length=128)
    prompts = [
        "introduce yourself",# * 15,
        "list all prime numbers within 100",# * 15,
        "give me your opinion on the impact of artificial intelligence on society",# * 15,
    ] #* 30
    # Chat checkpoints expect the prompt template, not just raw user text.
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    # outputs["text"] contains decoded completions in the original prompt order.
    generated_texts = outputs['text']

    for prompt, output in zip(prompts, generated_texts):
        print("\n")
        print(f"Prompt: {prompt}")
        print(f"Completion: {output}")


if __name__ == "__main__":
    main()
