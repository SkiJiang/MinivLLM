"""Checkpoint loading helpers for Hugging Face safetensors models."""

import torch
from torch import nn
import os
from safetensors import safe_open
from transformers import AutoConfig
import re


def default_weight_loader(param, weight):
    """Default weight loader that copies weight data to parameter."""
    # Direct loading is valid only when the checkpoint tensor already matches
    # this parameter's local shape.
    if param.shape != weight.shape:
        raise ValueError(f"Shape mismatch: param {param.shape} vs weight {weight.shape}")
    param.data.copy_(weight)


def load_weights_from_checkpoint(model: nn.Module, model_name_or_path: str):
    """
    Load weights from a Hugging Face model checkpoint into the custom model.

    Handles QKV and gate_up weight merging for optimized layers.

    Args:
        model: The target model to load weights into
        model_name_or_path: Path to local checkpoint or Hugging Face model name
    """
    from huggingface_hub import snapshot_download

    # Resolve model_name_or_path to a concrete local directory.  It can already
    # be a local path, or it can be a Hugging Face repository id.
    checkpoint_path = None

    # Prefer local paths so repeated runs do not need hub access.
    if model_name_or_path.startswith('~'):
        checkpoint_path = os.path.expanduser(model_name_or_path)
    elif os.path.isdir(model_name_or_path):
        checkpoint_path = model_name_or_path

    # If no local path exists, ask HF Hub for safetensors and config files only.
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        try:
            checkpoint_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors", "*.json"],
                ignore_patterns=["*.msgpack", "*.h5", "*.bin"]
            )
        except Exception as e:
            raise ValueError(
                f"Could not find or download model '{model_name_or_path}'. "
                f"Error: {e}\n"
                f"Please ensure the model name is correct or provide a valid local path."
            )

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint path not found: {checkpoint_path}")

    # A checkpoint may be sharded across multiple safetensors files.
    safetensor_files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]

    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {checkpoint_path}")

    # Load HF weights by their original parameter names on CPU.
    hf_weights = {}
    for file in sorted(safetensor_files):
        file_path = os.path.join(checkpoint_path, file)
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for weight_name in f.keys():
                hf_weights[weight_name] = f.get_tensor(weight_name)

    # Track loaded and skipped names for the final diagnostic report.
    loaded_params = set()
    skipped_params = []

    # Process every HF tensor and map it into the custom module layout.
    for hf_name, hf_weight in hf_weights.items():
        try:
            # 1. Merge q_proj/k_proj/v_proj into the fused qkv_projection weight.
            if '.self_attn.q_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    k_name = hf_name.replace('q_proj', 'k_proj')
                    v_name = hf_name.replace('q_proj', 'v_proj')

                    if k_name in hf_weights and v_name in hf_weights:
                        q_weight = hf_weight
                        k_weight = hf_weights[k_name]
                        v_weight = hf_weights[v_name]

                        # Fused projection layout is [Q rows, K rows, V rows].
                        qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.self_attn.qkv_projection.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(qkv_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(k_name)
                            loaded_params.add(v_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 2. Merge gate_proj/up_proj into the fused gate_up MLP weight.
            elif '.mlp.gate_proj.weight' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_name in hf_weights:
                        gate_weight = hf_weight
                        up_weight = hf_weights[up_name]

                        # SiluAndMul expects gate first, up/value second.
                        gate_up_weight = torch.cat([gate_weight, up_weight], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.weight"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_weight)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 3. Merge MLP biases the same way when the checkpoint includes them.
            elif '.mlp.gate_proj.bias' in hf_name:
                layer_match = re.search(r'layers\.(\d+)', hf_name)
                if layer_match:
                    layer_idx = layer_match.group(1)
                    up_bias_name = hf_name.replace('gate_proj', 'up_proj')

                    if up_bias_name in hf_weights:
                        gate_bias = hf_weight
                        up_bias = hf_weights[up_bias_name]
                        gate_up_bias = torch.cat([gate_bias, up_bias], dim=0)

                        custom_name = f"model.layers.{layer_idx}.mlp.gate_up.bias"
                        try:
                            param = model.get_parameter(custom_name)
                            param.data.copy_(gate_up_bias)
                            loaded_params.add(custom_name)
                            loaded_params.add(hf_name)
                            loaded_params.add(up_bias_name)
                        except AttributeError:
                            skipped_params.append((custom_name, "Parameter not found"))

            # 4. Companion tensors are consumed by the q_proj/gate_proj cases.
            elif any(x in hf_name for x in ['.k_proj.', '.v_proj.', '.up_proj.']):
                if hf_name not in loaded_params:
                    skipped_params.append((hf_name, "Merged into qkv_projection or gate_up"))

            # 5. Names that match the custom module tree can be loaded directly.
            else:
                try:
                    param = model.get_parameter(hf_name)
                    if param.shape != hf_weight.shape:
                        # Embedding/lm_head tensors may differ by padded vocab
                        # rows.  Copy the overlapping prefix instead of failing.
                        if len(param.shape) > 0 and len(hf_weight.shape) > 0:
                            min_size = min(param.shape[0], hf_weight.shape[0])
                            param.data[:min_size].copy_(hf_weight[:min_size])
                        else:
                            param.data.copy_(hf_weight)
                    else:
                        param.data.copy_(hf_weight)
                    loaded_params.add(hf_name)
                except AttributeError:
                    skipped_params.append((hf_name, "Parameter not found"))

        except Exception as e:
            skipped_params.append((hf_name, f"Error: {str(e)}"))

    # Check for custom model parameters that did not receive checkpoint data.
    unloaded_params = []
    for name, param in model.named_parameters():
        if name not in loaded_params:
            unloaded_params.append(name)

    print(f"\n{'='*80}")
    print(f"Weight Loading Summary:")
    print(f"{'='*80}")
    print(f"Successfully loaded: {len([p for p in loaded_params if not any(x in p for x in ['.k_proj.', '.v_proj.', '.up_proj.'])])} parameter groups")

    if unloaded_params:
        print(f"\n⚠️  WARNING: {len(unloaded_params)} model parameters NOT loaded from checkpoint:")
        for name in unloaded_params[:15]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name} (shape: {param.shape}, mean: {param.data.mean():.6f})")
        if len(unloaded_params) > 15:
            print(f"  ... and {len(unloaded_params) - 15} more")

    if skipped_params:
        # Group skipped entries by reason to keep logs readable.
        merged_skips = [s for s in skipped_params if "Merged" in s[1]]
        not_found_skips = [s for s in skipped_params if "not found" in s[1]]
        no_mapping_skips = [s for s in skipped_params if "No mapping" in s[1]]

        if merged_skips:
            print(f"Skipped (merged into other weights): {len(merged_skips)}")
        if not_found_skips:
            print(f"Skipped (not found in model): {len(not_found_skips)}")
            for name, reason in not_found_skips[:5]:
                print(f"  - {name}")
            if len(not_found_skips) > 5:
                print(f"  ... and {len(not_found_skips) - 5} more")
        if no_mapping_skips:
            print(f"Skipped (no mapping rule): {len(no_mapping_skips)}")
            for name, reason in no_mapping_skips[:5]:
                print(f"  - {name}")
            if len(no_mapping_skips) > 5:
                print(f"  ... and {len(no_mapping_skips) - 5} more")

    print(f"{'='*80}")
    return loaded_params
