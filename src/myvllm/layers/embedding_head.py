"""Tensor-parallel token embedding and language-model head layers."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from myvllm.utils import get_context


class VocabParallelEmbedding(nn.Module):
    """Embedding table sharded over vocabulary ids.

    Each rank owns a contiguous slice of token ids.  During forward, out-of-rank
    ids are masked to zero locally, then all ranks sum their partial embeddings.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.tp_size = dist.get_world_size()
        self.tp_rank = dist.get_rank()

        # Keep the original vocabulary size for trimming and range checks.
        self.num_embeddings = num_embeddings
        # Pad the global vocab so every rank owns the same number of rows.
        self.padded_num_embeddings = (num_embeddings + self.tp_size - 1) // self.tp_size * self.tp_size
        # Number of embedding rows allocated on this rank.
        self.num_embeddings_per_partition = self.padded_num_embeddings // self.tp_size
        self.embedding_dim = embedding_dim

        # The parameter shape is local-vocab x hidden-size.
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # Custom loaders know how to copy only this rank's shard from a full
        # checkpoint tensor.
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weights: torch.Tensor):
        """Load this rank's vocabulary shard from full embedding weights."""
        param_data = param.data

        # Global row range assigned to this rank after padding.
        offset = self.tp_rank * self.num_embeddings_per_partition
        shard_size = self.num_embeddings_per_partition

        # Some of the padded range may sit beyond the real vocabulary.
        actual_start = min(offset, self.num_embeddings)
        actual_end = min(offset + shard_size, self.num_embeddings)
        actual_size = max(0, actual_end - actual_start)

        if actual_size > 0:
            # Copy only real rows from the checkpoint.
            sharded_weights = loaded_weights.narrow(0, actual_start, actual_size)
            param_data[:actual_size].copy_(sharded_weights)

        # Padded rows should never contribute logits/embeddings.
        if actual_size < shard_size:
            param_data[actual_size:].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Look up embeddings with vocabulary-parallel masking."""
        # mask identifies token ids that belong to this rank and are not padding.
        mask = (x >= self.tp_rank * self.num_embeddings_per_partition) & \
               (x < (self.tp_rank + 1) * self.num_embeddings_per_partition) & \
               (x < self.num_embeddings)
        # Convert global token ids into this rank's local row indices.  Masked
        # entries become zero temporarily and are zeroed again after lookup.
        x = mask * (x - self.tp_rank * self.num_embeddings_per_partition)
        output = F.embedding(x, self.weight)

        if dist.get_world_size() > 1:
            # Without this mask, out-of-rank token ids would contribute row 0's
            # embedding from every non-owning rank.
            output = mask.unsqueeze(1) * output
            # Sum partial embeddings; exactly one rank contributes a nonzero
            # vector for each real token id.
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output

class ParallelLMHead(VocabParallelEmbedding):
    """Vocabulary-parallel output projection, optionally tied to embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project hidden states to logits and gather vocabulary shards on rank 0."""
        context = get_context()
        if context.is_prefill:
            # During prefill we only need logits for each sequence's last prompt
            # token; earlier prompt positions are not sampled.
            last_token = context.cu_seqlens_q[1:] - 1  # exclude the first element which is 0
            x = x[last_token].contiguous()

        # Local logits cover only this rank's vocabulary shard.
        logits = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            # Only rank 0 needs the full logits for sampling.
            all_logits = [torch.empty(logits.size(), device=logits.device) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, gather_list=all_logits, dst=0)
            if self.tp_rank == 0:
                # Concatenate vocabulary shards and trim padded rows.
                logits = torch.cat(all_logits, dim=-1)
                logits = logits[..., :self.num_embeddings]

        return logits
