"""Per-forward execution context shared by model layers.

The engine prepares attention metadata immediately before calling the model.
Layers read it through get_context() instead of threading many arguments through
every module in the transformer stack.
"""

from dataclasses import dataclass 
import torch 


@dataclass
class Context:
    """Metadata required by attention and output-head layers for one forward pass."""

    # True for prompt prefill, False for one-token decode.
    is_prefill: bool = False
    # Cumulative query/key sequence lengths for varlen prefill kernels.
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # Maximum query/key lengths in the current batch; kept for kernels that need
    # static launch parameters or validation.
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # Flattened physical cache slot for each token whose K/V should be written.
    slot_mapping: torch.Tensor | None = None
    # Decode-only full context length per sequence.
    context_lens: torch.Tensor | None = None
    # Decode/prefix-cache block table mapping logical block index to physical id.
    block_tables: torch.Tensor | None = None

# Module-level singleton.  ModelRunner resets it after every forward pass.
_context = Context()

def get_context() -> Context:
    """Return the current forward-pass context."""
    return _context

def reset_context():
    """Clear context to default values after a model run."""
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """Replace the current context with freshly prepared metadata."""
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
