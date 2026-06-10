"""Sampling and stopping configuration for one generation request."""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """Parameters copied into Sequence objects at submission time."""

    # Temperature rescales logits before categorical sampling.
    temperature: float = 1.0
    # Maximum number of generated tokens, excluding prompt tokens.
    max_tokens: int = 64
    # If true, EOS is treated like an ordinary generated token.
    ignore_eos: bool = False
    # Optional cap on total prompt + completion length.
    max_model_length: int | None = None

    def __post_init__(self):
        # The sampler currently assumes stochastic sampling; near-zero temperature
        # would approximate greedy decoding but can also create numeric issues.
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
