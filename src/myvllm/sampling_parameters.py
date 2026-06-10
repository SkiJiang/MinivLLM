"""单个生成请求的采样参数和停止条件配置。"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    """提交请求时会复制进 Sequence 对象的参数集合。"""

    # temperature 会在类别采样前缩放 logits。
    temperature: float = 1.0
    # 最多生成多少个 token，不包含 prompt token。
    max_tokens: int = 64
    # 如果为 True，EOS 会像普通 token 一样保留，不会触发停止。
    ignore_eos: bool = False
    # 可选的总长度上限，包含 prompt 和 completion。
    max_model_length: int | None = None

    def __post_init__(self):
        # 当前 sampler 假设使用随机采样；接近 0 的 temperature 会近似贪心解码，
        # 但也更容易带来数值问题。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
