"""模型层共享的单次 forward 执行上下文。

engine 会在调用模型前准备 attention 所需的元数据。各层通过 get_context()
读取这些信息，而不是把大量参数一层层传过整个 Transformer 栈。
"""

from dataclasses import dataclass 
import torch 


@dataclass
class Context:
    """一次 forward 中 attention 层和输出 head 层需要的元数据。"""

    # True 表示 prompt prefill，False 表示单 token decode。
    is_prefill: bool = False
    # varlen prefill kernel 使用的 query/key 累积序列长度。
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    # 当前 batch 内 query/key 的最大长度；供需要静态 launch 参数或校验的 kernel 使用。
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    # 每个需要写入 K/V 的 token 对应的扁平物理 cache slot。
    slot_mapping: torch.Tensor | None = None
    # decode 阶段每个序列的完整上下文长度。
    context_lens: torch.Tensor | None = None
    # decode 和 prefix cache 使用的 block 表：逻辑 block 下标 -> 物理 block id。
    block_tables: torch.Tensor | None = None

# 模块级单例。ModelRunner 会在每次 forward 后重置它。
_context = Context()

def get_context() -> Context:
    """返回当前 forward 的上下文。"""
    return _context

def reset_context():
    """模型运行结束后将上下文恢复到默认值。"""
    global _context
    _context = Context()

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    """用刚准备好的元数据替换当前上下文。"""
    global _context
    _context = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)
