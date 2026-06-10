"""CUDA/分布式模型执行与输入准备。

ModelRunner 的层级刻意低于 LLMEngine。它了解张量并行 rank、CUDA graph 捕获、
KV-cache 内存布局，以及用于让 worker rank 与 rank 0 执行同一方法的小型共享
内存 RPC 机制。
"""

import math
import torch
import pickle
import torch.distributed as dist
from pathlib import Path
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from myvllm.models.qwen3 import Qwen3ForCausalLM
from myvllm.models.llama import LlamaForCausalLM
from myvllm.layers.sampler import SamplerLayer
from myvllm.engine.sequence import Sequence
from myvllm.utils import *

class ModelRunner:
    """在单个 rank 上执行模型 forward，并协调其他 rank。"""

    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        # 在 rank 0 上 event 是 worker event 列表；在 worker rank 上，它是共享内存
        # 写入后用于唤醒当前 rank 的单个 event。
        self.event = event

        # 调度、cache 布局和 CUDA graph 捕获都会用到这些配置。
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        # enforce_eager 会禁用 CUDA graph replay，便于调试 shape 问题，
        # 或在不支持 graph capture 的环境中运行。
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        # 所有 rank 加入同一个 NCCL 进程组。张量并行层会查询该进程组，
        # 判断自己拥有哪一片权重。
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # 根据 checkpoint 目录/名称实例化匹配的模型架构。
        path_str = self.config['model_name_or_path']
        model_name = Path(path_str).name
        match model_name:
            case 'Qwen3-0.6B':
                self.model = Qwen3ForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    num_heads=config['num_heads'],
                    head_dim=config['head_dim'],
                    scale=config['scale'],
                    num_kv_heads=config['num_kv_heads'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    qkv_bias=config['qkv_bias'],
                    base=config['base'],
                    max_position=config['max_position'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    tie_word_embeddings=config['tie_word_embeddings'],
                    block_size=self.block_size,
                )
            case 'Llama-3.2-1B-Instruct':
                self.model = LlamaForCausalLM(
                    vocab_size=config['vocab_size'],
                    hidden_size=config['hidden_size'],
                    head_dim=config['head_dim'],
                    num_qo_heads=config['num_qo_heads'],
                    num_kv_heads=config['num_kv_heads'],
                    has_attn_bias=config['has_attn_bias'],
                    rms_norm_epsilon=config['rms_norm_epsilon'],
                    rope_base=config['rope_base'],
                    max_position_embeddings=config['max_position_embeddings'],
                    intermediate_size=config['intermediate_size'],
                    ffn_bias=config['ffn_bias'],
                    num_layers=config['num_layers'],
                    block_size=self.block_size,
                    tie_word_embeddings=config['tie_word_embeddings'],
                )
            case _:
                raise Exception(f"Unsupported model: {config['model_name_or_path']}")

        # 在加载 checkpoint 权重前，先把参数移动到当前 rank 对应的 GPU。
        self.model = self.model.cuda(rank)

        # 权重加载器知道如何处理自定义的融合 QKV 和 gate/up 层。
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # 另一种路线：先在 CPU 加载权重，再移动模型到 GPU。
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # cache 大小计算需要 itemsize，因此在后面恢复 default device/dtype 前先记住 dtype。
        self.default_dtype = torch.get_default_dtype()

        # 首次 decode 调试标志，保留用于本地快速诊断。
        self._first_decode = False

        # warmup 用于测量峰值 activation 显存；分配 KV cache 池前会从可用显存中扣除它。
        self.warmup_model()

        # 分配一个 paged KV-cache 池，并把每一层对应的 view 绑定到 attention 模块上。
        self.allocate_kv_cache()

        # decode shape 相对静态，适合通过 CUDA graph replay 降低开销。
        if not self.enforce_eager:
            self.capture_cudagraph()

        # warmup/capture 设置完成后恢复进程级默认 device/dtype。
        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        if self.world_size > 1:
            # 共享内存必须等所有 rank 完成模型和 cache 初始化后再创建；
            # 否则 worker 可能会在 master 准备好发送命令前进入 loop。
            dist.barrier()
            if self.rank == 0:
                # 如果之前崩溃留下了共享内存段，先尝试清理。
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # 只有共享内存段创建完成后，才放行 worker。
                dist.barrier()
            else:
                # worker 等待 rank 0 创建共享内存段，然后附着到该段。
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')

    def read_shm(self):
        """worker 侧从共享内存读取序列化的方法调用。"""
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        # event.wait() 会阻塞，直到 rank 0 写入方法调用字节。
        self.event.wait()
        # 前 4 个字节存 payload 长度；payload 是 (method_name, *args) 的 pickle。
        n = int.from_bytes(self.shm.buf[:4], 'little')
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, args: tuple):
        """rank 0 为 worker rank 写入序列化方法调用。"""
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # 展平 args，使 read_shm() 能返回 method_name 和普通参数列表。
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        # payload 完整写入后再唤醒所有 worker。
        for event in self.event:
            event.set()

    def exit(self):
        """释放 CUDA graph、共享内存和分布式资源。"""
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # 销毁前先确认进程组已经初始化。
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def loop(self):
        """worker 事件循环：镜像执行 rank 0 发来的方法调用。"""
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == 'exit':
                self.exit()
                break

    def call(self, method_name: str, *args: dict):
        """在本地分发具名方法；若当前是 rank 0，同时广播给 worker。"""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    def warmup_model(self):
        """运行一次合成 prefill，用于测量峰值 activation 显存。"""
        # 从干净的 CUDA allocator 统计开始，让 peak_memory_stats 反映 warmup 执行，
        # 而不是之前初始化阶段的分配。
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # 用最大长度的合成序列填满配置的 token budget。这里故意使用偏保守的 shape，
        # 以保证后续真实 batch 能放得下。
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]

        # 输出会被丢弃；这里关心的只是 allocator 统计。
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """分配 paged KV-cache 池，并绑定每层的 cache view。"""
        # 先扣除模型观测到的峰值 activation 显存，再计算还剩多少显存可以给 cache 使用。
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # 每个 cache block 要同时存储所有层、每个本地 KV head、block 内每个 token 位置、
        # 以及 head_dim 上的 K 和 V。
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # bytes/block = block_size * (K,V) * layers * local_kv_heads * head_dim *
        # dtype_size。这就是 paged attention 使用的物理页大小。
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # 在所有 rank 之间同步 max_cached_blocks。每个 rank 都会根据本地空闲显存独立
        # 计算 num_available_kv_blocks，但不同 rank 的结果可能略有差异：rank 0
        # 通常有额外开销（NCCL buffer、进程组状态），所以空闲显存可能更少。
        # 如果不同步，只在 rank 0 上运行的 scheduler 可能分配超过某些 rank 容量的 block，
        # 导致这些 rank 写 KV cache 时 OOM。
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # 使用 MIN 做 all_reduce：每个 rank 都得到最保守的上限，也就是显存最紧张的
            # rank 也能承受的 block 数。这个一致值会写回 config，使后续在 rank 0
            # 初始化的 Scheduler 不会分配超过任意 rank 物理容量的 block。
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # 单 GPU：不需要跨 rank 同步，直接使用本地值。
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # 分配一个连续大池，而不是为每个序列单独分配 tensor。
        # attention kernel 通过 block_tables 和 slot_mapping 定位其中的内容。
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                # 每个 attention 层拿到自己所在 layer 切片的 view：
                # (num_blocks, block_size, local_kv_heads, head_dim)。
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """构造拼接后的 prefill tensor 和 attention 上下文。

        已命中 prefix cache 的 token 会从 input_ids 和 slot_mapping 中省略，
        但它们的 block table 仍会提供给 attention，使其能把 cached K/V 作为前缀上下文读取。
        """
        # prefix cache 跳过后仍然需要送入模型 forward 的 token。
        input_ids = []
        # 每个新 token 的 K/V 应该写入的物理 cache slot。
        slot_mappings = []
        # 移除 cached prefix 后的 query 长度。
        seqlens_q = []
        # key 长度是完整序列长度，包含 cached prefix token。
        seqlens_k = []
        # query 累积长度用于在拼接后的 query tensor 中划分每个序列。
        cu_seqlens_q = [0]
        # key 累积长度用于划分每个序列完整的 key/value 长度。
        cu_seqlens_k = [0]
        # 读取 cached prefix block 时使用的、经过 padding 的物理 block id 表。
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # 实际 input_ids 中跳过 cached prefix token。
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # 只有未被 cache 的 block 需要 slot mapping，因为本次 prefill 只会写入这些 K/V。
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        # 完整 block：写入全部 block_size 个位置。
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        # 最后一个 block 可能不满：只写真实 token 对应的 slot。
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # query 长度小于 key 长度说明存在 cached prefix。
            # 这种情况下 attention 需要 block_tables 来读取 cached KV。
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                # 用 -1 padding，使所有行对 kernel 来说具有相同宽度。
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)

        # 使用 pinned CPU tensor 和 non_blocking 拷贝，使 host-to-device 传输在可能时
        # 可以和 CUDA 工作重叠。
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        # 将 attention 元数据存入进程级全局 context，供本次 forward 中的模型层和
        # Triton attention wrapper 读取。
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            cu_seqlens_k=torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            max_seqlen_q=max(seqlens_q),
            max_seqlen_k=max(seqlens_k),
            slot_mapping=slot_mapping_tensor,
            context_lens=None,
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids


    def prepare_decode(self, seqs: list[Sequence]) -> torch.Tensor:
        """构造单 token decode tensor 和 paged attention 上下文。"""
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            # decode 只输入最后一个 token；它之前的所有上下文都从该序列的 KV-cache block 中读取。
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            # 新 token 的 K/V 写入最后一个 block 的尾部 slot。
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            # 对行进行 padding，使 decode kernel 可以把 block_tables 当成稠密矩阵索引：
            # (batch_size, max_num_blocks)。
            block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
            block_tables.append(block_table)
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        set_context(
            is_prefill=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=0,
            max_seqlen_k=0,
            slot_mapping=torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            context_lens=torch.tensor(context_lens, dtype=torch.long, pin_memory=True).cuda(non_blocking=True),
            block_tables=torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) if block_tables else None,
        )
        return input_ids    

    def prepare_sample(self, seqs: list[Sequence]) -> None:
        """在当前 CUDA 设备上收集每个序列的 temperature。"""
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """以 eager 方式或捕获好的 decode graph 方式运行模型。"""
        if is_prefill or self.enforce_eager:
            # prefill 是变长的，token 数通常会变化，因此使用 1D 拼接 token tensor 走 eager。
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            # decode 每个序列只有一个 token，可以复用按 batch size 桶捕获的 graph。
            bs = input_ids.size(0)
            context = get_context()

            # 选择能容纳当前 decode batch 的最小已捕获 batch size。
            # replay 后未使用的尾部行会被忽略。
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars

            # CUDA graph 要求内存地址固定，因此新的 decode 元数据会先拷贝到预分配的
            # graph 变量中，再 replay。
            vars['input_ids'][:bs].copy_(input_ids)
            vars['slot_mapping'][:bs].fill_(-1)
            vars['slot_mapping'][:bs].copy_(context.slot_mapping)
            vars["context_lens"].zero_()
            vars['context_lens'][:bs].copy_(context.context_lens)
            vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables

            graph.replay()
            logits = self.model.compute_logits(vars['outputs'][:bs])

        return logits


    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """准备输入、执行模型、采样 token，并重置 context。"""
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)

        # 只有 rank 0 需要采样 token id 供 scheduler.postprocess 使用。
        # 其他 rank 仍然运行模型，以保持张量并行 collective 对齐。
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))

        # 全局 context 是单次 forward 的元数据；清空它可以避免下一批误用旧信息。
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """为常见 batch size 捕获 decode graph。

        捕获后的 graph 会在稳定输入 buffer 中拷入新数据，然后 replay 同一组 CUDA kernel。
        decode 每步很小且对延迟敏感，这样可以去掉 Python 调度开销。
        """
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)

        # decode 输入是每个序列一个 token id。
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # 每个新 token 的 K/V 要写入的物理 cache slot。
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # 每个序列的完整上下文长度，包含当前 token。
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # 每个序列的逻辑 block 到物理 block id 映射。
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # 每次 replay 都复用的 graph 输出 buffer。
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # 小的 2 次幂和 16 的倍数能覆盖常见 decode batch size，同时避免捕获数量过多。
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            # 将 context 绑定到该捕获 batch size 对应的静态 graph buffer 切片。
            set_context(
                is_prefill=False,
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                max_seqlen_q=0,
                max_seqlen_k=0,
                slot_mapping=slot_mapping[:batch_size],
                context_lens=context_lens[:batch_size],
                block_tables=block_tables[:batch_size],
            )

            # 捕获前先 eager 跑一次，使 lazy kernel 先完成物化。
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    # 多个捕获共享 graph memory pool，减少额外分配压力。
                    graph_pool = graph.pool()

            self.graphs[batch_size] = graph

            # 切换到下一个 capture size 前先同步。
            torch.cuda.synchronize()
            reset_context()

        # 保留静态 buffer 引用；如果释放它们，CUDA graph 记录的内存地址会失效。
        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
