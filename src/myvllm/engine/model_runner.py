"""CUDA/distributed model execution and input preparation.

ModelRunner is intentionally lower level than LLMEngine.  It knows about tensor
parallel ranks, CUDA graph capture, KV-cache memory layout, and the small
shared-memory RPC mechanism used to ask worker ranks to run the same method as
rank 0.
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
    """Run model forward passes on one rank and coordinate peer ranks."""

    def __init__(self, config: dict, rank: int, event: Event | list[Event]):
        self.config = config
        # On rank 0 this is a list of worker events.  On worker ranks it is the
        # single event used to wake this rank after shared memory is written.
        self.event = event

        # Configuration used by scheduling, cache layout, and CUDA graph capture.
        self.block_size = config['block_size']
        self.world_size = config['world_size']
        # enforce_eager disables CUDA graph replay, which is useful for debugging
        # shape issues or running environments that do not support graph capture.
        self.enforce_eager = config.get('enforce_eager', False)

        self.rank = rank
        # All ranks join the same NCCL group.  Tensor-parallel layers query this
        # process group to decide which weight shard they own.
        dist.init_process_group('nccl', "tcp://localhost:12345", world_size=config['world_size'], rank=rank)
        torch.cuda.set_device(rank)

        # Instantiate the architecture that matches the checkpoint directory/name.
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

        # Move parameters to this rank's GPU before loading checkpoint weights.
        self.model = self.model.cuda(rank)

        # Weight loading understands the custom fused QKV and gate/up layers.
        if config.get('model_name_or_path'):
            from myvllm.utils.loader import load_weights_from_checkpoint
            load_weights_from_checkpoint(self.model, config['model_name_or_path'])

        # Load weights in CPU (move the model to GPU after loading weights)
        # self.model = self.model.cuda(rank)

        self.sampler = SamplerLayer()

        # The cache size computation needs itemsize, so remember the dtype before
        # set_default_device/set_default_dtype are restored later.
        self.default_dtype = torch.get_default_dtype()

        # Debug flag for first decode step; kept for quick local diagnostics.
        self._first_decode = False

        # Warmup measures peak activation memory, which is subtracted from the
        # free memory budget before allocating the KV cache pool.
        self.warmup_model()

        # Allocate one paged KV-cache pool and attach layer-specific views.
        self.allocate_kv_cache()

        # Decode uses static shapes and benefits from CUDA graph replay.
        if not self.enforce_eager:
            self.capture_cudagraph()

        # Restore process-wide defaults after warmup/capture setup.
        torch.set_default_device(f'cuda:{rank}')
        torch.set_default_dtype(self.default_dtype)

        if self.world_size > 1:
            # Shared memory is created only after every rank has initialized the
            # model and cache; otherwise a worker could enter its loop before the
            # master is ready to send commands.
            dist.barrier()
            if self.rank == 0:
                # Clean up a previous crashed run's segment if it exists.
                try:
                    old_shm = SharedMemory(name='myvllm')
                    old_shm.close()
                    old_shm.unlink()
                except FileNotFoundError:
                    pass
                self.shm = SharedMemory(name='myvllm', create=True, size=2**20)
                # Release workers only after the segment exists.
                dist.barrier()
            else:
                # Workers wait for rank 0, then attach to the existing segment.
                dist.barrier()
                self.shm = SharedMemory(name='myvllm')

    def read_shm(self):
        """Worker-side read of a serialized method call from shared memory."""
        assert self.world_size > 1 and self.rank != 0, "read_shm can only be called when world_size > 1 and rank != 0"
        # event.wait() blocks until rank 0 has written method bytes.
        self.event.wait()
        # The first four bytes store payload length; the payload is a pickle of
        # (method_name, *args).
        n = int.from_bytes(self.shm.buf[:4], 'little')
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name: str, args: tuple):
        """Rank-0 write of a serialized method call for worker ranks."""
        assert self.world_size > 1 and self.rank == 0, "write_shm can only be called when world_size > 1 and rank == 0"
        # Flatten args so read_shm() can return method_name and a normal arg list.
        data = pickle.dumps((method_name, *args))
        n = len(data)
        self.shm.buf[:4] = n.to_bytes(4, 'little')
        self.shm.buf[4:n+4] = data
        # Wake every worker after the payload is fully written.
        for event in self.event:
            event.set()

    def exit(self):
        """Release CUDA graph, shared memory, and distributed resources."""
        if self.world_size > 1:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs
            del self.graph_vars
        torch.cuda.synchronize()
        # Check if process group exists before destroying
        if dist.is_initialized():
            dist.destroy_process_group()
    
    def loop(self):
        """Worker event loop that mirrors rank-0 method calls."""
        assert self.world_size > 1 and self.rank != 0, "loop can only be called when world_size > 1 and rank != 0"
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == 'exit':
                self.exit()
                break

    def call(self, method_name: str, *args: dict):
        """Dispatch a named method locally and broadcast it from rank 0."""
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, args)
        method = getattr(self, method_name, None)
        if method:
            return method(*args)
        raise ValueError(f"Unknown method: {method_name}")

    def warmup_model(self):
        """Run a synthetic prefill to measure peak activation memory."""
        # Start from a clean CUDA allocator snapshot so peak_memory_stats reflects
        # warmup execution rather than previous setup allocations.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Fill the configured token budget with max-length synthetic sequences.
        # The shapes are intentionally pessimistic so later real batches fit.
        max_tokens = self.config['max_num_batch_tokens']
        max_model_length = self.config['max_model_length']
        batch_size = max_tokens // max_model_length
        seqs = [Sequence(token_ids=[0]*max_model_length, block_size=self.config['block_size']) for _ in range(batch_size)]

        # The output is discarded; only allocator statistics matter.
        self.run(seqs, is_prefill=True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate the paged KV-cache pool and bind layer cache views."""
        # Determine how much memory can be used for cache after reserving room
        # for the model's observed peak activation usage.
        free_mem, total_mem = torch.cuda.mem_get_info()
        total_free_mem = free_mem * self.config['gpu_memory_utilization']
        peak_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.peak']
        current_mem_usage = torch.cuda.memory_stats()['allocated_bytes.all.current']
        available_mem = total_free_mem - (peak_mem_usage - current_mem_usage)
        
        # Each cache block stores both K and V for every layer, local KV head,
        # token position in the block, and head dimension.
        num_layers = self.config['num_layers']
        num_kv_heads = self.config['num_kv_heads'] // self.world_size
        head_dim = self.config['head_dim'] if 'head_dim' in self.config else self.config['hidden_size'] // self.config['num_heads']

        # bytes/block = block_size * (K,V) * layers * local_kv_heads * head_dim *
        # dtype_size.  This is the physical page size used by paged attention.
        block_bytes = self.block_size * 2 * num_layers * num_kv_heads * head_dim * self.default_dtype.itemsize
        num_available_kv_blocks = int(available_mem // block_bytes)
        assert num_available_kv_blocks >= 1, f'Not enough memory to hold at least one block of KV cache on rank {self.rank}'
        
        # Synchronize max_cached_blocks across all ranks.
        # Each rank independently computed num_available_kv_blocks from its own
        # free GPU memory. Ranks may differ slightly: rank-0 carries extra overhead
        # (NCCL buffers, process-group state) so it often has less free memory than
        # workers. Without sync, the scheduler (which runs only on rank-0) would use
        # rank-0's local value and could allocate more blocks than some rank can hold,
        # causing an OOM on that rank during KV cache writes.
        if self.world_size > 1:
            print(f"[Rank {self.rank}] Local max_cached_blocks: {num_available_kv_blocks}")
            per_rank_max_blocks_tensor = torch.tensor(
                num_available_kv_blocks,
                dtype=torch.long,
                device=f'cuda:{self.rank}'
            )
            # all_reduce with MIN: every rank learns the most conservative limit,
            # i.e. the block count that even the most memory-constrained rank can serve.
            # This single agreed-upon value is then stored in config so the Scheduler
            # (initialized afterwards on rank-0) never allocates more blocks than any
            # rank can physically hold.
            dist.all_reduce(per_rank_max_blocks_tensor, op=dist.ReduceOp.MIN)
            self.config['max_cached_blocks'] = per_rank_max_blocks_tensor.item()
        else:
            # Single GPU: no cross-rank sync needed; use the local value directly.
            self.config['max_cached_blocks'] = num_available_kv_blocks
        if self.rank == 0:
            print(f"[Rank 0] Global max_cached_blocks (min): {self.config['max_cached_blocks']}")

        # Allocate one contiguous pool instead of per-sequence tensors.  Attention
        # kernels address it through block_tables and slot_mapping.
        allocated_kv_cache = torch.zeros(2, self.config['num_layers'], self.config['max_cached_blocks'], self.block_size, num_kv_heads, head_dim, device=f'cuda:{self.rank}')
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, 'k_cache') and hasattr(module, 'v_cache'):
                # Each attention layer receives a view into its layer slice:
                # (num_blocks, block_size, local_kv_heads, head_dim).
                module.k_cache = allocated_kv_cache[0, layer_id]
                module.v_cache = allocated_kv_cache[1, layer_id]
                layer_id += 1

    def prepare_prefill(self, seqs: list[Sequence]) -> torch.Tensor:
        """Build concatenated prefill tensors and attention context.

        Prefix-cached tokens are omitted from input_ids and slot_mapping, but
        their block tables are still provided so attention can read cached keys
        and values as prefix context.
        """
        # Tokens that still need a model forward pass after prefix-cache skips.
        input_ids = []
        # Physical cache slots where each new token's K/V should be written.
        slot_mappings = []
        # Query lengths after removing cached prefix tokens.
        seqlens_q = []
        # Key lengths are full sequence lengths, including cached prefix tokens.
        seqlens_k = []
        # Prefix sums delimit each sequence in the concatenated query tensor.
        cu_seqlens_q = [0]
        # Prefix sums delimit each sequence's full key/value length.
        cu_seqlens_k = [0]
        # Padded physical block ids for reading cached prefix blocks.
        block_tables = []
        for seq in seqs:
            token_ids = seq.token_ids
            num_cached_tokens = seq.num_cached_tokens
            # Skip cached prefix tokens in the actual input ids.
            input_ids.extend(token_ids[num_cached_tokens:])
            seqlens_q.append(len(token_ids) - num_cached_tokens)
            seqlens_k.append(len(token_ids))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlens_q[-1])
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlens_k[-1])
            if seq.block_table:
                # Only uncached blocks need slot mappings because only those K/V
                # values are written during this prefill.
                for i, block_id in enumerate(seq.block_table[seq.num_cached_blocks:]):
                    if seq.num_cached_blocks + i != seq.num_blocks - 1:
                        # Full block: write all block_size positions.
                        slot_mappings.extend(list(range(block_id * self.block_size, (block_id+1) * self.block_size)))
                    else:
                        # Last block may be partial: write only real token slots.
                        slot_mappings.extend(list(range(block_id * self.block_size, block_id * self.block_size + seq.last_block_num_tokens)))
        if cu_seqlens_q[-1] < cu_seqlens_k[-1]:
            # A smaller query length than key length means some prefix was cached.
            # In that case attention needs block_tables to read the cached KV.
            all_block_tables = [seq.block_table for seq in seqs]
            max_num_blocks = max(len(bt) for bt in all_block_tables)
            for i, seq in enumerate(seqs):
                # Pad with -1 so all rows have the same width for the kernel.
                block_table = seq.block_table + [-1]*(max_num_blocks - len(seq.block_table))
                block_tables.append(block_table)

        # Use pinned CPU tensors plus non_blocking copies so host-to-device
        # transfer can overlap with CUDA work when possible.
        input_ids = torch.tensor(input_ids, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_tensor = torch.tensor(slot_mappings, dtype=torch.long, pin_memory=True).cuda(non_blocking=True)

        # Store attention metadata in a process-global context consumed by model
        # layers and Triton attention wrappers during this forward pass.
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
        """Build one-token decode tensors and paged-attention context."""
        input_ids = []
        context_lens = []   
        slot_mappings = []  
        block_tables = []
        for seq in seqs:
            # Decode feeds only the last token.  Everything before it is read
            # from the sequence's KV-cache blocks.
            input_ids.append(seq.last_token)
            context_lens.append(len(seq))
            # The new token's K/V is stored at the tail slot of the last block.
            slot_mappings.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        all_block_tables = [seq.block_table for seq in seqs]
        max_num_blocks = max(len(bt) for bt in all_block_tables)
        for i, seq in enumerate(seqs):
            # Pad rows so the decode kernel can index block_tables as a dense
            # matrix: (batch_size, max_num_blocks).
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
        """Collect per-sequence temperatures on the active CUDA device."""
        return torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """Run the model either eagerly or through a captured decode graph."""
        if is_prefill or self.enforce_eager:
            # Prefill is variable-length and usually has changing token counts,
            # so it runs eagerly with a 1D concatenated token tensor.
            hidden_states = self.model(input_ids)
            logits = self.model.compute_logits(hidden_states)
        else:
            # Decode has one token per sequence and can reuse a captured graph
            # for the next bucketed batch size.
            bs = input_ids.size(0)
            context = get_context()

            # Pick the smallest captured batch size that can hold this decode
            # batch.  The unused tail rows remain ignored after replay.
            graph = self.graphs[next(bs_ for bs_ in self.graphs.keys() if bs_ >= bs)]
            vars = self.graph_vars

            # CUDA graphs require fixed memory addresses, so new decode metadata
            # is copied into pre-allocated graph variables before replay.
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
        """Prepare inputs, execute the model, sample tokens, and reset context."""
        if is_prefill:
            input_ids = self.prepare_prefill(seqs)
        else:
            input_ids = self.prepare_decode(seqs)
        logits = self.run_model(input_ids, is_prefill)

        # Only rank 0 needs sampled token ids for scheduler.postprocess.  Other
        # ranks still run the model so tensor-parallel collectives stay aligned.
        token_ids = None
        if self.rank == 0:
            token_ids = self.sampler(logits, self.prepare_sample(seqs))

        # The global context is per-forward metadata; clear it to avoid accidental
        # reuse by the next batch.
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self) -> None:
        """Capture decode graphs for common batch sizes.

        Captured graphs replay the same CUDA kernel sequence with new data copied
        into stable input buffers.  This removes Python overhead from decode,
        where each step is small and latency-sensitive.
        """
        max_bs = self.config['max_num_seqs']
        max_len = self.config['max_model_length']
        max_num_blocks = math.ceil(max_len / self.block_size)

        # Decode input is one token id per sequence.
        input_ids = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # Where each new token writes K/V into the physical cache.
        slot_mapping = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # Full context length for each sequence, including the current token.
        context_lens = torch.zeros(max_bs, dtype=torch.long, device=f'cuda:{self.rank}')
        # Logical-to-physical block ids for each sequence.
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32, device=f'cuda:{self.rank}')
        # Graph output buffer reused by every replay.
        outputs = torch.zeros(max_bs, self.config['vocab_size'], device=f'cuda:{self.rank}')

        # Small powers of two and then multiples of sixteen cover common decode
        # batch sizes while keeping capture count modest.
        batch_sizes = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        graph_pool = None

        for batch_size in reversed(batch_sizes):
            graph = torch.cuda.CUDAGraph()
            # Bind the context to slices of the static graph buffers for this
            # capture size.
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

            # One eager warmup run materializes any lazy kernels before capture.
            outputs[:batch_size] = self.model(input_ids[:batch_size])

            with torch.cuda.graph(graph, graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size])
                if graph_pool is None:
                    # Share the graph memory pool across captures to reduce
                    # extra allocation pressure.
                    graph_pool = graph.pool()

            self.graphs[batch_size] = graph

            # Synchronize before switching context for the next capture size.
            torch.cuda.synchronize()
            reset_context()

        # Keep references to the static buffers.  Releasing them would invalidate
        # the memory addresses recorded inside the CUDA graphs.
        self.graph_vars = dict(
            input_ids=input_ids,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
