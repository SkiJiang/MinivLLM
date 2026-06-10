"""物理 KV-cache block 管理。

本模块负责维护逻辑序列 block 与预分配 KV cache 中物理 block 之间的映射。
它还通过“完整 token block + 前缀 hash”的方式实现 prefix cache：当两个序列的
局部 token 和前缀上下文都一致时，就可以共享之前已经计算好的物理 block。
"""

import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    """单个物理 KV-cache block 的元数据。"""

    def __init__(self, block_id):
        # block_id 是模型 KV cache 池中的稳定物理下标。
        self.block_id = block_id
        # hash == -1 表示“暂时不能 cache”或“没有 cache 身份”。
        # 不完整 block 会故意保持 -1，因为它的 token 范围在 decode 中还会继续增长。
        self.hash = -1 
        # ref_count 记录有多少活跃序列指向这个物理 block。
        # prefix cache 命中时会增加它，避免 block 被过早释放。
        self.ref_count = 0
        # token_ids 镜像该 block 存储的逻辑 token。它用于排除罕见 hash 碰撞，
        # 并确认 cache block 与请求的逻辑 block 在语义上完全一致。
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        # 只有同时记录 hash 和精确 token 内容后，一个 block 才具备可 cache 的身份。
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        # 这里只重置元数据。真正的 GPU KV 内存会在 attention 写入新 K/V 时被覆盖。
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    """为被调度的序列分配、共享和回收 KV-cache block。"""

    def __init__(self, num_blocks: int, block_size: int):
        # block_size 是一个 cache block 固定代表的 token 数。Sequence.block(i)
        # 使用同样的大小，因此逻辑视图和物理视图保持对齐。
        self.block_size: int = block_size
        # blocks 保存每个物理 block id 对应的元数据。
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash_to_block_id 是 prefix cache 索引。一个 hash 指向已经包含相同完整逻辑
        # block 的物理 block。
        self.hash_to_block_id: dict[int, int] = {}
        # 空闲 block id 用 deque 保存，使分配便宜且顺序确定。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used_block_ids 让分配器区分“仍在 cache 索引中但当前无人引用”的 block，
        # 以及“正被序列持有”的 block。
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """将一个完整逻辑 block 与前一个 block 的 hash 一起计算 hash。

        prefix hash 可以让相同 token 片段在不同前缀下得到不同 hash，
        避免同一局部 block token 在不同上下文里误命中 cache。
        """
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            # 前一个 block 的 hash 按固定 8 字节小端形式序列化，匹配 xxh64 的 digest 大小。
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        # 将 token id 转成紧凑且确定的字节表示，避免 Python list 对象身份影响 hash。
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """将一个空闲物理 block 移入 used 集合，并清空其元数据。"""
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        # 这里用 remove()，因为有时会分配某个指定的 cached block id，
        # 不一定总是 deque 最左边的空闲 id。
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """当没有序列引用某个 block 时，将它放回空闲队列。"""
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # 保留 block.hash 和 hash_to_block_id 中的索引，使 prefix cache 以后仍可能
        # 重新找到这个 block；这里只清掉活跃 token 列表。allocate() 查找时会校验
        # token_ids，因此过期条目或碰撞条目会被当成 cache miss。
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查 waiting prompt 是否能一次性预留它需要的所有 block。"""
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        """为新调度的 prompt 分配或共享它需要的所有 block。"""
        # h 是滚动 prefix hash。初始为 -1，因为第一个 block 没有前一个完整 block。
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # 完整 block 一旦分配后内容不可变，因此可以参与 prefix cache。
            # 最后的不完整 block 在 decode 中可能继续增长，所以这里不给它稳定 hash。
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # 只查 hash 不够：还要检查 token_ids，以防 hash 碰撞或已释放 block 的过期条目。
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # cache 命中：prefill 时可以跳过整个 block 的计算，因为物理 cache
                # 中已经存在对应 KV。
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # 如果命中的 cached block 当前空闲，就重新激活它；如果已经被使用，
                # 就通过增加 ref_count 来共享。
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # cache 未命中：取下一个空闲物理 block 并绑定到当前逻辑 block。
                # 完整 block 会登记到索引中供未来 prefix cache 命中；不完整 block 保持 h == -1。
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            # 序列存的是物理 block id，而不是逻辑下标。attention kernel 后续会用
            # 这张表读写 KV cache。
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        """释放某个序列引用的所有物理 block。"""
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # 清空序列上的逻辑到物理映射，使它后续重新调度时从干净状态开始。
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        """返回 decode 是否能为该序列追加下一个 token。"""
        # 如果当前长度正好是 block_size 的整数倍，下一个生成 token 会开启新物理 block，
        # 因而需要空闲容量；否则它会落在已经分配的尾部 block 中。
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        """调度一个 decode token 后更新 block 元数据。"""
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # 情况 1：刚追加的 token 正好填满尾部 block。此时内容已经不可变，
        # 可以计算最终 hash 并暴露给 prefix cache。
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # 情况 2：刚追加的 token 是新逻辑 block 的第一个 token。
        # 前一个 block 必须已经 finalized，然后为序列追加一个新的不完整物理 block。
        elif seq.num_tokens % self.block_size == 1:
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # 情况 3：刚追加的 token 位于已有的不完整 block 内。
        # GPU kernel 会把该 token 的 KV 写入已有物理 block 的相应 slot。
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
