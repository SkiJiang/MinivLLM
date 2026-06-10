"""Physical KV-cache block management.

This module owns the mapping between logical sequence blocks and physical
blocks in the pre-allocated KV cache.  It also implements prefix caching by
hashing full token blocks with their prefix hash, so two sequences can share a
previously computed block when both the local tokens and their prefix context
match.
"""

import xxhash
import numpy as np
from collections import deque

from myvllm.engine.sequence import Sequence

class Block:
    """Metadata for one physical KV-cache block."""

    def __init__(self, block_id):
        # block_id is the stable physical index into the model's KV cache pool.
        self.block_id = block_id
        # hash == -1 means either "not cacheable yet" or "no cache identity".
        # Partial blocks deliberately keep -1 because their token span can still
        # grow during decode.
        self.hash = -1 
        # ref_count tracks how many live sequences point at this physical block.
        # Prefix-cache hits increment it so the block is not freed too early.
        self.ref_count = 0
        # token_ids mirrors the logical tokens stored in the block.  It is used
        # to reject rare hash collisions and to know whether a cached block is
        # semantically identical to the requested logical block.
        self.token_ids = []


    def update(self, h: int, token_ids: list[int]):
        # A block becomes cache-identifiable only after its hash and exact token
        # payload are recorded together.
        self.hash = h 
        self.token_ids = token_ids

    def reset(self):
        # Reset only metadata.  The actual GPU KV memory is overwritten later
        # when attention stores new keys and values for this physical slot.
        self.hash = -1 
        self.ref_count = 0
        self.token_ids = []

class BlockManager:
    """Allocate, share, and recycle KV-cache blocks for scheduled sequences."""

    def __init__(self, num_blocks: int, block_size: int):
        # block_size is the fixed number of tokens represented by one cache
        # block.  Sequence.block(i) uses the same size, so the logical and
        # physical views stay aligned.
        self.block_size: int = block_size
        # blocks owns metadata for every physical block id.
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # hash_to_block_id is the prefix-cache index.  A hash points to a
        # physical block that already contains the same full logical block.
        self.hash_to_block_id: dict[int, int] = {}
        # Free ids are kept in a deque so allocation is cheap and deterministic.
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # used_block_ids lets the allocator distinguish blocks that are cached
        # but currently unreferenced from blocks actively owned by sequences.
        self.used_block_ids: set[int] = set()

    def compute_hash(self, token_ids: list[int], prefix_hash_value: int) -> int:
        """Hash one full logical block together with its previous block hash.

        The prefix hash makes equal token chunks distinct when they appear after
        different prefixes.  That prevents an accidental cache hit for the same
        local block tokens in a different context.
        """
        h = xxhash.xxh64()
        if prefix_hash_value != -1:
            # The previous block hash is serialized in a fixed 8-byte little
            # endian form, matching xxh64's digest size.
            h.update(prefix_hash_value.to_bytes(8, 'little'))
        # Convert token ids to a compact, deterministic byte representation so
        # Python list object identity never affects the hash.
        h.update(np.array(token_ids, dtype=np.int32).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Move a free physical block into the used set and clear metadata."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, "Block is already allocated"
        block.reset()
        # remove() is used because allocation sometimes targets a specific
        # cached block id, not necessarily the left-most free id.
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Return a block to the free queue once no sequence references it."""
        assert self.blocks[block_id].ref_count == 0, "Block is still in use"
        block = self.blocks[block_id]
        # Keep block.hash in hash_to_block_id so the prefix-cache directory can
        # still rediscover this block later; only the live token list is cleared.
        # allocate() verifies token_ids on lookup, so stale/colliding entries are
        # treated as cache misses.
        block.token_ids = []
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check if a waiting prompt can reserve all blocks it needs."""
        return len(self.free_block_ids) >= seq.num_blocks


    def allocate(self, seq: Sequence) -> None:
        """Allocate or share all blocks required by a newly scheduled prompt."""
        # h is the rolling prefix hash.  It starts at -1 because the first block
        # has no previous full block.
        h = -1
        for i in range(seq.num_blocks):
            no_cache_found = False

            token_ids = seq.block(i)
            # Full blocks are immutable once allocated, so they can participate
            # in prefix caching.  The final partial block may grow during decode,
            # therefore it never gets a stable cache hash here.
            h = self.compute_hash(token_ids=token_ids, prefix_hash_value=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            
            # A hash lookup alone is not enough: token_ids is checked to protect
            # against hash collisions and against stale entries for freed blocks.
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                no_cache_found = True

            if not no_cache_found:
                # Cache hit: this sequence can skip computing the whole block in
                # prefill because KV values already exist in the physical cache.
                seq.num_cached_tokens += self.block_size # which == len(token_ids)
                # If the cached block is currently free, make it live again.  If
                # it is already used, share it by increasing ref_count.
                if block_id not in self.used_block_ids:
                    block = self._allocate_block(block_id)
                else:
                    block = self.blocks[self.hash_to_block_id[h]]
                    block.ref_count += 1
            else:
                # Cache miss: take the next free physical block and associate it
                # with this logical block.  Full blocks are registered for future
                # prefix-cache hits; partial blocks keep h == -1.
                block = self._allocate_block(self.free_block_ids[0])
                block.update(h=h, token_ids=token_ids)
                if h != -1:
                    self.hash_to_block_id[h] = block.block_id
            # The sequence stores physical block ids, not logical indices.  The
            # attention kernels later use this table to read/write KV cache.
            seq.block_table.append(block.block_id)
        
    def deallocate(self, seq: Sequence) -> None:
        """Release every physical block referenced by a sequence."""
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        # Clear the logical-to-physical mapping on the sequence so rescheduling
        # starts from a clean state.
        seq.block_table = []
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        """Return whether decode can append the next token for this sequence."""
        # When the current length is exactly a multiple of block_size, the next
        # generated token starts a new physical block and therefore needs free
        # capacity.  Otherwise the token fits in the already allocated tail.
        if seq.num_tokens % self.block_size == 0:
            return len(self.free_block_ids) > 0
        return True

    def append(self, seq: Sequence) -> None:
        """Update block metadata after scheduling one decode token."""
        block_tables = seq.block_table
        last_block_for_seq_id = block_tables[-1]

        # Case 1: the appended token just filled the tail block.  Its contents
        # are now immutable, so finalize its hash and expose it to prefix cache.
        if seq.num_tokens % self.block_size == 0:
            h = self.compute_hash(token_ids = seq.block(seq.num_blocks - 1), prefix_hash_value = -1 if len(block_tables) == 1 else self.blocks[block_tables[-2]].hash)
            block = self.blocks[last_block_for_seq_id]
            block.update(h=h, token_ids=seq.block(seq.num_blocks - 1))
            self.hash_to_block_id[h] = block.block_id
        # Case 2: the appended token is the first token of a new logical block.
        # The previous block must have been finalized, and a fresh partial block
        # is added to the sequence's block table.
        elif seq.num_tokens % self.block_size == 1:
            assert self.blocks[last_block_for_seq_id].hash != -1
            block = self._allocate_block(self.free_block_ids[0])
            block_tables.append(block.block_id)
        # Case 3: the appended token is inside an existing partial block.  The
        # GPU kernel will write the token into the existing physical block slot.
        else:
            assert last_block_for_seq_id in self.used_block_ids, "Last block should be allocated"
            assert self.blocks[last_block_for_seq_id].hash == -1, "Last block should be partial block with hash -1"
