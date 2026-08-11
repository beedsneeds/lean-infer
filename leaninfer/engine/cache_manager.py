import torch
from torch import Tensor
from leaninfer.qwen3.model_config import ModelConfig
from leaninfer.engine.engine_config import EngineConfig
from collections import deque

# Utility helper
def blocks_for(n_tokens: int, block_size: int) -> int:
    """Blocks needed to hold n_tokens (ceil division)"""
    return -(-n_tokens // block_size)


class BlockAllocator:
    """Free List of physical block IDs
    Pool shape: [num_hidden_layers, num_blocks, block_size, num_key_value_heads, head_dim]
    But we keep a track only of the second dimension
    """

    def __init__(self, num_blocks: int) -> None:
        # Idle CUDA-graph rows still run cache.write so if we don't point them to a scratch node,
        # they'd scribble garbage KV into a live request's block 
        # Therefore, we reserve block 0 as the trash block and have all idle rows point to it
        # Reads are bounded by cache_seqlens so don't need this treatment
        # vLLM/nano-vLLM use a more elegant solution
        self.free: deque[int] = deque(range(1, num_blocks))

    @property
    def num_free(self) -> int:
        return len(self.free)

    def allocate(self, n: int) -> list[int]:
        """Take n blocks off free list. 
        Admission control is at the scheduler level. Check for valid num_free there
        
        Prefill: take ceil(prompt_len/block_size) blocks
        Decode: take 1 block if no space
        """
        assert n <= len(self.free), f"Insufficient free slots: asked {n}, free {self.free}"
        return [self.free.popleft() for _ in range(n)]

    def release(self, blocks: list[int]) -> None:
        self.free.extend(blocks)


class BlockCache:
    """
    Pool shape: [num_hidden_layers, num_blocks, block_size, num_key_value_heads, head_dim]
    
    Maps a sequence's logical block index to a physical block id
    Shape [B, max_blocks_this_batch]
    """

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
        shape = (model_config.num_hidden_layers, engine_config.num_blocks, engine_config.block_size,
                model_config.num_key_value_heads, model_config.head_dim)

        self.k = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
        self.v = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
        self.block_size = engine_config.block_size
        self.device = torch.device(engine_config.device)


    def write(self, layer: int, block_tables: Tensor, pos: Tensor, k: Tensor, v: Tensor) -> None:
        """ Scatter-on-Write
        k,v: [B, N_KV_HEADS, q_len, HEAD_DIM]
        logical column:   cols = pos + arange(q_len)   # same as SlotCache
        block number:     blk  = cols // block_size  # index into the block table
        offset:           off  = cols % block_size   # offset within the block
        physical block:   phys = block_table[blk]   # indirection
        """
        q_len = k.shape[2]
        cols = pos[:, None] + torch.arange(q_len, device=pos.device)  # [B, q_len] logical
        phys = block_tables.gather(1, cols // self.block_size)    # [B, q_len] physical block
        off = cols % self.block_size
        self.k[layer, phys, off] = k.transpose(1, 2)  # [B, q_len, N_KV_HEADS, HEAD_DIM]
        self.v[layer, phys, off] = v.transpose(1, 2)


    # def read(self, block_tables):
    #     """ Gather-on-Read all past K/V upto s
        
    #     TODO replace with paged kernel
    #     """
    #     pass


# class SlotCache:
#     """slot based cache
#     Pool shape: [num_hidden_layers, n_slots, slot_len, num_key_value_heads, head_dim]
#     """

#     def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
#         shape = (model_config.num_hidden_layers, engine_config.n_slots, engine_config.slot_len,
#                  model_config.num_key_value_heads, model_config.head_dim)
#         self.k = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
#         self.v = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
#         self.device = torch.device(engine_config.device)


#     def write(self, layer: int, slots: Tensor, pos: Tensor, k: Tensor, v: Tensor) -> None:
#         """Write into slots at position pos"""
#         # k,v shape: [B, N_KV_HEADS, q_len, HEAD_DIM]
#         # pool stores [.., col, heads, dim]
#         q_len = k.shape[2]
#         cols = pos[:, None] + torch.arange(q_len, device=pos.device)   # [B, q_len]
#         self.k[layer, slots[:, None], cols] = k.transpose(1, 2) # [B, q_len, N_KV_HEADS, HEAD_DIM]
#         self.v[layer, slots[:, None], cols] = v.transpose(1, 2)

#     def read(self, layer: int, slots: Tensor, s: int) -> tuple[Tensor, Tensor]:
#         """Read past K/V upto s. Let mask determine which columns to actually access

#         Statically shaped free view when the caller wants the whole pool over the whole window (CUDA Graphs)
#         Any narrower request falls back to advanced indexing
#         """
#         if s == self.k.shape[2] and slots.shape[0] == self.k.shape[1]:
#             return self.k[layer].transpose(1, 2), self.v[layer].transpose(1, 2)
#         k = self.k[layer, slots, :s]
#         v = self.v[layer, slots, :s]
#         return k.transpose(1, 2), v.transpose(1, 2)
