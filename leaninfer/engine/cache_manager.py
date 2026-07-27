import torch
from torch import Tensor

# ---- config (from config.json) ----
N_LAYERS     = 28
N_KV_HEADS   = 8 # GQA: each KV head is shared by N_HEADS // N_KV_HEADS = 2 query heads
HEAD_DIM     = 128 # decoupled! N_HEADS * HEAD_DIM = 2048 != HIDDEN




class SlotCache:
    """slot based cache
    Pool shape: [N_LAYERS, n_blocks, block_size, K_KV_HEADS, HEAD_DIM]
    """

    def __init__(self, n_slots: int, max_len: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        shape = (N_LAYERS, n_slots, max_len, N_KV_HEADS, HEAD_DIM)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.device = torch.device(device)


    def write(self, layer: int, slots: Tensor, pos: Tensor, k: Tensor, v: Tensor) -> None:
        """Write into slots at position pos"""
        # k,v shape: [B, N_KV_HEADS, q_len, HEAD_DIM]
        # pool stores [.., col, heads, dim]
        q_len = k.shape[2]
        cols = pos[:, None] + torch.arange(q_len, device=pos.device)   # [B, q_len]
        self.k[layer, slots[:, None], cols] = k.transpose(1, 2) # [B, q_len, N_KV_HEADS, HEAD_DIM]
        self.v[layer, slots[:, None], cols] = v.transpose(1, 2)

    def read(self, layer: int, slots: Tensor, s: int) -> tuple[Tensor, Tensor]:
        """Read past K/V upto s. Let mask determine which columns to actually access"""
        k = self.k[layer, slots, :s]
        v = self.v[layer, slots, :s]
        return k.transpose(1, 2), v.transpose(1, 2)
