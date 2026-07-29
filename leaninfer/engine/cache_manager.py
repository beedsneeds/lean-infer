import torch
from torch import Tensor
from leaninfer.qwen3.model_config import ModelConfig
from leaninfer.engine.engine_config import EngineConfig




class SlotCache:
    """slot based cache
    Pool shape: [num_hidden_layers, n_slots, slot_len, num_key_value_heads, head_dim]
    """

    def __init__(self, model_config: ModelConfig, engine_config: EngineConfig) -> None:
        shape = (model_config.num_hidden_layers, engine_config.n_slots, engine_config.slot_len,
                 model_config.num_key_value_heads, model_config.head_dim)
        self.k = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
        self.v = torch.zeros(shape, dtype=engine_config.dtype, device=engine_config.device)
        self.device = torch.device(engine_config.device)


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
