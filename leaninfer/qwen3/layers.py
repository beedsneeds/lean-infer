
import torch
from torch import Tensor
from leaninfer.qwen3.model_config import ModelConfig


def build_rope(config: ModelConfig, pos: Tensor, q_len: int, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """pos: [B] each row's start position (tokens already cached)"""
    dev = pos.device
    inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, device=dev).float() / config.head_dim)) # [D/2]
    positions = pos[:, None] + torch.arange(q_len, device=dev)      # [B, q_len] broadcast
    angles = positions[..., None].float() * inv_freq    # [B, q_len, D/2]
    emb = torch.cat((angles, angles), dim=-1)           # [B, q_len, D]
    # angles stay fp32 for precision; cast so apply_rope doesn't promote k back to fp32 and break the bf16 cache write
    return emb.cos()[:, None].to(dtype), emb.sin()[:, None].to(dtype)   # [B, 1, q_len, D]


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin

# def build_mask(pos: Tensor, q_len: int, s: int, dtype: torch.dtype) -> Tensor:
#     """returns additive float mask [B, 1, q_len, S] for SDPA (0=attend, -inf=forbid)"""
#     # 1 = real, 0 = pad
#     dev = pos.device
#     key_pos = torch.arange(s, device=dev)                                  # [S]
#     query_pos = pos[:, None] + torch.arange(q_len, device=dev)             # [B, q_len]
#     allowed = key_pos[None, None, :] <= query_pos[:, :, None]  # [B, q_len, S]
#     # cast so SDPA doesn't promote the whole attention to the mask's dtype
#     return torch.where(allowed, 0.0, float("-inf"))[:, None].to(dtype)   # [B, 1, q_len, S]




