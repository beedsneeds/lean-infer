
import torch
from torch import Tensor


HEAD_DIM     = 128 # decoupled. N_HEADS * HEAD_DIM = 2048 != HIDDEN

ROPE_THETA   = 1_000_000



def build_rope(pos: Tensor, q_len: int) -> tuple[Tensor, Tensor]:
    """pos: [B] each row's start position (tokens already cached)"""
    dev = pos.device
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM)) # [D/2]
    positions = pos[:, None] + torch.arange(q_len, device=dev)      # [B, q_len] broadcast
    angles = positions[..., None].float() * inv_freq    # [B, q_len, D/2]
    emb = torch.cat((angles, angles), dim=-1)           # [B, q_len, D]
    return emb.cos()[:, None], emb.sin()[:, None]       # cos/sin [B, 1, q_len, D]


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin

def build_mask(pos: Tensor, q_len: int, s: int) -> Tensor:
    """returns additive float mask [B, 1, q_len, S] for SDPA (0=attend, -inf=forbid)"""
    # 1 = real, 0 = pad
    dev = pos.device
    key_pos = torch.arange(s, device=dev)                                  # [S]
    query_pos = pos[:, None] + torch.arange(q_len, device=dev)             # [B, q_len]
    allowed = key_pos[None, None, :] <= query_pos[:, :, None]  # [B, q_len, S]
    return torch.where(allowed, 0.0, float("-inf"))[:, None]   # [B, 1, q_len, S]




