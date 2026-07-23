
import torch
from torch import Tensor


HEAD_DIM     = 128 # decoupled. N_HEADS * HEAD_DIM = 2048 != HIDDEN

ROPE_THETA   = 1_000_000



def build_rope(n: int, offset: int=0) -> tuple[Tensor, Tensor]:
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    pos = torch.arange(offset, offset + n).float()     # absolute positions offset..offset+n-1
    angles = torch.outer(pos, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    return x * cos + rotate_half(x) * sin

def build_mask(mask: Tensor, q_len: int) -> Tensor:
    # 1 = real, 0 = pad
    # returns additive float mask [B, 1, q_len, S] for SDPA (0=attend, -inf=forbid)
    # S = past + q_len
    _, s = mask.shape
    key = torch.where(mask.bool(), 0.0, float("-inf"))[:, None, None, :] # [B, 1, 1, S]
    if q_len == 1:
        return key # decode
    causal = torch.triu(torch.full((q_len, s), float("-inf")), diagonal=1)   # [q,S], q==S at prefill
    m = causal[None, None] + key
    diag = torch.arange(q_len)
    m[:, :, diag, diag] = 0.0 
    return m     



