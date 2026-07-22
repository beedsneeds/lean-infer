
import torch


HEAD_DIM     = 128 # decoupled. N_HEADS * HEAD_DIM = 2048 != HIDDEN

ROPE_THETA   = 1_000_000



def build_rope(n, offset=0):
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    pos = torch.arange(offset, offset + n).float()     # absolute positions offset..offset+n-1
    angles = torch.outer(pos, inv_freq)
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin



