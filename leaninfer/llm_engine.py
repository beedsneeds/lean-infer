import torch
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache
from torch import Tensor
from leaninfer.qwen3.model import Qwen3ForCausalLM

@torch.no_grad()
def greedy(model: Qwen3ForCausalLM, ids: Tensor, n: int=64) -> list[int]:
    cache = DynamicCache()
    out: list[int] = []
    x = ids
    for _ in range(n):
        logits = model(x, cache) # prefill
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        if nxt in STOP_IDS:
            break
        x = ids.new_tensor([nxt]) # consume only new token 
    return out
