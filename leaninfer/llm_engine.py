import torch
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache

@torch.no_grad()
def greedy(model, ids, n=64):
    cache = DynamicCache()
    out = []
    x = ids
    for _ in range(n):
        logits = model(x, cache) # prefill
        nxt = int(logits[-1].argmax())
        out.append(nxt)
        if nxt in STOP_IDS:
            break
        x = ids.new_tensor([nxt]) # consume only new token 
    return out
