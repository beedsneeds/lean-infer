import torch
from leaninfer.oracle import STOP_IDS

@torch.no_grad()
def greedy(model, ids, n=64):
    ids = ids.clone()
    out = []
    for _ in range(n):
        nxt = int(model(ids)[-1].argmax())
        out.append(nxt)
        if nxt in STOP_IDS:
            break
        ids = torch.cat([ids, ids.new_tensor([nxt])]) 
    return out
