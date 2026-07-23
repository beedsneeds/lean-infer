import torch
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache
from torch import Tensor
from leaninfer.qwen3.model import Qwen3ForCausalLM

STOP = torch.tensor(sorted(STOP_IDS))

@torch.no_grad()
def greedy(model: Qwen3ForCausalLM, ids: Tensor, mask: Tensor, n: int=64, pad_id: int=151643) -> Tensor:
    # ids, mask: [B, seq] — left-padded input_ids and its attention mask
    cache = DynamicCache()
    b = ids.shape[0]
    done = torch.zeros(b, dtype=torch.bool)
    out: list[Tensor] = []
    x = ids # prefill takes in full [B, seq]
    for _ in range(n):
        logits = model(x, cache, mask) # [B, q, VOCAB]
        nxt = logits[:, -1].argmax(-1) 
        nxt = torch.where(done, pad_id, nxt)  # emit pad if already finished
        out.append(nxt)
        done |= torch.isin(nxt, STOP) # mark rows that stopped this step
        if done.all():
            break
        x = nxt[:, None] # only the new token for decode
        mask = torch.cat([mask, mask.new_ones(b, 1)], dim=1)  # grow mask
    return torch.stack(out, dim=1)


def trim(row: list[int]) -> list[int]:
    for i, t in enumerate(row):
        if t in STOP_IDS:
            return row[:i + 1]      # keep the stop token, matching the oracle
    return row

# @torch.no_grad()
# def greedy(model: Qwen3ForCausalLM, ids: Tensor, n: int=64) -> list[int]:
#     cache = DynamicCache()
#     out: list[int] = []
#     x = ids
#     for _ in range(n):
#         logits = model(x, cache) # prefill
#         nxt = int(logits[-1].argmax())
#         out.append(nxt)
#         if nxt in STOP_IDS:
#             break
#         x = ids.new_tensor([nxt]) # consume only new token 
#     return out
