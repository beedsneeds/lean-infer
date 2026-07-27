import torch
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache
from torch import Tensor
from leaninfer.qwen3.model import Qwen3ForCausalLM
from leaninfer.engine.scheduler import Scheduler, Request
from leaninfer.engine.cache_manager import SlotCache
from dataclasses import dataclass

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



class LLMEngine:
    def __init__(self, n_slots: int, max_len: int, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        self.scheduler = Scheduler(n_slots)
        self.n_slots = n_slots
        self.max_len = max_len
        self.device = device
        self.dtype = dtype


    def run_step(self, model: Qwen3ForCausalLM, cache: SlotCache, reqs: list[Request], prefill: bool) -> None:               
            dev = cache.device
            if prefill:
                print("llm_engine.py: prefilling")
                r = reqs[0]     # [1, prompt_len]
                ids = torch.tensor([r.prompt], device=dev)
            else: 
                ids = torch.tensor([[r.out[-1]] for r in reqs], device=dev) # [B, 1]
            slots = torch.tensor([r.slot for r in reqs], device=dev)
            pos = torch.tensor([r.pos for r in reqs], device=dev)
            q_len = ids.shape[1]
            s = int((pos + q_len).max())

            logits: Tensor = model(ids, cache, slots, pos, s)        
            nxt = logits[:, -1].argmax(-1)
            for r, tok in zip(reqs, nxt.tolist()):
                r.pos += q_len
                r.out.append(tok)

    @torch.no_grad()
    def generate(self, model: Qwen3ForCausalLM, prompts: list[list[int]], max_new: int) -> list[Request]:
            print("llm_engine.py: engining")
            done: list[Request] = []
            for i, p in enumerate(prompts):
                 self.scheduler.add(Request(id=i, prompt=p, max_new=max_new))
            cache = SlotCache(self.n_slots, self.max_len, self.device, self.dtype)

            while self.scheduler.busy():
                req = self.scheduler.admit()
                if req is not None:
                    # cache.allocate skipping this since no need
                    self.run_step(model, cache, [req], prefill=True) # everything else stalls
                else:
                     self.run_step(model, cache, self.scheduler.running, prefill=False)

                for r in list(self.scheduler.running):
                     if r.out[-1] in STOP_IDS or len(r.out) >= r.max_new:
                        #   cache.free
                        self.scheduler.retire(r)
                        done.append(r)
                        print("llm_engine.py: retiring")


            return done




    # def run(self):
    #     """ Only one prompt is admitted per step.
    #     That step is dedicated to prefilling it (see engine/llm_engine.py)
    #     """
    #     while self.scheduler.busy():
    #         req = self.scheduler.admit()
    #         if req is not None:
    #             step_prefill(model, cache, req)          # B=1, q_len=len(prompt)
    #         else:
    #             step_decode(model, cache, self.scheduler.running) # B=len(running), q_len=1
    #         for r in [r for r in self.scheduler.running if stopped(r)]:
    #             self.scheduler.retire(r)



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
