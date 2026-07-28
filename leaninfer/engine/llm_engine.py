import torch, time
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache
from torch import Tensor
from leaninfer import metrics
from leaninfer.qwen3.model import Qwen3ForCausalLM
from leaninfer.engine.scheduler import Scheduler, Request
from leaninfer.engine.cache_manager import SlotCache
from leaninfer.engine.engine_config import EngineConfig


STOP = torch.tensor(sorted(STOP_IDS))


def trim(row: list[int]) -> list[int]:
    for i, t in enumerate(row):
        if t in STOP_IDS:
            return row[:i + 1]      # keep the stop token, matching the oracle
    return row



class LLMEngine:
    def __init__(self, engine_config: EngineConfig) -> None:
        self.engine_config = engine_config
        self.scheduler = Scheduler(engine_config)


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

            phase = "prefill" if prefill else "decode"
            with metrics.STEP_DURATION.labels(phase=phase).time():
                logits: Tensor = model(ids, cache, slots, pos, s)        
                nxt = logits[:, -1].argmax(-1)
                toks = nxt.tolist()

            if prefill:
                metrics.PROMPT_TOKENS.inc(q_len)
            # True for prefill as well since reqs is 1
            metrics.OUTPUT_TOKENS.inc(len(reqs))
                 
            now = time.perf_counter()
            for r, tok in zip(reqs, toks):
                r.pos += q_len
                r.out.append(tok)
                if not r.t_first:
                    r.t_first = now
                    metrics.TTFT.observe(now - r.t_arrival)
                r.t_last = now

    @torch.no_grad()
    def generate(self, model: Qwen3ForCausalLM, prompts: list[list[int]]) -> list[Request]:
            print("llm_engine.py: engining")
            done: list[Request] = []
            for i, p in enumerate(prompts):
                 self.scheduler.add(Request(id=i, prompt=p, max_new=self.engine_config.max_new))
            cache = SlotCache(model.config, self.engine_config)

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
                        metrics.TPOT.observe(r.tpot)

                metrics.KV_TOKENS.set(sum(r.pos for r in self.scheduler.running))

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
