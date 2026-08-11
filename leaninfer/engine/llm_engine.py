import torch, time
from leaninfer.oracle import STOP_IDS
from transformers import DynamicCache
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from leaninfer import metrics
from leaninfer.qwen3.model import Qwen3ForCausalLM
from leaninfer.engine.scheduler import Scheduler, Request
from leaninfer.engine.cache_manager import BlockCache
from leaninfer.engine.engine_config import EngineConfig


STOP = torch.tensor(sorted(STOP_IDS))


def trim(row: list[int]) -> list[int]:
    for i, t in enumerate(row):
        if t in STOP_IDS:
            return row[:i + 1]  # keep the stop token, matching the oracle
    return row


class DecodeGraph:
    """Capture one decode step into a CUDA graph
    """

    def __init__(self, model: Qwen3ForCausalLM, cache: BlockCache, config: EngineConfig) -> None:
        self.model = model
        self.cache = cache
        n_slots = config.n_slots
        max_blocks = -(-config.slot_len // config.block_size) # longest admissible sequence

        self.graph: torch.cuda.CUDAGraph | None = None
        self.logits: Tensor

        # staging buffers
        self.host_ids = torch.zeros(n_slots, 1, dtype=torch.long, pin_memory=True)
        self.host_pos = torch.zeros(n_slots, dtype=torch.long, pin_memory=True)
        self.host_bt = torch.zeros(n_slots, max_blocks, dtype=torch.int32, pin_memory=True)
        self.host_ctx = torch.ones(n_slots, dtype=torch.int32, pin_memory=True)
        self.np_ids, self.np_pos = self.host_ids.numpy(), self.host_pos.numpy()
        self.np_bt, self.np_ctx = self.host_bt.numpy(), self.host_ctx.numpy()

        dev = cache.device
        self.ids = torch.zeros(n_slots, 1, dtype=torch.long, device=dev)
        self.pos = torch.zeros(n_slots, dtype=torch.long, device=dev)
        self.bt = torch.zeros(n_slots, max_blocks, dtype=torch.int32, device=dev)
        self.ctx = torch.ones(n_slots, dtype=torch.int32, device=dev)

    def _capture(self) -> torch.cuda.CUDAGraph:
        # Warm up off the default stream first, as capture requires
        # This also lets FA do its one-time allocations outside the graph's private memory pool
        # This is why we set aside block 0 as trash block
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self.model(self.ids, self.cache, self.bt, self.pos, self.ctx)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.logits = self.model(self.ids, self.cache, self.bt, self.pos, self.ctx)
        return graph

    def step(self, reqs: list[Request]) -> list[int]:
        """Next token for every slot, indexed by slot number. Idle slots hold garbage"""
        self.np_pos[:] = 0
        self.np_ctx[:] = 1  # Start at 1, see cache_manager.py
        self.np_bt[:] = 0

        for r in reqs:
            self.np_ids[r.slot, 0] = r.out[-1]
            self.np_pos[r.slot] = r.pos
            self.np_bt[r.slot, :len(r.block_table)] = r.block_table
            self.np_ctx[r.slot] = r.pos + 1  # keys 0..pos: the token being written

        self.ids.copy_(self.host_ids, non_blocking=True)
        self.pos.copy_(self.host_pos, non_blocking=True)
        self.bt.copy_(self.host_bt, non_blocking=True)
        self.ctx.copy_(self.host_ctx, non_blocking=True)

        if self.graph is None:
            self.graph = self._capture() # warm up on this step's inputs
        self.graph.replay()

        return self.logits[:, -1].argmax(-1).tolist()


class LLMEngine:
    def __init__(self, engine_config: EngineConfig) -> None:
        self.engine_config = engine_config
        self.scheduler = Scheduler(engine_config)
        self.decode_graph: DecodeGraph | None = None
        self._last_decode: float | None = None


    def run_step(self, model: Qwen3ForCausalLM, cache: BlockCache, reqs: list[Request], prefill: bool) -> None:
            dev = cache.device
            q_len = len(reqs[0].prompt) if prefill else 1

            if prefill:
                r = reqs[0]
                with metrics.STEP_DURATION.labels(phase="prefill").time():
                    logits: Tensor = model(
                        torch.tensor([r.prompt], device=dev), # [1, prompt_len]
                        cache,
                        torch.tensor([r.block_table], dtype=torch.int32, device=dev), # [1, n_blocks]
                        torch.tensor([r.pos], device=dev),
                        None, # prefill starts at pos 0
                    )
                    toks = logits[:, -1].argmax(-1).tolist()
                
                metrics.PROMPT_TOKENS.inc(q_len)
            else:
                if self.engine_config.cuda_graphs and self.decode_graph is None:
                    self.decode_graph = DecodeGraph(model, cache, self.engine_config)
                with metrics.STEP_DURATION.labels(phase="decode").time():
                    if self.decode_graph is not None:
                        by_slot = self.decode_graph.step(reqs)
                        toks = [by_slot[r.slot] for r in reqs]
                    else:
                        # ragged tables -> rectangular [B, max_blocks]; pad id 0 is a valid
                        # block, and FA never reads past each row's cache_seqlens anyway.
                        # int32 throughout: FA requires it, and it still indexes cache.write
                        bt = pad_sequence([torch.tensor(r.block_table, dtype=torch.int32) for r in reqs],
                                          batch_first=True).to(dev)
                        pos = torch.tensor([r.pos for r in reqs], device=dev)
                        logits = model(
                            torch.tensor([[r.out[-1]] for r in reqs], device=dev),  # [B, 1]
                            cache,
                            bt,
                            pos,
                            (pos + q_len).to(torch.int32), # keys 0..pos inclusive: the token being written
                        )
                        toks = logits[:, -1].argmax(-1).tolist()

            # True for prefill as well since reqs is 1
            metrics.OUTPUT_TOKENS.inc(len(reqs))

            now = time.perf_counter()
            for r, tok in zip(reqs, toks):
                r.pos += q_len
                r.out.append(tok)
                if not r.t_first:
                    r.t_first = now
                    metrics.TTFT.observe(r.ttft)
                    metrics.PREFILL.observe(r.prefill_time)
                r.t_last = now

    @torch.no_grad()
    def generate(self, model: Qwen3ForCausalLM, prompts: list[list[int]], max_new_tokens: list[int]) -> list[Request]:
            print("llm_engine.py: engining")
            done: list[Request] = []
            for i, (p, n) in enumerate(zip(prompts, max_new_tokens)):
                self.scheduler.add(Request(id=i, prompt=p, max_new_tokens=n))
            cache = BlockCache(model.config, self.engine_config)

            while self.scheduler.busy():
                req = self.scheduler.admit()
                if req is not None:
                    print("llm_engine.py: admiting")
                    self.run_step(model, cache, [req], prefill=True) # everything else stalls
                else:
                    # may cause a preempt, so must run before the decode batch is read off running
                    self.scheduler.grow_for_decode()
                    metrics.KV_UTILIZATION.observe(
                        self.scheduler.reserved_tokens / self.engine_config.kv_capacity
                    )

                    self.run_step(model, cache, self.scheduler.running, prefill=False)
                    
                    # This takes into account prefill stall as well since we only measure delta between two decodes
                    now = time.perf_counter()
                    if self._last_decode is not None:
                        metrics.ITL.observe(now - self._last_decode)
                    self._last_decode = now

                for r in list(self.scheduler.running):
                    if r.out[-1] in STOP_IDS or len(r.out) >= r.max_new_tokens:
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
