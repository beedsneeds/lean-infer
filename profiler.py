"""Profile prefill, decode, or both.

Divergence from main.py: fills every slot by prefilling n_slots requests (B=1, q_len=PROMPT_LEN each),
then profiles a handful of decode steps at full batch. Either phase can be profiled on its own

    python profiler.py              # both phases
    python profiler.py --prefill    # prefill only
    python profiler.py --decode     # decode only
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from leaninfer.engine.cache_manager import BlockCache
from leaninfer.engine.engine_config import EngineConfig
from leaninfer.engine.llm_engine import LLMEngine
from leaninfer.engine.scheduler import Request
from leaninfer.loader import load_model
from leaninfer.qwen3.model import Qwen3ForCausalLM
from leaninfer.qwen3.model_config import ModelConfig

PROMPT_LEN = 768
WAIT, WARMUP, ACTIVE = 1, 3, 6      # 10 steps per phase, the last 6 recorded
STEPS = WAIT + WARMUP + ACTIVE
MAX_NEW = 768


def profiler() -> profile:
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1),
        record_shapes=True,
        with_stack=True,        # attributes kernels back to the .py line that launched them
    )


def fill_slots(
    llm: LLMEngine,
    model: Qwen3ForCausalLM,
    cache: BlockCache,
    n_slots: int,
    prof: profile | None,
) -> list[Request]:
    """Only the last STEPS of them are stepped through the schedule: the earlier
    ones run with the profiler idle and serve as extra warmup.
    """
    rng = np.random.default_rng(0)
    for i in range(n_slots):
        prompt = rng.integers(0, 10_000, size=PROMPT_LEN).tolist()
        llm.scheduler.add(Request(id=i, prompt=prompt, max_new_tokens=MAX_NEW))
        req = llm.scheduler.admit()
        assert req is not None, "n_slots requests should all be admitted"
        llm.run_step(model, cache, [req], prefill=True)
        if prof is not None and i >= n_slots - STEPS:
            prof.step()
    return llm.scheduler.running


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefill", action="store_true", help="profile prefill")
    ap.add_argument("--decode", action="store_true", help="profile decode")
    args = ap.parse_args()
    if not (args.prefill or args.decode):
        args.prefill = args.decode = True       # neither flag given: profile both
    return args


@torch.no_grad()
def main() -> None:
    args = parse_args()
    model_config = ModelConfig.from_pretrained()
    # Longest admissible sequence, just as main.py derives it
    # n_slots=78 is a profiler-only override. Its the batch size the KV pool sustains under main.py's current workload
    engine_config = replace(EngineConfig(), slot_len=PROMPT_LEN + MAX_NEW, n_slots=78)
    model = load_model(model_config, engine_config)
    llm = LLMEngine(engine_config)
    cache = BlockCache(model_config, engine_config)
    n_slots = engine_config.n_slots

    if args.prefill and n_slots < STEPS:
        raise SystemExit(f"n_slots={n_slots} leaves no recorded prefill step, {STEPS} needed")

    with (profiler() if args.prefill else nullcontext()) as prof:
        running = fill_slots(llm, model, cache, n_slots, prof)
    if prof is not None:
        prof.export_chrome_trace("prefill_trace.json")

    if args.decode:
        # Decode cost tracks total KV bytes, and off prefill every sequence still
        # sits at its prompt length, so recording there reads about a third low
        # against a running engine. Advancing to mid-budget first also absorbs the
        # DecodeGraph capture, which WAIT alone was not keeping out of the trace.
        for _ in range(MAX_NEW // 2):
            llm.scheduler.grow_for_decode()
            llm.run_step(model, cache, running, prefill=False)
        with profiler() as prof:
            for _ in range(STEPS):
                llm.scheduler.grow_for_decode()
                llm.run_step(model, cache, running, prefill=False)
                prof.step()
        prof.export_chrome_trace("decode_trace.json")


if __name__ == "__main__":
    main()
