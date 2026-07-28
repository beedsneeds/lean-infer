import time
import numpy as np
from leaninfer.oracle import tokenizer        # HF Qwen3, fp32, weights loaded
from leaninfer.loader import load_model
from leaninfer.engine.llm_engine import trim, LLMEngine
from prometheus_client import start_http_server
from leaninfer import metrics
from dataclasses import replace
from leaninfer.engine.engine_config import EngineConfig
from leaninfer.qwen3.model_config import ModelConfig


PROMPTS = [
    "1 + 1 =",
    "The mitochondria is the",
    "Elect a clown, expect a",
    "Who said this and why? 'Educate, agitate, organize'",
]
# PROMPTS = [
#     "1 + 1 =",
#     "The mitochondria is the",
#     "Elect a clown, expect a",
#     "Who said this and why? 'Educate, agitate, organize'",
# ]

SYNTHETIC = True # Set false for prompts above
NUM_SEQS = 256
INPUT_LEN = 16
INPUT_JITTER = 8
MAX_NEW_JITTER = 16 # If 0, requests retire in lockstep synchronicity. Doesn't provide much value
LINGER = 30.0



def make_batch(
    num_seqs: int, input_len: int, max_new: int, input_jitter: int = 0, max_new_jitter: int = 0, seed: int = 0
) -> tuple[list[list[int]], list[int]]:
    """Synthetic workload. input_jitter=0 gives every request the same length."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(input_len - input_jitter, input_len + input_jitter + 1, size=num_seqs)
    budgets = rng.integers(max_new - max_new_jitter, max_new + max_new_jitter + 1, size=num_seqs)
    prompts = [rng.integers(0, 10_000, size=int(n)).tolist() for n in lens]
    return prompts, np.clip(budgets, 1, None).tolist()

# For lognormal: (cast to int)
# sigma = 0.5
# budgets = rng.lognormal(np.log(max_new) - sigma**2 / 2, sigma, num_seqs)


def main() -> None:
    print("Hello from lean-infer!")
    start_http_server(8000)

    model_config = ModelConfig.from_pretrained()
    engine_config = EngineConfig()


    if SYNTHETIC:
        prompts, budgets = make_batch(
            NUM_SEQS, INPUT_LEN, engine_config.max_new, INPUT_JITTER, MAX_NEW_JITTER
        )
    else:
        prompts = [tokenizer(p).input_ids for p in PROMPTS]
        budgets = [engine_config.max_new] * len(prompts)


    engine_config = replace(engine_config, max_len=max(len(p) + n for p, n in zip(prompts, budgets)))

    model = load_model(model_config, engine_config)
    llm = LLMEngine(engine_config)
    print("main.py: generating")

    metrics.KV_CAPACITY.set(engine_config.kv_capacity)

    t0 = time.perf_counter()
    done = llm.generate(model, prompts, budgets)
    elapsed = time.perf_counter() - t0
    done.sort(key=lambda r: r.id)                              # retire order != prompt order

    out_tokens = sum(len(r.out) for r in done)
    print(
            f"{len(done)} reqs, {out_tokens} tokens in {elapsed:.1f}s "
            f"= {out_tokens / elapsed:.1f} tok/s"
    )

    if not SYNTHETIC:
        from leaninfer.oracle import greedy as ref_greedy;        # HF Qwen3, fp32, weights loaded

        for r in done:
            eng_row = trim(r.out)
            ids = tokenizer(PROMPTS[r.id], return_tensors="pt").input_ids   # oracle wants a tensor. [1, seq]
            ref_row = ref_greedy(ids)                                   # oracle, run solo
            ok = eng_row == ref_row
            print(f"[{'ok ' if ok else 'BAD'}] {PROMPTS[r.id]!r} -> {tokenizer.decode(eng_row)!r}")

    print(f"pausing for {LINGER} seconds until final scrape completes")
    time.sleep(LINGER)


if __name__ == "__main__":
    main()
