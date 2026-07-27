import torch, time
import numpy as np
from torch import nn
from leaninfer.oracle import tokenizer        # HF Qwen3, fp32, weights loaded
from leaninfer.qwen3.model import Qwen3MLP, Qwen3Attention, HIDDEN
from leaninfer.qwen3.layers import build_rope, HEAD_DIM
from leaninfer.loader import load_model
from leaninfer.engine.llm_engine import greedy, trim, LLMEngine
from prometheus_client import start_http_server
from leaninfer import metrics


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
MAX_NEW = 64
N_SLOTS = 16
LINGER = 30.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

def make_batch(
    num_seqs: int, input_len: int, jitter: int = 0, seed: int = 0
) -> list[list[int]]:
    """Synthetic prompts. jitter=0 gives every request the same length."""
    rng = np.random.default_rng(seed)
    lens = rng.integers(input_len - jitter, input_len + jitter + 1, size=num_seqs)
    return [rng.integers(0, 10_000, size=int(n)).tolist() for n in lens]


def main() -> None:
    print("Hello from lean-infer!")
    start_http_server(8000)

    model = load_model(device=DEVICE, dtype=DTYPE)

    prompts: list[list[int]] = (
        make_batch(NUM_SEQS, INPUT_LEN, 0)
        if SYNTHETIC
        else [tokenizer(p).input_ids for p in PROMPTS]
    )

    max_len = max(len(p) for p in prompts) + MAX_NEW
    llm = LLMEngine(n_slots=N_SLOTS, max_len=max_len, device=DEVICE, dtype=DTYPE)
    print("main.py: generating")

    metrics.KV_CAPACITY.set(N_SLOTS * max_len)

    t0 = time.perf_counter()
    done = llm.generate(model, prompts, MAX_NEW)
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
