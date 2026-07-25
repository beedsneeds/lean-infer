import torch
from torch import nn
from leaninfer.oracle import model as ref, tokenizer, greedy as ref_greedy        # HF Qwen3, fp32, weights loaded
from leaninfer.qwen3.model import Qwen3MLP, Qwen3Attention, HIDDEN
from leaninfer.qwen3.layers import build_rope, HEAD_DIM
from leaninfer.loader import load_model
from leaninfer.engine.llm_engine import greedy, trim, LLMEngine

from transformers import DynamicCache


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



def main() -> None:
    print("Hello from lean-infer!")

    model = load_model()

    prompts = [tokenizer(p).input_ids for p in PROMPTS]   # list[list[int]]
    max_len = max(len(p) for p in prompts) + 64 # this changes with max_new in LLMEngine.generate()

    llm = LLMEngine(n_slots=5, max_len=max_len)
    print("main.py: generating")
    done = llm.generate(model, prompts)

    done.sort(key=lambda r: r.id)                              # retire order != prompt order

    for r in done:
        eng_row = trim(r.out)
        ids = tokenizer(PROMPTS[r.id], return_tensors="pt").input_ids   # oracle wants a tensor. [1, seq]
        ref_row = ref_greedy(ids)                                   # oracle, run solo
        ok = eng_row == ref_row
        print(f"[{'ok ' if ok else 'BAD'}] {PROMPTS[r.id]!r} -> {tokenizer.decode(eng_row)!r}")



if __name__ == "__main__":
    main()
