import torch
from torch import nn
from leaninfer.oracle import model as ref, tokenizer, greedy as ref_greedy        # HF Qwen3, fp32, weights loaded
from leaninfer.qwen3.model import Qwen3MLP, Qwen3Attention, HIDDEN
from leaninfer.qwen3.layers import build_rope, HEAD_DIM
from leaninfer.loader import load_model
from leaninfer.llm_engine import greedy, trim

from transformers import DynamicCache


PROMPTS = [
    "1 + 1 =",
    "The mitochondria is",
    "Elect a clown, expect",
    "Who said this and why: Educate, agitate, organize",
]



def main() -> None:
    print("Hello from lean-infer!")

    model = load_model()

    tokenizer.padding_side = "left"
    batch = tokenizer(PROMPTS, return_tensors="pt", padding=True)   # input_ids/attention_mask: [B, seq]

    eng = greedy(model, batch.input_ids, batch.attention_mask)      # [B, n_gen]

    for i, prompt in enumerate(PROMPTS):
        eng_row = trim(eng[i].tolist())
        ids = tokenizer(prompt, return_tensors="pt").input_ids      # [1, seq], no padding
        ref_row = ref_greedy(ids)                                   # oracle, run solo
        ok = eng_row == ref_row
        print(f"[{'ok ' if ok else 'BAD'}] {prompt!r} -> {tokenizer.decode(eng_row)!r}")


if __name__ == "__main__":
    main()
