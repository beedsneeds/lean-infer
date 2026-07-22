import torch
from torch import nn
from leaninfer.oracle import model as ref, tokenizer, greedy as ref_greedy        # HF Qwen3, fp32, weights loaded
from leaninfer.qwen3.model import Qwen3MLP, Qwen3Attention, HIDDEN
from leaninfer.qwen3.layers import build_rope, HEAD_DIM
from leaninfer.loader import load_model
from leaninfer.llm_engine import greedy

from transformers import DynamicCache





def main() -> None:
    print("Hello from lean-infer!")

    model = load_model()
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids   # [1, seq]

    ref_out = ref_greedy(ids)
    eng_out = greedy(model, ids[0])
    print(eng_out == ref_out)
    print(tokenizer.decode(eng_out))


if __name__ == "__main__":
    main()
