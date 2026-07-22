"""Qwen3-0.6B forward pass, from scratch. Fill in the bodies.

Correctness check (no cache yet): run forward over the prompt and confirm
argmax-per-position == HF oracle's argmax-per-position (teacher forcing).
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from leaninfer.qwen3.model import Qwen3ForCausalLM 


# ---- weights ----
         # dict[str, Tensor], dtype bf16 -> cast slices to fp32 as you use them
# ---- weight layout ----
# proj weights are [out, in] and bias-free -> nn.Linear(in, out, bias=False); F.linear(x, W) = x @ W.T
# embed_tokens.weight [VOCAB, HIDDEN] is TIED to lm_head (no separate lm_head weight in the checkpoint)
# every *_layernorm / q_norm / k_norm is a single RMSNorm scale vector -> declared as `.weight`
# the inner `model` submodule reproduces the `model.` prefix in every checkpoint key.
# params load as fp32 (bf16 checkpoint is cast on copy_), matching the fp32 oracle.


def load_model(path: str | None = None) -> Qwen3ForCausalLM:
    if path is None:
        path = hf_hub_download("Qwen/Qwen3-0.6B", "model.safetensors")

    
    model = Qwen3ForCausalLM()
    with safe_open(path, "pt", "cpu") as f:
        W: dict[str, Tensor] = {name: f.get_tensor(name) for name in f.keys()}
    W.pop("lm_head.weight")        # redundant duplicate of model.embed_tokens.weight (tied)
    # alternatively just declare lm_head
    model.load_state_dict(W)
    return model.eval()


if __name__ == "__main__":
    # from leaninfer.oracle import tok, model as ref
    load_model()
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    ids  = tokenizer("The capital of France is", return_tensors="pt").input_ids[0]
    # mine = forward(ids, W).argmax(-1)                 # [seq]
    # gold = ref(ids[None]).logits[0].argmax(-1)        # HF per-position argmax
    # print(torch.equal(mine, gold), mine.tolist(), gold.tolist())
    ...
