"""HuggingFace greedy decode — the correctness oracle. Match its tokens exactly."""

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"
STOP_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# fp32 on purpose: CPU fp16 kernels are missing/slow, and HF upcasts norms/softmax to
# fp32 internally anyway — matching it in bf16 is MORE work, not less. Revisit only on GPU.
# eager = the naive softmax(QKᵀ/√d)·V path, i.e. the same math our engine will write.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager"
).eval()


@torch.no_grad()
def greedy(ids: Tensor, n: int=64) -> list[int]:
    out = []
    for _ in range(n):
        nxt = int(model(ids).logits[0, -1].argmax())
        out.append(nxt)
        if nxt in STOP_IDS:
            break
        ids = torch.cat([ids, ids.new_tensor([[nxt]])], dim=1)
    return out

# def first_divergence(ref, eng):
#     for i, (a, b) in enumerate(zip(ref, eng)):
#         if a != b:
#             return i
#     return None if len(ref) == len(eng) else min(len(ref), len(eng))


# from leaninfer.oracle import greedy, tokenizer

# ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
# ref = greedy(ids)          # the tokens your engine must reproduce
# # eng = my_engine_greedy(ids)
# # assert eng == ref


if __name__ == "__main__":
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids
    print(tokenizer.decode(greedy(ids)))
