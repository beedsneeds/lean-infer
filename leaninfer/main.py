import torch
from torch import nn
from leaninfer.oracle import model as ref, tokenizer, greedy as ref_greedy        # HF Qwen3, fp32, weights loaded
from leaninfer.qwen3.model import Qwen3MLP, Qwen3Attention, HIDDEN
from leaninfer.qwen3.layers import build_rope, HEAD_DIM
from leaninfer.loader import load_model
from leaninfer.llm_engine import greedy


def main():
    print("Hello from lean-infer!")
    # hf   = ref.model.layers[0].input_layernorm        # Qwen3RMSNorm, shape [1024]
    # mine = nn.RMSNorm(1024, eps=1e-6)
    # mine.load_state_dict(hf.state_dict())              # copies the single `weight` tensor

    # x = torch.randn(5, 1024)
    # print(torch.allclose(mine(x), hf(x), atol=1e-6))   # True -> identical

    # hf   = ref.model.layers[0].mlp
    # mine = Qwen3MLP()
    # mine.load_state_dict(hf.state_dict())          # gate/up/down_proj.weight copied by name

    # x = torch.randn(5, HIDDEN)
    # print(torch.allclose(mine(x), hf(x), atol=1e-5))   # True

    # seq = 7
    # cos, sin = build_rope(seq)                                    # ours: [seq, HEAD_DIM]
    # hf_cos, hf_sin = ref.model.rotary_emb(torch.zeros(1, seq, HEAD_DIM), torch.arange(seq)[None])
    # print("cos:", torch.allclose(cos, hf_cos[0], atol=1e-5))
    # print("sin:", torch.allclose(sin, hf_sin[0], atol=1e-5))     # both True

    # seq = 7
    # x = torch.randn(seq, HIDDEN)
    # hf_attn = ref.model.layers[0].self_attn
    # mine = Qwen3Attention(); mine.load_state_dict(hf_attn.state_dict())

    # cos, sin = build_rope(seq)
    # mine_out = mine(x, cos, sin)                                   # [seq, HIDDEN]

    # xb   = x[None]                                                 # HF wants a batch dim
    # hcos, hsin = ref.model.rotary_emb(xb, torch.arange(seq)[None])
    # mask = torch.triu(torch.full((seq, seq), float("-inf")), 1)[None, None]   # [1,1,seq,seq] additive causal
    # hf_out, _ = hf_attn(xb, (hcos, hsin), mask)

    # print("allclose:", torch.allclose(mine_out, hf_out[0], atol=1e-4),
    #   "max|diff|:", (mine_out - hf_out[0]).abs().max().item())


    # model = load_model()
    # ids = tokenizer("The capital of France is", return_tensors="pt").input_ids[0]   # [seq]

    # with torch.no_grad():
    #     mine = model(ids).argmax(-1)                     # [seq]
    #     gold = ref(ids[None]).logits[0].argmax(-1)       # [seq]
    # print(torch.equal(mine, gold), mine.tolist(), gold.tolist())

    model = load_model()
    ids = tokenizer("The capital of France is", return_tensors="pt").input_ids   # [1, seq]

    ref_out = ref_greedy(ids)          # HF oracle
    eng_out = greedy(model, ids[0])    # ours (1-D)
    print(eng_out == ref_out)
    print(tokenizer.decode(eng_out))


if __name__ == "__main__":
    main()
