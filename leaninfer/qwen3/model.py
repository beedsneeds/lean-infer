from torch import nn
from torch.nn import functional as F
from leaninfer.qwen3.layers import apply_rope, build_rope
from transformers import Qwen3Config
from transformers import AutoModelForCausalLM


# ---- config (from config.json) ----
N_LAYERS     = 28
HIDDEN       = 1024
N_HEADS      = 16
N_KV_HEADS   = 8 # GQA: each KV head is shared by N_HEADS // N_KV_HEADS = 2 query heads
HEAD_DIM     = 128 # decoupled! N_HEADS * HEAD_DIM = 2048 != HIDDEN
INTERMEDIATE = 3072
VOCAB        = 151936
EPS          = 1e-6
ROPE_THETA   = 1_000_000



#   model.embed_tokens.weight                        [VOCAB, HIDDEN]
#   model.layers.{L}.input_layernorm.weight          [HIDDEN]
#   model.layers.{L}.self_attn.q_proj.weight         [2048, HIDDEN]
#   model.layers.{L}.self_attn.k_proj.weight         [1024, HIDDEN]
#   model.layers.{L}.self_attn.v_proj.weight         [1024, HIDDEN]
#   model.layers.{L}.self_attn.q_norm.weight         [HEAD_DIM]
#   model.layers.{L}.self_attn.k_norm.weight         [HEAD_DIM]
#   model.layers.{L}.self_attn.o_proj.weight         [HIDDEN, 2048]
#   model.layers.{L}.post_attention_layernorm.weight [HIDDEN]
#   model.layers.{L}.mlp.gate_proj.weight            [INTERMEDIATE, HIDDEN]
#   model.layers.{L}.mlp.up_proj.weight              [INTERMEDIATE, HIDDEN]
#   model.layers.{L}.mlp.down_proj.weight            [HIDDEN, INTERMEDIATE]
#   model.norm.weight                                [HIDDEN]


class Qwen3MLP(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x, cos, sin):
        seq = x.shape[0]
        q = self.q_norm(self.q_proj(x).view(seq, N_HEADS, HEAD_DIM)).transpose(0, 1)
        k = self.k_norm(self.k_proj(x).view(seq, N_KV_HEADS, HEAD_DIM)).transpose(0, 1)
        v = self.v_proj(x).view(seq, N_KV_HEADS, HEAD_DIM).transpose(0, 1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        o = o.transpose(0, 1).reshape(seq, N_HEADS * HEAD_DIM)
        return self.o_proj(o)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn = Qwen3Attention()
        self.mlp = Qwen3MLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)

    def forward(self, h, cos, sin):
        h = h + self.self_attn(self.input_layernorm(h), cos, sin)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h


class Qwen3Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(Qwen3DecoderLayer() for _ in range(N_LAYERS))
        self.norm = nn.RMSNorm(HIDDEN, eps=EPS)

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        cos, sin = build_rope(input_ids.shape[0])
        for layer in self.layers:
            h = layer(h, cos, sin)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = Qwen3Model()

    def forward(self, input_ids):
        h = self.model(input_ids)
        # reusing embedding matrix as op proj since checkpoint doesn't have a lm_head.weight
        return F.linear(h, self.model.embed_tokens.weight)
