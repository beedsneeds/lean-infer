from torch import nn
from torch.nn import functional as F
from leaninfer.qwen3.layers import apply_rope, build_rope, build_mask
from torch import Tensor
from leaninfer.engine.cache_manager import SlotCache


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
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down_proj = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.k_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: SlotCache, layer_idx: int, attn_mask: Tensor, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, N_HEADS, HEAD_DIM)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, n, N_KV_HEADS, HEAD_DIM)).transpose(1, 2)
        v = self.v_proj(x).view(b, n, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        cache.write(layer_idx, slots, pos, k, v)
        k, v = cache.read(layer_idx, slots, s)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        o = o.transpose(1, 2).reshape(b, n, N_HEADS * HEAD_DIM)
        return self.o_proj(o)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention()
        self.mlp = Qwen3MLP()
        self.input_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)
        self.post_attention_layernorm = nn.RMSNorm(HIDDEN, eps=EPS)

    def forward(self, h: Tensor, cos: Tensor, sin: Tensor, cache: SlotCache, layer_idx: int, attn_mask: Tensor, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, cache, layer_idx, attn_mask, slots, pos, s)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h



class Qwen3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList(Qwen3DecoderLayer() for _ in range(N_LAYERS))
        self.norm = nn.RMSNorm(HIDDEN, eps=EPS)

    def forward(self, input_ids: Tensor, cache: SlotCache, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        q_len = input_ids.shape[1]
        h = self.embed_tokens(input_ids)
        cos, sin = build_rope(pos, q_len)
        m = build_mask(pos, q_len, s)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache, i, m, slots, pos, s)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Qwen3Model()

    def forward(self, input_ids: Tensor, cache: SlotCache, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        h = self.model(input_ids, cache, slots, pos, s)
        return F.linear(h, self.model.embed_tokens.weight)
