from torch import nn, Tensor
from torch.nn import functional as F
from leaninfer.qwen3.layers import apply_rope, build_rope, build_mask
from leaninfer.engine.cache_manager import SlotCache
from leaninfer.qwen3.model_config import ModelConfig



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
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, cache: SlotCache, layer_idx: int, attn_mask: Tensor, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        config = self.config
        b, n, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(b, n, config.num_attention_heads, config.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(b, n, config.num_key_value_heads, config.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(b, n, config.num_key_value_heads, config.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        cache.write(layer_idx, slots, pos, k, v)
        k, v = cache.read(layer_idx, slots, s)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=True)
        o = o.transpose(1, 2).reshape(b, n, config.num_attention_heads * config.head_dim)
        return self.o_proj(o)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h: Tensor, cos: Tensor, sin: Tensor, cache: SlotCache, layer_idx: int, attn_mask: Tensor, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        h = h + self.self_attn(self.input_layernorm(h), cos, sin, cache, layer_idx, attn_mask, slots, pos, s)
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h



class Qwen3Model(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: Tensor, cache: SlotCache, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        q_len = input_ids.shape[1]
        h = self.embed_tokens(input_ids)
        cos, sin = build_rope(self.config, pos, q_len)
        m = build_mask(pos, q_len, s)
        for i, layer in enumerate(self.layers):
            h = layer(h, cos, sin, cache, i, m, slots, pos, s)
        return self.norm(h)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

    def forward(self, input_ids: Tensor, cache: SlotCache, slots: Tensor, pos: Tensor, s: int) -> Tensor:
        h = self.model(input_ids, cache, slots, pos, s)
        return F.linear(h, self.model.embed_tokens.weight)
