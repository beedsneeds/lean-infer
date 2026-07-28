from dataclasses import dataclass
from typing import cast

from transformers import Qwen3Config

MODEL_ID = "Qwen/Qwen3-0.6B"

# # ---- config (from config.json) ----
# N_LAYERS     = 28
# HIDDEN       = 1024
# N_HEADS      = 16
# N_KV_HEADS   = 8 # GQA: each KV head is shared by N_HEADS // N_KV_HEADS = 2 query heads
# HEAD_DIM     = 128 # decoupled! N_HEADS * HEAD_DIM = 2048 != HIDDEN
# INTERMEDIATE = 3072
# VOCAB        = 151936
# EPS          = 1e-6
# ROPE_THETA   = 1_000_000


@dataclass(frozen=True)
class ModelConfig:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float

    # Strong typing HF's optional
    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID) -> "ModelConfig":
        hf = Qwen3Config.from_pretrained(model_id)
        rope = cast(dict[str, float], hf.rope_parameters)
        return cls(
            num_hidden_layers=hf.num_hidden_layers,
            hidden_size=hf.hidden_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=cast(int, hf.num_key_value_heads),
            head_dim=cast(int, hf.head_dim),
            intermediate_size=hf.intermediate_size,
            vocab_size=hf.vocab_size,
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=rope["rope_theta"],
        )
