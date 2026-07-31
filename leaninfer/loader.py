from torch import  Tensor
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from leaninfer.qwen3.model import Qwen3ForCausalLM 
from leaninfer.qwen3.model_config import ModelConfig, MODEL_ID
from leaninfer.engine.engine_config import EngineConfig


# ---- weights ----
         # dict[str, Tensor], dtype bf16 -> cast slices to fp32 as you use them
# ---- weight layout ----
# proj weights are [out, in] and bias-free -> nn.Linear(in, out, bias=False); F.linear(x, W) = x @ W.T
# embed_tokens.weight [VOCAB, HIDDEN] is TIED to lm_head (no separate lm_head weight in the checkpoint)
# every *_layernorm / q_norm / k_norm is a single RMSNorm scale vector -> declared as `.weight`
# the inner `model` submodule reproduces the `model.` prefix in every checkpoint key.
# params load in engine_config.dtype, bf16 by default -- so no longer bit-comparable to the fp32 oracle.


def load_model(model_config: ModelConfig, engine_config: EngineConfig, path: str | None = None) -> Qwen3ForCausalLM:
    if path is None:
        path = hf_hub_download(MODEL_ID, "model.safetensors")

    
    model = Qwen3ForCausalLM(model_config)
    with safe_open(path, "pt", "cpu") as f:
        W: dict[str, Tensor] = {name: f.get_tensor(name) for name in f.keys()}
    W.pop("lm_head.weight")        # redundant duplicate of model.embed_tokens.weight (tied)
    # alternatively just declare lm_head
    model.load_state_dict(W)
    return model.eval().to(device=engine_config.device, dtype=engine_config.dtype)


