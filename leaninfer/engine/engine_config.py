from dataclasses import dataclass, field

import torch



@dataclass(frozen=True)
class EngineConfig:
    n_slots: int = 64
    slot_len: int = 2048  # tokens per slot (prompt + output); sizes the KV pool. Overridden by replace() in main.py
    device: torch.device = field(
            default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    # Diverging from fp32 since it locks SDPA to the math backend
    dtype: torch.dtype = torch.bfloat16
    # Capture decode into a CUDA graph
    cuda_graphs: bool = True

    @property
    def kv_capacity(self) -> int:
        return self.n_slots * self.slot_len
