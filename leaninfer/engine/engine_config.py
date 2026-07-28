from dataclasses import dataclass, field

import torch



@dataclass(frozen=True)
class EngineConfig:
    n_slots: int = 16
    max_len: int = 512
    max_new: int = 64
    device: torch.device = field(
            default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    dtype: torch.dtype = torch.float32

    @property
    def kv_capacity(self) -> int:
        return self.n_slots * self.max_len
