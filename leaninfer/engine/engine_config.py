from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class EngineConfig:
    n_slots: int = 128
    slot_len: int = 2048  # tokens per slot (prompt + output); sizes the KV pool. Overridden by replace() in main.py
    block_size: int = 256 # tokens per KV block. flash attn's paged KV requires a multiple of 256
    num_blocks: int = 512 # Matches naive slot based cache size @ 2048*64
    # Admission headroom: I'm artificially maximizing goodput because I already know how large each prompt is
    # This is just a stopgap solution until I plug in real prompts after chunked prefill
    watermark: float = 1.0
    device: torch.device = field(
            default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    # Diverging from fp32 since it locks SDPA to the math backend
    dtype: torch.dtype = torch.bfloat16
    # Capture decode into a CUDA graph: block_tables + cache_seqlens are staged buffers
    cuda_graphs: bool = True

    @property
    def usable_blocks(self) -> int:
        return self.num_blocks - 1  # block 0 is allocator's trash block. See cache_manager.py

    @property
    def kv_capacity(self) -> int:
        return self.block_size * self.usable_blocks
