import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from leaninfer import metrics
from leaninfer.engine.engine_config import EngineConfig
from leaninfer.engine.cache_manager import BlockAllocator, blocks_for

class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new_tokens: int
    state: State = State.WAITING
    slot: int = -1 # admission number. No longer addresses memory like slot-based cache
    block_table: list[int] = field(default_factory=list) # physical KV block ids, logical order
    pos: int = 0 # tokens of this request in the cache
    out: list[int] = field(default_factory=list)
    t_arrival: float = 0.0
    t_admit: float = 0.0
    t_first: float = 0.0
    t_last: float = 0.0

    @property
    def prefill_time(self) -> float:
        return self.t_first - self.t_admit
    
    @property
    def ttft(self) -> float:
        return self.t_first - self.t_arrival

    @property
    def tpot(self) -> float:
        n = len(self.out) - 1
        return (self.t_last - self.t_first) / n if n > 0 else 0.0


class Scheduler:
    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.free_slots: deque[int] = deque(range(config.n_slots))
        self.alloc = BlockAllocator(config.num_blocks)
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        req.t_arrival = time.perf_counter()
        self.waiting.append(req)
        metrics.WAITING.set(len(self.waiting))
    

    def _outstanding(self) -> int:
        """Blocks that running requests will need to reach their budgets
        Remove once you move away from synthetic prompts"""
        bs = self.config.block_size
        return sum(blocks_for(len(r.prompt) + r.max_new_tokens, bs) - len(r.block_table)
                   for r in self.running)

    @property
    def reserved_tokens(self) -> int:
        held = self.config.usable_blocks - self.alloc.num_free
        return held * self.config.block_size

    def _publish_kv(self) -> None:
        """Blocks currently held. Must be republished wherever the pool moves
        """
        metrics.KV_RESERVED.set(self.reserved_tokens)

    def admit(self) -> Request | None:
        """Admit the head of the queue if a slot and its prompt's blocks are available.
        Temporary: reserve _outstanding too
        Blocks for decode are claimed later, as the sequence grows (grow_for_decode)"""
        if not self.waiting or not self.free_slots:
            return None
        need = blocks_for(len(self.waiting[0].prompt), self.config.block_size)
        reserve = int(self.config.watermark * self._outstanding())
        if self.alloc.num_free - need < reserve:
            return None  # FCFS, head-of-line blocks / convoy
        req = self.waiting.popleft()
        req.block_table = self.alloc.allocate(need)
        req.slot = self.free_slots.popleft()
        req.state = State.RUNNING
        self.running.append(req)
        req.t_admit = time.perf_counter()
        metrics.QUEUE_DELAY.observe(req.t_admit - req.t_arrival)
        metrics.WAITING.set(len(self.waiting))
        metrics.RUNNING.set(len(self.running))
        self._publish_kv()
        return req

    def grow_for_decode(self) -> None:
        """Give every running request a block for the token it's about to write.
        Preempts from the back (youngest) when the pool is exhausted"""
        for r in list(self.running):  # oldest first
            if r.state is not State.RUNNING:  # preempted earlier in this pass
                continue
            if r.pos == len(r.block_table) * self.config.block_size:
                while self.alloc.num_free < 1:
                    assert len(self.running) > 1, "one sequence exceeds total KV capacity"
                    self.preempt()  # youngest (may be r itself)
                if r.state is State.RUNNING:
                    r.block_table += self.alloc.allocate(1)
        self._publish_kv()

    def preempt(self) -> None:
        """Evict the youngest running request
        """
        victim = self.running.pop()
        metrics.PREEMPTIONS.inc()
        metrics.RECOMPUTE_TOKENS.inc(victim.pos)
        self.alloc.release(victim.block_table)
        victim.block_table = []
        victim.out = []
        victim.pos = 0
        victim.state = State.WAITING
        self.free_slots.append(victim.slot)
        victim.slot = -1
        self.waiting.appendleft(victim)  # was running: outranks fresh arrivals
        metrics.WAITING.set(len(self.waiting))
        metrics.RUNNING.set(len(self.running))
        self._publish_kv()

    def retire(self, req: Request) -> None:
        req.state = State.FINISHED
        self.alloc.release(req.block_table)
        req.block_table = []
        self.free_slots.append(req.slot)
        self.running.remove(req)
        metrics.RUNNING.set(len(self.running))
        self._publish_kv()

    def busy(self) -> bool:
        return bool(self.waiting or self.running)

