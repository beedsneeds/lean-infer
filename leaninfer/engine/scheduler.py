import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from leaninfer import metrics
from leaninfer.engine.engine_config import EngineConfig

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
    slot: int = -1 # scheduler slot number
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
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        req.t_arrival = time.perf_counter()
        self.waiting.append(req)
        metrics.WAITING.set(len(self.waiting))
    

    def admit(self) -> Request | None:
        """Admit a Request into a slot. 
        Returns None when either no more work or no free slots"""
        if not self.waiting or not self.free_slots:
            return None
        req = self.waiting.popleft()
        req.slot = self.free_slots.popleft()
        req.state = State.RUNNING
        self.running.append(req)
        req.t_admit = time.perf_counter()
        metrics.QUEUE_DELAY.observe(req.t_admit - req.t_arrival)
        metrics.WAITING.set(len(self.waiting))
        metrics.RUNNING.set(len(self.running))
        return req

    # Push the slot back into free so it can be used
    # TODO rename to deallocate?
    def retire(self, req: Request) -> None:
        req.state = State.FINISHED
        self.free_slots.append(req.slot)
        self.running.remove(req)
        metrics.RUNNING.set(len(self.running))

    def busy(self) -> bool:
        return bool(self.waiting or self.running)

