
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Request:
    id: int
    prompt: list[int]
    max_new: int = 64
    state: State = State.WAITING
    slot: int = -1 # scheduler slot number
    pos: int = 0 # tokens of this request in the cache
    out: list[int] = field(default_factory=list)

class Scheduler:
    def __init__(self, n_slots: int) -> None:
        self.free_slots: deque[int] = deque(range(n_slots))
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

    def add(self, req: Request) -> None:
        self.waiting.append(req)

    # TODO: add bulk method?


    def admit(self) -> Request | None:
        """Admit a Request into a slot. 
        Returns None when either no more work or no free slots"""
        if not self.waiting or not self.free_slots:
            return None
        req = self.waiting.popleft()
        req.slot = self.free_slots.popleft()
        req.state = State.RUNNING
        self.running.append(req)
        return req

    # Push the slot back into free so it can be used
    # TODO rename to deallocate?
    def retire(self, req: Request) -> None:
        req.state = State.FINISHED
        self.free_slots.append(req.slot)
        self.running.remove(req)

    def busy(self) -> bool:
        return bool(self.waiting or self.running)

