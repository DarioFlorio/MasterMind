from __future__ import annotations
import time
from dataclasses import dataclass, field


@dataclass
class SessionUsage:
    turns:        int   = 0
    prompt_tok:   int   = 0
    completion_tok: int = 0
    start_time:   float = field(default_factory=time.time)

    def add_turn(self, p: int = 0, c: int = 0) -> None:
        self.turns          += 1
        self.prompt_tok     += p
        self.completion_tok += c

    @property
    def total_tok(self) -> int:
        return self.prompt_tok + self.completion_tok

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> str:
        e = self.elapsed
        return (f"Session: {self.turns} turns | "
                f"~{self.total_tok} tokens | "
                f"{e:.0f}s elapsed")
