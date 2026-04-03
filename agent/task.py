"""agent/task.py — Task lifecycle tracking."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETE  = "complete"
    FAILED    = "failed"
    ABORTED   = "aborted"


@dataclass
class Task:
    description: str
    max_turns:   int    = 20
    status:      TaskStatus = TaskStatus.PENDING
    started_at:  float  = 0.0
    ended_at:    float  = 0.0
    turns_used:  int    = 0
    error:       str    = ""

    def start(self) -> None:
        self.status     = TaskStatus.RUNNING
        self.started_at = time.time()

    def complete(self) -> None:
        self.status   = TaskStatus.COMPLETE
        self.ended_at = time.time()

    def fail(self, error: str) -> None:
        self.status   = TaskStatus.FAILED
        self.ended_at = time.time()
        self.error    = error

    def abort(self) -> None:
        self.status   = TaskStatus.ABORTED
        self.ended_at = time.time()

    @property
    def elapsed(self) -> float:
        end = self.ended_at or time.time()
        return end - self.started_at if self.started_at else 0.0
