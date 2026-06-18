"""
agent/goal_tracker.py — Goal locking and drift prevention for EVE.

Pins the active goal at the top of every prompt injection so the model
cannot drift, pretend completion, or answer unrelated questions while
a task is in flight.
"""
from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GoalStep:
    description: str
    status: str = "pending"   # pending | done | failed | skipped
    ts_start: float = field(default_factory=time.time)
    ts_end: Optional[float] = None
    error: str = ""

    def complete(self) -> None:
        self.status = "done"
        self.ts_end = time.time()

    def fail(self, error: str = "") -> None:
        self.status = "failed"
        self.ts_end = time.time()
        self.error = error


class GoalTracker:
    """
    Tracks the current goal, its sub-steps, and injects a status block
    into every prompt so EVE always knows where she is.
    """

    _PERSIST_PATH = "memdir/current_goal.json"

    def __init__(self, working_dir: str = "") -> None:
        self._working_dir = working_dir
        self.goal: str = ""
        self.steps: list[GoalStep] = []
        self.turn_count: int = 0
        self.start_ts: float = 0.0
        self.completed: bool = False
        self._load()

    # ── Goal management ───────────────────────────────────────────────────────

    # Short messages that are NOT goals
    _TRIVIAL_RE = re.compile(
        r"^(oi|hi|hey|hello|yo|sup|ok|okay|sure|yes|no|thanks|ty|"
        r"what|who|where|when|how|why|cool|nice|good|"
        r"who am i|continue|go on)$",
        re.I,
    )

    def set_goal(self, goal: str, steps: Optional[list[str]] = None) -> None:
        """Only set a goal if the message is a real task, not a greeting/trivial query."""
        stripped = goal.strip()
        # Don't replace an active goal with a trivial message
        if self._TRIVIAL_RE.match(stripped) or len(stripped) < 20:
            return
        # Don't overwrite an active unfinished goal with a trivial follow-up
        if self.goal and not self.completed and len(stripped) < 40:
            return
        self.goal = stripped
        self.steps = [GoalStep(s) for s in (steps or [])]
        self.turn_count = 0
        self.start_ts = time.time()
        self.completed = False
        self._save()

    def update_steps(self, steps: list[str]) -> None:
        existing = {s.description: s for s in self.steps}
        new_steps = []
        for s in steps:
            if s in existing:
                new_steps.append(existing[s])
            else:
                new_steps.append(GoalStep(s))
        self.steps = new_steps
        self._save()

    def mark_step_done(self, idx: int) -> None:
        if 0 <= idx < len(self.steps):
            self.steps[idx].complete()
            self._save()

    def mark_step_failed(self, idx: int, error: str = "") -> None:
        if 0 <= idx < len(self.steps):
            self.steps[idx].fail(error)
            self._save()

    def complete_goal(self) -> None:
        self.completed = True
        for s in self.steps:
            if s.status == "pending":
                s.status = "skipped"
        self._save()

    def clear(self) -> None:
        self.goal = ""
        self.steps = []
        self.completed = False
        self._save()

    def tick(self) -> None:
        self.turn_count += 1

    # ── Injection ─────────────────────────────────────────────────────────────

    def inject(self, user_message: str, force: bool = False) -> str:
        """
        Prepend a compact goal reminder. Only fires on turn 1 and every
        2 turns after that — not every single turn — to keep token cost low.
        """
        if not self.goal or self.completed:
            return user_message
        # Only inject on turn 1, or every 2 turns, or if forced
        if not force and self.turn_count > 1 and self.turn_count % 2 != 0:
            return user_message
        block = self._status_block()
        return f"{block}\n{user_message}"

    def _status_block(self) -> str:
        """Compact single-line goal reminder — minimal token cost."""
        pending = [s for s in self.steps if s.status == "pending"]
        failed  = [s for s in self.steps if s.status == "failed"]
        next_step = f" Next: {pending[0].description[:50]}" if pending else ""
        fail_note = f" [{len(failed)} failed—try differently]" if failed else ""
        return (
            f"[GOAL t{self.turn_count}] {self.goal[:80]}{fail_note}{next_step} "
            f"— do NOT stop until complete."
        )

    def persistence_reminder(self) -> str:
        """Inject into tool results to keep EVE on track."""
        if not self.goal or self.completed:
            return ""
        pending_steps = [s for s in self.steps if s.status == "pending"]
        next_step = pending_steps[0].description if pending_steps else "continue toward goal"
        return (
            f"\n[GoalTracker] Goal still active: {self.goal[:80]}\n"
            f"Next action: {next_step}\n"
            f"Keep going — do NOT stop until complete.\n"
        )

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _persist_path(self) -> Path:
        base = Path(self._working_dir) if self._working_dir else Path(".")
        return base / self._PERSIST_PATH

    def _save(self) -> None:
        try:
            p = self._persist_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "goal": self.goal,
                "steps": [
                    {"desc": s.description, "status": s.status, "error": s.error}
                    for s in self.steps
                ],
                "turn_count": self.turn_count,
                "start_ts": self.start_ts,
                "completed": self.completed,
            }
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load(self) -> None:
        try:
            p = self._persist_path()
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            self.goal = data.get("goal", "")
            self.steps = [
                GoalStep(
                    description=s["desc"],
                    status=s.get("status", "pending"),
                    error=s.get("error", ""),
                )
                for s in data.get("steps", [])
            ]
            self.turn_count = data.get("turn_count", 0)
            self.start_ts = data.get("start_ts", 0.0)
            self.completed = data.get("completed", False)
        except Exception:
            pass
