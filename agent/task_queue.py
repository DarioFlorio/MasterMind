"""
agent/task_queue.py — BabyAGI-style task queue with objective-driven loop.

Provides:
  - ObjectiveTaskQueue: maintains a prioritised list of tasks toward a goal
  - TaskAgent: creates new tasks from results, re-prioritises, executes
  - Integrates with QueryEngine via the /plan slash command or TaskQueueTool

Unlike UltraPlan (which blueprints before execution), the task queue is
*adaptive* — new tasks emerge from what the agent discovers during execution.

Usage:
    queue = ObjectiveTaskQueue(objective="Build a REST API for user auth")
    queue.add_task("Research best Python auth libraries")
    queue.run(engine, max_iterations=10)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING
import json

if TYPE_CHECKING:
    from agent.query_engine import QueryEngine


class TaskStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class QueuedTask:
    id:          str
    description: str
    priority:    int   = 50       # 1 (highest) → 100 (lowest)
    status:      TaskStatus = TaskStatus.PENDING
    result:      str   = ""
    created_at:  float = field(default_factory=time.time)
    completed_at: float = 0.0
    parent_id:   str   = ""       # which task spawned this one

    def to_dict(self) -> dict:
        return {
            "id": self.id, "description": self.description,
            "priority": self.priority, "status": self.status.value,
            "result": self.result[:300], "parent_id": self.parent_id,
        }


class ObjectiveTaskQueue:
    """
    Maintains a prioritised queue of tasks toward a single objective.
    New tasks are synthesised from prior results (BabyAGI pattern).
    """

    _SAVE_DIR = Path(__file__).parent.parent / "memdir" / "task_queues"

    def __init__(self, objective: str, queue_id: str | None = None) -> None:
        self.objective  = objective
        self.queue_id   = queue_id or uuid.uuid4().hex[:8]
        self._tasks: list[QueuedTask] = []
        self._done_results: list[str] = []   # results of completed tasks

    # ── Task management ───────────────────────────────────────────────────────

    def add_task(self, description: str, priority: int = 50,
                 parent_id: str = "") -> QueuedTask:
        t = QueuedTask(
            id=uuid.uuid4().hex[:6],
            description=description,
            priority=priority,
            parent_id=parent_id,
        )
        self._tasks.append(t)
        self._tasks.sort(key=lambda x: x.priority)
        return t

    def next_pending(self) -> QueuedTask | None:
        for t in self._tasks:
            if t.status == TaskStatus.PENDING:
                return t
        return None

    def pending_count(self) -> int:
        return sum(1 for t in self._tasks if t.status == TaskStatus.PENDING)

    def summary(self) -> str:
        done    = sum(1 for t in self._tasks if t.status == TaskStatus.DONE)
        pending = self.pending_count()
        failed  = sum(1 for t in self._tasks if t.status == TaskStatus.FAILED)
        return (f"Objective: {self.objective}\n"
                f"Tasks: {done} done, {pending} pending, {failed} failed, "
                f"{len(self._tasks)} total")

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, engine: "QueryEngine", max_iterations: int = 10,
            on_update: callable | None = None) -> str:
        """
        Run the task loop: execute → create new tasks → prioritise → repeat.
        Returns a final summary.
        """
        iterations = 0
        while self.pending_count() > 0 and iterations < max_iterations:
            iterations += 1
            task = self.next_pending()
            if task is None:
                break

            task.status = TaskStatus.RUNNING
            if on_update:
                on_update(f"[task {task.id}] Running: {task.description}")

            # ── Execute the task ──────────────────────────────────────────────
            prompt = self._build_execution_prompt(task)
            try:
                result = engine.submit_message(prompt)
                task.result = result
                task.status = TaskStatus.DONE
                task.completed_at = time.time()
                self._done_results.append(f"[{task.id}] {task.description}: {result[:400]}")
            except Exception as exc:
                task.result = str(exc)
                task.status = TaskStatus.FAILED

            if on_update:
                on_update(f"[task {task.id}] {'Done' if task.status == TaskStatus.DONE else 'Failed'}")

            # ── Create new tasks from result ──────────────────────────────────
            if task.status == TaskStatus.DONE and self.pending_count() < 8:
                new_tasks = self._generate_new_tasks(engine, task)
                for desc, prio in new_tasks:
                    self.add_task(desc, priority=prio, parent_id=task.id)
                    if on_update:
                        on_update(f"[queue] New task added: {desc}")

            # ── Reprioritise remaining tasks ──────────────────────────────────
            self._reprioritise(engine)

        self.save()
        return self._final_summary(engine)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_execution_prompt(self, task: QueuedTask) -> str:
        context = ""
        if self._done_results:
            last = "\n".join(self._done_results[-5:])
            context = f"\n\nContext from completed tasks:\n{last}"
        return (
            f"Objective: {self.objective}\n"
            f"Current task: {task.description}{context}\n\n"
            f"Complete this task concisely and thoroughly."
        )

    def _generate_new_tasks(self, engine: "QueryEngine",
                            completed: QueuedTask) -> list[tuple[str, int]]:
        """Ask the model to suggest follow-up tasks based on the result."""
        existing = [t.description for t in self._tasks
                    if t.status == TaskStatus.PENDING]
        existing_str = "\n".join(f"- {d}" for d in existing[:10]) or "(none)"

        prompt = (
            f"Objective: {self.objective}\n"
            f"Just completed: {completed.description}\n"
            f"Result summary: {completed.result[:300]}\n\n"
            f"Already pending tasks:\n{existing_str}\n\n"
            f"List up to 3 NEW tasks that would meaningfully advance the objective "
            f"given what we just learned. Format: one task per line, starting with "
            f"a number 1-100 (priority, lower=higher priority) followed by a colon "
            f"and the task description. Return ONLY the task lines, nothing else. "
            f"If no new tasks are needed, return DONE."
        )
        try:
            raw = engine.submit_message(prompt)
        except Exception:
            return []

        if "DONE" in raw.upper():
            return []

        tasks: list[tuple[str, int]] = []
        for line in raw.strip().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Parse "50: description" or "50. description"
            import re
            m = re.match(r'^(\d+)\s*[:.]\s*(.+)$', line)
            if m:
                prio = max(1, min(100, int(m.group(1))))
                desc = m.group(2).strip()
                if desc and desc not in [t.description for t in self._tasks]:
                    tasks.append((desc, prio))
        return tasks[:3]

    def _reprioritise(self, engine: "QueryEngine") -> None:
        """Re-sort pending tasks by priority (keep user-set priorities stable)."""
        self._tasks.sort(key=lambda t: (t.priority, t.created_at))

    def _final_summary(self, engine: "QueryEngine") -> str:
        done_tasks  = [t for t in self._tasks if t.status == TaskStatus.DONE]
        failed_tasks = [t for t in self._tasks if t.status == TaskStatus.FAILED]

        if not done_tasks:
            return f"No tasks completed for objective: {self.objective}"

        results = "\n".join(
            f"- {t.description}: {t.result[:200]}" for t in done_tasks[-10:]
        )
        prompt = (
            f"Objective: {self.objective}\n\n"
            f"Completed tasks and results:\n{results}\n\n"
            f"Provide a brief synthesis of what was accomplished toward the objective."
        )
        try:
            return engine.submit_message(prompt)
        except Exception:
            return self.summary()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self._SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path = self._SAVE_DIR / f"{self.queue_id}.json"
        data = {
            "queue_id": self.queue_id,
            "objective": self.objective,
            "tasks": [t.to_dict() for t in self._tasks],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, queue_id: str) -> "ObjectiveTaskQueue":
        path = cls._SAVE_DIR / f"{queue_id}.json"
        data = json.loads(path.read_text())
        q = cls(objective=data["objective"], queue_id=data["queue_id"])
        for td in data.get("tasks", []):
            t = QueuedTask(
                id=td["id"], description=td["description"],
                priority=td.get("priority", 50),
                status=TaskStatus(td.get("status", "pending")),
                result=td.get("result", ""),
                parent_id=td.get("parent_id", ""),
            )
            q._tasks.append(t)
        return q

    @classmethod
    def list_queues(cls) -> list[dict]:
        cls._SAVE_DIR.mkdir(parents=True, exist_ok=True)
        out = []
        for p in sorted(cls._SAVE_DIR.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                tasks = d.get("tasks", [])
                done = sum(1 for t in tasks if t.get("status") == "done")
                out.append({
                    "id": d["queue_id"],
                    "objective": d["objective"][:60],
                    "tasks": len(tasks),
                    "done": done,
                })
            except Exception:
                pass
        return out
