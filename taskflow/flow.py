"""
Task Flow — durable multi-step pipeline orchestration.
MasterMind durable pipeline orchestration.

Sits above background tasks (cron/swarm). Manages durable multi-step flows
with their own state, revision tracking, and sync semantics.

Two modes:
  - Managed:  Task Flow owns the lifecycle, creates/drives tasks.
  - Mirrored: Task Flow observes externally created tasks.

State persists to disk so flows survive process restarts.
"""
from __future__ import annotations
import asyncio
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional


_STATE_DIR = Path.home() / ".mastermind" / "taskflows"


class FlowStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    CANCELLED = "cancelled"


class StepStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    SUCCEEDED = "succeeded"
    FAILED    = "failed"
    SKIPPED   = "skipped"


class SyncMode(str, Enum):
    MANAGED  = "managed"   # Flow creates and drives tasks
    MIRRORED = "mirrored"  # Flow observes externally created tasks


@dataclass
class FlowStep:
    id: str
    name: str
    status: StepStatus = StepStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    task_fn: Optional[Callable] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("task_fn", None)
        return d


@dataclass
class FlowState:
    flow_id: str
    name: str
    status: FlowStatus = FlowStatus.PENDING
    sync_mode: SyncMode = SyncMode.MANAGED
    steps: list[FlowStep] = field(default_factory=list)
    current_step: int = 0
    revision: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_intent: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "flow_id": self.flow_id,
            "name": self.name,
            "status": self.status.value,
            "sync_mode": self.sync_mode.value,
            "steps": [s.to_dict() for s in self.steps],
            "current_step": self.current_step,
            "revision": self.revision,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "cancel_intent": self.cancel_intent,
            "metadata": self.metadata,
        }


class FlowCancelledError(Exception):
    pass

class FlowStepError(Exception):
    def __init__(self, step_name: str, cause: Exception):
        super().__init__(f"Step '{step_name}' failed: {cause}")
        self.step_name = step_name
        self.cause = cause


class TaskFlow:
    """
    A durable multi-step pipeline flow.

    Usage (simple):
        flow = TaskFlow("weekly-report")
        flow.add_step("gather", gather_data_fn)
        flow.add_step("generate", generate_report_fn)
        flow.add_step("deliver", deliver_fn)
        await flow.run()

    Usage (mirrored):
        flow = TaskFlow("morning-ops", mode=SyncMode.MIRRORED)
        flow.mirror_task("backup-job", check_backup_fn)
        flow.mirror_task("email-report", check_email_fn)
        await flow.run()
    """

    def __init__(
        self,
        name: str,
        flow_id: str | None = None,
        mode: SyncMode = SyncMode.MANAGED,
        state_dir: Path | None = None,
        on_step_start: Optional[Callable] = None,
        on_step_complete: Optional[Callable] = None,
        on_flow_complete: Optional[Callable] = None,
    ):
        self._state_dir = state_dir or _STATE_DIR
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Try to resume existing flow
        fid = flow_id or name.lower().replace(" ", "-")
        existing = self._load_state(fid)
        if existing:
            self._state = existing
        else:
            self._state = FlowState(
                flow_id=fid,
                name=name,
                sync_mode=mode,
            )

        self._on_step_start = on_step_start
        self._on_step_complete = on_step_complete
        self._on_flow_complete = on_flow_complete
        self._step_fns: dict[str, Callable] = {}

    # ── Step management ────────────────────────────────────────────────────────

    def add_step(self, name: str, fn: Callable, step_id: str | None = None) -> "TaskFlow":
        """Add a managed step to the flow."""
        sid = step_id or f"step-{len(self._state.steps):03d}-{name.lower().replace(' ', '-')}"
        # Don't re-add if already completed in a resumed flow
        existing = next((s for s in self._state.steps if s.id == sid), None)
        if existing and existing.status == StepStatus.SUCCEEDED:
            self._step_fns[sid] = fn
            return self
        if not existing:
            step = FlowStep(id=sid, name=name)
            self._state.steps.append(step)
        self._step_fns[sid] = fn
        return self

    def mirror_task(self, name: str, check_fn: Callable, step_id: str | None = None) -> "TaskFlow":
        """Add a mirrored step (observes an external task)."""
        self._state.sync_mode = SyncMode.MIRRORED
        return self.add_step(name, check_fn, step_id)

    # ── Execution ──────────────────────────────────────────────────────────────

    async def run(self) -> dict[str, Any]:
        """Execute all pending steps. Resumes from last successful step."""
        with self._lock:
            if self._state.cancel_intent:
                raise FlowCancelledError(f"Flow {self._state.name} is cancelled")
            self._state.status = FlowStatus.RUNNING
            self._state.updated_at = time.time()
            self._save_state()

        results = {}
        for i, step in enumerate(self._state.steps):
            if self._state.cancel_intent:
                raise FlowCancelledError(f"Flow cancelled at step '{step.name}'")

            if step.status == StepStatus.SUCCEEDED:
                results[step.name] = step.result
                continue

            if step.status in (StepStatus.FAILED, StepStatus.PENDING):
                await self._run_step(i, step, results)
                results[step.name] = step.result

        with self._lock:
            self._state.status = FlowStatus.SUCCEEDED
            self._state.updated_at = time.time()
            self._state.revision += 1
            self._save_state()

        if self._on_flow_complete:
            self._on_flow_complete(self._state, results)

        return results

    async def _run_step(self, idx: int, step: FlowStep, context: dict) -> None:
        fn = self._step_fns.get(step.id)
        if not fn:
            step.status = StepStatus.SKIPPED
            return

        with self._lock:
            step.status = StepStatus.RUNNING
            step.started_at = time.time()
            self._state.current_step = idx
            self._state.updated_at = time.time()
            self._save_state()

        if self._on_step_start:
            self._on_step_start(step)

        try:
            if asyncio.iscoroutinefunction(fn):
                result = await fn(context)
            else:
                result = await asyncio.get_event_loop().run_in_executor(None, fn, context)

            with self._lock:
                step.status = StepStatus.SUCCEEDED
                step.result = result
                step.finished_at = time.time()
                self._state.revision += 1
                self._state.updated_at = time.time()
                self._save_state()

            if self._on_step_complete:
                self._on_step_complete(step)

        except Exception as e:
            with self._lock:
                step.status = StepStatus.FAILED
                step.error = str(e)
                step.finished_at = time.time()
                self._state.status = FlowStatus.FAILED
                self._state.revision += 1
                self._state.updated_at = time.time()
                self._save_state()
            raise FlowStepError(step.name, e)

    # ── Control ────────────────────────────────────────────────────────────────

    def cancel(self) -> None:
        """Signal cancellation. Active steps will stop at the next checkpoint."""
        with self._lock:
            self._state.cancel_intent = True
            self._state.status = FlowStatus.CANCELLED
            self._state.updated_at = time.time()
            self._save_state()

    def reset_step(self, step_name: str) -> bool:
        """Reset a failed step so it can be retried."""
        for step in self._state.steps:
            if step.name == step_name and step.status == StepStatus.FAILED:
                step.status = StepStatus.PENDING
                step.error = None
                self._save_state()
                return True
        return False

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def status(self) -> FlowStatus:
        return self._state.status

    @property
    def flow_id(self) -> str:
        return self._state.flow_id

    def summary(self) -> dict:
        return self._state.to_dict()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _state_path(self, flow_id: str) -> Path:
        return self._state_dir / f"{flow_id}.json"

    def _save_state(self) -> None:
        path = self._state_path(self._state.flow_id)
        path.write_text(json.dumps(self._state.to_dict(), indent=2))

    def _load_state(self, flow_id: str) -> Optional[FlowState]:
        path = self._state_path(flow_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            state = FlowState(
                flow_id=data["flow_id"],
                name=data["name"],
                status=FlowStatus(data["status"]),
                sync_mode=SyncMode(data["sync_mode"]),
                current_step=data["current_step"],
                revision=data["revision"],
                created_at=data["created_at"],
                updated_at=data["updated_at"],
                cancel_intent=data.get("cancel_intent", False),
                metadata=data.get("metadata", {}),
            )
            for sd in data.get("steps", []):
                state.steps.append(FlowStep(
                    id=sd["id"],
                    name=sd["name"],
                    status=StepStatus(sd["status"]),
                    result=sd.get("result"),
                    error=sd.get("error"),
                    started_at=sd.get("started_at"),
                    finished_at=sd.get("finished_at"),
                ))
            return state
        except Exception:
            return None


# ── Flow Registry ─────────────────────────────────────────────────────────────

class FlowRegistry:
    """Lists and manages all flows."""

    def __init__(self, state_dir: Path | None = None):
        self._dir = state_dir or _STATE_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    def list_flows(self) -> list[dict]:
        flows = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                flows.append({
                    "flow_id": data["flow_id"],
                    "name": data["name"],
                    "status": data["status"],
                    "steps": len(data.get("steps", [])),
                    "updated_at": data.get("updated_at"),
                })
            except Exception:
                pass
        return sorted(flows, key=lambda f: f.get("updated_at", 0), reverse=True)

    def get_flow(self, flow_id: str) -> Optional[dict]:
        path = self._dir / f"{flow_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
        return None

    def cancel_flow(self, flow_id: str) -> bool:
        path = self._dir / f"{flow_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            data["cancel_intent"] = True
            data["status"] = FlowStatus.CANCELLED.value
            path.write_text(json.dumps(data, indent=2))
            return True
        except Exception:
            return False

    def delete_flow(self, flow_id: str) -> bool:
        path = self._dir / f"{flow_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
