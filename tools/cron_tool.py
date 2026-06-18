# -*- coding: utf-8 -*-
"""
tools/cron_tool.py — Cron job scheduler tool.

MasterMind built-in's ScheduleCronTool.
Manages scheduled tasks using Python's schedule library or system cron.
Jobs are persisted to ~/.mastermind/cron_jobs.json.
"""
from __future__ import annotations
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult

log = logging.getLogger("tools.cron")
CRON_PATH = Path.home() / ".mastermind" / "cron_jobs.json"


@dataclass
class CronJob:
    job_id: str
    name: str
    schedule: str   # e.g. "every 5 minutes", "every day at 09:00", "*/5 * * * *"
    command: str
    enabled: bool = True
    last_run: float = 0.0
    run_count: int = 0
    created_at: float = field(default_factory=time.time)


class CronScheduler:
    """In-process cron scheduler using a background thread."""

    def __init__(self):
        self._jobs: dict[str, CronJob] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._load()

    def _load(self) -> None:
        if CRON_PATH.exists():
            try:
                data = json.loads(CRON_PATH.read_text())
                for d in data:
                    j = CronJob(**d)
                    self._jobs[j.job_id] = j
                log.info("Cron: loaded %d jobs", len(self._jobs))
            except Exception as e:
                log.warning("Cron load failed: %s", e)

    def _save(self) -> None:
        CRON_PATH.parent.mkdir(parents=True, exist_ok=True)
        CRON_PATH.write_text(json.dumps([asdict(j) for j in self._jobs.values()], indent=2))

    def add(self, name: str, schedule: str, command: str) -> CronJob:
        job = CronJob(job_id=str(uuid.uuid4())[:8], name=name, schedule=schedule, command=command)
        self._jobs[job.job_id] = job
        self._save()
        self._ensure_running()
        return job

    def delete(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def list_jobs(self) -> list[CronJob]:
        return list(self._jobs.values())

    def enable(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].enabled = True
            self._save()
            return True
        return False

    def disable(self, job_id: str) -> bool:
        if job_id in self._jobs:
            self._jobs[job_id].enabled = False
            self._save()
            return True
        return False

    def _parse_interval_seconds(self, schedule: str) -> int | None:
        """Parse simple schedule strings into seconds."""
        s = schedule.lower().strip()
        if "every" in s:
            parts = s.replace("every", "").strip().split()
            try:
                n = int(parts[0])
                unit = parts[1] if len(parts) > 1 else "seconds"
                multipliers = {
                    "second": 1, "seconds": 1, "sec": 1,
                    "minute": 60, "minutes": 60, "min": 60,
                    "hour": 3600, "hours": 3600,
                    "day": 86400, "days": 86400,
                }
                return n * multipliers.get(unit, 60)
            except (ValueError, IndexError):
                return None
        return None

    def _should_run(self, job: CronJob, now: float) -> bool:
        interval = self._parse_interval_seconds(job.schedule)
        if interval is None:
            return False  # complex cron expressions need crontab
        return now - job.last_run >= interval

    def _run_job(self, job: CronJob) -> None:
        import subprocess
        log.info("Cron: running job %s (%s)", job.name, job.job_id)
        try:
            subprocess.run(job.command, shell=True, timeout=300,
                           capture_output=True, text=True)
        except Exception as e:
            log.error("Cron job %s error: %s", job.name, e)
        job.last_run = time.time()
        job.run_count += 1
        self._save()

        from hooks.manager import hook_manager
        hook_manager.fire("cron_fire", job_id=job.job_id, name=job.name)

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=5):
            now = time.time()
            for job in list(self._jobs.values()):
                if job.enabled and self._should_run(job, now):
                    t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
                    t.start()

    def _ensure_running(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="cron")
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()


_scheduler = CronScheduler()


class CronCreateTool(BaseTool):
    name = "cron_create"
    description = "Create a scheduled cron job to run a command on a schedule."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Job name"},
            "schedule": {"type": "string", "description": "Schedule: 'every 5 minutes', 'every hour', 'every day'"},
            "command": {"type": "string", "description": "Shell command to run"},
        },
        "required": ["name", "schedule", "command"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        job = _scheduler.add(inp.get("name", ""), inp.get("schedule", ""), inp.get("command", ""))
        return ToolResult(output=f"Cron job created: {job.job_id} — '{job.name}' {job.schedule}")


class CronListTool(BaseTool):
    name = "cron_list"
    description = "List all scheduled cron jobs."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, inp: dict) -> ToolResult:
        jobs = _scheduler.list_jobs()
        if not jobs:
            return ToolResult(output="No cron jobs scheduled")
        lines = ["ID       Name              Schedule              Runs  Status"]
        lines.append("-" * 65)
        for j in jobs:
            status = "enabled" if j.enabled else "disabled"
            lines.append(f"{j.job_id:<8} {j.name:<17} {j.schedule:<22} {j.run_count:<5} {status}")
        return ToolResult(output="\n".join(lines))


class CronDeleteTool(BaseTool):
    name = "cron_delete"
    description = "Delete a scheduled cron job by ID."
    input_schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string", "description": "Job ID to delete"},
        },
        "required": ["job_id"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        job_id = inp.get("job_id", "")
        if _scheduler.delete(job_id):
            return ToolResult(output=f"Job {job_id} deleted")
        return ToolResult(output=f"Job {job_id} not found", is_error=True)
