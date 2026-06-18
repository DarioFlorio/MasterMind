# -*- coding: utf-8 -*-
"""telemetry/logger.py — Structured telemetry event logger."""
from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("telemetry")
TELEMETRY_ENABLED = os.environ.get("MM_TELEMETRY", "1") == "1"
TELEMETRY_PATH = Path(os.environ.get("MM_TELEMETRY_PATH",
                      str(Path.home() / ".mastermind" / "telemetry.jsonl")))


@dataclass
class TelemetryEvent:
    event: str
    ts: float = field(default_factory=time.time)
    session_id: str = ""
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class TelemetryLogger:
    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self._events: list[TelemetryEvent] = []
        self._file = None
        if TELEMETRY_ENABLED:
            TELEMETRY_PATH.parent.mkdir(parents=True, exist_ok=True)
            try:
                self._file = open(TELEMETRY_PATH, "a", encoding="utf-8")
            except Exception as e:
                log.warning("Cannot open telemetry file: %s", e)

    def emit(self, event: str, **data: Any) -> None:
        if not TELEMETRY_ENABLED:
            return
        ev = TelemetryEvent(event=event, session_id=self.session_id, data=data)
        self._events.append(ev)
        if self._file:
            try:
                self._file.write(ev.to_json() + "\n")
                self._file.flush()
            except Exception:
                pass

    # Convenience telemetry methods
    def session_start(self, **kw) -> None: self.emit("session_start", **kw)
    def session_end(self, **kw) -> None: self.emit("session_end", **kw)
    def tool_use(self, tool: str, **kw) -> None: self.emit("tool_use", tool=tool, **kw)
    def tool_error(self, tool: str, error: str, **kw) -> None:
        self.emit("tool_error", tool=tool, error=error, **kw)
    def skill_invoke(self, skill: str, **kw) -> None: self.emit("skill_invoke", skill=skill, **kw)
    def mcp_call(self, server: str, tool: str, **kw) -> None:
        self.emit("mcp_call", server=server, tool=tool, **kw)
    def swarm_event(self, kind: str, **kw) -> None: self.emit(f"swarm_{kind}", **kw)

    def summary(self) -> dict:
        from collections import Counter
        counts = Counter(e.event for e in self._events)
        return {"total": len(self._events), "by_event": dict(counts)}

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


# Session-level singleton (replaced in main.py with proper session_id)
telemetry = TelemetryLogger()
