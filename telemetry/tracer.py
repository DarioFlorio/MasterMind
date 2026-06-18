# -*- coding: utf-8 -*-
"""telemetry/tracer.py — Lightweight performance tracer (Perfetto-compatible)."""
from __future__ import annotations
import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator


@dataclass
class Span:
    name: str
    start: float = field(default_factory=time.perf_counter)
    end: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end - self.start) * 1000

    def finish(self) -> "Span":
        self.end = time.perf_counter()
        return self


class Tracer:
    """
    Minimal performance tracer.
    Usage:
        with tracer.span("tool_use") as span:
            span.meta["tool"] = "bash"
            do_work()
        tracer.report()
    """

    def __init__(self):
        self._spans: list[Span] = []

    @contextlib.contextmanager
    def span(self, name: str, **meta) -> Generator[Span, None, None]:
        s = Span(name=name, meta=meta)
        try:
            yield s
        finally:
            s.finish()
            self._spans.append(s)

    def record(self, name: str, duration_ms: float, **meta) -> None:
        s = Span(name=name, end=0.0, meta=meta)
        s.start = 0.0
        s.end = duration_ms / 1000
        self._spans.append(s)

    def report(self) -> str:
        if not self._spans:
            return "(no spans recorded)"
        lines = [f"{'Span':<40} {'Duration':>10}"]
        lines.append("-" * 52)
        for s in sorted(self._spans, key=lambda x: x.duration_ms, reverse=True)[:20]:
            lines.append(f"{s.name:<40} {s.duration_ms:>9.1f}ms")
        total = sum(s.duration_ms for s in self._spans)
        lines.append(f"\nTotal tracked: {total:.1f}ms across {len(self._spans)} spans")
        return "\n".join(lines)

    def to_perfetto(self, path: Path | None = None) -> str:
        """Export as Perfetto/Chrome tracing format JSON."""
        events = []
        for s in self._spans:
            events.append({
                "name": s.name,
                "ph": "X",
                "ts": s.start * 1_000_000,
                "dur": (s.end - s.start) * 1_000_000,
                "pid": 0, "tid": 0,
                "args": s.meta,
            })
        out = json.dumps({"traceEvents": events}, indent=2)
        if path:
            path.write_text(out)
        return out

    def clear(self) -> None:
        self._spans.clear()


tracer = Tracer()
