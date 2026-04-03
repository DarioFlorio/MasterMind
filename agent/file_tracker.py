"""
agent/file_tracker.py — Claude Code-style file change tracker.

Tracks which files were read and written during a session. Plugs in as a
thin wrapper around tool execution — no changes to existing tools needed.

Usage in query_engine.py:
    tracker = FileTracker()
    # after each tool call:
    tracker.record(tool_name, inp, result)
    # at any point:
    print(tracker.summary())
"""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FileOp:
    path: str
    op: str          # "read" | "write" | "edit" | "delete"
    ts: float = field(default_factory=time.time)
    ok: bool  = True


class FileTracker:
    """Records file operations performed by tools during a session."""

    _READ_TOOLS  = {"read_file", "grep", "glob", "list_dir"}
    _WRITE_TOOLS = {"write_file", "edit_file"}
    _DELETE_TOOLS: set[str] = set()

    def __init__(self) -> None:
        self._ops: list[FileOp] = []
        self._reads:  set[str]  = set()
        self._writes: set[str]  = set()

    def record(self, tool_name: str, inp: dict, is_error: bool = False) -> None:
        """Call this after every tool execution to record file operations."""
        op = self._classify(tool_name, inp)
        if op is None:
            return
        path_str, op_type = op
        self._ops.append(FileOp(path=path_str, op=op_type, ok=not is_error))
        if op_type == "read":
            self._reads.add(path_str)
        elif op_type in ("write", "edit"):
            self._writes.add(path_str)

    def _classify(self, tool_name: str, inp: dict) -> tuple[str, str] | None:
        if tool_name in self._READ_TOOLS:
            path = (inp.get("path") or inp.get("file_path") or
                    inp.get("pattern") or inp.get("directory") or "")
            return (str(path), "read") if path else None
        if tool_name in self._WRITE_TOOLS:
            path = inp.get("path") or inp.get("file_path") or ""
            return (str(path), tool_name.replace("_file", "")) if path else None
        # Bash tool: detect file operations from command string
        if tool_name == "bash":
            cmd = inp.get("command", "")
            return self._classify_bash(cmd)
        return None

    def _classify_bash(self, cmd: str) -> tuple[str, str] | None:
        import re
        # Write patterns: redirection, cp, mv, touch, mkdir
        m = re.search(r'(?:>>?|tee)\s+"?([^\s"]+)"?', cmd)
        if m:
            return (m.group(1), "write")
        m = re.search(r'\b(?:cp|mv|touch|mkdir)\b.*\s+"?([^\s"]+)"?\s*$', cmd)
        if m:
            return (m.group(1), "write")
        # Read patterns: cat, head, tail, less, more
        m = re.search(r'\b(?:cat|head|tail|less|more|wc|diff)\s+"?([^\s"]+)"?', cmd)
        if m:
            return (m.group(1), "read")
        return None

    # ── Reporting ─────────────────────────────────────────────────────────────

    @property
    def reads(self) -> set[str]:
        return self._reads

    @property
    def writes(self) -> set[str]:
        return self._writes

    def modified_files(self) -> list[str]:
        return sorted(self._writes)

    def summary(self) -> str:
        if not self._ops:
            return "No file operations recorded."
        lines = []
        if self._writes:
            lines.append(f"Modified ({len(self._writes)}):")
            for p in sorted(self._writes):
                lines.append(f"  ✎ {p}")
        if self._reads - self._writes:
            read_only = self._reads - self._writes
            lines.append(f"Read ({len(read_only)}):")
            for p in sorted(read_only):
                lines.append(f"  ◌ {p}")
        return "\n".join(lines)

    def reset(self) -> None:
        self._ops.clear()
        self._reads.clear()
        self._writes.clear()
