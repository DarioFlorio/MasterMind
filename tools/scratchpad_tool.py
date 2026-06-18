"""
tools/scratchpad_tool.py — MasterMind-style in-session working memory.

A per-session key/value notepad the model can write intermediate plans,
observations, and reasoning to. Contents do NOT enter conversation history,
so they don't consume context tokens.

Usage model calls:
    scratchpad write key="plan" value="1. Do X\n2. Do Y"
    scratchpad read key="plan"
    scratchpad list
    scratchpad clear key="plan"
"""
from __future__ import annotations

import time
from tools.base_tool import BaseTool, ToolResult


class ScratchpadTool(BaseTool):
    name = "scratchpad"
    description = (
        "In-session working memory notepad. Write intermediate plans, notes, "
        "and observations that you want to reference later WITHOUT adding them "
        "to the conversation. Contents are private to this session. "
        "ops: write (key + value), read (key), list, clear (key or all)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["write", "read", "list", "clear"],
                "description": "Operation: write, read, list, or clear.",
            },
            "key": {
                "type": "string",
                "description": "Entry name (e.g. 'plan', 'observations', 'todo').",
            },
            "value": {
                "type": "string",
                "description": "Content to write (only for op=write).",
            },
        },
        "required": ["op"],
    }

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def execute(self, inp: dict) -> ToolResult:
        op    = (inp.get("op") or "list").lower().strip()
        key   = (inp.get("key") or "").strip()
        value = inp.get("value") or ""

        if op == "write":
            if not key:
                return ToolResult("'key' is required for write.", is_error=True)
            self._store[key] = {"value": value, "updated": time.strftime("%H:%M:%S")}
            return ToolResult(f"Scratchpad '{key}' saved ({len(value)} chars).")

        if op == "read":
            if not key:
                return ToolResult("'key' is required for read.", is_error=True)
            entry = self._store.get(key)
            if entry is None:
                return ToolResult(f"No scratchpad entry '{key}'. Use list to see keys.")
            return ToolResult(f"[{key}] (saved {entry['updated']})\n{entry['value']}")

        if op == "list":
            if not self._store:
                return ToolResult("Scratchpad is empty.")
            lines = ["Scratchpad entries:"]
            for k, v in self._store.items():
                preview = v["value"][:60].replace("\n", " ")
                lines.append(f"  {k} ({v['updated']}): {preview}…")
            return ToolResult("\n".join(lines))

        if op == "clear":
            if key:
                removed = self._store.pop(key, None)
                return ToolResult(
                    f"Cleared '{key}'." if removed else f"No entry '{key}'."
                )
            self._store.clear()
            return ToolResult("Scratchpad cleared.")

        return ToolResult(f"Unknown op: {op}", is_error=True)
