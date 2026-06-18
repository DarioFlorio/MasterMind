# -*- coding: utf-8 -*-
"""tools/sleep_tool.py — Pause execution for N seconds."""
from __future__ import annotations
import time
from tools.base_tool import BaseTool, ToolResult


class SleepTool(BaseTool):
    name = "sleep"
    description = "Pause execution for a specified number of seconds. Useful in loops or when waiting for external processes."
    input_schema = {
        "type": "object",
        "properties": {
            "seconds": {"type": "number", "description": "Seconds to sleep (max 300)"},
            "reason": {"type": "string", "description": "Optional reason for sleeping"},
        },
        "required": ["seconds"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        seconds = float(inp.get("seconds", 1))
        seconds = min(max(0, seconds), 300)
        reason = inp.get("reason", "")
        msg = f"Sleeping {seconds}s" + (f" — {reason}" if reason else "")
        time.sleep(seconds)
        return ToolResult(output=f"{msg} [done]")
