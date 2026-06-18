"""
tools/task_output_tool.py — TaskOutputTool

Lets background or long-running tasks emit mid-task progress output
that is visible to the coordinator/parent agent without waiting for
the task to complete.

Ported from src TaskOutputTool pattern.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Callable, Optional

from tools.base_tool import BaseTool, ToolResult

log = logging.getLogger("tools.task_output")


# Global output buffer (task_id → list of output events)
_output_buffers: dict[str, list[dict]] = {}
_output_lock = threading.Lock()
_output_listeners: list[Callable[[str, dict], None]] = []


def register_output_listener(fn: Callable[[str, dict], None]) -> None:
    """Register a callback for task output events: fn(task_id, event)."""
    with _output_lock:
        _output_listeners.append(fn)


def get_task_outputs(task_id: str) -> list[dict]:
    """Return all output events for a given task_id."""
    with _output_lock:
        return list(_output_buffers.get(task_id, []))


def clear_task_outputs(task_id: str) -> None:
    """Clear buffered outputs for a task (e.g., after reading them)."""
    with _output_lock:
        _output_buffers.pop(task_id, None)


class TaskOutputTool(BaseTool):
    """
    Emit progress output from a background task.

    Background agents use this to send intermediate status updates,
    partial results, or progress messages to the coordinator without
    finishing the task. The output is buffered and can be polled
    via TaskGetTool or retrieved by the coordinator.
    """

    name = "task_output"
    description = (
        "Emit a progress message or partial result from a background task. "
        "Use this inside long-running agents to report intermediate status to "
        "the coordinator/parent without ending the task. "
        "Input: {\"task_id\": \"...\", \"message\": \"...\", \"level\": \"info|progress|result\"}"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": "The ID of the current background task (from TaskCreateTool).",
            },
            "message": {
                "type": "string",
                "description": "The progress message or partial result to emit.",
            },
            "level": {
                "type": "string",
                "enum": ["info", "progress", "result", "warning", "error"],
                "description": "Severity/type of the output. Defaults to 'progress'.",
            },
            "data": {
                "type": "object",
                "description": "Optional structured data payload accompanying the message.",
            },
        },
        "required": ["task_id", "message"],
    }

    def execute(self, inp: dict) -> ToolResult:
        task_id = (inp.get("task_id") or "").strip()
        message = (inp.get("message") or "").strip()
        level = (inp.get("level") or "progress").strip().lower()
        data = inp.get("data")

        if not task_id:
            return ToolResult("Error: 'task_id' is required.", is_error=True)
        if not message:
            return ToolResult("Error: 'message' is required.", is_error=True)

        valid_levels = {"info", "progress", "result", "warning", "error"}
        if level not in valid_levels:
            level = "progress"

        event = {
            "task_id": task_id,
            "message": message,
            "level": level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if data is not None:
            event["data"] = data

        # Buffer the event
        with _output_lock:
            if task_id not in _output_buffers:
                _output_buffers[task_id] = []
            _output_buffers[task_id].append(event)
            listeners = list(_output_listeners)

        # Notify listeners
        for fn in listeners:
            try:
                fn(task_id, event)
            except Exception as exc:
                log.debug("TaskOutputTool: listener error: %s", exc)

        log.debug("TaskOutputTool: task=%s level=%s: %s", task_id, level, message[:80])
        return ToolResult(f"[{level.upper()}] {message}")
