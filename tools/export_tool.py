"""
tools/export_tool.py — MasterMind session export.

Exports the current conversation to a readable Markdown file.
Also available as /export slash command in the REPL.
"""
from __future__ import annotations

import time
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult


def export_session(session, path: str | None = None) -> str:
    """
    Export session messages to a Markdown file.
    Returns the path of the written file.
    """
    ts   = time.strftime("%Y%m%d_%H%M%S")
    dest = Path(path) if path else Path(f"session_{ts}.md")

    lines = [
        f"# Session Export — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
    ]
    for msg in session._messages:
        role = msg.role.upper()
        if msg.meta.get("tool_result"):
            lines.append(f"### ⚙ Tool result")
            lines.append(f"```\n{msg.content[:2000]}\n```")
        elif role == "USER":
            lines.append(f"### You")
            lines.append(msg.content)
        elif role == "ASSISTANT":
            lines.append(f"### Assistant")
            lines.append(msg.content)
        lines.append("")

    if session._summary:
        lines.insert(2, f"> **Compressed context summary:**\n> {session._summary[:400]}\n")

    dest.write_text("\n".join(lines), encoding="utf-8")
    return str(dest)


class ExportTool(BaseTool):
    name = "export_session"
    description = (
        "Export the current conversation to a Markdown file. "
        "Useful for saving research sessions, code reviews, or planning discussions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Output file path (default: session_TIMESTAMP.md).",
            },
        },
        "required": [],
    }

    def __init__(self, session=None) -> None:
        self._session = session

    def set_session(self, session) -> None:
        self._session = session

    def execute(self, inp: dict) -> ToolResult:
        if self._session is None:
            return ToolResult("Session not attached to ExportTool.", is_error=True)
        path = inp.get("path")
        try:
            dest = export_session(self._session, path)
            return ToolResult(f"Session exported to: {dest}")
        except Exception as exc:
            return ToolResult(f"Export failed: {exc}", is_error=True)
