from __future__ import annotations
import json
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR

_TODO_FILE = Path.home() / ".mastermind_todos.json"


def _load() -> list[dict]:
    try:
        return json.loads(_TODO_FILE.read_text()) if _TODO_FILE.exists() else []
    except Exception:
        return []


def _save(todos: list[dict]) -> None:
    _TODO_FILE.write_text(json.dumps(todos, indent=2))


class TodoWriteTool(BaseTool):
    name = "todo_write"
    description = (
        "Create or update the todo list for this session. "
        "Replaces entire list. Use to track task progress."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of todo items",
                "items": {
                    "type": "object",
                    "properties": {
                        "id":      {"type": "string"},
                        "content": {"type": "string"},
                        "status":  {"type": "string", "enum": ["pending", "in_progress", "completed"]},
                        "priority":{"type": "string", "enum": ["high", "medium", "low"]},
                    },
                },
            }
        },
        "required": ["todos"],
    }

    def execute(self, inp: dict) -> ToolResult:
        todos = inp.get("todos", [])
        _save(todos)
        counts = {}
        for t in todos:
            s = t.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in counts.items())
        return ToolResult(f"Todo list updated: {len(todos)} items ({summary})")


class TodoReadTool(BaseTool):
    name = "todo_read"
    description = "Read the current todo list."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, inp: dict) -> ToolResult:
        todos = _load()
        if not todos:
            return ToolResult("No todos yet.")
        icons = {"completed": "✓", "in_progress": "→", "pending": "○"}
        pri   = {"high": "!", "medium": "~", "low": " "}
        lines = []
        for t in todos:
            icon = icons.get(t.get("status", "pending"), "○")
            p    = pri.get(t.get("priority", "medium"), " ")
            lines.append(f"[{icon}][{p}] {t.get('id','?')}: {t.get('content','')}")
        return ToolResult("\n".join(lines))
