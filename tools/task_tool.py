# -*- coding: utf-8 -*-
"""
tools/task_tool.py — Task lifecycle management (Create/Get/List/Update/Stop).

MasterMind built-in's TaskCreateTool, TaskGetTool, TaskListTool,
TaskUpdateTool, TaskStopTool. Backed by a SQLite store.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult

_DB_PATH = Path.home() / ".mastermind" / "tasks.db"
_lock = threading.Lock()


def _db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            status TEXT DEFAULT 'pending',
            priority TEXT DEFAULT 'normal',
            tags TEXT DEFAULT '[]',
            created_at REAL,
            updated_at REAL,
            completed_at REAL,
            metadata TEXT DEFAULT '{}'
        )
    """)
    conn.commit()
    return conn


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    status: str = "pending"   # pending | in_progress | done | cancelled | blocked
    priority: str = "normal"  # low | normal | high | urgent
    tags: list[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: float | None = None
    metadata: dict = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if not self.created_at:
            self.created_at = time.time()
        self.updated_at = time.time()


def _row_to_task(row) -> Task:
    return Task(
        id=row["id"], title=row["title"], description=row["description"],
        status=row["status"], priority=row["priority"],
        tags=json.loads(row["tags"] or "[]"),
        created_at=row["created_at"], updated_at=row["updated_at"],
        completed_at=row["completed_at"],
        metadata=json.loads(row["metadata"] or "{}"),
    )


class TaskCreateTool(BaseTool):
    name = "task_create"
    description = "Create a new task with title, description, priority, and tags."
    input_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Task title"},
            "description": {"type": "string", "description": "Task details"},
            "priority": {"type": "string", "description": "low | normal | high | urgent"},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags"},
        },
        "required": ["title"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        task = Task(
            id=str(uuid.uuid4())[:8],
            title=inp.get("title", ""),
            description=inp.get("description", ""),
            priority=inp.get("priority", "normal"),
            tags=inp.get("tags", []),
        )
        with _lock:
            conn = _db()
            conn.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?,?,?,?,?)",
                (task.id, task.title, task.description, task.status,
                 task.priority, json.dumps(task.tags),
                 task.created_at, task.updated_at, task.completed_at,
                 json.dumps(task.metadata))
            )
            conn.commit()
            conn.close()
        return ToolResult(output=f"Task created: {task.id} — {task.title}")


class TaskGetTool(BaseTool):
    name = "task_get"
    description = "Get details of a specific task by ID."
    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Task ID"}},
        "required": ["id"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        task_id = inp.get("id", "")
        with _lock:
            conn = _db()
            row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
            conn.close()
        if not row:
            return ToolResult(output=f"Task {task_id!r} not found", is_error=True)
        task = _row_to_task(row)
        return ToolResult(output=json.dumps(asdict(task), indent=2))


class TaskListTool(BaseTool):
    name = "task_list"
    description = "List tasks, optionally filtered by status or priority."
    input_schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "description": "Filter by status"},
            "priority": {"type": "string", "description": "Filter by priority"},
            "tag": {"type": "string", "description": "Filter by tag"},
            "limit": {"type": "integer", "description": "Max results (default: 50)"},
        },
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        status = inp.get("status", "")
        priority = inp.get("priority", "")
        tag = inp.get("tag", "")
        limit = int(inp.get("limit", 50))

        query = "SELECT * FROM tasks WHERE 1=1"
        params = []
        if status:
            query += " AND status=?"; params.append(status)
        if priority:
            query += " AND priority=?"; params.append(priority)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with _lock:
            conn = _db()
            rows = conn.execute(query, params).fetchall()
            conn.close()

        tasks = [_row_to_task(r) for r in rows]
        if tag:
            tasks = [t for t in tasks if tag in t.tags]

        if not tasks:
            return ToolResult(output="No tasks found")

        lines = [f"{'ID':<10} {'Status':<12} {'Priority':<8} {'Title'}"]
        lines.append("-" * 60)
        for t in tasks:
            lines.append(f"{t.id:<10} {t.status:<12} {t.priority:<8} {t.title[:35]}")
        return ToolResult(output="\n".join(lines))


class TaskUpdateTool(BaseTool):
    name = "task_update"
    description = "Update a task's status, priority, or description."
    input_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Task ID"},
            "status": {"type": "string", "description": "New status"},
            "priority": {"type": "string", "description": "New priority"},
            "description": {"type": "string", "description": "New description"},
            "title": {"type": "string", "description": "New title"},
        },
        "required": ["id"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        # Accept both 'id' and 'task_id' since EVE sometimes uses task_id
        task_id = inp.get("id", inp.get("task_id", "")).strip()
        updates = []
        params = []
        for field in ("status", "priority", "description", "title"):
            if field in inp:
                updates.append(f"{field}=?")
                params.append(inp[field])
        if not updates:
            return ToolResult(output="No fields to update", is_error=True)

        now = time.time()
        updates.append("updated_at=?"); params.append(now)
        if inp.get("status") == "done":
            updates.append("completed_at=?"); params.append(now)

        params.append(task_id)
        with _lock:
            conn = _db()
            r = conn.execute(f"UPDATE tasks SET {', '.join(updates)} WHERE id=?", params)
            conn.commit(); conn.close()
        if r.rowcount == 0:
            return ToolResult(output=f"Task {task_id!r} not found — no changes made")
        return ToolResult(output=f"Task {task_id} updated")

class TaskStopTool(BaseTool):
    name = "task_stop"
    description = "Cancel a task (sets status to 'cancelled')."
    input_schema = {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "Task ID to cancel"}},
        "required": ["id"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        task_id = inp.get("id", "")
        with _lock:
            conn = _db()
            r = conn.execute(
                "UPDATE tasks SET status='cancelled', updated_at=? WHERE id=?",
                (time.time(), task_id)
            )
            conn.commit(); conn.close()
        if r.rowcount == 0:
            return ToolResult(output=f"Task {task_id!r} not found", is_error=True)
        return ToolResult(output=f"Task {task_id} cancelled")
