# ─────────────────────────────────────────────────────────────────────────────
# USAGE REFERENCE — what the agent can now do
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
 
"""
AGENT USAGE EXAMPLES
====================
 
1. START OF SESSION — retrieve project context without clogging context window:
   pm(action="brief")                               # list all projects
   pm(action="brief", project_id="abc123")          # full project brief
   pm(action="checkpoint", project_id="abc123")     # last stable checkpoint
 
2. SEARCH — like web_search but for your own project history:
   pm(action="search", project_id="abc123", query="MCP authentication")
   pm(action="search", project_id="abc123", query="broken sqlite")
 
3. LOG ATTEMPTS — so nothing is lost:
   pm(action="new_log", project_id="abc123", log_type="attempt",
      title="Tried requests-based MCP client",
      content="Used requests library to POST to MCP server",
      outcome="FAILED — MCP requires SSE, not plain HTTP POST")
 
   pm(action="new_log", project_id="abc123", log_type="solution",
      title="MCP via httpx with SSE stream",
      content="Use httpx.AsyncClient with stream=True for MCP SSE endpoint",
      outcome="WORKS — connected successfully")
 
4. SAVE WORKING CODE:
   pm(action="new_snippet", project_id="abc123",
      title="MCP SSE connection", language="python", snippet_status="working",
      code="async with httpx.AsyncClient() as c:\\n    async for chunk in c.stream('POST', url):\\n        ...",
      tags=["mcp", "sse", "working"])
 
5. MARK A CHECKPOINT:
   pm(action="checkpoint", project_id="abc123",
      title="MCP tool registration working",
      content="All 5 tools register correctly. SSE streaming stable.",
      outcome="Stable — can continue from here next session")
 
6. KANBAN TASK MANAGEMENT:
   pm(action="new_task", project_id="abc123",
      title="Implement MCP dispatcher", status="in-progress", priority="high",
      checklist=["Parse SSE stream", "Route to correct tool", "Return result"])
 
   pm(action="update_task", task_id="t1a2b3c4", status="done")
   pm(action="update_task", task_id="t1a2b3c4", status="blocked",
      description="Blocked: MCP server returns 403 on localhost")
 
7. SPRINTS:
   pm(action="new_sprint", project_id="abc123",
      name="Sprint 1", goal="Get MCP tools working end-to-end",
      start_date="2026-04-03", end_date="2026-04-17", status="active")
 
8. PROJECT DOCS — persistent architecture/decision notes:
   pm(action="set_docs", project_id="abc123",
      docs="## Architecture\\nUsing llama-cpp direct mode.\\n\\n## Known Issues\\n- MCP timeout on large responses")
 
   pm(action="get_docs", project_id="abc123")
 
 
LIFECYCLE PATTERN (recommended agent workflow)
=============================================
 
Session start:
  1. pm(action="brief")                    → which project to work on?
  2. pm(action="brief", project_id=PID)    → what's the state?
  3. pm(action="checkpoint", project_id=PID) → where did we leave off?
 
During work:
  4. pm(action="new_log", log_type="session", title="Starting session", ...)
  5. pm(action="new_log", log_type="attempt", ...) for each thing tried
  6. pm(action="new_log", log_type="failure", ...) if it fails
  7. pm(action="new_log", log_type="success", ...) if it works
  8. pm(action="new_snippet", snippet_status="working", ...) for good code
  9. pm(action="update_task", task_id=..., status="done") when tasks complete
 
Session end:
  10. pm(action="checkpoint", ...) to mark where you are
  11. pm(action="update_task", ...) to update task board state
 
Between projects:
  12. pm(action="search", query="...") to find relevant past solutions
  13. pm(action="list_snippets", snippet_status="working") to reuse code
"""

"""
tools/pm_tool.py — Persistent Project Management for PyClaudeCode

SQLite-backed PM suite integrated as a native agent tool.
Supports full Agile/Scrum/Kanban lifecycle with agent retrieval.

DB: memdir/pm_suite.db (alongside eve_journal.db)

Actions:
  Retrieval (read, low-context):
    brief         — compact project summary for agent context
    search        — search across tasks, logs, snippets
    checkpoint    — get or set a stable checkpoint

  Projects:
    new_project   — create a project
    list_projects — list all projects
    update_project— update project fields

  Tasks / Kanban:
    new_task      — add a task
    update_task   — change status, priority, fields
    list_tasks    — list tasks (filter by status)

  Logs / Session tracking:
    new_log       — record attempt/success/failure/solution/note
    list_logs     — recent log entries

  Sprints:
    new_sprint    — create a sprint
    update_sprint — change sprint status
    list_sprints  — list sprints

  Snippets (code library):
    new_snippet   — save a code snippet
    list_snippets — list snippets (filter by status)
    update_snippet— mark snippet working/broken/etc.

  Docs:
    get_docs      — read project documentation
    set_docs      — write/append project documentation
"""



import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool, ToolResult

# ── DB Location ───────────────────────────────────────────────────────────────
_DB_DIR  = Path(__file__).parent.parent / "memdir"
_DB_PATH = _DB_DIR / "pm_suite.db"

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    status      TEXT DEFAULT 'active',
    tags        TEXT DEFAULT '[]',
    docs        TEXT DEFAULT '',
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    description TEXT DEFAULT '',
    status      TEXT DEFAULT 'backlog',
    priority    TEXT DEFAULT 'medium',
    sprint_id   TEXT,
    start_date  TEXT,
    end_date    TEXT,
    tags        TEXT DEFAULT '[]',
    checklist   TEXT DEFAULT '[]',
    created     TEXT NOT NULL,
    updated     TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS logs (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    task_id     TEXT,
    type        TEXT DEFAULT 'note',
    title       TEXT DEFAULT '',
    content     TEXT DEFAULT '',
    outcome     TEXT DEFAULT '',
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS snippets (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    title       TEXT NOT NULL,
    language    TEXT DEFAULT 'python',
    code        TEXT DEFAULT '',
    status      TEXT DEFAULT 'partial',
    tags        TEXT DEFAULT '[]',
    created     TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE TABLE IF NOT EXISTS sprints (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    goal        TEXT DEFAULT '',
    start_date  TEXT,
    end_date    TEXT,
    status      TEXT DEFAULT 'planning',
    created     TEXT NOT NULL,
    FOREIGN KEY (project_id) REFERENCES projects(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_project   ON tasks(project_id, status);
CREATE INDEX IF NOT EXISTS idx_logs_project    ON logs(project_id, type, timestamp);
CREATE INDEX IF NOT EXISTS idx_snippets_project ON snippets(project_id, status);
"""

VALID_TASK_STATUSES  = {"backlog","todo","in-progress","blocked","done","failed"}
VALID_LOG_TYPES      = {"session","attempt","success","failure","solution","checkpoint","note"}
VALID_SNIPPET_STATUS = {"working","partial","broken","deprecated"}
VALID_PRIORITIES     = {"low","medium","high","critical"}
VALID_SPRINT_STATUS  = {"planning","active","completed"}
VALID_PROJECT_STATUS = {"active","paused","completed","archived"}

LOG_ICONS = {
    "session": "🚀", "attempt": "⚡", "success": "✅",
    "failure": "❌", "solution": "💡", "checkpoint": "🏁", "note": "📝",
}

# ── DB helpers ────────────────────────────────────────────────────────────────

def _init_db() -> None:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    with _conn() as c:
        c.executescript(_SCHEMA)

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def _uid() -> str:
    return str(uuid.uuid4())[:8]

def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _fmt(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        return datetime.fromisoformat(iso.replace("Z","")).strftime("%d %b %Y")
    except Exception:
        return iso

def _row(r) -> dict:
    d = dict(r)
    for k in ("tags", "checklist"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except Exception:
                d[k] = []
    return d

def _trunc(s: str | None, n: int = 120) -> str:
    if not s:
        return ""
    return s[:n] + ("…" if len(s) > n else "")

_init_db()


# ── Core functions ────────────────────────────────────────────────────────────

class _DB:
    """Thin data-access layer used by the tool."""

    # ── Projects ─────────────────────────────────────────────────────────────

    @staticmethod
    def new_project(name: str, description: str = "", status: str = "active",
                    tags: list | None = None) -> dict:
        p = {
            "id": _uid(), "name": name, "description": description,
            "status": status, "tags": json.dumps(tags or []),
            "docs": "", "created": _ts(), "updated": _ts(),
        }
        with _conn() as c:
            c.execute(
                "INSERT INTO projects VALUES"
                "(:id,:name,:description,:status,:tags,:docs,:created,:updated)", p
            )
        p["tags"] = tags or []
        return p

    @staticmethod
    def list_projects(status: str | None = None) -> list[dict]:
        with _conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM projects WHERE status=? ORDER BY updated DESC", (status,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM projects ORDER BY updated DESC"
                ).fetchall()
        return [_row(r) for r in rows]

    @staticmethod
    def get_project(pid: str) -> dict | None:
        with _conn() as c:
            r = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return _row(r) if r else None

    @staticmethod
    def update_project(pid: str, **kwargs) -> str:
        allowed = {"name","description","status","tags","docs"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed:
                updates[k] = json.dumps(v) if k == "tags" and isinstance(v, list) else v
        if not updates:
            return "Nothing to update."
        updates["updated"] = _ts()
        clause = ", ".join(f"{k}=?" for k in updates)
        with _conn() as c:
            c.execute(f"UPDATE projects SET {clause} WHERE id=?", (*updates.values(), pid))
        return f"Project {pid} updated."

    # ── Tasks ─────────────────────────────────────────────────────────────────

    @staticmethod
    def new_task(project_id: str, title: str, description: str = "",
                 status: str = "backlog", priority: str = "medium",
                 sprint_id: str | None = None, start_date: str | None = None,
                 end_date: str | None = None, tags: list | None = None,
                 checklist: list | None = None) -> dict:
        checklist_data = [{"id": _uid(), "text": c, "done": False} for c in (checklist or [])]
        t = {
            "id": _uid(), "project_id": project_id, "title": title,
            "description": description, "status": status, "priority": priority,
            "sprint_id": sprint_id, "start_date": start_date, "end_date": end_date,
            "tags": json.dumps(tags or []),
            "checklist": json.dumps(checklist_data),
            "created": _ts(), "updated": _ts(),
        }
        with _conn() as c:
            c.execute(
                "INSERT INTO tasks VALUES"
                "(:id,:project_id,:title,:description,:status,:priority,"
                ":sprint_id,:start_date,:end_date,:tags,:checklist,:created,:updated)", t
            )
        return _row(dict(t))

    @staticmethod
    def list_tasks(project_id: str, status: str | None = None) -> list[dict]:
        with _conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM tasks WHERE project_id=? AND status=? ORDER BY created",
                    (project_id, status)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM tasks WHERE project_id=? ORDER BY created",
                    (project_id,)
                ).fetchall()
        return [_row(r) for r in rows]

    @staticmethod
    def update_task(task_id: str, **kwargs) -> str:
        allowed = {"title","description","status","priority","sprint_id",
                   "start_date","end_date","tags","checklist"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed:
                updates[k] = json.dumps(v) if k in ("tags","checklist") and isinstance(v, list) else v
        if not updates:
            return "Nothing to update."
        updates["updated"] = _ts()
        clause = ", ".join(f"{k}=?" for k in updates)
        with _conn() as c:
            c.execute(f"UPDATE tasks SET {clause} WHERE id=?", (*updates.values(), task_id))
        return f"Task {task_id} updated."

    # ── Logs ──────────────────────────────────────────────────────────────────

    @staticmethod
    def new_log(project_id: str, log_type: str = "note", title: str = "",
                content: str = "", outcome: str = "",
                task_id: str | None = None) -> dict:
        if log_type not in VALID_LOG_TYPES:
            log_type = "note"
        entry = {
            "id": _uid(), "project_id": project_id, "task_id": task_id,
            "type": log_type, "title": title, "content": content,
            "outcome": outcome, "timestamp": _ts(),
        }
        with _conn() as c:
            c.execute(
                "INSERT INTO logs VALUES"
                "(:id,:project_id,:task_id,:type,:title,:content,:outcome,:timestamp)",
                entry
            )
        return entry

    @staticmethod
    def list_logs(project_id: str, n: int = 20,
                  log_type: str | None = None) -> list[dict]:
        with _conn() as c:
            if log_type:
                rows = c.execute(
                    "SELECT * FROM logs WHERE project_id=? AND type=? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (project_id, log_type, n)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM logs WHERE project_id=? ORDER BY timestamp DESC LIMIT ?",
                    (project_id, n)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Sprints ───────────────────────────────────────────────────────────────

    @staticmethod
    def new_sprint(project_id: str, name: str, goal: str = "",
                   start_date: str | None = None, end_date: str | None = None,
                   status: str = "planning") -> dict:
        s = {
            "id": _uid(), "project_id": project_id, "name": name, "goal": goal,
            "start_date": start_date, "end_date": end_date,
            "status": status, "created": _ts(),
        }
        with _conn() as c:
            c.execute(
                "INSERT INTO sprints VALUES"
                "(:id,:project_id,:name,:goal,:start_date,:end_date,:status,:created)", s
            )
        return s

    @staticmethod
    def list_sprints(project_id: str) -> list[dict]:
        with _conn() as c:
            rows = c.execute(
                "SELECT * FROM sprints WHERE project_id=? ORDER BY start_date, created",
                (project_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    @staticmethod
    def update_sprint(sprint_id: str, **kwargs) -> str:
        allowed = {"name","goal","start_date","end_date","status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return "Nothing to update."
        clause = ", ".join(f"{k}=?" for k in updates)
        with _conn() as c:
            c.execute(f"UPDATE sprints SET {clause} WHERE id=?", (*updates.values(), sprint_id))
        return f"Sprint {sprint_id} updated."

    # ── Snippets ──────────────────────────────────────────────────────────────

    @staticmethod
    def new_snippet(project_id: str, title: str, language: str = "python",
                    code: str = "", status: str = "partial",
                    tags: list | None = None) -> dict:
        s = {
            "id": _uid(), "project_id": project_id, "title": title,
            "language": language, "code": code, "status": status,
            "tags": json.dumps(tags or []), "created": _ts(),
        }
        with _conn() as c:
            c.execute(
                "INSERT INTO snippets VALUES"
                "(:id,:project_id,:title,:language,:code,:status,:tags,:created)", s
            )
        s["tags"] = tags or []
        return s

    @staticmethod
    def list_snippets(project_id: str, status: str | None = None) -> list[dict]:
        with _conn() as c:
            if status:
                rows = c.execute(
                    "SELECT * FROM snippets WHERE project_id=? AND status=? ORDER BY created DESC",
                    (project_id, status)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM snippets WHERE project_id=? ORDER BY created DESC",
                    (project_id,)
                ).fetchall()
        return [_row(r) for r in rows]

    @staticmethod
    def update_snippet(snippet_id: str, **kwargs) -> str:
        allowed = {"title","language","code","status","tags"}
        updates: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in allowed:
                updates[k] = json.dumps(v) if k == "tags" and isinstance(v, list) else v
        if not updates:
            return "Nothing to update."
        clause = ", ".join(f"{k}=?" for k in updates)
        with _conn() as c:
            c.execute(f"UPDATE snippets SET {clause} WHERE id=?", (*updates.values(), snippet_id))
        return f"Snippet {snippet_id} updated."

    # ── Search ────────────────────────────────────────────────────────────────

    @staticmethod
    def search(project_id: str, query: str) -> dict:
        q = f"%{query}%"
        with _conn() as c:
            tasks = [_row(r) for r in c.execute(
                "SELECT * FROM tasks WHERE project_id=? AND "
                "(title LIKE ? OR description LIKE ?) ORDER BY updated DESC LIMIT 10",
                (project_id, q, q)
            ).fetchall()]
            logs = [dict(r) for r in c.execute(
                "SELECT * FROM logs WHERE project_id=? AND "
                "(title LIKE ? OR content LIKE ? OR outcome LIKE ?) "
                "ORDER BY timestamp DESC LIMIT 10",
                (project_id, q, q, q)
            ).fetchall()]
            snippets = [_row(r) for r in c.execute(
                "SELECT * FROM snippets WHERE project_id=? AND "
                "(title LIKE ? OR code LIKE ? OR tags LIKE ?) ORDER BY created DESC LIMIT 5",
                (project_id, q, q, q)
            ).fetchall()]
        return {"tasks": tasks, "logs": logs, "snippets": snippets}


# ── Agent retrieval functions ─────────────────────────────────────────────────

def _brief(project_id: str) -> str:
    """Compact, agent-readable project summary — use like web_search for your project."""
    p = _DB.get_project(project_id)
    if not p:
        return f"[PM] Project '{project_id}' not found. Use list_projects to see available projects."

    all_tasks = _DB.list_tasks(project_id)
    recent_logs = _DB.list_logs(project_id, n=10)
    working_snips = _DB.list_snippets(project_id, status="working")
    sprints = _DB.list_sprints(project_id)

    by_status: dict[str, list] = {}
    for t in all_tasks:
        by_status.setdefault(t["status"], []).append(t)

    active_sprint = next((s for s in sprints if s["status"] == "active"), None)
    checkpoints = _DB.list_logs(project_id, n=1, log_type="checkpoint")
    last_cp = checkpoints[0] if checkpoints else None

    lines = [
        f"# PM BRIEF — {p['name']}",
        f"ID: {p['id']} | Status: {p['status']} | Created: {_fmt(p['created'])}",
        f"Description: {p['description'] or '(none)'}",
        "",
        "## TASK SUMMARY",
        (f"  ✅ Done: {len(by_status.get('done',[]))}  "
         f"🔄 In-Progress: {len(by_status.get('in-progress',[]))}  "
         f"⛔ Blocked: {len(by_status.get('blocked',[]))}  "
         f"❌ Failed: {len(by_status.get('failed',[]))}  "
         f"📋 Backlog: {len(by_status.get('backlog',[]))}  "
         f"📌 Total: {len(all_tasks)}"),
    ]

    if active_sprint:
        lines += [
            "",
            f"## ACTIVE SPRINT: {active_sprint['name']}",
            f"  Goal: {active_sprint['goal']}",
            f"  Period: {_fmt(active_sprint['start_date'])} → {_fmt(active_sprint['end_date'])}",
        ]

    if last_cp:
        lines += [
            "",
            f"## LAST CHECKPOINT [{last_cp['timestamp'][:16]}]",
            f"  {last_cp['title']}",
        ]
        if last_cp["content"]:
            lines.append(f"  {_trunc(last_cp['content'], 200)}")
        if last_cp["outcome"]:
            lines.append(f"  Outcome: {_trunc(last_cp['outcome'], 100)}")

    blocked = by_status.get("blocked", [])
    if blocked:
        lines += ["", "## BLOCKED — NEEDS ATTENTION"]
        for t in blocked:
            lines.append(f"  ⛔ [{t['id']}] {t['title']}: {_trunc(t['description'], 100)}")

    in_progress = by_status.get("in-progress", [])
    if in_progress:
        lines += ["", "## IN PROGRESS"]
        for t in in_progress:
            lines.append(f"  🔄 [{t['id']}] {t['title']}")

    failed = by_status.get("failed", [])
    if failed:
        lines += ["", "## TRIED & FAILED / ABANDONED"]
        for t in failed:
            lines.append(f"  ❌ [{t['id']}] {t['title']}: {_trunc(t['description'], 100)}")

    if working_snips:
        lines += ["", "## WORKING SOLUTIONS (code snippets)"]
        for s in working_snips:
            lines.append(f"  💡 [{s['id']}] {s['title']} [{s['language']}]")

    if recent_logs:
        lines += ["", "## RECENT LOG (last 10)"]
        for l in recent_logs:
            icon = LOG_ICONS.get(l["type"], "📝")
            ts16 = l["timestamp"][:16]
            lines.append(f"  {icon} [{ts16}] {l['type'].upper()}: {l['title']}")
            if l["content"]:
                lines.append(f"       → {_trunc(l['content'], 140)}")
            if l["outcome"]:
                lines.append(f"       Result: {_trunc(l['outcome'], 80)}")

    if p.get("docs"):
        docs_preview = p["docs"][:400] + ("…" if len(p["docs"]) > 400 else "")
        lines += ["", "## PROJECT DOCS (preview)", docs_preview]

    lines += ["", f"--- Generated {_ts()} | DB: {_DB_PATH} ---"]
    return "\n".join(lines)


def _search_brief(project_id: str, query: str) -> str:
    """Agent-readable search results."""
    results = _DB.search(project_id, query)
    lines = [f"## PM SEARCH — '{query}' in project {project_id}", ""]

    tasks = results["tasks"]
    if tasks:
        lines.append(f"### Tasks ({len(tasks)})")
        for t in tasks:
            lines.append(f"  [{t['id']}] [{t['status'].upper()}] [{t['priority']}] {t['title']}")
            if t["description"]:
                lines.append(f"       {_trunc(t['description'], 100)}")

    logs = results["logs"]
    if logs:
        lines.append(f"\n### Logs ({len(logs)})")
        for l in logs:
            icon = LOG_ICONS.get(l["type"], "📝")
            lines.append(f"  {icon} [{l['id']}] [{l['type'].upper()}] {l['title']}")
            if l["content"]:
                lines.append(f"       {_trunc(l['content'], 100)}")
            if l["outcome"]:
                lines.append(f"       Result: {_trunc(l['outcome'], 80)}")

    snippets = results["snippets"]
    if snippets:
        lines.append(f"\n### Snippets ({len(snippets)})")
        for s in snippets:
            lines.append(f"  [{s['id']}] [{s['status']}] {s['title']} ({s['language']})")

    if not any([tasks, logs, snippets]):
        lines.append("No results found.")

    return "\n".join(lines)


# ── The Tool ──────────────────────────────────────────────────────────────────

class PMTool(BaseTool):
    """
    Persistent project management. Supports Agile/Scrum/Kanban across all sessions.
    Think of 'brief' and 'search' as web_search for your own project history.
    """

    name = "pm"

    description = (
        "Persistent project management across all sessions. "
        "Track tasks (Kanban), sprints (Scrum), attempts/failures/solutions (log), "
        "and working code (snippets). "
        "Use 'brief' to retrieve a compact project summary without loading everything. "
        "Use 'search' to find specific past work. "
        "Use 'checkpoint' to mark stable points you can return to. "
        "All data persists in SQLite across sessions."
    )

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Action to perform. One of: "
                    "brief | search | checkpoint | "
                    "new_project | list_projects | update_project | "
                    "new_task | update_task | list_tasks | "
                    "new_log | list_logs | "
                    "new_sprint | update_sprint | list_sprints | "
                    "new_snippet | update_snippet | list_snippets | "
                    "get_docs | set_docs"
                ),
            },
            # Common
            "project_id": {"type": "string", "description": "Project ID (from list_projects or brief)"},
            "query":      {"type": "string", "description": "Search query (for search action)"},

            # Project fields
            "name":        {"type": "string"},
            "description": {"type": "string"},
            "status":      {"type": "string", "description": "Project/task/sprint status"},
            "tags":        {"type": "array",  "items": {"type": "string"}},
            "docs":        {"type": "string", "description": "Documentation content (set_docs)"},

            # Task fields
            "task_id":     {"type": "string"},
            "title":       {"type": "string"},
            "priority":    {"type": "string", "description": "low | medium | high | critical"},
            "sprint_id":   {"type": "string"},
            "start_date":  {"type": "string", "description": "YYYY-MM-DD"},
            "end_date":    {"type": "string", "description": "YYYY-MM-DD"},
            "checklist":   {"type": "array", "items": {"type": "string"},
                            "description": "List of checklist item strings"},

            # Log fields
            "log_id":      {"type": "string"},
            "log_type":    {"type": "string",
                            "description": "session | attempt | success | failure | solution | checkpoint | note"},
            "content":     {"type": "string"},
            "outcome":     {"type": "string"},

            # Sprint fields
            "sprint_id_update": {"type": "string", "description": "Sprint ID to update"},
            "goal":        {"type": "string"},

            # Snippet fields
            "snippet_id":  {"type": "string"},
            "language":    {"type": "string", "description": "python | typescript | bash | json | other"},
            "code":        {"type": "string"},
            "snippet_status": {"type": "string",
                               "description": "working | partial | broken | deprecated"},

            # Filters
            "limit":       {"type": "integer", "description": "Max items to return"},
        },
        "required": ["action"],
    }

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def execute(self, inp: dict) -> ToolResult:
        action = (inp.get("action") or "").strip().lower()
        pid    = inp.get("project_id", "").strip()

        try:
            if action == "brief":
                return ToolResult(_brief(pid) if pid else self._list_projects_brief())

            elif action == "search":
                if not pid:
                    return ToolResult("[PM] 'project_id' required for search.", is_error=True)
                query = (inp.get("query") or "").strip()
                if not query:
                    return ToolResult("[PM] 'query' required for search.", is_error=True)
                return ToolResult(_search_brief(pid, query))

            elif action == "checkpoint":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                title   = inp.get("title") or inp.get("name") or "Checkpoint"
                content = inp.get("content") or inp.get("description") or ""
                outcome = inp.get("outcome") or ""
                if content or outcome:
                    # Writing a checkpoint
                    entry = _DB.new_log(pid, "checkpoint", title, content, outcome)
                    return ToolResult(
                        f"✅ Checkpoint saved: [{entry['id']}] {title}\n"
                        f"   at {entry['timestamp']}"
                    )
                else:
                    # Reading last checkpoint
                    cps = _DB.list_logs(pid, n=1, log_type="checkpoint")
                    if not cps:
                        return ToolResult("[PM] No checkpoints yet for this project.")
                    cp = cps[0]
                    return ToolResult(
                        f"🏁 LAST CHECKPOINT [{cp['timestamp'][:16]}]\n"
                        f"   Title: {cp['title']}\n"
                        f"   Notes: {cp['content']}\n"
                        f"   Outcome: {cp['outcome']}"
                    )

            # ── Projects ──────────────────────────────────────────────────────

            elif action == "new_project":
                name = (inp.get("name") or "").strip()
                if not name:
                    return ToolResult("[PM] 'name' required.", is_error=True)
                status = inp.get("status") or "active"
                if status not in VALID_PROJECT_STATUS:
                    status = "active"
                p = _DB.new_project(
                    name=name,
                    description=inp.get("description") or "",
                    status=status,
                    tags=inp.get("tags") or [],
                )
                return ToolResult(
                    f"✅ Project created: [{p['id']}] {p['name']}\n"
                    f"   Status: {p['status']} | Use project_id='{p['id']}' for all further actions."
                )

            elif action == "list_projects":
                status_filter = inp.get("status")
                projects = _DB.list_projects(status_filter)
                if not projects:
                    return ToolResult("[PM] No projects yet. Use new_project to create one.")
                lines = [f"Projects ({len(projects)}):"]
                for p in projects:
                    tasks = _DB.list_tasks(p["id"])
                    done  = sum(1 for t in tasks if t["status"] == "done")
                    lines.append(
                        f"  [{p['id']}] [{p['status'].upper():8}] {p['name']} "
                        f"— {done}/{len(tasks)} tasks done | updated {p['updated'][:10]}"
                    )
                return ToolResult("\n".join(lines))

            elif action == "update_project":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                kwargs = {k: inp[k] for k in ("name","description","status","tags") if k in inp}
                return ToolResult(_DB.update_project(pid, **kwargs))

            # ── Tasks ─────────────────────────────────────────────────────────

            elif action == "new_task":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                title = (inp.get("title") or "").strip()
                if not title:
                    return ToolResult("[PM] 'title' required.", is_error=True)
                status = inp.get("status") or "backlog"
                if status not in VALID_TASK_STATUSES:
                    status = "backlog"
                priority = inp.get("priority") or "medium"
                if priority not in VALID_PRIORITIES:
                    priority = "medium"
                t = _DB.new_task(
                    project_id=pid, title=title,
                    description=inp.get("description") or "",
                    status=status, priority=priority,
                    sprint_id=inp.get("sprint_id"),
                    start_date=inp.get("start_date"),
                    end_date=inp.get("end_date"),
                    tags=inp.get("tags") or [],
                    checklist=inp.get("checklist") or [],
                )
                return ToolResult(
                    f"✅ Task added: [{t['id']}] {t['title']}\n"
                    f"   Status: {t['status']} | Priority: {t['priority']}"
                )

            elif action == "update_task":
                tid = (inp.get("task_id") or "").strip()
                if not tid:
                    return ToolResult("[PM] 'task_id' required.", is_error=True)
                kwargs = {}
                for k in ("title","description","status","priority","sprint_id",
                          "start_date","end_date","tags","checklist"):
                    if k in inp:
                        kwargs[k] = inp[k]
                if "status" in kwargs and kwargs["status"] not in VALID_TASK_STATUSES:
                    return ToolResult(
                        f"[PM] Invalid status. Valid: {', '.join(sorted(VALID_TASK_STATUSES))}",
                        is_error=True
                    )
                return ToolResult(_DB.update_task(tid, **kwargs))

            elif action == "list_tasks":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                status_filter = inp.get("status")
                tasks = _DB.list_tasks(pid, status_filter)
                if not tasks:
                    return ToolResult(f"[PM] No tasks{' with status=' + status_filter if status_filter else ''} in project {pid}.")
                STATUS_ICONS = {"backlog":"📋","todo":"📌","in-progress":"🔄",
                                "blocked":"⛔","done":"✅","failed":"❌"}
                PRIO_ICONS   = {"critical":"🔴","high":"🟠","medium":"🟡","low":"⚪"}
                lines = [f"Tasks in {pid}{' [' + status_filter + ']' if status_filter else ''} ({len(tasks)}):"]
                for t in tasks:
                    icon_s = STATUS_ICONS.get(t["status"], "•")
                    icon_p = PRIO_ICONS.get(t["priority"], "•")
                    cl     = t.get("checklist") or []
                    done_c = sum(1 for c in cl if c.get("done"))
                    cl_str = f" [{done_c}/{len(cl)}✓]" if cl else ""
                    lines.append(
                        f"  {icon_s}{icon_p} [{t['id']}] {t['title']}{cl_str}"
                    )
                    if t["description"]:
                        lines.append(f"       {_trunc(t['description'], 100)}")
                return ToolResult("\n".join(lines))

            # ── Logs ──────────────────────────────────────────────────────────

            elif action == "new_log":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                log_type = inp.get("log_type") or inp.get("type") or "note"
                if log_type not in VALID_LOG_TYPES:
                    log_type = "note"
                title   = inp.get("title") or ""
                content = inp.get("content") or inp.get("description") or ""
                outcome = inp.get("outcome") or ""
                task_id = inp.get("task_id") or None
                entry = _DB.new_log(pid, log_type, title, content, outcome, task_id)
                icon = LOG_ICONS.get(log_type, "📝")
                return ToolResult(
                    f"{icon} Log entry saved: [{entry['id']}] {log_type.upper()}: {title}\n"
                    f"   at {entry['timestamp']}"
                )

            elif action == "list_logs":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                n = int(inp.get("limit") or 15)
                log_type = inp.get("log_type") or inp.get("type") or None
                logs = _DB.list_logs(pid, n, log_type)
                if not logs:
                    return ToolResult("[PM] No log entries yet.")
                lines = [f"Logs for {pid} (last {n}{', type=' + log_type if log_type else ''}):"]
                for l in logs:
                    icon = LOG_ICONS.get(l["type"], "📝")
                    lines.append(f"  {icon} [{l['timestamp'][:16]}] {l['type'].upper()}: {l['title']}")
                    if l["content"]:
                        lines.append(f"       → {_trunc(l['content'], 120)}")
                    if l["outcome"]:
                        lines.append(f"       Result: {_trunc(l['outcome'], 80)}")
                return ToolResult("\n".join(lines))

            # ── Sprints ───────────────────────────────────────────────────────

            elif action == "new_sprint":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                name = (inp.get("name") or "").strip()
                if not name:
                    return ToolResult("[PM] 'name' required.", is_error=True)
                s = _DB.new_sprint(
                    project_id=pid, name=name,
                    goal=inp.get("goal") or "",
                    start_date=inp.get("start_date"),
                    end_date=inp.get("end_date"),
                    status=inp.get("status") or "planning",
                )
                return ToolResult(
                    f"✅ Sprint created: [{s['id']}] {s['name']}\n"
                    f"   Goal: {s['goal']} | {_fmt(s['start_date'])} → {_fmt(s['end_date'])}"
                )

            elif action == "update_sprint":
                sid = (inp.get("sprint_id") or inp.get("sprint_id_update") or "").strip()
                if not sid:
                    return ToolResult("[PM] 'sprint_id' required.", is_error=True)
                kwargs = {k: inp[k] for k in ("name","goal","start_date","end_date","status") if k in inp}
                return ToolResult(_DB.update_sprint(sid, **kwargs))

            elif action == "list_sprints":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                sprints = _DB.list_sprints(pid)
                if not sprints:
                    return ToolResult("[PM] No sprints yet.")
                STATUS_ICONS = {"planning":"📅","active":"🏃","completed":"🏁"}
                lines = [f"Sprints for {pid} ({len(sprints)}):"]
                for s in sprints:
                    icon = STATUS_ICONS.get(s["status"], "•")
                    lines.append(
                        f"  {icon} [{s['id']}] [{s['status'].upper():9}] {s['name']}"
                        f"  {_fmt(s['start_date'])} → {_fmt(s['end_date'])}"
                    )
                    if s["goal"]:
                        lines.append(f"       Goal: {s['goal']}")
                return ToolResult("\n".join(lines))

            # ── Snippets ──────────────────────────────────────────────────────

            elif action == "new_snippet":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                title = (inp.get("title") or "").strip()
                if not title:
                    return ToolResult("[PM] 'title' required.", is_error=True)
                snip_status = inp.get("snippet_status") or inp.get("status") or "partial"
                if snip_status not in VALID_SNIPPET_STATUS:
                    snip_status = "partial"
                s = _DB.new_snippet(
                    project_id=pid, title=title,
                    language=inp.get("language") or "python",
                    code=inp.get("code") or "",
                    status=snip_status,
                    tags=inp.get("tags") or [],
                )
                return ToolResult(
                    f"💾 Snippet saved: [{s['id']}] {s['title']} [{s['language']}] "
                    f"status={s['status']}"
                )

            elif action == "update_snippet":
                sid = (inp.get("snippet_id") or "").strip()
                if not sid:
                    return ToolResult("[PM] 'snippet_id' required.", is_error=True)
                kwargs = {}
                if "snippet_status" in inp:
                    kwargs["status"] = inp["snippet_status"]
                elif "status" in inp:
                    kwargs["status"] = inp["status"]
                for k in ("title","language","code","tags"):
                    if k in inp:
                        kwargs[k] = inp[k]
                return ToolResult(_DB.update_snippet(sid, **kwargs))

            elif action == "list_snippets":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                snip_status = inp.get("snippet_status") or inp.get("status") or None
                snippets = _DB.list_snippets(pid, snip_status)
                if not snippets:
                    return ToolResult("[PM] No snippets yet.")
                STATUS_ICONS = {"working":"💡","partial":"🔧","broken":"💥","deprecated":"🗑️"}
                lines = [f"Snippets for {pid}{' [' + snip_status + ']' if snip_status else ''} ({len(snippets)}):"]
                for s in snippets:
                    icon = STATUS_ICONS.get(s["status"], "•")
                    lines.append(f"  {icon} [{s['id']}] [{s['status']:10}] {s['title']} ({s['language']})")
                return ToolResult("\n".join(lines))

            # ── Docs ──────────────────────────────────────────────────────────

            elif action == "get_docs":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                p = _DB.get_project(pid)
                if not p:
                    return ToolResult(f"[PM] Project {pid} not found.", is_error=True)
                docs = p.get("docs") or ""
                return ToolResult(docs if docs else "[PM] No docs yet for this project.")

            elif action == "set_docs":
                if not pid:
                    return ToolResult("[PM] 'project_id' required.", is_error=True)
                content = inp.get("docs") or inp.get("content") or ""
                append  = inp.get("append", False)
                if append:
                    p = _DB.get_project(pid)
                    existing = (p or {}).get("docs") or ""
                    content  = existing + ("\n\n" if existing else "") + content
                _DB.update_project(pid, docs=content)
                return ToolResult(f"✅ Docs updated for project {pid} ({len(content)} chars).")

            else:
                valid = (
                    "brief, search, checkpoint, "
                    "new_project, list_projects, update_project, "
                    "new_task, update_task, list_tasks, "
                    "new_log, list_logs, "
                    "new_sprint, update_sprint, list_sprints, "
                    "new_snippet, update_snippet, list_snippets, "
                    "get_docs, set_docs"
                )
                return ToolResult(
                    f"[PM] Unknown action '{action}'. Valid actions:\n  {valid}",
                    is_error=True
                )

        except Exception as exc:
            return ToolResult(f"[PM] Error in action '{action}': {exc}", is_error=True)

    def _list_projects_brief(self) -> str:
        projects = _DB.list_projects()
        if not projects:
            return "[PM] No projects yet. Use action='new_project' to create one."
        lines = ["PM PROJECTS — use project_id to drill into a specific project:", ""]
        for p in projects:
            tasks = _DB.list_tasks(p["id"])
            done  = sum(1 for t in tasks if t["status"] == "done")
            blocked = sum(1 for t in tasks if t["status"] == "blocked")
            lines.append(
                f"  [{p['id']}] {p['name']} ({p['status']}) "
                f"— {done}/{len(tasks)} done, {blocked} blocked"
            )
            if p["description"]:
                lines.append(f"       {_trunc(p['description'], 80)}")
        lines += ["", f"Use action='brief' with project_id='<id>' for full details."]
        return "\n".join(lines)