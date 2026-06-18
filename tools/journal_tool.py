"""
tools/journal_tool.py — MasterMind persistent SQLite memory, ported as a tool.

Tables (from MasterMind's db_export.sql schema):
  sessions      — one row per boot, tracks session UUID + timestamps
  journal       — free-text entries with tags (what happened, decisions, facts)
  breadcrumbs   — ordered action log within a session (step N: action → outcome)
  world_model   — key/value entity knowledge store with confidence scores
  bg_tasks      — background task tracking (status, result, error)

Usage via skill_tool:
  {"tool": "journal", "action": "write",   "content": "...", "tags": ["dev"]}
  {"tool": "journal", "action": "read",    "limit": 20, "tag": "dev"}
  {"tool": "journal", "action": "crumb",   "summary": "called web_search", "outcome": "ok"}
  {"tool": "journal", "action": "know",    "entity": "dario", "fact": "developer, owner"}
  {"tool": "journal", "action": "recall",  "entity": "dario"}
  {"tool": "journal", "action": "status"}
"""
from __future__ import annotations
import json
import sqlite3
import time
import uuid
from pathlib import Path

from tools.base_tool import BaseTool, ToolResult
from utils.episode_log import ep as _ep

# ── DB location ───────────────────────────────────────────────────────────────
_DB_DIR  = Path(__file__).parent.parent / "memdir"
_DB_PATH = _DB_DIR / "eve_journal.db"

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    note        TEXT
);

CREATE TABLE IF NOT EXISTS journal (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    ts          REAL    NOT NULL,
    ts_human    TEXT    NOT NULL,
    content     TEXT    NOT NULL,
    tags        TEXT    DEFAULT '[]',
    FOREIGN KEY(session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS breadcrumbs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    step            INTEGER NOT NULL DEFAULT 0,
    ts              REAL    NOT NULL,
    ts_human        TEXT    NOT NULL,
    action_summary  TEXT    NOT NULL,
    outcome         TEXT,
    journal_id      INTEGER,
    FOREIGN KEY(session_id) REFERENCES sessions(id),
    FOREIGN KEY(journal_id) REFERENCES journal(id)
);

CREATE TABLE IF NOT EXISTS world_model (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    entity      TEXT    NOT NULL,
    fact        TEXT    NOT NULL,
    confidence  REAL    DEFAULT 1.0,
    source      TEXT,
    ts          REAL    NOT NULL,
    ts_human    TEXT    NOT NULL,
    UNIQUE(entity, fact) ON CONFLICT REPLACE
);

CREATE TABLE IF NOT EXISTS bg_tasks (
    task_id     TEXT PRIMARY KEY,
    description TEXT,
    user_input  TEXT,
    status      TEXT DEFAULT 'pending',
    result      TEXT,
    error       TEXT,
    started_at  REAL,
    finished_at REAL,
    created_at  REAL
);

CREATE INDEX IF NOT EXISTS idx_journal_ts      ON journal(ts DESC);
CREATE INDEX IF NOT EXISTS idx_journal_session ON journal(session_id);
CREATE INDEX IF NOT EXISTS idx_crumbs_session  ON breadcrumbs(session_id, step);
CREATE INDEX IF NOT EXISTS idx_world_entity    ON world_model(entity);
"""

# ── Connection helper ─────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    _DB_DIR.mkdir(exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def _ts() -> tuple[float, str]:
    """Return (unix_float, iso_string)."""
    t = time.time()
    from datetime import datetime, timezone
    h = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return t, h


# ── Session management (module-level singleton) ───────────────────────────────
_SESSION_ID: str = ""

def _get_or_create_session() -> str:
    global _SESSION_ID
    if _SESSION_ID:
        return _SESSION_ID
    _SESSION_ID = str(uuid.uuid4())
    t, h = _ts()
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at, note) VALUES (?,?,?)",
                (_SESSION_ID, t, "auto-created by journal_tool")
            )
    except Exception:
        pass
    return _SESSION_ID


def _next_step(session_id: str) -> int:
    try:
        with _conn() as c:
            row = c.execute(
                "SELECT MAX(step) as m FROM breadcrumbs WHERE session_id=?", (session_id,)
            ).fetchone()
            return (row["m"] or 0) + 1
    except Exception:
        return 0


# ── Tool class ────────────────────────────────────────────────────────────────

class JournalTool(BaseTool):
    name = "journal"
    description = (
        "Persistent SQLite memory — read/write journal entries, breadcrumb actions, "
        "store entity facts in the world model, and track background tasks. "
        "MasterMind-compatible schema: sessions → journal → breadcrumbs + world_model."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "crumb", "know", "recall", "status",
                         "task_create", "task_update", "task_list",
                         "remember", "episode", "ep_get", "ep_stats"],
                "description": (
                    "write=add journal entry | read=fetch recent entries | "
                    "crumb=log a breadcrumb step | know=store entity fact | "
                    "recall=retrieve entity facts | status=session summary | "
                    "task_create/update/list=background task management | "
                    "remember=search episodic memory ('remember that parsing error...') | "
                    "episode=manually bookmark an event | "
                    "ep_get=full detail of episode by id | "
                    "ep_stats=episode log summary"
                )
            },
            "content": {"type": "string", "description": "Journal entry text (write) or episode title (episode)"},
            "tags":    {"type": "array",  "items": {"type": "string"},
                        "description": "Tags for filtering (write/read/episode)"},
            "tag":     {"type": "string", "description": "Filter by single tag (read)"},
            "limit":   {"type": "integer","description": "Max entries to return (read/remember, default 20)"},
            "summary": {"type": "string", "description": "Action summary (crumb)"},
            "outcome": {"type": "string", "description": "Outcome of action (crumb)"},
            "entity":  {"type": "string", "description": "Entity name (know/recall)"},
            "fact":    {"type": "string", "description": "Fact about entity (know)"},
            "confidence": {"type": "number", "description": "Confidence 0-1 (know, default 1.0)"},
            "task_id":    {"type": "string", "description": "Task ID (task_update/list)"},
            "description": {"type": "string", "description": "Task description (task_create)"},
            "status":  {"type": "string", "description": "Task status (task_update)"},
            "result":  {"type": "string", "description": "Task result (task_update)"},
            "error":   {"type": "string", "description": "Task error (task_update)"},
            "query":   {"type": "string", "description": "Natural language search query (remember)"},
            "type":    {"type": "string", "description": "Episode type filter: error/tool_error/token_limit/crash/interrupted/power_cut (remember)"},
            "project": {"type": "string", "description": "Project name filter (remember/episode)"},
            "detail":  {"type": "string", "description": "Episode detail / full description (episode)"},
            "severity":{"type": "string", "description": "Episode severity: info/warning/error/critical (episode)"},
            "id":      {"type": "integer","description": "Episode ID (ep_get)"},
        },
        "required": ["action"],
    }

    def execute(self, inp: dict) -> ToolResult:
        action = inp.get("action", "").strip().lower()
        session_id = _get_or_create_session()

        try:
            if action == "write":
                return self._write(inp, session_id)
            elif action == "read":
                return self._read(inp)
            elif action == "crumb":
                return self._crumb(inp, session_id)
            elif action == "know":
                return self._know(inp)
            elif action == "recall":
                return self._recall(inp)
            elif action == "status":
                return self._status(session_id)
            elif action == "task_create":
                return self._task_create(inp)
            elif action == "task_update":
                return self._task_update(inp)
            elif action == "task_list":
                return self._task_list(inp)
            # ── episodic memory ───────────────────────────────────────────────
            elif action == "remember":
                return self._remember(inp)
            elif action == "episode":
                return self._log_episode(inp)
            elif action == "ep_get":
                return self._ep_get(inp)
            elif action == "ep_stats":
                return ToolResult(_ep.stats())
            else:
                return ToolResult(f"Unknown action: {action}", is_error=True)
        except Exception as e:
            return ToolResult(f"Journal error ({action}): {e}", is_error=True)

    # ── write ──────────────────────────────────────────────────────────────

    def _write(self, inp: dict, session_id: str) -> ToolResult:
        content = (inp.get("content") or "").strip()
        if not content:
            return ToolResult("'content' required for write.", is_error=True)
        tags = json.dumps(inp.get("tags") or [])
        t, h = _ts()
        with _conn() as c:
            cur = c.execute(
                "INSERT INTO journal(session_id,ts,ts_human,content,tags) VALUES(?,?,?,?,?)",
                (session_id, t, h, content, tags)
            )
            jid = cur.lastrowid
        return ToolResult(f"Journal entry #{jid} saved. [{h}]")

    # ── read ───────────────────────────────────────────────────────────────

    def _read(self, inp: dict) -> ToolResult:
        limit = int(inp.get("limit", 20))
        tag   = inp.get("tag") or inp.get("filter_tag") or ""
        with _conn() as c:
            if tag:
                rows = c.execute(
                    "SELECT id,ts_human,content,tags FROM journal "
                    "WHERE tags LIKE ? ORDER BY ts DESC LIMIT ?",
                    (f"%{tag}%", limit)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT id,ts_human,content,tags FROM journal "
                    "ORDER BY ts DESC LIMIT ?", (limit,)
                ).fetchall()

        if not rows:
            return ToolResult("No journal entries found.")

        lines = [f"Journal ({len(rows)} entries):\n"]
        for r in rows:
            tags_raw = r["tags"]
            try:
                tags_list = json.loads(tags_raw) if tags_raw else []
                tag_str = f" [{', '.join(tags_list)}]" if tags_list else ""
            except Exception:
                tag_str = ""
            lines.append(f"#{r['id']} {r['ts_human']}{tag_str}")
            lines.append(f"  {r['content']}")
            lines.append("")
        return ToolResult("\n".join(lines))

    # ── crumb ──────────────────────────────────────────────────────────────

    def _crumb(self, inp: dict, session_id: str) -> ToolResult:
        summary = (inp.get("summary") or inp.get("action_summary") or "").strip()
        if not summary:
            return ToolResult("'summary' required for crumb.", is_error=True)
        outcome = inp.get("outcome")
        t, h    = _ts()
        step    = _next_step(session_id)
        with _conn() as c:
            c.execute(
                "INSERT INTO breadcrumbs(session_id,step,ts,ts_human,action_summary,outcome) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, step, t, h, summary[:500], outcome)
            )
        return ToolResult(f"Breadcrumb step {step} logged: {summary[:80]}")

    # ── know (world model) ────────────────────────────────────────────────

    def _know(self, inp: dict) -> ToolResult:
        entity = (inp.get("entity") or "").strip()
        fact   = (inp.get("fact") or "").strip()
        if not entity or not fact:
            return ToolResult("'entity' and 'fact' required for know.", is_error=True)
        confidence = float(inp.get("confidence", 1.0))
        source     = inp.get("source") or "agent"
        t, h = _ts()
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO world_model"
                "(entity,fact,confidence,source,ts,ts_human) VALUES(?,?,?,?,?,?)",
                (entity.lower(), fact, confidence, source, t, h)
            )
        return ToolResult(f"World model updated: '{entity}' → '{fact[:80]}' (conf={confidence})")

    # ── recall (world model) ──────────────────────────────────────────────

    def _recall(self, inp: dict) -> ToolResult:
        entity = (inp.get("entity") or "").strip().lower()
        with _conn() as c:
            if entity:
                rows = c.execute(
                    "SELECT entity,fact,confidence,ts_human FROM world_model "
                    "WHERE entity=? ORDER BY confidence DESC",
                    (entity,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT entity,fact,confidence,ts_human FROM world_model "
                    "ORDER BY ts DESC LIMIT 50"
                ).fetchall()

        if not rows:
            return ToolResult(f"No world model entries for '{entity or '(all)'}'. ")

        lines = [f"World model — '{entity or 'all'}' ({len(rows)} facts):\n"]
        for r in rows:
            lines.append(f"  [{r['entity']}] conf={r['confidence']:.2f} @ {r['ts_human']}")
            lines.append(f"    {r['fact']}")
        return ToolResult("\n".join(lines))

    # ── status ────────────────────────────────────────────────────────────

    def _status(self, session_id: str) -> ToolResult:
        with _conn() as c:
            n_journal = c.execute("SELECT COUNT(*) as n FROM journal").fetchone()["n"]
            n_crumbs  = c.execute(
                "SELECT COUNT(*) as n FROM breadcrumbs WHERE session_id=?",
                (session_id,)
            ).fetchone()["n"]
            n_world   = c.execute("SELECT COUNT(*) as n FROM world_model").fetchone()["n"]
            n_tasks   = c.execute("SELECT COUNT(*) as n FROM bg_tasks").fetchone()["n"]
            n_sessions = c.execute("SELECT COUNT(*) as n FROM sessions").fetchone()["n"]
            recent = c.execute(
                "SELECT ts_human, content FROM journal ORDER BY ts DESC LIMIT 3"
            ).fetchall()

        lines = [
            "── Journal Status ──────────────────────────────",
            f"  Session ID   : {session_id[:16]}…",
            f"  Sessions     : {n_sessions}",
            f"  Journal entries: {n_journal}",
            f"  Breadcrumbs (this session): {n_crumbs}",
            f"  World model facts: {n_world}",
            f"  Background tasks: {n_tasks}",
            f"  DB: {_DB_PATH}",
            "",
            "── Recent journal ──────────────────────────────",
        ]
        for r in recent:
            lines.append(f"  {r['ts_human']}: {r['content'][:80]}")
        return ToolResult("\n".join(lines))

    # ── bg tasks ──────────────────────────────────────────────────────────

    def _task_create(self, inp: dict) -> ToolResult:
        task_id = str(uuid.uuid4())[:8]
        desc    = (inp.get("description") or "").strip()
        user_in = inp.get("user_input") or ""
        t, _ = _ts()
        with _conn() as c:
            c.execute(
                "INSERT INTO bg_tasks(task_id,description,user_input,status,created_at) "
                "VALUES(?,?,?,'pending',?)",
                (task_id, desc, user_in, t)
            )
        return ToolResult(f"Background task created: {task_id} — {desc[:60]}")

    def _task_update(self, inp: dict) -> ToolResult:
        task_id = (inp.get("task_id") or "").strip()
        if not task_id:
            return ToolResult("'task_id' required for task_update.", is_error=True)
        status  = inp.get("status")
        result  = inp.get("result")
        error   = inp.get("error")
        t, _ = _ts()
        updates = []
        vals = []
        if status:  updates.append("status=?");      vals.append(status)
        if result:  updates.append("result=?");      vals.append(result)
        if error:   updates.append("error=?");       vals.append(error)
        if status in ("done", "complete", "error", "failed"):
            updates.append("finished_at=?"); vals.append(t)
        if not updates:
            return ToolResult("Nothing to update.", is_error=True)
        vals.append(task_id)
        with _conn() as c:
            c.execute(f"UPDATE bg_tasks SET {', '.join(updates)} WHERE task_id=?", vals)
        return ToolResult(f"Task {task_id} updated → {status or '?'}")

    def _task_list(self, inp: dict) -> ToolResult:
        status_filter = inp.get("status")
        with _conn() as c:
            if status_filter:
                rows = c.execute(
                    "SELECT task_id,description,status,created_at FROM bg_tasks "
                    "WHERE status=? ORDER BY created_at DESC LIMIT 20",
                    (status_filter,)
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT task_id,description,status,created_at FROM bg_tasks "
                    "ORDER BY created_at DESC LIMIT 20"
                ).fetchall()
        if not rows:
            return ToolResult("No background tasks.")
        lines = [f"Background tasks ({len(rows)}):"]
        for r in rows:
            lines.append(f"  [{r['status']:8}] {r['task_id']} — {r['description'] or '(no desc)'}[:60]")
        return ToolResult("\n".join(lines))

    # ── episodic memory ────────────────────────────────────────────────────

    def _remember(self, inp: dict) -> ToolResult:
        """
        Natural-language search over all episodes.
        e.g. {"action":"remember","query":"parsing error in project xyz"}
             {"action":"remember","query":"token limit","type":"token_limit"}
             {"action":"remember","query":"interrupted","project":"Mind_EVE"}
        """
        query          = (inp.get("query") or inp.get("content") or "").strip()
        type_filter    = inp.get("type") or ""
        project_filter = inp.get("project") or ""
        severity_filter= inp.get("severity") or ""
        limit          = int(inp.get("limit") or 15)
        result = _ep.search(query, type_filter, project_filter, severity_filter, limit)
        return ToolResult(result)

    def _log_episode(self, inp: dict) -> ToolResult:
        """Manually bookmark an episode from within the agent."""
        title    = (inp.get("content") or inp.get("title") or "").strip()
        if not title:
            return ToolResult("'content' (title) required for episode.", is_error=True)
        detail   = inp.get("detail") or ""
        tags     = inp.get("tags") or []
        project  = inp.get("project") or ""
        severity = inp.get("severity") or "info"
        ep_type  = inp.get("type") or "note"
        eid = _ep.log(ep_type, title, detail, project=project, tags=tags, severity=severity)
        return ToolResult(f"Episode #{eid} logged: {title[:80]}")

    def _ep_get(self, inp: dict) -> ToolResult:
        """Retrieve full detail of one episode by numeric ID."""
        eid = inp.get("id")
        if eid is None:
            return ToolResult("'id' required for ep_get.", is_error=True)
        return ToolResult(_ep.get_episode(int(eid)))