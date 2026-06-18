# -*- coding: utf-8 -*-
"""
utils/episode_log.py — EVE Episodic Memory Engine
══════════════════════════════════════════════════

Gives EVE human-like autobiographical memory: every error, tool call,
token-limit hit, crash, accidental CTRL+C, or power-cut is recorded as
a searchable "episode".  On the next boot, EVE automatically detects
whether the last session ended cleanly and, if not, logs an "interrupted"
episode so you can ask "what happened?" and get a real answer.

Natural-language recall:
    "remember that parsing error in project xyz?"
    "what happened last time we hit token limit?"
    "that time I accidentally stopped everything"
    "what were you doing when the power cut happened?"

Public API (use the singleton `ep`):
    ep.log(type, title, detail, project, tool, tags, severity)
    ep.log_tool_error(tool_name, inp, error_text)
    ep.log_token_limit(provider, model, usage_dict)
    ep.log_crash(exc, tb_str, context)
    ep.log_interrupted(last_action, last_tool)
    ep.search(query, type_filter, project_filter, limit) -> str
    ep.detect_resume(current_session_id, cwd) -> str | None
    ep.session_start(session_id, cwd, model)
    ep.session_end(session_id)

WINDOWS NOTE — querying the episode DB directly:
    `sqlite3` is NOT available as a shell command on Windows.
    `Invoke-SqliteQuery` is NOT a standard PowerShell cmdlet.
    ALWAYS use Python's built-in sqlite3 module:

        bash: python -c "
        import sqlite3, pathlib
        db = pathlib.Path(r'C:/Users/dario/OneDrive/Documenten/Mind_EVE/memdir/eve_episodes.db')
        con = sqlite3.connect(db)
        rows = con.execute('SELECT ts_human, type, title FROM episodes ORDER BY ts DESC LIMIT 10').fetchall()
        for r in rows: print(r)
        con.close()
        "

    Or better — use the ep singleton directly:
        from utils.episode_log import ep
        print(ep.search("last task", limit=5))
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import sqlite3
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── DB location (same dir as the rest of EVE's memory) ───────────────────────
_DB_DIR  = Path(__file__).parent.parent / "memdir"
_DB_PATH = _DB_DIR / "eve_episodes.db"

# ── Episode types ─────────────────────────────────────────────────────────────
class EpType:
    SESSION_START  = "session_start"
    SESSION_END    = "session_end"
    ERROR          = "error"
    TOOL_ERROR     = "tool_error"
    TOKEN_LIMIT    = "token_limit"
    CRASH          = "crash"
    INTERRUPTED    = "interrupted"   # CTRL+C / kill signal
    POWER_CUT      = "power_cut"     # no clean shutdown detected on resume
    TASK_ABORT     = "task_abort"
    NOTE           = "note"          # manually logged
    PROVIDER_FAIL  = "provider_fail"
    IMPORT_ERROR   = "import_error"

# ── Schema ────────────────────────────────────────────────────────────────────
_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- One row per physical boot
CREATE TABLE IF NOT EXISTS ep_sessions (
    id          TEXT PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    cwd         TEXT,
    model       TEXT,
    clean_exit  INTEGER DEFAULT 0   -- 1 = session_end was logged
);

-- Every noteworthy event
CREATE TABLE IF NOT EXISTS episodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL,
    ts          REAL    NOT NULL,
    ts_human    TEXT    NOT NULL,
    type        TEXT    NOT NULL,
    title       TEXT    NOT NULL,
    detail      TEXT    DEFAULT '',
    project     TEXT    DEFAULT '',
    tool_name   TEXT    DEFAULT '',
    tags        TEXT    DEFAULT '[]',
    severity    TEXT    DEFAULT 'info',   -- info/warning/error/critical
    resolved    INTEGER DEFAULT 0
);

-- Full-text search over title + detail + tags + project + tool_name
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    title, detail, tags, project, tool_name,
    content=episodes, content_rowid=id,
    tokenize='porter unicode61'
);

-- Keep FTS in sync
CREATE TRIGGER IF NOT EXISTS ep_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, title, detail, tags, project, tool_name)
    VALUES (new.id, new.title, new.detail, new.tags, new.project, new.tool_name);
END;
CREATE TRIGGER IF NOT EXISTS ep_ad AFTER DELETE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, detail, tags, project, tool_name)
    VALUES ('delete', old.id, old.title, old.detail, old.tags, old.project, old.tool_name);
END;
CREATE TRIGGER IF NOT EXISTS ep_au AFTER UPDATE ON episodes BEGIN
    INSERT INTO episodes_fts(episodes_fts, rowid, title, detail, tags, project, tool_name)
    VALUES ('delete', old.id, old.title, old.detail, old.tags, old.project, old.tool_name);
    INSERT INTO episodes_fts(rowid, title, detail, tags, project, tool_name)
    VALUES (new.id, new.title, new.detail, new.tags, new.project, new.tool_name);
END;

CREATE INDEX IF NOT EXISTS ep_idx_ts      ON episodes(ts DESC);
CREATE INDEX IF NOT EXISTS ep_idx_type    ON episodes(type);
CREATE INDEX IF NOT EXISTS ep_idx_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS ep_idx_project ON episodes(project);
"""

# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> tuple[float, str]:
    t = time.time()
    h = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return t, h


def _conn() -> sqlite3.Connection:
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    return c


def _project_from_cwd(cwd: str = "") -> str:
    """Derive a short project name from the working directory."""
    if not cwd:
        cwd = os.getcwd()
    return Path(cwd).name or "unknown"


# ── Core engine ───────────────────────────────────────────────────────────────

class EpisodeLog:
    """Singleton episodic memory engine."""

    def __init__(self):
        self._session_id: str = ""
        self._project:    str = ""
        self._shutdown_registered = False
        self._last_tool: str  = ""
        self._last_inp:  dict = {}

    # ── Session lifecycle ─────────────────────────────────────────────────

    def session_start(self, session_id: str, cwd: str = "", model: str = "auto"):
        self._session_id = session_id
        self._project    = _project_from_cwd(cwd)
        t, h = _now()
        try:
            with _conn() as c:
                c.execute(
                    "INSERT OR IGNORE INTO ep_sessions(id,started_at,cwd,model) VALUES(?,?,?,?)",
                    (session_id, t, cwd, model)
                )
            self.log(
                EpType.SESSION_START,
                f"Session started — {self._project}",
                f"model={model}  cwd={cwd}",
                severity="info",
            )
        except Exception:
            pass

        if not self._shutdown_registered:
            atexit.register(self._atexit_handler)
            self._shutdown_registered = True

    def session_end(self, session_id: str = ""):
        sid = session_id or self._session_id
        if not sid:
            return
        t, h = _now()
        try:
            with _conn() as c:
                c.execute(
                    "UPDATE ep_sessions SET ended_at=?, clean_exit=1 WHERE id=?",
                    (t, sid)
                )
            self.log(EpType.SESSION_END, "Session ended cleanly", severity="info")
        except Exception:
            pass

    def _atexit_handler(self):
        """Called on normal Python exit — marks the session as clean."""
        self.session_end()

    # ── Resume / interruption detection ──────────────────────────────────

    def detect_resume(self, current_session_id: str, cwd: str = "") -> Optional[str]:
        """
        Call at boot.  Returns a human-readable description of what happened
        in the previous session (if it ended unexpectedly), or None if last
        exit was clean.  Automatically logs an 'interrupted' or 'power_cut'
        episode if needed.
        """
        try:
            with _conn() as c:
                prev = c.execute(
                    "SELECT id, started_at, ended_at, clean_exit, cwd, model "
                    "FROM ep_sessions WHERE id != ? "
                    "ORDER BY started_at DESC LIMIT 1",
                    (current_session_id,)
                ).fetchone()

            if not prev or prev["clean_exit"]:
                return None   # clean exit last time

            # Unclean — figure out what the last logged episode was
            with _conn() as c:
                last_ep = c.execute(
                    "SELECT type, title, detail, ts_human FROM episodes "
                    "WHERE session_id=? ORDER BY ts DESC LIMIT 1",
                    (prev["id"],)
                ).fetchone()

            # Decide episode type: if ended_at is None, likely a hard kill/power cut
            ep_type = EpType.POWER_CUT if prev["ended_at"] is None else EpType.INTERRUPTED
            ep_title = (
                "Power cut / hard kill detected — previous session had no clean exit"
                if ep_type == EpType.POWER_CUT
                else "Previous session was interrupted (CTRL+C or signal)"
            )

            last_str = ""
            if last_ep:
                last_str = (
                    f"Last recorded event: [{last_ep['type']}] {last_ep['title']} "
                    f"@ {last_ep['ts_human']}"
                )

            detail = (
                f"Previous session: {prev['id'][:16]}…\n"
                f"Started: {datetime.fromtimestamp(prev['started_at'], tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
                f"Project: {_project_from_cwd(prev['cwd'] or '')}\n"
                f"Model:   {prev['model'] or 'unknown'}\n"
                f"{last_str}"
            )

            self.log(ep_type, ep_title, detail, severity="warning")
            return f"⚡ {ep_title}\n  {last_str}"

        except Exception as e:
            return None

    # ── Core log ──────────────────────────────────────────────────────────

    def log(
        self,
        type:      str,
        title:     str,
        detail:    str  = "",
        project:   str  = "",
        tool:      str  = "",
        tags:      list = None,
        severity:  str  = "info",
    ) -> int:
        """Write one episode. Returns the new row id."""
        t, h = _now()
        sid  = self._session_id or "bootstrap"
        proj = project or self._project or _project_from_cwd()
        tags_json = json.dumps(tags or [])
        try:
            with _conn() as c:
                cur = c.execute(
                    "INSERT INTO episodes"
                    "(session_id,ts,ts_human,type,title,detail,project,tool_name,tags,severity)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (sid, t, h, type, title[:400], detail[:4000],
                     proj[:200], tool[:100], tags_json, severity)
                )
                return cur.lastrowid or 0
        except Exception:
            return 0

    # ── Convenience loggers ───────────────────────────────────────────────

    def log_tool_error(self, tool_name: str, inp: dict, error_text: str):
        inp_snippet = json.dumps(inp)[:300]
        self.log(
            EpType.TOOL_ERROR,
            f"Tool error — {tool_name}",
            f"Input: {inp_snippet}\nError: {error_text[:1000]}",
            tool=tool_name,
            tags=["tool_error", tool_name],
            severity="error",
        )

    def log_token_limit(self, provider: str, model: str, usage: dict = None):
        detail = f"Provider: {provider}\nModel: {model}"
        if usage:
            detail += f"\nUsage: {json.dumps(usage)}"
        self.log(
            EpType.TOKEN_LIMIT,
            f"Token/quota limit hit — {provider} ({model})",
            detail,
            tags=["token_limit", provider, model],
            severity="warning",
        )

    def log_crash(self, exc: BaseException, tb_str: str = "", context: str = ""):
        exc_name = type(exc).__name__
        self.log(
            EpType.CRASH,
            f"Unhandled exception — {exc_name}: {str(exc)[:120]}",
            f"Context: {context}\n\nTraceback:\n{tb_str or traceback.format_exc()}",
            tags=["crash", exc_name],
            severity="critical",
        )

    def log_interrupted(self, last_action: str = "", last_tool: str = ""):
        detail = ""
        if last_action:
            detail += f"Last action: {last_action}\n"
        if last_tool:
            detail += f"Last tool:   {last_tool}\n"
        self.log(
            EpType.INTERRUPTED,
            "Session interrupted (CTRL+C / SIGTERM)",
            detail.strip(),
            tool=last_tool,
            tags=["interrupted"],
            severity="warning",
        )

    def log_provider_fail(self, provider: str, status_code: int, error: str):
        self.log(
            EpType.PROVIDER_FAIL,
            f"Cloud provider failed — {provider} (HTTP {status_code})",
            error[:500],
            tags=["provider_fail", provider],
            severity="error",
        )

    # ── Signal handlers ───────────────────────────────────────────────────

    def install_signal_handlers(self):
        """Install SIGINT/SIGTERM handlers that log before exiting."""
        _orig_sigint  = signal.getsignal(signal.SIGINT)
        _orig_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle(signum, frame):
            sig_name = "SIGINT (CTRL+C)" if signum == signal.SIGINT else "SIGTERM"
            self.log_interrupted(last_action=f"received {sig_name}", last_tool=self._last_tool)
            # Don't mark as clean — the atexit handler will still run but
            # we've already stored the interrupt episode above.
            # Remove ourselves then re-raise the original handler.
            signal.signal(signum, _orig_sigint if signum == signal.SIGINT else _orig_sigterm)
            os.kill(os.getpid(), signum)

        try:
            signal.signal(signal.SIGINT,  _handle)
            signal.signal(signal.SIGTERM, _handle)
        except (OSError, ValueError):
            pass   # can't set signals in non-main thread

    def install_excepthook(self):
        """Catch any unhandled exception and log it before Python prints it."""
        _orig = sys.excepthook

        def _hook(exc_type, exc_value, exc_tb):
            if exc_type is KeyboardInterrupt:
                _orig(exc_type, exc_value, exc_tb)
                return
            tb_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            self.log_crash(exc_value, tb_str, context="sys.excepthook")
            _orig(exc_type, exc_value, exc_tb)

        sys.excepthook = _hook

    # ── Track last tool (for interrupt context) ───────────────────────────

    def on_tool_start(self, tool_name: str, inp: dict):
        self._last_tool = tool_name
        self._last_inp  = inp

    def on_tool_end(self, tool_name: str, result):
        if result.is_error:
            self.log_tool_error(tool_name, self._last_inp, result.output or "")

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        type_filter:    str  = "",
        project_filter: str  = "",
        severity_filter: str = "",
        limit:          int  = 15,
    ) -> str:
        """
        Full-text search over all episodes.
        Returns a formatted string ready to be injected into the model's context.
        """
        try:
            with _conn() as c:
                # Build WHERE clause
                fts_q = query.strip()
                params: list = []

                if fts_q:
                    base = (
                        "SELECT e.id, e.ts_human, e.type, e.title, e.detail, "
                        "       e.project, e.tool_name, e.severity "
                        "FROM episodes_fts fts "
                        "JOIN episodes e ON e.id = fts.rowid "
                        "WHERE episodes_fts MATCH ? "
                    )
                    params.append(fts_q)
                else:
                    base = (
                        "SELECT id, ts_human, type, title, detail, "
                        "       project, tool_name, severity "
                        "FROM episodes WHERE 1=1 "
                    )

                if type_filter:
                    base += " AND e.type=? " if fts_q else " AND type=? "
                    params.append(type_filter)
                if project_filter:
                    base += " AND e.project LIKE ? " if fts_q else " AND project LIKE ? "
                    params.append(f"%{project_filter}%")
                if severity_filter:
                    base += " AND e.severity=? " if fts_q else " AND severity=? "
                    params.append(severity_filter)

                order = (
                    " ORDER BY rank LIMIT ?" if fts_q
                    else " ORDER BY ts DESC LIMIT ?"
                )
                params.append(limit)

                rows = c.execute(base + order, params).fetchall()

        except Exception as e:
            return f"[episode_log] Search error: {e}"

        if not rows:
            return f"No episodes found matching: '{query}'"

        lines = [f"── Episode recall: '{query}' ({len(rows)} match{'es' if len(rows)!=1 else ''}) ──\n"]
        for r in rows:
            sev_icon = {"critical": "💥", "error": "✖", "warning": "⚠", "info": "·"}.get(r["severity"], "·")
            proj = f"  [{r['project']}]" if r["project"] else ""
            tool = f"  via {r['tool_name']}" if r["tool_name"] else ""
            lines.append(
                f"{sev_icon} #{r['id']}  {r['ts_human']}  {r['type'].upper()}{proj}{tool}"
            )
            lines.append(f"  {r['title']}")
            if r["detail"]:
                # Show first 2 lines of detail
                detail_lines = [l for l in r["detail"].splitlines() if l.strip()][:2]
                for dl in detail_lines:
                    lines.append(f"    {dl}")
            lines.append("")

        return "\n".join(lines)

    def get_episode(self, episode_id: int) -> str:
        """Retrieve the full detail of a single episode by ID."""
        try:
            with _conn() as c:
                r = c.execute(
                    "SELECT * FROM episodes WHERE id=?", (episode_id,)
                ).fetchone()
        except Exception as e:
            return f"[episode_log] DB error: {e}"

        if not r:
            return f"No episode with id={episode_id}."

        sev_icon = {"critical": "💥", "error": "✖", "warning": "⚠", "info": "·"}.get(r["severity"], "·")
        lines = [
            f"── Episode #{r['id']} ──────────────────────────────────────",
            f"  Time     : {r['ts_human']}",
            f"  Type     : {r['type']}",
            f"  Severity : {sev_icon} {r['severity']}",
            f"  Project  : {r['project'] or '(none)'}",
            f"  Tool     : {r['tool_name'] or '(none)'}",
            f"  Tags     : {r['tags']}",
            f"  Resolved : {'yes' if r['resolved'] else 'no'}",
            "",
            f"  Title: {r['title']}",
            "",
            "  Detail:",
        ]
        for line in (r["detail"] or "").splitlines():
            lines.append(f"    {line}")
        return "\n".join(lines)

    def stats(self) -> str:
        """Return a brief summary of the episode log."""
        try:
            with _conn() as c:
                total   = c.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                by_type = c.execute(
                    "SELECT type, COUNT(*) as n FROM episodes GROUP BY type ORDER BY n DESC"
                ).fetchall()
                sessions = c.execute("SELECT COUNT(*) FROM ep_sessions").fetchone()[0]
                last5 = c.execute(
                    "SELECT ts_human, type, title FROM episodes ORDER BY ts DESC LIMIT 5"
                ).fetchall()
        except Exception as e:
            return f"[episode_log] stats error: {e}"

        lines = [
            "── Episode Log Stats ────────────────────────────────",
            f"  Total episodes : {total}",
            f"  Sessions       : {sessions}",
            f"  DB             : {_DB_PATH}",
            "",
            "  By type:",
        ]
        for row in by_type:
            lines.append(f"    {row['type']:20s} {row['n']}")
        lines.append("\n  Recent episodes:")
        for r in last5:
            lines.append(f"    {r['ts_human']}  [{r['type']}]  {r['title'][:60]}")
        return "\n".join(lines)


# ── Singleton ─────────────────────────────────────────────────────────────────
ep: EpisodeLog = EpisodeLog()