"""
tools/investigate_tool.py — Self-investigation and session recovery tool.

EVE calls this whenever it needs to reconstruct what happened in a previous
session, recover from an unclean exit, or produce an audit trail.

It reads every available evidence source in the project:
  - data/mastermind.db         (user ↔ assistant conversation messages)
  - memdir/eve_journal.db      (sessions, journal entries, breadcrumbs)
  - memdir/journal.json        (heartbeat/activity log)
  - temp/context_offload/*.md  (task checkpoint snapshots)
  - logs/*.log                 (system logs)
  - memdir/*.json / *.md       (goal state, error knowledge, flags)

And writes three output files to memdir/:
  - task_log.md        — every significant action with timestamp + hash
  - metadata.json      — structured session index, blockers, file hashes
  - journal_transcript.md — full verbatim interaction transcript

Usage via tool call:
  {"tool": "investigate", "action": "full"}
  {"tool": "investigate", "action": "transcript"}
  {"tool": "investigate", "action": "task_log"}
  {"tool": "investigate", "action": "metadata"}
  {"tool": "investigate", "action": "status"}     ← quick summary only, no writes
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.base_tool import BaseTool, ToolResult

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).parent.parent
_MEMDIR    = _ROOT / "memdir"
_LOGS      = _ROOT / "logs"
_OFFLOAD   = _ROOT / "temp" / "context_offload"
_MASTERDB  = _ROOT / "data" / "mastermind.db"
_JOURNALDB = _MEMDIR / "eve_journal.db"
_JOURNAL_JSON = _MEMDIR / "journal.json"

_OUT_TASK_LOG    = _MEMDIR / "task_log.md"
_OUT_METADATA    = _MEMDIR / "metadata.json"
_OUT_TRANSCRIPT  = _MEMDIR / "journal_transcript.md"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_human() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _file_hash(path: Path) -> str:
    """MD5 of file content, first 8 chars."""
    try:
        h = hashlib.md5(path.read_bytes()).hexdigest()
        return h[:8]
    except Exception:
        return "????????"


def _str_hash(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _ts_human(ts_float: float) -> str:
    return datetime.fromtimestamp(ts_float, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _safe_read(path: Path, max_bytes: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_bytes]
    except Exception:
        return ""


# ── Source readers ─────────────────────────────────────────────────────────────

def _read_mastermind() -> list[dict]:
    """Return all messages from mastermind.db ordered by created_at."""
    if not _MASTERDB.exists():
        return []
    try:
        conn = sqlite3.connect(str(_MASTERDB))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT m.id, m.conv_id, m.role, m.text, m.tools_json, m.created_at,
                   c.title as conv_title
            FROM messages m
            LEFT JOIN conversations c ON c.id = m.conv_id
            ORDER BY m.created_at
            """
        )
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        return [{"_error": str(e)}]


def _read_eve_journal() -> dict:
    """Return sessions, journal entries, breadcrumbs from eve_journal.db."""
    out = {"sessions": [], "entries": [], "breadcrumbs": []}
    if not _JOURNALDB.exists():
        return out
    try:
        conn = sqlite3.connect(str(_JOURNALDB))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM sessions ORDER BY started_at")
        out["sessions"] = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM journal ORDER BY ts")
        out["entries"] = [dict(r) for r in cur.fetchall()]
        cur.execute("SELECT * FROM breadcrumbs ORDER BY ts")
        out["breadcrumbs"] = [dict(r) for r in cur.fetchall()]
        conn.close()
    except Exception as e:
        out["_error"] = str(e)
    return out


def _read_journal_json() -> list[dict]:
    if not _JOURNAL_JSON.exists():
        return []
    try:
        return json.loads(_JOURNAL_JSON.read_text(encoding="utf-8"))
    except Exception:
        return []


def _read_logs() -> list[dict]:
    """Read all .log files from logs/."""
    results = []
    if not _LOGS.exists():
        return results
    for p in sorted(_LOGS.glob("*.log")):
        results.append({
            "filename": p.name,
            "modified": _ts_human(p.stat().st_mtime),
            "content": _safe_read(p, 50_000),
        })
    return results


def _read_context_offloads() -> list[dict]:
    """Read all .md files from temp/context_offload/."""
    results = []
    if not _OFFLOAD.exists():
        return results
    for p in sorted(_OFFLOAD.glob("*.md"), key=lambda x: x.stat().st_mtime):
        results.append({
            "filename": p.name,
            "modified": _ts_human(p.stat().st_mtime),
            "hash": _file_hash(p),
            "content": _safe_read(p, 30_000),
        })
    return results


def _read_memdir_flags() -> list[dict]:
    """Read .flag, .lock, .txt, .json files from memdir/."""
    results = []
    for pat in ("*.flag", "*.lock", "*.txt", "*.json"):
        for p in sorted(_MEMDIR.glob(pat)):
            if p.name in ("metadata.json", "task_log.md", "journal_transcript.md"):
                continue
            results.append({
                "filename": p.name,
                "modified": _ts_human(p.stat().st_mtime),
                "size": p.stat().st_size,
                "hash": _file_hash(p),
                "content_preview": _safe_read(p, 500),
            })
    return results


# ── Writers ────────────────────────────────────────────────────────────────────

def _write_task_log(
    messages: list[dict],
    offloads: list[dict],
    logs: list[dict],
    flags: list[dict],
) -> str:
    lines = [
        "# Mind EVE — Task Log (auto-generated by investigate_tool)",
        f"# Generated: {_now_human()}",
        "# Format: [TIMESTAMP UTC] [HASH8] ACTION_TYPE | detail",
        "#",
    ]

    # ── Mastermind messages
    if messages:
        lines.append("\n## CONVERSATION MESSAGES (mastermind.db)\n")
        current_conv = None
        for m in messages:
            if "_error" in m:
                lines.append(f"[ERROR] {m['_error']}")
                continue
            ts = _ts_human(m["created_at"] / 1000) if m.get("created_at") else "?"
            conv = m.get("conv_id", "?")
            if conv != current_conv:
                current_conv = conv
                title = m.get("conv_title") or conv
                lines.append(f"\n### Conversation: {title}")
            role = (m.get("role") or "?").upper()
            h = _str_hash(str(m.get("id", "")) + str(m.get("created_at", "")))
            tools = []
            if m.get("tools_json"):
                try:
                    tj = json.loads(m["tools_json"])
                    tools = [t.get("name", "?") for t in tj if isinstance(t, dict)]
                except Exception:
                    pass
            tool_str = f"  [tools: {', '.join(tools)}]" if tools else ""
            # Full text — no truncation so task_log captures every word
            full_text = (m.get("text") or "").strip().replace("\n", " ↵ ")
            lines.append(f"[{ts}] [{h}] {role}{tool_str} | {full_text}")

    # ── Context offloads
    if offloads:
        lines.append("\n## CONTEXT OFFLOAD CHECKPOINTS (temp/context_offload/)\n")
        for o in offloads:
            first_line = (o["content"].split("\n")[0] if o["content"] else "").strip()
            lines.append(
                f"[{o['modified']}] [{o['hash']}] CONTEXT_OFFLOAD | {o['filename']} — {first_line}"
            )

    # ── Logs
    if logs:
        lines.append("\n## SYSTEM LOGS (logs/)\n")
        for log in logs:
            for line in log["content"].splitlines():
                line = line.strip()
                if not line:
                    continue
                h = _str_hash(log["filename"] + line)[:8]
                action = "LOG_ERROR" if "ERROR" in line else "LOG_INFO"
                lines.append(f"[{log['modified']}] [{h}] {action} | [{log['filename']}] {line}")

    # ── Flags / state files
    if flags:
        lines.append("\n## MEMDIR STATE FILES\n")
        for f in flags:
            lines.append(
                f"[{f['modified']}] [{f['hash']}] MEMDIR_FILE | {f['filename']} ({f['size']} bytes)"
            )

    content = "\n".join(lines)
    _OUT_TASK_LOG.write_text(content, encoding="utf-8")
    return content


def _write_metadata(
    messages: list[dict],
    eve_journal: dict,
    offloads: list[dict],
    flags: list[dict],
) -> dict:
    # Group messages into conversations
    convs: dict[str, list] = {}
    for m in messages:
        if "_error" in m:
            continue
        cid = m.get("conv_id", "unknown")
        convs.setdefault(cid, []).append(m)

    sessions_meta = []
    for cid, msgs in convs.items():
        title = msgs[0].get("conv_title") or cid
        start_ts = min(m["created_at"] for m in msgs if m.get("created_at"))
        end_ts   = max(m["created_at"] for m in msgs if m.get("created_at"))
        tools_all = []
        for m in msgs:
            if m.get("tools_json"):
                try:
                    tj = json.loads(m["tools_json"])
                    tools_all += [t.get("name", "?") for t in tj if isinstance(t, dict)]
                except Exception:
                    pass
        sessions_meta.append({
            "conv_id": cid,
            "title": title,
            "started": _ts_human(start_ts / 1000),
            "ended": _ts_human(end_ts / 1000),
            "message_count": len(msgs),
            "tool_calls": list(dict.fromkeys(tools_all)),
            "hash": _str_hash(cid),
        })

    # DB sessions
    db_sessions = [
        {
            "id": s["id"],
            "started": _ts_human(s["started_at"]),
            "ended": _ts_human(s["ended_at"]) if s.get("ended_at") else None,
            "note": s.get("note"),
        }
        for s in eve_journal.get("sessions", [])
    ]

    # Detect blockers from logs
    blockers = []
    for f in flags:
        if "error" in f["filename"].lower() or "block" in f["content_preview"].lower():
            blockers.append({"file": f["filename"], "preview": f["content_preview"][:200]})

    # File hashes for key outputs
    file_hashes = {}
    for p in _ROOT.glob("**/*.py"):
        if p.parent.name in ("__pycache__", ".git"):
            continue
        rel = str(p.relative_to(_ROOT))
        if p.stat().st_size < 200_000:
            file_hashes[rel] = _file_hash(p)

    meta = {
        "generated": _now_human(),
        "project_root": str(_ROOT),
        "conversations": sessions_meta,
        "db_sessions": db_sessions,
        "context_offloads": [
            {"filename": o["filename"], "modified": o["modified"], "hash": o["hash"]}
            for o in offloads
        ],
        "state_flags": [
            {"filename": f["filename"], "size": f["size"], "hash": f["hash"]}
            for f in flags
        ],
        "blockers_detected": blockers,
        "key_file_hashes": file_hashes,
    }
    _OUT_METADATA.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _write_transcript(
    messages: list[dict],
    eve_journal: dict,
    offloads: list[dict],
    journal_heartbeats: list[dict],
) -> str:
    lines = [
        "# Mind EVE — Full Interaction Transcript",
        f"# Auto-generated by investigate_tool on {_now_human()}",
        "# Sources: mastermind.db · eve_journal.db · context_offload/*.md · journal.json",
        "#",
        "",
    ]

    # ── Mastermind conversations (real utterances)
    if messages and not any("_error" in m for m in messages):
        lines.append("═" * 70)
        lines.append("MASTERMIND CONVERSATIONS — Verbatim user ↔ assistant messages")
        lines.append("═" * 70)

        current_conv = None
        for m in messages:
            cid = m.get("conv_id", "?")
            if cid != current_conv:
                current_conv = cid
                title = m.get("conv_title") or cid
                lines += ["", f"─── Conversation: {title} ───", ""]

            ts = _ts_human(m["created_at"] / 1000) if m.get("created_at") else "?"
            role = (m.get("role") or "?").upper()
            text = (m.get("text") or "").strip()

            tools = []
            if m.get("tools_json"):
                try:
                    tj = json.loads(m["tools_json"])
                    tools = [t.get("name", "?") for t in tj if isinstance(t, dict)]
                except Exception:
                    pass

            lines.append(f"[{ts}] {role}")
            if tools:
                lines.append(f"  Tools: {', '.join(tools)}")
            lines.append(text if text else "[no text body]")
            lines.append("")

    # ── Eve journal entries (structured)
    if eve_journal.get("sessions"):
        lines += ["", "═" * 70, "EVE JOURNAL DB — Sessions and breadcrumbs", "═" * 70, ""]
        for s in eve_journal["sessions"]:
            lines.append(
                f"Session {s['id']} | started {_ts_human(s['started_at'])} | {s.get('note','')}"
            )
        if eve_journal.get("entries"):
            lines.append("\nJournal entries:")
            for e in eve_journal["entries"]:
                # Full content — no cap
                lines.append(f"  [{e.get('ts_human','?')}] {e.get('content','')}")
        if eve_journal.get("breadcrumbs"):
            lines.append("\nBreadcrumbs:")
            for b in eve_journal["breadcrumbs"]:
                # Full action_summary and outcome — no cap
                lines.append(
                    f"  step {b.get('step','?')} [{b.get('ts_human','?')}] "
                    f"{b.get('action_summary','')} → {b.get('outcome','')}"
                )

    # ── Context offloads (task checkpoints)
    if offloads:
        lines += ["", "═" * 70, "CONTEXT OFFLOAD CHECKPOINTS — Task progress snapshots", "═" * 70]
        for o in offloads:
            lines += [
                "",
                f"─── {o['filename']} [{o['modified']}] ───",
                o["content"].strip(),
                "",
            ]

    # ── Heartbeat journal — every entry, no sampling
    if journal_heartbeats:
        lines += ["", "═" * 70, "HEARTBEAT JOURNAL (journal.json)", "═" * 70, ""]
        lines.append(f"Total entries: {len(journal_heartbeats)}")
        lines.append("")
        for e in journal_heartbeats:
            lines.append(f"  [{e.get('ts','')}] {e.get('note','')}")

    content = "\n".join(lines)
    _OUT_TRANSCRIPT.write_text(content, encoding="utf-8")
    return content


def _quick_status(
    messages: list[dict],
    eve_journal: dict,
    offloads: list[dict],
    flags: list[dict],
    heartbeats: list[dict],
) -> str:
    conv_count = len({m.get("conv_id") for m in messages if "conv_id" in m})
    msg_count  = len([m for m in messages if "_error" not in m])
    sess_count = len(eve_journal.get("sessions", []))
    offload_count = len(offloads)
    flag_count = len(flags)
    heartbeat_count = len(heartbeats)

    last_offload = offloads[-1]["filename"] if offloads else "none"
    last_offload_ts = offloads[-1]["modified"] if offloads else "?"

    lines = [
        "=== Mind EVE — Investigate Status ===",
        f"  Conversations  : {conv_count}",
        f"  Messages       : {msg_count}",
        f"  DB sessions    : {sess_count}",
        f"  Context offloads: {offload_count}",
        f"  Memdir flags   : {flag_count}",
        f"  Heartbeats     : {heartbeat_count}",
        f"  Last checkpoint: {last_offload} @ {last_offload_ts}",
    ]

    # Check for unclean exit indicators
    lock_files = [f["filename"] for f in flags if f["filename"].endswith(".lock")]
    if lock_files:
        lines.append(f"  Lock files     : {', '.join(lock_files)}")

    # Check for BLOCKED/ERROR indicators
    error_logs = [
        f["filename"] for f in flags
        if "error" in f["content_preview"].lower() or "blocked" in f["content_preview"].lower()
    ]
    if error_logs:
        lines.append(f"  Possible issues: {', '.join(error_logs)}")

    return "\n".join(lines)


# ── Tool class ─────────────────────────────────────────────────────────────────

class InvestigateTool(BaseTool):
    name = "investigate"
    description = (
        "Self-investigation: reads mastermind.db, eve_journal.db, context offloads, "
        "logs, and memdir state files to reconstruct what happened and write "
        "task_log.md, metadata.json, and journal_transcript.md to memdir/. "
        "Call after unclean exits or when recovering session state."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "What to generate: 'full' (all three files), 'transcript', "
                    "'task_log', 'metadata', or 'status' (no writes, quick summary)."
                ),
                "enum": ["full", "transcript", "task_log", "metadata", "status"],
            }
        },
        "required": ["action"],
    }

    def execute(self, inp: dict) -> ToolResult:
        action = inp.get("action", "full")

        # Gather all evidence sources
        messages      = _read_mastermind()
        eve_journal   = _read_eve_journal()
        offloads      = _read_context_offloads()
        logs          = _read_logs()
        flags         = _read_memdir_flags()
        heartbeats    = _read_journal_json()

        if action == "status":
            result = _quick_status(messages, eve_journal, offloads, flags, heartbeats)
            return ToolResult(output=result)

        written = []

        if action in ("full", "task_log"):
            _write_task_log(messages, offloads, logs, flags)
            written.append(str(_OUT_TASK_LOG))

        if action in ("full", "metadata"):
            _write_metadata(messages, eve_journal, offloads, flags)
            written.append(str(_OUT_METADATA))

        if action in ("full", "transcript"):
            _write_transcript(messages, eve_journal, offloads, heartbeats)
            written.append(str(_OUT_TRANSCRIPT))

        summary = _quick_status(messages, eve_journal, offloads, flags, heartbeats)
        output = (
            f"investigate({action}) complete.\n"
            f"Files written:\n"
            + "\n".join(f"  {w}" for w in written)
            + f"\n\n{summary}"
        )
        return ToolResult(output=output)
