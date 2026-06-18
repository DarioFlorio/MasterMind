"""
utils/db.py — SQLite persistence layer for MasterMind
======================================================
Tables
  conversations  — id, title, model, created_at, updated_at
  messages       — conv_id FK, role, text, think, image_data_url, tools_json
  settings       — key, JSON value

Uses WAL mode + per-thread connections for safe concurrent access.
"""
from __future__ import annotations
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

_DB_PATH = Path(__file__).parent.parent / "data" / "mastermind.db"
_local   = threading.local()


# ── Connection ─────────────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA synchronous=NORMAL")
        _local.conn = c
    return _local.conn


# ── Schema ────────────────────────────────────────────────────────────────────

def init() -> None:
    _conn().executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT    PRIMARY KEY,
            title      TEXT    NOT NULL DEFAULT 'New conversation',
            model      TEXT    DEFAULT '',
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            conv_id       TEXT    NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            role          TEXT    NOT NULL,
            text          TEXT    NOT NULL DEFAULT '',
            think         TEXT    DEFAULT '',
            image_data_url TEXT   DEFAULT '',
            tools_json    TEXT    DEFAULT '[]',
            created_at    INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conv_id, created_at);

        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT    PRIMARY KEY,
            value      TEXT    NOT NULL,
            updated_at INTEGER NOT NULL
        );
    """)
    _conn().commit()


# ── Conversations ─────────────────────────────────────────────────────────────

def list_conversations() -> list[dict]:
    rows = _conn().execute(
        "SELECT id, title, model, created_at, updated_at "
        "FROM conversations ORDER BY updated_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def create_conversation(id: str, title: str = "New conversation",
                        model: str = "") -> dict:
    now = _now()
    _conn().execute(
        "INSERT OR IGNORE INTO conversations "
        "(id, title, model, created_at, updated_at) VALUES (?,?,?,?,?)",
        (id, title, model, now, now),
    )
    _conn().commit()
    return {"id": id, "title": title, "model": model,
            "created_at": now, "updated_at": now}


def update_conversation(id: str, title: str | None = None,
                        model: str | None = None) -> None:
    now = _now()
    if title is not None:
        _conn().execute(
            "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
            (title, now, id),
        )
    if model is not None:
        _conn().execute(
            "UPDATE conversations SET model=?, updated_at=? WHERE id=?",
            (model, now, id),
        )
    _conn().commit()


def touch_conversation(id: str) -> None:
    _conn().execute(
        "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), id)
    )
    _conn().commit()


def delete_conversation(id: str) -> None:
    _conn().execute("DELETE FROM conversations WHERE id=?", (id,))
    _conn().commit()


# ── Messages ──────────────────────────────────────────────────────────────────

def get_messages(conv_id: str) -> list[dict]:
    rows = _conn().execute(
        "SELECT id, role, text, think, image_data_url, tools_json, created_at "
        "FROM messages WHERE conv_id=? ORDER BY created_at ASC",
        (conv_id,),
    ).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["tools"] = json.loads(d.pop("tools_json", "[]") or "[]")
        result.append(d)
    return result


def add_message(
    conv_id: str,
    role: str,
    text: str = "",
    think: str = "",
    image_data_url: str = "",
    tools: list | None = None,
) -> dict:
    now = _now()
    cur = _conn().execute(
        "INSERT INTO messages "
        "(conv_id, role, text, think, image_data_url, tools_json, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (conv_id, role, text, think or "", image_data_url or "",
         json.dumps(tools or []), now),
    )
    _conn().commit()
    touch_conversation(conv_id)
    return {
        "id": cur.lastrowid, "role": role, "text": text,
        "think": think, "image_data_url": image_data_url,
        "tools": tools or [], "created_at": now,
    }


# ── Settings ──────────────────────────────────────────────────────────────────

def get_settings() -> dict[str, Any]:
    rows = _conn().execute("SELECT key, value FROM settings").fetchall()
    result: dict[str, Any] = {}
    for r in rows:
        try:
            result[r["key"]] = json.loads(r["value"])
        except Exception:
            result[r["key"]] = r["value"]
    return result


def save_settings(updates: dict[str, Any]) -> None:
    now = _now()
    for k, v in updates.items():
        _conn().execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?,?,?)",
            (k, json.dumps(v), now),
        )
    _conn().commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> int:
    return int(time.time() * 1000)