"""
memory/manager.py — Persistent cross-session memory backed by a JSON file.
Keeps a rolling journal of sessions and a key/value fact store.
"""
from __future__ import annotations
import json, time
from pathlib import Path

_MEM_DIR  = Path(__file__).parent.parent / "memdir"
_JOURNAL  = _MEM_DIR / "journal.json"
_FACTS    = _MEM_DIR / "facts.json"
_MAX_ENTRIES = 200   # keep last N journal entries


def _ensure() -> None:
    _MEM_DIR.mkdir(exist_ok=True)


def _load_journal() -> list[dict]:
    _ensure()
    try:
        return json.loads(_JOURNAL.read_text()) if _JOURNAL.exists() else []
    except Exception:
        return []


def _save_journal(entries: list[dict]) -> None:
    _ensure()
    _JOURNAL.write_text(json.dumps(entries[-_MAX_ENTRIES:], indent=2))


def _load_facts() -> dict:
    _ensure()
    try:
        return json.loads(_FACTS.read_text()) if _FACTS.exists() else {}
    except Exception:
        return {}


def _save_facts(facts: dict) -> None:
    _ensure()
    _FACTS.write_text(json.dumps(facts, indent=2))


# ── Public API ─────────────────────────────────────────────────────────────────

def append_session(note: str) -> None:
    """Append a session note to the journal."""
    entries = _load_journal()
    entries.append({"ts": time.strftime("%Y-%m-%d %H:%M"), "note": note[:500]})
    _save_journal(entries)


def save_fact(key: str, content: str) -> None:
    """Save a key/value fact to persistent memory."""
    facts = _load_facts()
    facts[key] = {"content": content, "saved": time.strftime("%Y-%m-%d %H:%M")}
    _save_facts(facts)


def load_fact(key: str) -> str:
    """Load a specific fact by key."""
    facts = _load_facts()
    entry = facts.get(key)
    if not entry:
        return ""
    return f"[{key}] ({entry.get('saved','?')}): {entry['content']}"


def load_context(max_entries: int = 20) -> str:
    """Load memory context for injection into the system prompt."""
    journal = _load_journal()
    facts   = _load_facts()

    parts: list[str] = []

    if facts:
        parts.append("## Remembered Facts")
        for k, v in list(facts.items())[-20:]:
            parts.append(f"- [{k}]: {v['content'][:200]}")

    if journal:
        recent = journal[-max_entries:]
        parts.append("\n## Recent Session Notes")
        for e in recent:
            parts.append(f"- {e['ts']}: {e['note']}")

    return "\n".join(parts) if parts else ""


def clear_all() -> None:
    """Wipe all memory."""
    if _JOURNAL.exists():
        _JOURNAL.unlink()
    if _FACTS.exists():
        _FACTS.unlink()
