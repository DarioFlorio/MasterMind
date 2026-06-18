"""
services/daily_digest.py — Daily digest + scheduled automations + proactive surfacing.

GAPs IMPLEMENTED:
  1. Proactive surfacing — no cron, no timed insights.
     EVE now surfaces relevant insights proactively.
  2. Scheduled automations — daily digest, reminders, checks.
     EVE generates a daily digest and can be set up with reminders.

The digest is generated:
  - On startup if last digest was > DIGEST_INTERVAL_H hours ago
  - On demand via get_startup_digest()
  - Daily at DIGEST_TIME via the CronScheduler integration

Usage:
    from services.daily_digest import DailyDigest
    digest = DailyDigest(working_dir=WORKING_DIR)
    digest.set_llm(llm_call_fn)

    # On startup
    startup_msg = digest.get_startup_digest()
    if startup_msg:
        print(startup_msg)

    # Schedule the daily digest
    digest.schedule_daily()

    # Add a user reminder
    digest.add_reminder("Review Q3 report", "2026-06-01 09:00")
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.daily_digest")

DIGEST_INTERVAL_H = 20    # Generate at most once per 20 hours
REMINDER_CHECK_S  = 60    # Check reminders every minute
_DIGEST_FILE      = "memdir/last_digest.json"
_REMINDERS_FILE   = "memdir/reminders.json"


class DailyDigest:
    """
    Generates and manages daily digests and scheduled reminders.
    Reads from the memory journal/facts to build a structured recap.
    """

    def __init__(
        self,
        working_dir: str = "",
        llm_call_fn=None,
    ) -> None:
        self._base = Path(working_dir) if working_dir else Path.cwd()
        self._llm  = llm_call_fn
        self._digest_path    = self._base / _DIGEST_FILE
        self._reminders_path = self._base / _REMINDERS_FILE
        self._pending_reminders: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def set_llm(self, fn) -> None:
        self._llm = fn

    def get_startup_digest(self) -> str:
        """
        Returns a digest string if one is due (hasn't been generated recently).
        Returns "" if the digest is fresh.
        """
        last = self._load_last_digest()
        now  = time.time()
        if last and now - last.get("ts", 0) < DIGEST_INTERVAL_H * 3600:
            return ""  # fresh

        # Generate a new digest
        digest_text = self._generate_digest()
        if digest_text:
            self._save_digest(digest_text)
        return digest_text

    def force_digest(self) -> str:
        """Force-generate a digest regardless of timing."""
        digest_text = self._generate_digest()
        if digest_text:
            self._save_digest(digest_text)
        return digest_text

    def add_reminder(self, text: str, when: str = "") -> dict:
        """
        Add a reminder.
        when: ISO datetime string like "2026-06-01 09:00" or relative like "in 2 hours"
        Returns the reminder dict.
        """
        reminders = self._load_reminders()
        ts = self._parse_when(when)
        reminder = {
            "id":      f"rem_{int(time.time())}",
            "text":    text,
            "when":    when,
            "ts":      ts,
            "created": time.time(),
            "fired":   False,
        }
        reminders.append(reminder)
        self._save_reminders(reminders)
        log.info("Reminder added: %s at %s", text[:50], when)
        return reminder

    def check_reminders(self) -> list[str]:
        """
        Return list of reminder texts that are now due.
        Marks them as fired so they don't re-trigger.
        """
        reminders = self._load_reminders()
        now = time.time()
        due: list[str] = []
        changed = False

        for r in reminders:
            if not r.get("fired") and r.get("ts", 0) <= now:
                due.append(f"⏰ REMINDER: {r['text']}")
                r["fired"] = True
                changed = True

        if changed:
            self._save_reminders(reminders)
        return due

    def list_reminders(self) -> list[dict]:
        """Return all upcoming (non-fired) reminders."""
        return [r for r in self._load_reminders() if not r.get("fired")]

    def schedule_daily(self, time_str: str = "08:00") -> None:
        """
        Register a daily digest job with the CronScheduler.
        Call once after CronScheduler is available.
        """
        try:
            from tools.cron_tool import _global_scheduler
            if _global_scheduler is not None:
                _global_scheduler.add(
                    name="daily_digest",
                    schedule=f"every day at {time_str}",
                    command="__daily_digest__",
                )
                log.info("Daily digest scheduled at %s", time_str)
        except Exception as exc:
            log.debug("schedule_daily: %s", exc)

    def get_pending_reminders_for_startup(self) -> list[str]:
        """Return any overdue reminders to surface on startup."""
        return self.check_reminders()

    # ── Digest generation ─────────────────────────────────────────────────────

    def _generate_digest(self) -> str:
        """Build the daily digest from journal, facts, and memory tiers."""
        parts: list[str] = []

        # Source 1: journal entries from the last 24 hours
        try:
            from memory.manager import _load_journal, _load_facts
            journal = _load_journal()
            facts   = _load_facts()
            now     = datetime.now()
            yesterday = now - timedelta(hours=24)

            recent_entries = []
            for entry in journal[-30:]:
                ts_str = entry.get("ts", "")
                try:
                    ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M")
                    if ts >= yesterday:
                        recent_entries.append(entry["note"])
                except Exception:
                    recent_entries.append(entry["note"])

            if recent_entries:
                parts.append("Recent activity:\n" + "\n".join(f"• {e}" for e in recent_entries[-10:]))

            if facts:
                key_facts = list(facts.values())[-5:]
                parts.append("Key facts on record:\n" + "\n".join(
                    f"• {v['content'][:150]}" for v in key_facts
                ))
        except Exception as exc:
            log.debug("Digest journal load: %s", exc)

        # Source 2: ThreeTierMemory semantic tier (stable facts)
        try:
            from memory.three_tier import ThreeTierMemory
            mem_dir = str(self._base / "memdir" / "three_tier")
            # Quick read-only access
            import chromadb
            client = chromadb.PersistentClient(path=mem_dir)
            sem_col = client.get_or_create_collection("semantic_memory")
            if sem_col.count() > 0:
                res = sem_col.get(include=["documents"])
                docs = res.get("documents", [])
                if docs:
                    parts.append("Semantic knowledge:\n" + "\n".join(
                        f"• {d[:200]}" for d in docs[-5:]
                    ))
        except Exception as exc:
            log.debug("Digest semantic tier: %s", exc)

        if not parts:
            return ""

        blob = "\n\n".join(parts)

        # If LLM available, produce a polished digest
        if self._llm:
            try:
                prompt = (
                    "Generate a concise daily digest for the EVE AI agent based on "
                    "the following memory data. Format as:\n"
                    "**Daily Digest — {date}**\n"
                    "• [3-5 bullet points: key accomplishments, open tasks, important facts]\n"
                    "Keep it under 150 words. Date: "
                    f"{datetime.now().strftime('%Y-%m-%d')}.\n\n"
                    f"Memory data:\n{blob[:2000]}"
                )
                result = self._llm(prompt, temperature=0.3, max_tokens=300)
                if isinstance(result, str) and result.strip():
                    return result.strip()
            except Exception as exc:
                log.debug("LLM digest: %s", exc)

        # Fallback: raw digest without LLM
        date_str = datetime.now().strftime("%Y-%m-%d")
        return f"**Daily Digest — {date_str}**\n\n{blob[:500]}"

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_last_digest(self) -> dict:
        try:
            if self._digest_path.exists():
                return json.loads(self._digest_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return {}

    def _save_digest(self, text: str) -> None:
        try:
            self._digest_path.parent.mkdir(parents=True, exist_ok=True)
            self._digest_path.write_text(
                json.dumps({"ts": time.time(), "text": text}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Save digest: %s", exc)

    def _load_reminders(self) -> list[dict]:
        try:
            if self._reminders_path.exists():
                return json.loads(self._reminders_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return []

    def _save_reminders(self, reminders: list[dict]) -> None:
        try:
            self._reminders_path.parent.mkdir(parents=True, exist_ok=True)
            self._reminders_path.write_text(
                json.dumps(reminders, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            log.debug("Save reminders: %s", exc)

    def _parse_when(self, when: str) -> float:
        """Parse a time string into a Unix timestamp."""
        if not when:
            return time.time() + 3600  # default: 1 hour from now

        when = when.strip().lower()

        # Relative: "in 2 hours", "in 30 minutes", "in 1 day"
        import re
        m = re.match(r"in\s+(\d+)\s+(minute|hour|day|week)s?", when)
        if m:
            n    = int(m.group(1))
            unit = m.group(2)
            mult = {"minute": 60, "hour": 3600, "day": 86400, "week": 604800}[unit]
            return time.time() + n * mult

        # Absolute: "2026-06-01 09:00" or "2026-06-01"
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M"):
            try:
                dt = datetime.strptime(when, fmt)
                if fmt == "%H:%M":
                    now = datetime.now()
                    dt  = dt.replace(year=now.year, month=now.month, day=now.day)
                    if dt < now:
                        dt += timedelta(days=1)
                return dt.timestamp()
            except ValueError:
                pass

        return time.time() + 3600  # fallback: 1 hour
