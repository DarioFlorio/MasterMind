"""
kairos.py — Kairos: persistent background daemon (Claude Code leak feature).

From the leak:
  "A persistent background daemon that runs even when the terminal is closed,
   using periodic 'tick' prompts to surface important insights proactively."

Kairos runs as a separate OS-level process (not just a thread) so it persists
after the terminal/main process exits. It uses a lock file + state file to
coordinate with the main EVE process.

Architecture:
  ┌─────────────┐   spawn once   ┌──────────────────────────────┐
  │  main.py    │ ─────────────► │  kairos_daemon.py (subprocess)│
  │  (terminal) │                │  runs as detached process     │
  └─────────────┘                │  writes insights to          │
       ▲                         │  memdir/kairos_insights.json  │
       │  reads on next boot     └──────────────────────────────┘
       └──────────────────────────────────────────────────────────

The daemon:
  - Reads pending journal entries every TICK_INTERVAL_S
  - Runs lightweight pattern checks (no LLM needed — pure heuristic)
  - Surfaces "insights" — things Dario should know when he returns
  - Writes insights to kairos_insights.json
  - Main process reads + displays insights on boot

Usage in main.py:
    from kairos import Kairos
    k = Kairos()
    k.ensure_running()          # spawn daemon if not running
    insights = k.pop_insights() # read + clear pending insights on boot
    k.push_context(session_summary)  # give daemon context at end of session
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

log = logging.getLogger("kairos")

# ── Config ────────────────────────────────────────────────────────────────────
TICK_INTERVAL_S  = 300     # daemon checks every 5 minutes
MAX_INSIGHTS     = 20      # cap stored insights
DAEMON_TIMEOUT_S = 10      # seconds to wait for daemon to start


class Kairos:
    """
    Persistent background daemon coordinator.
    Call ensure_running() on boot and pop_insights() to retrieve proactive tips.
    """

    def __init__(self, mem_dir: Path | str | None = None):
        self._mem_dir      = Path(mem_dir) if mem_dir else Path(__file__).parent / "memdir"
        self._lock_file    = self._mem_dir / "kairos.pid"
        self._insights_file= self._mem_dir / "kairos_insights.json"
        self._context_file = self._mem_dir / "kairos_context.json"
        self._mem_dir.mkdir(exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    def ensure_running(self) -> bool:
        """Spawn the Kairos daemon if it is not already running. Returns True if running."""
        if self._is_running():
            log.debug("Kairos daemon already running (pid=%s)", self._read_pid())
            return True
        return self._spawn()

    def pop_insights(self) -> list[str]:
        """Read pending insights and clear the queue. Returns list of insight strings."""
        if not self._insights_file.exists():
            return []
        try:
            data = json.loads(self._insights_file.read_text(encoding="utf-8"))
            insights = data.get("insights", [])
            # Clear after reading
            self._insights_file.write_text(
                json.dumps({"insights": [], "last_cleared": time.strftime("%Y-%m-%d %H:%M")}),
                encoding="utf-8"
            )
            return insights
        except Exception:
            return []

    def push_context(self, summary: str) -> None:
        """
        Feed the daemon a session summary so its next tick has fresh context.
        Called at end of each conversation turn.
        """
        try:
            existing = {}
            if self._context_file.exists():
                existing = json.loads(self._context_file.read_text(encoding="utf-8"))
            existing["last_summary"] = summary[:500]
            existing["last_push"]    = time.strftime("%Y-%m-%d %H:%M")
            self._context_file.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        except Exception:
            pass

    def stop(self) -> None:
        """Send SIGTERM to the daemon process."""
        pid = self._read_pid()
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
                log.info("Kairos daemon stopped (pid=%d)", pid)
            except ProcessLookupError:
                pass
            self._lock_file.unlink(missing_ok=True)

    def status(self) -> str:
        running = self._is_running()
        pid     = self._read_pid()
        n_insights = 0
        if self._insights_file.exists():
            try:
                n_insights = len(json.loads(self._insights_file.read_text()).get("insights", []))
            except Exception:
                pass
        return (
            f"Kairos | {'🟢 running' if running else '🔴 stopped'} | "
            f"pid={pid or 'n/a'} | pending_insights={n_insights}"
        )

    # ── Internal ───────────────────────────────────────────────────────────

    def _read_pid(self) -> int | None:
        if not self._lock_file.exists():
            return None
        try:
            return int(self._lock_file.read_text().strip())
        except Exception:
            return None

    def _is_running(self) -> bool:
        pid = self._read_pid()
        if not pid:
            return False
        try:
            os.kill(pid, 0)   # signal 0 = existence check
            return True
        except ProcessLookupError:
            self._lock_file.unlink(missing_ok=True)
            return False
        except PermissionError:
            return True  # process exists but we don't own it

    def _spawn(self) -> bool:
        daemon_script = Path(__file__).parent / "kairos_daemon.py"
        if not daemon_script.exists():
            log.warning("Kairos daemon script not found: %s", daemon_script)
            return False
        try:
            kwargs: dict = dict(
                args=[sys.executable, str(daemon_script), str(self._mem_dir)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Detach from terminal on all platforms
            if sys.platform == "win32":
                kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP |
                    subprocess.DETACHED_PROCESS
                )
            else:
                kwargs["start_new_session"] = True

            proc = subprocess.Popen(**kwargs)
            # Brief wait for daemon to write its PID
            for _ in range(20):
                time.sleep(0.25)
                if self._is_running():
                    log.info("Kairos daemon spawned (pid=%d)", proc.pid)
                    return True
            log.warning("Kairos daemon spawned but PID file not written in time")
            return False
        except Exception as e:
            log.error("Failed to spawn Kairos daemon: %s", e)
            return False


# ─────────────────────────────────────────────────────────────────────────────
# kairos_daemon.py logic — embedded here so it can write itself to disk
# ─────────────────────────────────────────────────────────────────────────────

DAEMON_SOURCE = '''#!/usr/bin/env python3
"""
kairos_daemon.py — Kairos background daemon process.
Spawned by kairos.py; runs independently of the terminal.
Reads journal/facts, surfaces insights, writes them for main process to read.
"""
import json, os, re, signal, sys, time
from pathlib import Path

def _write_pid(mem_dir):
    (mem_dir / "kairos.pid").write_text(str(os.getpid()))

def _load_json(p):
    try: return json.loads(p.read_text(encoding="utf-8"))
    except: return None

def _save_insights(mem_dir, insights):
    p = mem_dir / "kairos_insights.json"
    existing = _load_json(p) or {"insights": []}
    all_i = existing.get("insights", []) + insights
    all_i = all_i[-20:]  # cap
    p.write_text(json.dumps({"insights": all_i, "updated": time.strftime("%Y-%m-%d %H:%M")}, indent=2))

def _tick(mem_dir):
    """One tick: scan memory and generate insights."""
    insights = []

    # Read journal
    journal_path = mem_dir / "journal.json"
    journal = _load_json(journal_path) or []

    # Read context from main process
    ctx_path = mem_dir / "kairos_context.json"
    context  = _load_json(ctx_path) or {}

    # Read facts
    facts_path = mem_dir / "facts.json"
    facts = _load_json(facts_path) or {}

    now = time.time()

    # ── Heuristic 1: Unresolved TODOs in journal ────────────────────────
    todo_re = re.compile(r"\\b(TODO|FIXME|todo|fixme|need to|should|must|remember to)\\b")
    todo_entries = [e.get("note","") for e in journal[-50:] if todo_re.search(e.get("note",""))]
    if todo_entries:
        sample = todo_entries[-1][:80]
        insights.append(f"📋 Unresolved TODO detected in recent journal: \'{sample}\'")

    # ── Heuristic 2: Long idle since last push ───────────────────────────
    last_push = context.get("last_push", "")
    if last_push:
        try:
            from datetime import datetime
            lp = datetime.strptime(last_push, "%Y-%m-%d %H:%M")
            idle_h = (datetime.now() - lp).total_seconds() / 3600
            if idle_h > 8:
                insights.append(f"⏰ EVE has been idle for {idle_h:.1f}h. Last summary: {context.get('last_summary','(none)')[:60]}")
        except Exception:
            pass

    # ── Heuristic 3: Facts that haven\'t been accessed recently ──────────
    if len(facts) > 10:
        old_keys = sorted(facts, key=lambda k: facts[k].get("saved",""))[:3]
        for k in old_keys:
            insights.append(f"📚 Stale fact: [{k}] → {facts[k].get('content','')[:60]}")

    # ── Heuristic 4: Session summary surfacing ───────────────────────────
    last_summary = context.get("last_summary","")
    if last_summary and last_summary not in [i for i in insights]:
        insights.append(f"💡 Last session note: {last_summary[:100]}")

    return insights


def main():
    if len(sys.argv) < 2:
        mem_dir = Path(__file__).parent / "memdir"
    else:
        mem_dir = Path(sys.argv[1])
    mem_dir.mkdir(exist_ok=True)

    _write_pid(mem_dir)

    def _shutdown(sig, frame):
        (mem_dir / "kairos.pid").unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    try:
        signal.signal(signal.SIGINT, _shutdown)
    except Exception:
        pass

    tick_interval = int(os.environ.get("KAIROS_TICK_S", "300"))

    while True:
        try:
            insights = _tick(mem_dir)
            if insights:
                _save_insights(mem_dir, insights)
        except Exception:
            pass
        time.sleep(tick_interval)


if __name__ == "__main__":
    main()
'''


def write_daemon_script(dest: Path | None = None) -> Path:
    """Write kairos_daemon.py next to kairos.py (call once on first run)."""
    target = dest or (Path(__file__).parent / "kairos_daemon.py")
    if not target.exists():
        target.write_text(DAEMON_SOURCE, encoding="utf-8")
        log.info("Kairos daemon script written to %s", target)
    return target
