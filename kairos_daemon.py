#!/usr/bin/env python3
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
    todo_re = re.compile(r"\b(TODO|FIXME|todo|fixme|need to|should|must|remember to)\b")
    todo_entries = [e.get("note","") for e in journal[-50:] if todo_re.search(e.get("note",""))]
    if todo_entries:
        sample = todo_entries[-1][:80]
        insights.append(f"📋 Unresolved TODO detected in recent journal: '{sample}'")

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

    # ── Heuristic 3: Facts that haven't been accessed recently ──────────
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
