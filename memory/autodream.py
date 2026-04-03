"""
memory/autodream.py — AutoDream: idle-time memory consolidation (Claude Code leak feature).

AutoDream runs as a background daemon thread. When EVE has been idle for
IDLE_TRIGGER_S seconds it "dreams" — scanning the journal/facts store,
deduplicating entries, merging related facts, and writing a consolidated
summary back to persistent memory.

The leak described AutoDream as:
  "Dreams during idle times by scanning, deduplicating, and updating stored
   information to maintain structured memory without active user input."

This implementation:
  1. Monitors last-activity timestamp (ping() called on every user turn)
  2. After IDLE_TRIGGER_S of silence, spawns a dream cycle
  3. Dream cycle:
     a. Load all journal entries + facts
     b. Deduplicate near-identical entries (Jaccard similarity)
     c. Cluster related entries by keyword overlap
     d. Write a consolidated "dream summary" back to journal
     e. Prune entries older than MAX_AGE_DAYS if we're over MAX_ENTRIES cap
  4. Cooldown DREAM_COOLDOWN_S before next dream (won't spam on long idle)

Usage in main.py:
    from memory.autodream import AutoDream
    dream = AutoDream(memory_manager)
    dream.start()          # starts background thread
    # ... in input loop:
    dream.ping()           # reset idle timer on each user message
    dream.stop()           # on exit
"""
from __future__ import annotations
import json
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("memory.autodream")

# ── Config ────────────────────────────────────────────────────────────────────
IDLE_TRIGGER_S   = 120    # seconds of silence before dreaming
DREAM_COOLDOWN_S = 600    # minimum gap between dream cycles
MAX_ENTRIES      = 200    # prune journal when it exceeds this
MAX_AGE_DAYS     = 30     # purge entries older than N days if over cap
SIM_THRESHOLD    = 0.65   # Jaccard similarity threshold for dedup


class AutoDream:
    """
    Idle-time memory consolidation daemon.
    Attach to the existing memory manager; call ping() on every user turn.
    """

    def __init__(self, mem_dir: Path | str | None = None):
        self._mem_dir    = Path(mem_dir) if mem_dir else Path(__file__).parent.parent / "memdir"
        self._last_ping  = time.time()
        self._last_dream = 0.0
        self._stop_evt   = threading.Event()
        self._thread: threading.Thread | None = None
        self._dreaming   = False
        self._dream_count = 0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="autodream"
        )
        self._thread.start()
        log.info("AutoDream started (idle_trigger=%ds)", IDLE_TRIGGER_S)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3)
        log.info("AutoDream stopped. Dreams completed: %d", self._dream_count)

    def ping(self) -> None:
        """Call on every user interaction to reset the idle clock."""
        self._last_ping = time.time()

    def force_dream(self) -> str:
        """Manually trigger a dream cycle and return the summary."""
        return self._dream_cycle()

    def status(self) -> str:
        idle  = time.time() - self._last_ping
        since = time.time() - self._last_dream if self._last_dream else -1
        return (
            f"AutoDream | idle={idle:.0f}s | "
            f"last_dream={'never' if since < 0 else f'{since:.0f}s ago'} | "
            f"cycles={self._dream_count} | "
            f"{'💤 dreaming' if self._dreaming else '👁 watching'}"
        )

    # ── Watch loop ─────────────────────────────────────────────────────────

    def _watch_loop(self) -> None:
        while not self._stop_evt.wait(timeout=15):
            idle   = time.time() - self._last_ping
            cd_ok  = (time.time() - self._last_dream) >= DREAM_COOLDOWN_S
            if idle >= IDLE_TRIGGER_S and cd_ok and not self._dreaming:
                log.info("AutoDream: idle for %.0fs — starting dream cycle", idle)
                try:
                    self._dreaming = True
                    summary = self._dream_cycle()
                    self._dream_count += 1
                    self._last_dream = time.time()
                    log.info("AutoDream: dream complete — %s", summary[:120])
                except Exception as e:
                    log.error("AutoDream: dream cycle error: %s", e)
                finally:
                    self._dreaming = False

    # ── Dream cycle ────────────────────────────────────────────────────────

    def _dream_cycle(self) -> str:
        steps = []

        # 1. Load journal
        journal = self._load_journal()
        facts   = self._load_facts()
        steps.append(f"loaded {len(journal)} journal entries, {len(facts)} facts")

        if len(journal) < 3 and len(facts) < 3:
            return "nothing to consolidate"

        # 2. Deduplicate journal entries
        deduped, removed = _dedup_entries(journal)
        steps.append(f"deduped: removed {removed} near-duplicate entries")

        # 3. Cluster and summarise
        clusters = _cluster_entries(deduped)
        steps.append(f"formed {len(clusters)} topic clusters")

        # 4. Build dream summary
        dream_note = _build_dream_note(clusters, facts)

        # 5. Prune if over cap
        pruned = 0
        if len(deduped) > MAX_ENTRIES:
            cutoff = time.time() - (MAX_AGE_DAYS * 86400)
            before = len(deduped)
            deduped = [e for e in deduped if e.get("ts", 0) >= cutoff]
            pruned = before - len(deduped)
            steps.append(f"pruned {pruned} entries older than {MAX_AGE_DAYS}d")

        # 6. Write back consolidated journal + dream entry
        deduped.append({
            "ts":   time.strftime("%Y-%m-%d %H:%M"),
            "note": f"[AutoDream #{self._dream_count + 1}] {dream_note}",
        })
        self._save_journal(deduped)

        # 7. Deduplicate facts store
        deduped_facts, f_removed = _dedup_facts(facts)
        if f_removed:
            self._save_facts(deduped_facts)
            steps.append(f"deduped {f_removed} redundant facts")

        summary = " | ".join(steps)
        return summary

    # ── Persistence helpers ────────────────────────────────────────────────

    def _load_journal(self) -> list[dict]:
        p = self._mem_dir / "journal.json"
        if not p.exists():
            return []
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save_journal(self, entries: list[dict]) -> None:
        self._mem_dir.mkdir(exist_ok=True)
        p = self._mem_dir / "journal.json"
        p.write_text(json.dumps(entries[-MAX_ENTRIES:], indent=2), encoding="utf-8")

    def _load_facts(self) -> dict:
        p = self._mem_dir / "facts.json"
        if not p.exists():
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_facts(self, facts: dict) -> None:
        self._mem_dir.mkdir(exist_ok=True)
        p = self._mem_dir / "facts.json"
        p.write_text(json.dumps(facts, indent=2), encoding="utf-8")


# ── Dedup helpers (pure functions) ────────────────────────────────────────────

def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dedup_entries(entries: list[dict]) -> tuple[list[dict], int]:
    """Remove near-duplicate journal entries by Jaccard similarity."""
    kept: list[dict] = []
    removed = 0
    for entry in entries:
        note = entry.get("note", "")
        toks = _tokenise(note)
        is_dup = False
        for k in kept[-20:]:   # only compare against recent 20 to keep O(n) manageable
            if _jaccard(toks, _tokenise(k.get("note", ""))) >= SIM_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            kept.append(entry)
        else:
            removed += 1
    return kept, removed


def _cluster_entries(entries: list[dict], top_n: int = 6) -> list[list[dict]]:
    """
    Simple greedy clustering: group entries that share ≥2 keywords.
    Returns up to top_n clusters.
    """
    if not entries:
        return []
    clusters: list[list[dict]] = []
    assigned = set()
    for i, entry in enumerate(entries):
        if i in assigned:
            continue
        toks_i = _tokenise(entry.get("note", ""))
        cluster = [entry]
        assigned.add(i)
        for j, other in enumerate(entries):
            if j in assigned:
                continue
            toks_j = _tokenise(other.get("note", ""))
            if len(toks_i & toks_j) >= 2:
                cluster.append(other)
                assigned.add(j)
        clusters.append(cluster)
        if len(clusters) >= top_n:
            break
    return clusters


def _build_dream_note(clusters: list[list[dict]], facts: dict) -> str:
    parts = []
    for i, cluster in enumerate(clusters[:4]):
        # Use most recent entry as cluster representative
        rep = cluster[-1].get("note", "")[:100]
        parts.append(f"[topic {i+1}: {len(cluster)} entries] {rep}")
    fact_count = len(facts)
    parts.append(f"[facts: {fact_count} stored]")
    return " | ".join(parts)


def _dedup_facts(facts: dict) -> tuple[dict, int]:
    """Merge facts whose content is near-identical."""
    keys   = list(facts.keys())
    remove = set()
    for i in range(len(keys)):
        if keys[i] in remove:
            continue
        ci = _tokenise(facts[keys[i]].get("content", ""))
        for j in range(i + 1, len(keys)):
            if keys[j] in remove:
                continue
            cj = _tokenise(facts[keys[j]].get("content", ""))
            if _jaccard(ci, cj) >= SIM_THRESHOLD:
                # Keep newer entry
                ti = facts[keys[i]].get("saved", "")
                tj = facts[keys[j]].get("saved", "")
                remove.add(keys[i] if ti < tj else keys[j])
    deduped = {k: v for k, v in facts.items() if k not in remove}
    return deduped, len(remove)
