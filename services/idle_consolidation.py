"""
services/idle_consolidation.py — Active memory curation background service.

GAP IMPLEMENTED: Active memory curation — no idle consolidation pass.
EVE now runs a background service that curates memory when idle, preventing
stale or redundant data from accumulating.

What it does every IDLE_TRIGGER_S seconds of silence:
  1. Triggers ThreeTierMemory working→episodic→semantic consolidation
  2. Runs AutoDream deduplication and clustering
  3. Prunes stale working-memory entries
  4. Computes and caches an insight digest for proactive surfacing

Usage in main.py / webui.py:
    from services.idle_consolidation import IdleConsolidation
    consolidation = IdleConsolidation(working_dir=WORKING_DIR)
    consolidation.start()

    # Call on every user message to reset idle timer
    consolidation.ping()

    # On shutdown
    consolidation.stop()

    # To get the latest proactive insight (inject on next message)
    insight = consolidation.pop_insight()
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.idle_consolidation")

# ── Config ────────────────────────────────────────────────────────────────────
IDLE_TRIGGER_S    = 90     # seconds of silence before consolidating
CONSOLIDATION_GAP = 600    # minimum gap between consolidation cycles (10 min)
PRUNE_WORKING_AGE = 3600   # prune working-memory entries older than 1 hour


class IdleConsolidation:
    """
    Background daemon that runs memory curation when EVE is idle.

    Combines AutoDream (deduplication) + ThreeTierMemory (tier promotion)
    into a single orchestrated idle service.
    """

    def __init__(
        self,
        working_dir: str = "",
        three_tier_memory=None,  # ThreeTierMemory instance
        llm_call_fn=None,
    ) -> None:
        self._working_dir = Path(working_dir) if working_dir else Path.cwd()
        self._mem_dir     = self._working_dir / "memdir"
        self._three_tier  = three_tier_memory   # set later via set_memory()
        self._llm         = llm_call_fn
        self._last_ping   = time.time()
        self._last_consolidate = 0.0
        self._stop_evt    = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_insight: str = ""
        self._lock = threading.Lock()
        self._autodream = None  # lazy init

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def set_memory(self, three_tier_memory) -> None:
        """Inject the ThreeTierMemory instance after startup."""
        self._three_tier = three_tier_memory
        if self._autodream is None:
            try:
                from memory.autodream import AutoDream
                self._autodream = AutoDream(mem_dir=str(self._mem_dir))
            except Exception as exc:
                log.warning("AutoDream init failed: %s", exc)

    def set_llm(self, fn) -> None:
        """Inject LLM call function for insight generation."""
        self._llm = fn
        if self._three_tier:
            self._three_tier.set_llm(fn)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        # Init AutoDream
        if self._autodream is None:
            try:
                from memory.autodream import AutoDream
                self._autodream = AutoDream(mem_dir=str(self._mem_dir))
            except Exception as exc:
                log.warning("AutoDream init failed: %s", exc)

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="idle_consolidation",
        )
        self._thread.start()
        log.info("IdleConsolidation started (trigger=%ds)", IDLE_TRIGGER_S)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("IdleConsolidation stopped")

    def ping(self) -> None:
        """Reset idle timer. Call on every user message."""
        self._last_ping = time.time()
        if self._autodream:
            try:
                self._autodream.ping()
            except Exception:
                pass

    def pop_insight(self) -> str:
        """
        Return and clear the latest proactive insight.
        Returns "" if nothing is pending.
        Called by query_engine before each assistant turn.
        """
        with self._lock:
            insight = self._pending_insight
            self._pending_insight = ""
        return insight

    # ── Internal ──────────────────────────────────────────────────────────────

    def _watch_loop(self) -> None:
        while not self._stop_evt.is_set():
            time.sleep(15)
            idle = time.time() - self._last_ping
            gap  = time.time() - self._last_consolidate

            if idle >= IDLE_TRIGGER_S and gap >= CONSOLIDATION_GAP:
                log.info("Idle consolidation triggered (idle=%.0fs)", idle)
                self._consolidate()
                self._last_consolidate = time.time()

    def _consolidate(self) -> None:
        """Run full consolidation cycle."""
        # 1. AutoDream — dedup + cluster journal/facts
        if self._autodream:
            try:
                self._autodream._dream_cycle()
                log.debug("AutoDream cycle complete")
            except Exception as exc:
                log.debug("AutoDream error: %s", exc)

        # 2. ThreeTierMemory — force tier promotion
        if self._three_tier and self._three_tier._ready:
            try:
                fake_turn = int(time.time() / 60)  # synthetic turn count
                self._three_tier.maybe_consolidate(fake_turn)
                log.debug("ThreeTierMemory consolidation triggered at turn=%d", fake_turn)
            except Exception as exc:
                log.debug("ThreeTierMemory error: %s", exc)

        # 3. Prune stale working memory
        self._prune_stale_working()

        # 4. Generate proactive insight
        self._generate_insight()

    def _prune_stale_working(self) -> None:
        """Remove working-memory entries older than PRUNE_WORKING_AGE."""
        if not (self._three_tier and self._three_tier._ready):
            return
        try:
            col = self._three_tier._working
            if col is None or col.count() == 0:
                return
            cutoff = time.time() - PRUNE_WORKING_AGE
            results = col.get(include=["metadatas", "ids"])
            ids_to_delete = [
                doc_id
                for doc_id, meta in zip(results["ids"], results["metadatas"])
                if meta.get("ts", time.time()) < cutoff
            ]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
                log.info("Pruned %d stale working-memory entries", len(ids_to_delete))
        except Exception as exc:
            log.debug("Prune working: %s", exc)

    def _generate_insight(self) -> None:
        """
        Scan recent memory for a proactive insight to surface to the user.
        Stores result in _pending_insight for pop_insight() to return.

        GAP: Proactive surfacing — no cron, no timed insights.
        """
        if not self._three_tier or not self._three_tier._ready:
            return
        if not self._llm:
            return
        try:
            snippets = self._three_tier.retrieve(
                "recent work progress important findings", k=5
            )
            if not snippets:
                return
            blob = "\n".join(f"- {s}" for s in snippets)
            prompt = (
                "Based on these recent memory snippets, generate ONE concise "
                "proactive insight or reminder (1–2 sentences max) that would be "
                "useful to surface to the user when they return. "
                "Focus on: unfinished tasks, important decisions, or key findings. "
                "If nothing is worth surfacing, output only 'NONE'.\n\n"
                f"{blob}"
            )
            result = self._llm(prompt, temperature=0.3, max_tokens=100)
            if isinstance(result, str) and result.strip().upper() != "NONE":
                with self._lock:
                    self._pending_insight = result.strip()
                log.debug("Proactive insight generated: %s", result[:80])
        except Exception as exc:
            log.debug("Insight generation: %s", exc)
