"""
memory/trajectory_store.py — Successful action trajectory caching for EVE.

Stores sequences of (task, actions, outcome) so the agent can recall what
worked for similar tasks and inject those patterns into future decisions.

This is what makes MCTS regret matching actually useful: the trajectory bonus
signal in _heuristic_score comes from this store.

INTEGRATION (query_engine.py)
──────────────────────────────
    from memory.trajectory_store import TrajectoryStore

    # In QueryEngine.__init__:
    self._trajectories = TrajectoryStore(
        db_path=str(Path(working_dir) / "memdir" / "trajectories")
    )

    # After a successful tool execution (success confirmed):
    self._trajectories.store(
        task=self._goal_text or "",
        actions=[{"tool": name, "args": inp, "result": result.output[:200]}],
        outcome="success",
        summary=f"{name} on {self._goal_text[:60]} succeeded",
    )

    # After a failure:
    self._trajectories.store(
        task=self._goal_text or "",
        actions=[{"tool": name, "args": inp}],
        outcome="failure",
        summary=f"{name} failed: {result.output[:80]}",
    )

    # In prompt building or MCTS scoring:
    similar = self._trajectories.retrieve_similar(current_task, k=2)
    for s in similar:
        if s["outcome"] == "success":
            prompt += f"\\n[Past success] {s['summary']}"
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger("memory.trajectory_store")

_EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
_DIST_THRESHOLD   = 0.65
_WORD_OVERLAP_MIN = 0.20   # prevent cross-task poisoning


class TrajectoryStore:
    """
    ChromaDB-backed store for successful (and failed) tool-call trajectories.
    Includes a word-overlap guard to prevent cross-task trajectory poisoning
    (the bug where "search web" trajectories bled into "write file" tasks).
    """

    def __init__(
        self,
        db_path: str = "memdir/trajectories",
        embedding_model: str = _EMBEDDING_MODEL,
    ) -> None:
        self._db_path  = db_path
        self._emb_name = embedding_model
        self._embedder  = None
        self._col       = None
        self._ready     = False
        self._lock      = threading.RLock()
        threading.Thread(target=self._init, daemon=True,
                         name="trajectory_init").start()

    # ── Init ───────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            self._embedder = SentenceTransformer(self._emb_name)
            Path(self._db_path).mkdir(parents=True, exist_ok=True)
            client    = chromadb.PersistentClient(path=self._db_path)
            self._col = client.get_or_create_collection(
                "trajectories",
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
            log.info("TrajectoryStore ready  (existing: %d)", self._col.count())
        except ImportError as exc:
            log.warning("TrajectoryStore: %s", exc)
        except Exception as exc:
            log.warning("TrajectoryStore init: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(
        self,
        task:    str,
        actions: List[dict],
        outcome: str,          # "success" | "failure"
        summary: str = "",
    ) -> None:
        """Persist a trajectory entry."""
        if not self._ready:
            return
        try:
            summary = summary or f"{outcome}: {task[:60]}"
            doc = json.dumps({
                "task": task,
                "actions": actions,
                "outcome": outcome,
                "summary": summary,
            })
            doc_id = (
                f"traj_{int(time.time()*1000)}_"
                f"{hashlib.md5(task.encode()).hexdigest()[:8]}"
            )
            emb = self._embed(summary)
            with self._lock:
                self._col.add(
                    documents=[doc],
                    embeddings=[emb],
                    metadatas=[{
                        "outcome": outcome,
                        "task":    task[:200],
                        "ts":      time.time(),
                    }],
                    ids=[doc_id],
                )
            log.debug("Trajectory stored %s  outcome=%s", doc_id[:12], outcome)
        except Exception as exc:
            log.debug("trajectory store: %s", exc)

    def retrieve_similar(
        self,
        task: str,
        k:    int = 2,
        word_overlap_min: float = _WORD_OVERLAP_MIN,
    ) -> List[dict]:
        """
        Return up to k similar past trajectories.

        The word-overlap guard (default 20%) prevents cross-task poisoning:
        a "web search" trajectory will not be returned for a "write file" task
        even if their embeddings happen to be close.
        """
        if not self._ready:
            return []
        try:
            emb  = self._embed(task)
            with self._lock:
                total = self._col.count()
                if total == 0:
                    return []
                r = self._col.query(
                    query_embeddings=[emb],
                    n_results=min(k * 3, total),
                    include=["documents", "metadatas", "distances"],
                )
            docs   = r.get("documents",  [[]])[0]
            metas  = r.get("metadatas",  [[]])[0]
            dists  = r.get("distances",  [[]])[0]

            task_words = set(task.lower().split())
            filtered   = []
            for doc, meta, dist in zip(docs, metas, dists):
                if dist >= _DIST_THRESHOLD:
                    continue
                try:
                    entry = json.loads(doc)
                except Exception:
                    continue
                entry_words = set(entry.get("task", "").lower().split())
                if not entry_words:
                    continue
                overlap = len(task_words & entry_words) / max(len(task_words), 1)
                if overlap < word_overlap_min:
                    continue     # cross-task poisoning guard
                entry["score"] = round(1.0 - dist, 3)
                filtered.append(entry)
                if len(filtered) >= k:
                    break
            return filtered
        except Exception as exc:
            log.debug("trajectory retrieve: %s", exc)
            return []

    def format_hint(self, trajectories: List[dict]) -> str:
        """Format similar past trajectories as a prompt injection block."""
        if not trajectories:
            return ""
        lines = ["PAST SOLUTIONS (adapt — do NOT copy blindly):"]
        for t in trajectories[:2]:
            outcome = "✓ SUCCESS" if t.get("outcome") == "success" else "✗ FAILED"
            score   = t.get("score", 0)
            task    = t.get("task", "")[:80]
            summary = t.get("summary", "")[:120]
            actions = t.get("actions", [])
            action_str = ""
            if actions:
                a = actions[0]
                action_str = f"  tool={a.get('tool','')}  args={str(a.get('args',''))[:80]}"
            lines.append(f"  [{outcome}] score={score:.2f}  task: {task}")
            lines.append(f"  summary: {summary}")
            if action_str:
                lines.append(f"  action:{action_str}")
        lines.append("Apply lessons — avoid repeating failures.")
        return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            raise RuntimeError("embedder not ready")
        return self._embedder.encode(text, show_progress_bar=False).tolist()
