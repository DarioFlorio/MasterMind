"""
memory/error_rag.py — ChromaDB-backed error RAG for EVE.

Upgrades EVE's existing ErrorLearner (JSON keyword-matching) with semantic
vector search so that:
  • Errors are retrieved by MEANING, not just keyword match
  • "FileNotFoundError writing stress_test.py" retrieves past failures about
    "permission denied on agent_workspace/output.py" because they're semantically
    close, even though no keywords overlap
  • Fixes are stored alongside errors so the agent gets both warning and remedy

RELATIONSHIP TO ErrorLearner
────────────────────────────
This does NOT replace ErrorLearner — it augments it.
ErrorLearner handles session-level dedup and tool-specific alt strategies.
ErrorRAG handles cross-session semantic retrieval.

Use them together:
  error_learner.record_failure(tool, inp, error)  # session dedup + JSON persist
  error_rag.store(task, code_or_tool, error)       # vector persist
  hints = error_rag.recall(task, code_or_tool)     # semantic retrieval

INTEGRATION (query_engine.py)
──────────────────────────────
    from memory.error_rag import ErrorRAG

    # In QueryEngine.__init__:
    self._error_rag = ErrorRAG(
        db_path=str(Path(working_dir) / "memdir" / "error_rag")
    )

    # In _run_one_tool, after recording failure in error_learner:
    self._error_rag.store(
        task=self._goal_text or "",
        action=f"{name}: {json.dumps(inp)[:200]}",
        error=result.output,
    )

    # When building system prompt / before ACT decision:
    past_hints = self._error_rag.recall(
        task=self._goal_text or "",
        action=pending_action_description,
    )
    if past_hints:
        prompt += "\\n" + self._error_rag.format_hints(past_hints)
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

log = logging.getLogger("memory.error_rag")

_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_DIST_THRESHOLD  = 0.55    # cosine distance — lower = closer match
_MAX_RESULTS     = 3


class ErrorRAG:
    """
    Semantic error memory using ChromaDB.
    Stores (task, action, error, fix) tuples, retrievable by cosine similarity.
    """

    def __init__(
        self,
        db_path: str = "memdir/error_rag",
        embedding_model: str = _EMBEDDING_MODEL,
    ) -> None:
        self._db_path  = db_path
        self._emb_name = embedding_model
        self._embedder  = None
        self._col       = None
        self._ready     = False
        self._lock      = threading.RLock()
        threading.Thread(target=self._init, daemon=True, name="error_rag_init").start()

    # ── Init ───────────────────────────────────────────────────────────────────

    def _init(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            self._embedder = SentenceTransformer(self._emb_name)
            Path(self._db_path).mkdir(parents=True, exist_ok=True)
            client    = chromadb.PersistentClient(path=self._db_path)
            self._col = client.get_or_create_collection(
                "agent_errors",
                metadata={"hnsw:space": "cosine"},
            )
            self._ready = True
            log.info("ErrorRAG ready at %s  (existing: %d)", self._db_path, self._col.count())
        except ImportError as exc:
            log.warning("ErrorRAG: %s — running without vector error memory", exc)
        except Exception as exc:
            log.warning("ErrorRAG init: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(
        self,
        task:     str,
        action:   str,
        error:    str,
        fix:      str = "",
        resolved: bool = False,
    ) -> None:
        """Persist an error + optional fix to ChromaDB."""
        if not self._ready:
            return
        try:
            doc = f"TASK: {task}\nACTION: {action}\nERROR: {error}\nFIX: {fix}"
            doc_id = hashlib.md5(
                f"{task[:100]}{action[:100]}{error[:80]}".encode()
            ).hexdigest()
            emb = self._embed(doc)
            with self._lock:
                self._col.upsert(
                    ids=[doc_id],
                    documents=[doc],
                    embeddings=[emb],
                    metadatas=[{
                        "task":     task[:200],
                        "error":    error[:300],
                        "fix":      fix[:300],
                        "resolved": str(resolved),
                        "ts":       int(time.time()),
                    }],
                )
            log.debug("ErrorRAG stored %s…", doc_id[:8])
        except Exception as exc:
            log.debug("ErrorRAG store: %s", exc)

    def mark_resolved(self, task: str, action: str, error: str, fix: str) -> None:
        """Update an existing error entry with its fix and mark resolved."""
        self.store(task, action, error, fix=fix, resolved=True)

    def recall(
        self,
        task:   str,
        action: str = "",
        n:      int = _MAX_RESULTS,
    ) -> List[dict]:
        """
        Retrieve semantically similar past errors.
        Returns list of dicts with keys: text, task, error, fix, score.
        """
        if not self._ready:
            return []
        try:
            query = f"TASK: {task}\nACTION: {action}"
            emb   = self._embed(query)
            with self._lock:
                r = self._col.query(
                    query_embeddings=[emb],
                    n_results=min(n, max(self._col.count(), 1)),
                    include=["documents", "metadatas", "distances"],
                )
            docs   = r.get("documents",  [[]])[0]
            metas  = r.get("metadatas",  [[]])[0]
            dists  = r.get("distances",  [[]])[0]
            out = []
            for doc, meta, dist in zip(docs, metas, dists):
                if dist < _DIST_THRESHOLD:
                    out.append({
                        "text":  doc,
                        "task":  meta.get("task",  ""),
                        "error": meta.get("error", ""),
                        "fix":   meta.get("fix",   ""),
                        "score": round(1.0 - dist, 3),
                    })
            return out
        except Exception as exc:
            log.debug("ErrorRAG recall: %s", exc)
            return []

    def format_hints(self, past_errors: List[dict], max_show: int = 2) -> str:
        """Format recalled errors as a warning block for injection into prompts."""
        if not past_errors:
            return ""
        lines = ["⚠ PAST ERRORS on similar tasks (do NOT repeat these):"]
        for e in past_errors[:max_show]:
            lines.append(f"  • Error: {e['error'][:150]}")
            if e.get("fix"):
                lines.append(f"    Fix:   {e['fix'][:150]}")
        lines.append("Choose a different approach to avoid repeating these failures.")
        return "\n".join(lines)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            raise RuntimeError("embedder not ready")
        return self._embedder.encode(text, show_progress_bar=False).tolist()
