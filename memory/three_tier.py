"""
memory/three_tier.py — 3-tier hierarchical memory for EVE.

Ports the HierarchicalMemory architecture from CFE v12.1.

TIERS
─────
  Working   (ChromaDB collection "working_memory")
    Raw turn-level events — tool calls, results, user messages.
    Volatile: evicted / consolidated after CONSOLIDATION_INTERVAL turns.

  Episodic  (ChromaDB collection "episodic_memory")
    LLM-generated 2-3 sentence summaries of working-memory batches.
    "What happened in the last N turns" compressed into a paragraph.
    Indexed by embedding for semantic retrieval.

  Semantic  (ChromaDB collection "semantic_memory")
    Stable, session-spanning facts extracted from episodic batches.
    "EVE always uses Gemma-4", "User prefers pygame for games".
    Highest-quality, lowest-volume, longest-lived.

RETRIEVAL
─────────
    memory.retrieve("how to write a fibonacci function", k=4)
    → returns up to 4 strings drawn from all three tiers,
      ranked by cosine similarity, with tier label prefix.

INTEGRATION (query_engine.py)
──────────────────────────────
    from memory.three_tier import ThreeTierMemory

    # In QueryEngine.__init__:
    self._memory = ThreeTierMemory(
        db_path=str(Path(working_dir) / "memdir" / "three_tier"),
        llm_call_fn=None,        # set later via set_llm()
    )

    # In _run_loop, after session.add_user(user_text):
    self._memory.store(user_text, role="user", turn=self._turn_count)
    self._memory.maybe_consolidate(self._turn_count)

    # After tool result:
    self._memory.store(result_text, role="tool", turn=self._turn_count)

    # When building system prompt context:
    snippets = self._memory.retrieve(current_task, k=4)
    if snippets:
        prompt += "\\n\\n## Retrieved Memory\\n" + "\\n".join(f"- {s}" for s in snippets)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

log = logging.getLogger("memory.three_tier")

# ── Constants ──────────────────────────────────────────────────────────────────
CONSOLIDATION_INTERVAL = 10   # working → episodic every N turns
SEMANTIC_INTERVAL      = 50   # episodic → semantic every N turns
MEMORY_TOP_K           = 4
EMBEDDING_MODEL        = "all-MiniLM-L6-v2"
_DIST_THRESH_WORKING   = 1.20
_DIST_THRESH_EPISODIC  = 1.30
_DIST_THRESH_SEMANTIC  = 1.40


def _is_failure(text: str) -> bool:
    return any(w in text.lower()
               for w in ["error", "fail", "exception", "traceback",
                         "not found", "unknown tool"])


class ThreeTierMemory:
    """
    Three-tier persistent memory backed by ChromaDB + SentenceTransformers.
    Falls back gracefully if ChromaDB is not installed.
    """

    def __init__(
        self,
        db_path: str = "memdir/three_tier",
        llm_call_fn=None,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self._db_path  = db_path
        self._llm      = llm_call_fn
        self._emb_name = embedding_model
        self._embedder  = None
        self._working   = None
        self._episodic  = None
        self._semantic  = None
        self._buffer: List[dict] = []
        self._lock = threading.RLock()
        self._ready = False
        # Initialise in background thread to not block startup
        threading.Thread(target=self._init_db, daemon=True, name="three_tier_init").start()

    def set_llm(self, fn) -> None:
        self._llm = fn

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
        except ImportError as exc:
            log.warning("ThreeTierMemory: %s — running without vector memory", exc)
            return
        try:
            self._embedder = SentenceTransformer(self._emb_name)
            Path(self._db_path).mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=self._db_path)
            self._working  = client.get_or_create_collection(
                "working_memory",  metadata={"hnsw:space": "cosine"})
            self._episodic = client.get_or_create_collection(
                "episodic_memory", metadata={"hnsw:space": "cosine"})
            self._semantic = client.get_or_create_collection(
                "semantic_memory", metadata={"hnsw:space": "cosine"})
            self._ready = True
            log.info("ThreeTierMemory ready at %s", self._db_path)
        except Exception as exc:
            log.warning("ThreeTierMemory init failed: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def store(self, text: str, role: str = "agent", turn: int = 0) -> None:
        """Add a raw event to working memory."""
        with self._lock:
            self._buffer.append({"text": text[:800], "role": role, "turn": turn})
        if not self._ready:
            return
        try:
            doc = text[:1000]
            emb = self._embed(doc)
            doc_id = f"w_{turn}_{int(time.time()*1000)}"
            self._working.add(
                documents=[doc], embeddings=[emb],
                metadatas=[{"role": role, "turn": turn,
                            "failure": int(_is_failure(text)), "ts": time.time()}],
                ids=[doc_id],
            )
        except Exception as exc:
            log.debug("working store: %s", exc)

    def maybe_consolidate(self, turn: int) -> None:
        """Trigger tier promotions based on turn count."""
        if turn > 0 and turn % CONSOLIDATION_INTERVAL == 0:
            threading.Thread(
                target=self._working_to_episodic,
                args=(turn,), daemon=True,
                name="consolidate_episodic"
            ).start()
        if turn > 0 and turn % SEMANTIC_INTERVAL == 0:
            threading.Thread(
                target=self._episodic_to_semantic,
                args=(turn,), daemon=True,
                name="consolidate_semantic"
            ).start()

    def retrieve(self, query: str, k: int = MEMORY_TOP_K) -> List[str]:
        """Return up to k relevant strings from all three tiers."""
        if not self._ready:
            return self._buffer_fallback(query, k)
        try:
            emb = self._embed(query[:800])
            results: List[Tuple[float, str]] = []

            def _qcol(col, label: str, max_k: int, thresh: float) -> None:
                if col is None or col.count() == 0:
                    return
                n = min(max_k, col.count())
                r = col.query(query_embeddings=[emb], n_results=n,
                              include=["documents", "distances"])
                for doc, dist in zip(r["documents"][0], r["distances"][0]):
                    if dist < thresh:
                        results.append((dist, f"[{label}] {doc}"))

            _qcol(self._working,  "working",  k,   _DIST_THRESH_WORKING)
            _qcol(self._episodic, "episodic", 3,   _DIST_THRESH_EPISODIC)
            _qcol(self._semantic, "semantic", 3,   _DIST_THRESH_SEMANTIC)
            results.sort(key=lambda x: x[0])
            return [t for _, t in results[:k]]
        except Exception as exc:
            log.debug("retrieve: %s", exc)
            return self._buffer_fallback(query, k)

    # ── Internal: tier promotion ───────────────────────────────────────────────

    def _working_to_episodic(self, turn: int) -> None:
        with self._lock:
            recent = list(self._buffer[-CONSOLIDATION_INTERVAL:])
        if not recent:
            return
        blob = " | ".join(f"[{e['role']}] {e['text'][:200]}" for e in recent)
        summary = self._summarise(blob, style="episodic")
        if not summary:
            return
        try:
            emb   = self._embed(summary)
            ep_id = f"ep_{turn}_{int(time.time())}"
            self._episodic.add(
                documents=[summary], embeddings=[emb],
                metadatas=[{"turn_range": f"{turn-CONSOLIDATION_INTERVAL}-{turn}",
                            "ts": time.time()}],
                ids=[ep_id],
            )
            log.debug("Promoted to episodic: %s…", summary[:60])
        except Exception as exc:
            log.debug("episodic store: %s", exc)
        # Keep only recent buffer tail
        with self._lock:
            self._buffer = self._buffer[-5:]

    def _episodic_to_semantic(self, turn: int) -> None:
        if self._episodic is None or self._episodic.count() < 3:
            return
        try:
            res = self._episodic.get(include=["documents"])
            docs = res.get("documents", [])
            if not docs:
                return
            blob = "\n".join(docs[-10:])
            facts_raw = self._summarise(blob, style="semantic")
            if not facts_raw:
                return
            for line in facts_raw.splitlines():
                fact = line.strip("- *").strip()
                if len(fact) < 10:
                    continue
                emb    = self._embed(fact)
                sem_id = (f"sem_{int(time.time()*1000)}_"
                          f"{abs(hash(fact)) % 99999}")
                try:
                    self._semantic.add(
                        documents=[fact], embeddings=[emb],
                        metadatas=[{"ts": time.time()}],
                        ids=[sem_id],
                    )
                except Exception:
                    pass
            log.debug("Promoted facts to semantic tier at turn %d", turn)
        except Exception as exc:
            log.debug("semantic promote: %s", exc)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _embed(self, text: str) -> List[float]:
        if self._embedder is None:
            raise RuntimeError("Embedder not ready")
        return self._embedder.encode(text, show_progress_bar=False).tolist()

    def _summarise(self, blob: str, style: str = "episodic") -> str:
        if self._llm is None:
            # Simple keyword-based fallback
            words = re.sub(r"\s+", " ", blob).split()
            return " ".join(words[:50]) + ("…" if len(words) > 50 else "")
        if style == "episodic":
            system = ("Capture key actions, decisions, file names, errors, and "
                      "outcomes. Be concrete. 2-3 sentences.")
            prompt = f"Summarise these agent events:\n\n{blob[:1200]}"
        else:
            system = ("Extract stable, reusable facts from these session memories. "
                      "One fact per line. Only knowledge that persists across sessions.")
            prompt = f"Extract stable facts:\n\n{blob[:1200]}"
        try:
            return self._llm(prompt, system=system, temperature=0.0, max_tokens=250)
        except Exception as exc:
            log.debug("summarise failed: %s", exc)
            return blob[:200]

    def _buffer_fallback(self, query: str, k: int) -> List[str]:
        """Simple keyword-overlap retrieval from the in-memory buffer."""
        qwords = set(query.lower().split())
        scored = []
        for e in self._buffer:
            ewords = set(e["text"].lower().split())
            score  = len(qwords & ewords)
            if score > 0:
                scored.append((score, f"[working] {e['text'][:200]}"))
        scored.sort(reverse=True)
        return [t for _, t in scored[:k]]
