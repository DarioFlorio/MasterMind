"""
Memory Core — hybrid FTS + vector search with dreaming consolidation.
MasterMind hybrid memory search system.

Architecture:
  - SQLite FTS5 for keyword search (BM25 ranking)
  - Optional sqlite-vec or numpy for vector similarity search
  - Dreaming: nightly cron-style promotion of frequently recalled memories
  - Temporal decay: recent memories score higher
  - MMR (Maximal Marginal Relevance) for diversity in results
"""
from __future__ import annotations
import json
import math
import os
import re
import sqlite3
import threading
import time
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

_DB_PATH_ENV = "MASTERMIND_MEMORY_DB"
_DEFAULT_DB = Path.home() / ".mastermind" / "memory.db"


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class MemoryChunk:
    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    source: str = "file"
    created_at: float = field(default_factory=time.time)
    recall_count: int = 0
    last_recalled: float = 0.0

@dataclass
class MemorySearchResult:
    chunk: MemoryChunk
    score: float
    match_type: str  # "keyword", "vector", "hybrid"
    snippet: str = ""


# ── BM25 scoring helpers ───────────────────────────────────────────────────────

def _bm25_rank_to_score(rank: float) -> float:
    if not math.isfinite(rank):
        return 1 / (1 + 999)
    if rank < 0:
        relevance = -rank
        return relevance / (1 + relevance)
    return 1 / (1 + rank)

def _build_fts_query(raw: str) -> Optional[str]:
    tokens = re.findall(r'[\w]+', raw)
    if not tokens:
        return None
    quoted = [f'"{t}"' for t in tokens]
    return " AND ".join(quoted)


# ── Temporal decay ─────────────────────────────────────────────────────────────

def _temporal_decay_score(created_at: float, half_life_days: float = 30.0) -> float:
    age_days = (time.time() - created_at) / 86400
    return math.exp(-age_days * math.log(2) / half_life_days)


# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────────

def _mmr_rerank(
    results: list[MemorySearchResult],
    lambda_param: float = 0.7,
    top_k: int = 10,
) -> list[MemorySearchResult]:
    """Re-rank results using MMR to balance relevance and diversity."""
    if len(results) <= top_k:
        return results

    def simple_similarity(a: str, b: str) -> float:
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    selected = []
    remaining = list(results)

    while len(selected) < top_k and remaining:
        if not selected:
            best = remaining.pop(0)
            selected.append(best)
            continue

        best_score = -1.0
        best_idx = 0

        for i, candidate in enumerate(remaining):
            relevance = candidate.score
            max_sim = max(
                simple_similarity(candidate.chunk.text, s.chunk.text)
                for s in selected
            )
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i

        selected.append(remaining.pop(best_idx))

    return selected


# ── Memory Manager ─────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages a SQLite-backed memory store with hybrid FTS + vector search.
    Thread-safe via a reentrant lock.
    """

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = Path(db_path or os.environ.get(_DB_PATH_ENV, _DEFAULT_DB))
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db: Optional[sqlite3.Connection] = None
        self._vector_dims: Optional[int] = None
        self._embedding_fn: Optional[Callable[[str], list[float]]] = None
        self._open()
        self._ensure_schema()

    # ── Connection management ──────────────────────────────────────────────────

    def _open(self) -> None:
        self._db = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA cache_size=-32000")

    def _ensure_schema(self) -> None:
        with self._lock:
            self._db.executescript("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          TEXT PRIMARY KEY,
                    path        TEXT NOT NULL,
                    start_line  INTEGER NOT NULL DEFAULT 0,
                    end_line    INTEGER NOT NULL DEFAULT 0,
                    text        TEXT NOT NULL,
                    source      TEXT NOT NULL DEFAULT 'file',
                    created_at  REAL NOT NULL DEFAULT (unixepoch('now', 'subsec')),
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_recalled REAL NOT NULL DEFAULT 0
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    id UNINDEXED,
                    text,
                    content='chunks',
                    content_rowid='rowid'
                );

                CREATE TABLE IF NOT EXISTS chunk_embeddings (
                    chunk_id    TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    embedding   BLOB NOT NULL,
                    dims        INTEGER NOT NULL,
                    model       TEXT NOT NULL DEFAULT 'default'
                );

                CREATE TABLE IF NOT EXISTS memory_meta (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS chunks_after_insert
                    AFTER INSERT ON chunks BEGIN
                        INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
                    END;

                CREATE TRIGGER IF NOT EXISTS chunks_after_delete
                    AFTER DELETE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, id, text)
                            VALUES ('delete', old.rowid, old.id, old.text);
                    END;

                CREATE TRIGGER IF NOT EXISTS chunks_after_update
                    AFTER UPDATE ON chunks BEGIN
                        INSERT INTO chunks_fts(chunks_fts, rowid, id, text)
                            VALUES ('delete', old.rowid, old.id, old.text);
                        INSERT INTO chunks_fts(rowid, id, text) VALUES (new.rowid, new.id, new.text);
                    END;
            """)
            self._db.commit()

    def set_embedding_fn(self, fn: Callable[[str], list[float]], dims: int) -> None:
        """Register an embedding function for vector search."""
        self._embedding_fn = fn
        self._vector_dims = dims

    # ── Write API ──────────────────────────────────────────────────────────────

    def upsert(self, path: str, text: str, source: str = "file",
               start_line: int = 0, end_line: int = 0) -> str:
        """Index a memory chunk. Returns the chunk ID."""
        chunk_id = hashlib.sha256(f"{path}:{start_line}:{end_line}:{text[:200]}".encode()).hexdigest()[:16]
        with self._lock:
            self._db.execute("""
                INSERT INTO chunks (id, path, start_line, end_line, text, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    text=excluded.text,
                    source=excluded.source
            """, (chunk_id, path, start_line, end_line, text, source))
            self._db.commit()

        if self._embedding_fn:
            self._index_embedding(chunk_id, text)

        return chunk_id

    def _index_embedding(self, chunk_id: str, text: str) -> None:
        try:
            vec = self._embedding_fn(text)
            blob = _vec_to_blob(vec)
            with self._lock:
                self._db.execute("""
                    INSERT OR REPLACE INTO chunk_embeddings (chunk_id, embedding, dims)
                    VALUES (?, ?, ?)
                """, (chunk_id, blob, len(vec)))
                self._db.commit()
        except Exception as e:
            pass  # Vector indexing is best-effort

    def delete(self, chunk_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self._db.commit()

    def ingest_file(self, path: str, chunk_size: int = 50) -> list[str]:
        """Ingest a text file in chunks of `chunk_size` lines."""
        try:
            content = Path(path).read_text(errors="replace")
        except Exception:
            return []
        lines = content.splitlines()
        ids = []
        for start in range(0, len(lines), chunk_size):
            end = min(start + chunk_size, len(lines))
            chunk_text = "\n".join(lines[start:end])
            if chunk_text.strip():
                cid = self.upsert(path, chunk_text, source="file",
                                   start_line=start, end_line=end)
                ids.append(cid)
        return ids

    def ingest_text(self, text: str, label: str = "note") -> str:
        """Ingest an arbitrary text blob."""
        return self.upsert(label, text, source="note")

    # ── Search API ─────────────────────────────────────────────────────────────

    def search_keyword(self, query: str, limit: int = 20) -> list[MemorySearchResult]:
        fts_query = _build_fts_query(query)
        if not fts_query:
            return []
        with self._lock:
            rows = self._db.execute("""
                SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
                       c.created_at, c.recall_count, c.last_recalled,
                       bm25(chunks_fts) as rank,
                       snippet(chunks_fts, 1, '<b>', '</b>', '…', 20) as snip
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """, (fts_query, limit)).fetchall()

        results = []
        for row in rows:
            chunk = MemoryChunk(
                id=row['id'], path=row['path'],
                start_line=row['start_line'], end_line=row['end_line'],
                text=row['text'], source=row['source'],
                created_at=row['created_at'], recall_count=row['recall_count'],
                last_recalled=row['last_recalled'],
            )
            score = _bm25_rank_to_score(row['rank'])
            results.append(MemorySearchResult(chunk=chunk, score=score,
                                              match_type="keyword", snippet=row['snip']))
        return results

    def search_vector(self, query: str, limit: int = 20) -> list[MemorySearchResult]:
        if not self._embedding_fn:
            return []
        try:
            query_vec = self._embedding_fn(query)
        except Exception:
            return []
        dims = len(query_vec)

        with self._lock:
            rows = self._db.execute("""
                SELECT c.id, c.path, c.start_line, c.end_line, c.text, c.source,
                       c.created_at, c.recall_count, c.last_recalled,
                       e.embedding
                FROM chunk_embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                WHERE e.dims = ?
                LIMIT 5000
            """, (dims,)).fetchall()

        if not rows:
            return []

        try:
            import numpy as np
            # Build (n, dims) float32 matrix from raw blobs in one allocation —
            # one BLAS matmul replaces N sequential dot products.
            n = len(rows)
            matrix = np.frombuffer(
                b"".join(row["embedding"] for row in rows), dtype=np.float32
            ).reshape(n, dims)
            q = np.array(query_vec, dtype=np.float32)
            row_norms = np.linalg.norm(matrix, axis=1)  # (n,)
            q_norm = float(np.linalg.norm(q)) + 1e-9
            scores = (matrix @ q) / (row_norms * q_norm + 1e-9)  # (n,)
            top_idx = np.argsort(scores)[::-1][:limit]
            results = []
            for i in top_idx:
                row = rows[int(i)]
                chunk = MemoryChunk(
                    id=row["id"], path=row["path"],
                    start_line=row["start_line"], end_line=row["end_line"],
                    text=row["text"], source=row["source"],
                    created_at=row["created_at"], recall_count=row["recall_count"],
                    last_recalled=row["last_recalled"],
                )
                results.append(MemorySearchResult(
                    chunk=chunk, score=float(scores[int(i)]),
                    match_type="vector", snippet=row["text"][:300],
                ))
            return results

        except ImportError:
            # Fallback: pure-Python row-by-row (slow but functional)
            results = []
            for row in rows:
                stored_vec = _blob_to_vec(row["embedding"])
                sim = _cosine_similarity(query_vec, stored_vec)
                chunk = MemoryChunk(
                    id=row["id"], path=row["path"],
                    start_line=row["start_line"], end_line=row["end_line"],
                    text=row["text"], source=row["source"],
                    created_at=row["created_at"], recall_count=row["recall_count"],
                    last_recalled=row["last_recalled"],
                )
                results.append(MemorySearchResult(
                    chunk=chunk, score=sim, match_type="vector",
                    snippet=row["text"][:300],
                ))
            results.sort(key=lambda r: r.score, reverse=True)
            return results[:limit]

    def search_hybrid(
        self,
        query: str,
        limit: int = 10,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
        use_mmr: bool = True,
        use_temporal_decay: bool = True,
        half_life_days: float = 30.0,
    ) -> list[MemorySearchResult]:
        """Hybrid keyword + vector search with MMR and temporal decay."""
        kw_results = self.search_keyword(query, limit=limit * 3)
        vec_results = self.search_vector(query, limit=limit * 3)

        # Merge by chunk id
        by_id: dict[str, dict] = {}
        for r in kw_results:
            by_id.setdefault(r.chunk.id, {"chunk": r.chunk, "kw": 0.0, "vec": 0.0, "snippet": r.snippet})
            by_id[r.chunk.id]["kw"] = r.score
        for r in vec_results:
            by_id.setdefault(r.chunk.id, {"chunk": r.chunk, "kw": 0.0, "vec": 0.0, "snippet": r.snippet})
            by_id[r.chunk.id]["vec"] = r.score

        merged = []
        for info in by_id.values():
            score = keyword_weight * info["kw"] + vector_weight * info["vec"]
            if use_temporal_decay:
                decay = _temporal_decay_score(info["chunk"].created_at, half_life_days)
                score = 0.85 * score + 0.15 * decay
            merged.append(MemorySearchResult(
                chunk=info["chunk"], score=score, match_type="hybrid",
                snippet=info["snippet"]
            ))

        merged.sort(key=lambda r: r.score, reverse=True)

        if use_mmr:
            merged = _mmr_rerank(merged, top_k=limit)
        else:
            merged = merged[:limit]

        # Record recall
        if merged:
            ids = [r.chunk.id for r in merged]
            with self._lock:
                for cid in ids:
                    self._db.execute("""
                        UPDATE chunks SET recall_count = recall_count + 1,
                                         last_recalled = unixepoch('now', 'subsec')
                        WHERE id = ?
                    """, (cid,))
                self._db.commit()

        return merged

    # ── Dreaming (memory consolidation) ───────────────────────────────────────

    def dream(
        self,
        llm_fn: Callable[[str], str],
        min_recall_count: int = 2,
        min_score: float = 0.6,
        limit: int = 10,
    ) -> list[str]:
        """
        Consolidate frequently-recalled memories into higher-level summaries.
        MasterMind idle-time memory consolidation.
        Returns IDs of newly created consolidation chunks.
        """
        with self._lock:
            candidates = self._db.execute("""
                SELECT id, path, text, recall_count, created_at
                FROM chunks
                WHERE recall_count >= ?
                ORDER BY recall_count DESC
                LIMIT ?
            """, (min_recall_count, limit * 3)).fetchall()

        if not candidates:
            return []

        # Group by path/topic
        groups: dict[str, list] = {}
        for row in candidates:
            key = row['path']
            groups.setdefault(key, []).append(row)

        new_ids = []
        for path, chunks in list(groups.items())[:limit]:
            texts = "\n---\n".join(c['text'][:500] for c in chunks)
            prompt = (
                f"These memory fragments have been frequently recalled. "
                f"Synthesize them into a concise, high-level summary that "
                f"captures the key insights:\n\n{texts}"
            )
            try:
                summary = llm_fn(prompt)
                cid = self.upsert(
                    path=f"dream:{path}",
                    text=summary,
                    source="dream",
                )
                new_ids.append(cid)
            except Exception:
                pass

        return new_ids

    # ── Status ─────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = self._db.execute("SELECT COUNT(*) FROM chunk_embeddings").fetchone()[0]
            sources = self._db.execute(
                "SELECT source, COUNT(*) as cnt FROM chunks GROUP BY source"
            ).fetchall()
        return {
            "total_chunks": total,
            "embedded_chunks": embedded,
            "sources": {row['source']: row['cnt'] for row in sources},
            "db_path": str(self._db_path),
            "vector_enabled": self._embedding_fn is not None,
        }

    def close(self) -> None:
        if self._db:
            self._db.close()
            self._db = None


# ── Vector helpers ─────────────────────────────────────────────────────────────

def _vec_to_blob(vec: list[float]) -> bytes:
    import struct
    return struct.pack(f"{len(vec)}f", *vec)

def _blob_to_vec(blob: bytes) -> list[float]:
    import struct
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """numpy-accelerated cosine similarity (~100x faster than pure Python)."""
    try:
        import numpy as np
        a_arr = np.array(a, dtype=np.float32)
        b_arr = np.array(b, dtype=np.float32)
        denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-9
        return float(np.dot(a_arr, b_arr) / denom)
    except ImportError:
        # Pure-Python fallback if numpy unavailable
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)


# ── Module-level singleton ─────────────────────────────────────────────────────

_GLOBAL_MANAGER: Optional[MemoryManager] = None
_GLOBAL_LOCK = threading.Lock()

def get_memory_manager(db_path: str | Path | None = None) -> MemoryManager:
    global _GLOBAL_MANAGER
    with _GLOBAL_LOCK:
        if _GLOBAL_MANAGER is None:
            _GLOBAL_MANAGER = MemoryManager(db_path)
        return _GLOBAL_MANAGER
