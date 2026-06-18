"""
memory_core/embeddings.py — Lazy embedding model loader.

Supports two backends (pick one via .env):
  EMBED_BACKEND=sentence_transformers  (default, CPU-friendly, 80MB)
  EMBED_BACKEND=llama_cpp              (reuses existing llama-cpp-python binary)

EMBED_MODEL controls the model name / path:
  sentence_transformers → HuggingFace model id, default "all-MiniLM-L6-v2"
  llama_cpp             → path to a GGUF embed model, e.g. nomic-embed-text-v1.5.Q4_K_M.gguf

Call get_embedding_fn() at startup; it returns (fn, dims) or (None, 0) if
neither backend is available, so the rest of the app degrades gracefully to
keyword-only search.
"""
from __future__ import annotations
import os
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

_BACKEND = os.environ.get("EMBED_BACKEND", "sentence_transformers").strip().lower()
_MODEL   = os.environ.get("EMBED_MODEL",   "all-MiniLM-L6-v2").strip()

# Module-level cache so we load the model exactly once.
_cached_fn:   Optional[Callable[[str], list[float]]] = None
_cached_dims: int = 0
_loaded:      bool = False


def get_embedding_fn() -> tuple[Optional[Callable[[str], list[float]]], int]:
    """
    Return (embed_fn, dims).  embed_fn(text) -> list[float].
    Returns (None, 0) if no backend is available.
    """
    global _cached_fn, _cached_dims, _loaded
    if _loaded:
        return _cached_fn, _cached_dims

    _loaded = True  # mark even on failure so we don't retry every call

    if _BACKEND == "sentence_transformers":
        _cached_fn, _cached_dims = _load_sentence_transformers()
    elif _BACKEND == "llama_cpp":
        _cached_fn, _cached_dims = _load_llama_cpp()
    else:
        log.warning("EMBED_BACKEND=%r unknown; vector search disabled", _BACKEND)

    if _cached_fn:
        log.info(
            "Embedding backend=%s model=%s dims=%d",
            _BACKEND, _MODEL, _cached_dims,
        )
    else:
        log.info(
            "No embedding backend loaded — memory search falls back to keyword-only. "
            "Install sentence-transformers or set EMBED_BACKEND=llama_cpp."
        )

    return _cached_fn, _cached_dims


# ── Backend loaders ────────────────────────────────────────────────────────────

def _load_sentence_transformers() -> tuple[Optional[Callable], int]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        log.debug("sentence-transformers not installed (pip install sentence-transformers)")
        return None, 0

    try:
        model = SentenceTransformer(_MODEL)
        dims  = model.get_sentence_embedding_dimension() or 384

        def embed(text: str) -> list[float]:
            return model.encode(text, normalize_embeddings=True).tolist()

        return embed, dims
    except Exception as exc:
        log.warning("Failed to load SentenceTransformer %r: %s", _MODEL, exc)
        return None, 0


def _load_llama_cpp() -> tuple[Optional[Callable], int]:
    if not _MODEL:
        log.warning("EMBED_BACKEND=llama_cpp but EMBED_MODEL not set")
        return None, 0

    try:
        from llama_cpp import Llama  # type: ignore
    except ImportError:
        log.debug("llama-cpp-python not installed")
        return None, 0

    try:
        llm = Llama(
            model_path=_MODEL,
            embedding=True,
            n_ctx=512,
            n_threads=int(os.environ.get("N_THREADS", "0")) or None,
            verbose=False,
        )
        # Probe dims with a dummy embed
        probe = llm.create_embedding("probe")
        dims  = len(probe["data"][0]["embedding"])

        def embed(text: str) -> list[float]:
            result = llm.create_embedding(text[:512])
            return result["data"][0]["embedding"]

        return embed, dims
    except Exception as exc:
        log.warning("Failed to load llama-cpp embed model %r: %s", _MODEL, exc)
        return None, 0
