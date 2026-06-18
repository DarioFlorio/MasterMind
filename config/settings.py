"""
config/settings.py — All configuration via environment variables.
No hardcoded user paths — safe to distribute.
"""
from __future__ import annotations
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env = Path(__file__).parent.parent / ".env"
    load_dotenv(_env if _env.exists() else Path(__file__).parent.parent / ".env.example")
except Exception:
    pass

def _bool(key: str, default: bool) -> bool:
    v = os.environ.get(key, "")
    return v.lower() in ("1", "true", "yes") if v else default

def _int(key: str, default: int) -> int:
    try: return int(os.environ.get(key, ""))
    except: return default

def _float(key: str, default: float) -> float:
    try: return float(os.environ.get(key, ""))
    except: return default

# ── Model ──────────────────────────────────────────────────────────────────
MODEL_PATH: str    = os.environ.get("MODEL_PATH", "")
MODEL_DISPLAY: str = Path(MODEL_PATH).stem if MODEL_PATH else "no model set"

# ── Inference mode ─────────────────────────────────────────────────────────
DIRECT_MODE: bool      = _bool("DIRECT_MODE", True)
LLAMA_SERVER_URL: str  = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
LLAMA_SERVER_PORT: int = _int("LLAMA_SERVER_PORT", 8080)

# 🆕 API credentials for cloud endpoints (used when DIRECT_MODE=0)
API_URL: str = os.environ.get("API_URL", LLAMA_SERVER_URL + "/completion")
API_KEY: str = os.environ.get("API_KEY", "")

# ── Context ────────────────────────────────────────────────────────────────
CONTEXT_SIZE      = _int("CONTEXT_SIZE", 16384)
UNLIMITED_CONTEXT = _bool("UNLIMITED_CONTEXT", True)
MAX_TOKENS        = _int("MAX_TOKENS", 4096)
TEMPERATURE       = _float("TEMPERATURE", 1.0)
TOP_K             = _int("TOP_K", 64)
TOP_P             = _float("TOP_P", 0.95)
MIN_P             = _float("MIN_P", 0.0)
REPEAT_PENALTY    = _float("REPEAT_PENALTY", 1.0)

# ── Agent ──────────────────────────────────────────────────────────────────
MAX_TURNS       = _int("MAX_TURNS", 50)
PERMISSION_MODE = os.environ.get("PERMISSION_MODE", "auto")

# ── CPU tuning ─────────────────────────────────────────────────────────────
def _physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or max(1, (os.cpu_count() or 4) // 2)
    except ImportError:
        return max(1, (os.cpu_count() or 4) // 2)

N_THREADS       = _int("N_THREADS", _physical_cores())
N_THREADS_BATCH = _int("N_THREADS_BATCH", os.cpu_count() or N_THREADS)
BATCH_SIZE      = _int("BATCH_SIZE", 2048)
N_GPU_LAYERS    = _int("N_GPU_LAYERS", -1)   # -1 = auto, 0 = CPU, >0 = explicit

# ── CPU memory/bandwidth optimisations ─────────────────────────────────────
USE_MLOCK     = _bool("USE_MLOCK", True)
FLASH_ATTN    = _bool("FLASH_ATTN", True)
KV_CACHE_TYPE = _int("KV_CACHE_TYPE", 8)
DEFRAG_THOLD  = _float("DEFRAG_THOLD", 0.1)
MMPROJ_PATH   = os.environ.get("MMPROJ_PATH", "")

# ── Timeouts ───────────────────────────────────────────────────────────────
API_TIMEOUT_S  = None
BASH_TIMEOUT_S = _int("BASH_TIMEOUT_S", 31536000)

# ── Verbose & working dir ──────────────────────────────────────────────────
VERBOSE: bool    = _bool("VERBOSE", False)
WORKING_DIR: str = os.environ.get("WORKING_DIR", str(Path.cwd()))

# ── System prompt extra ───────────────────────────────────────────────────
SYSTEM_PROMPT_EXTRA: str = ""
# ── New feature flags (GAP implementations) ────────────────────────────────

# Cross-session memory: auto-load last session within this many hours
SESSION_RESUME_HOURS  = _float("SESSION_RESUME_HOURS", 8.0)

# Active memory curation: idle trigger in seconds
IDLE_CONSOLIDATION_S  = _int("IDLE_CONSOLIDATION_S", 90)

# Resumable plan artifacts: persist plans to disk
PERSIST_PLANS         = _bool("PERSIST_PLANS", True)

# Candidate scoring on retry: how many candidates to generate
MAX_RETRY_CANDIDATES  = _int("MAX_RETRY_CANDIDATES", 3)

# Proactive surfacing: daily digest enabled
DAILY_DIGEST_ENABLED  = _bool("DAILY_DIGEST_ENABLED", True)
DAILY_DIGEST_TIME     = os.environ.get("DAILY_DIGEST_TIME", "08:00")

# PowerShell native: prefer PowerShell over bash on Windows
PREFER_POWERSHELL_WINDOWS = _bool("PREFER_POWERSHELL_WINDOWS", True)

# Context quality: active pruning threshold
CONTEXT_PRUNE_ENABLED = _bool("CONTEXT_PRUNE_ENABLED", True)

# Skill overhead: max skill ratio before switching to direct tools
MAX_SKILL_RATIO       = _float("MAX_SKILL_RATIO", 0.7)


# ── Performance speed tweaks (no sampling/logic changes) ──────────────────
DRAFT_MODEL_PATH = os.environ.get("DRAFT_MODEL_PATH", "")          # e.g. "draft.gguf"
DRAFT_P_MIN      = float(os.environ.get("DRAFT_P_MIN", "0.05"))
NO_KV_OFFLOAD    = _bool("NO_KV_OFFLOAD", True)                    # True = don't offload KV cache
UBATCH_SIZE      = _int("UBATCH_SIZE", 256)
CPU_MASK         = os.environ.get("CPU_MASK", "")                  # e.g. "0xFF" for cores 0‑7
NO_MMAP          = _bool("NO_MMAP", True)                          # True = use anonymous memory
NO_PERF          = _bool("NO_PERF", True)                          # True = disable perf counters
SAMPLERS_STR     = os.environ.get("SAMPLERS", "top_k")
SAMPLERS         = [s.strip() for s in SAMPLERS_STR.split(",") if s.strip()]