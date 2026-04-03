"""
config/settings.py — All configuration via environment variables.
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
# FIXED: Use your actual model path
#MODEL_PATH: str        = os.environ.get("MODEL_PATH", r"C:\Users\dario\.cache\eve\Qwen_Qwen3-1.7B-Q4_K_M.gguf")
MODEL_PATH: str = r"C:\Users\dario\.cache\eve\Qwen_Qwen3-1.7B-Q4_K_M.gguf"
MODEL_DISPLAY: str     = Path(MODEL_PATH).stem

# ── Inference mode ─────────────────────────────────────────────────────────
DIRECT_MODE: bool      = _bool("DIRECT_MODE", True)
LLAMA_SERVER_URL: str  = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8080")
LLAMA_SERVER_PORT: int = _int("LLAMA_SERVER_PORT", 8080)

# ── Context (maxed out) ──────────────────────────────────────────────────
CONTEXT_SIZE = 32768          # Qwen3‑1.7B official max (32k)
UNLIMITED_CONTEXT = True      # sliding compression beyond 32k
MAX_TOKENS = 99999999            # max tokens to generate
TEMPERATURE = 0.7             # deterministic; set 0.7 for creativity

# ── Agent ──────────────────────────────────────────────────────────────────
MAX_TURNS = 99999999               # long agentic chains
PERMISSION_MODE = "auto"      # no confirmations

# ── CPU tuning (max out your cores) ───────────────────────────────────────
import os
N_THREADS = os.cpu_count()          # all physical + logical cores
N_THREADS_BATCH = N_THREADS         # same for batch processing
BATCH_SIZE = 512                     # CPU sweet spot (higher may hurt)
N_GPU_LAYERS = 0                     # ❗ FORCE CPU – no GPU offload

# ── Timeouts (effectively infinite) ───────────────────────────────────────
API_TIMEOUT_S = 31536000              # 1 year
BASH_TIMEOUT_S = 31536000             # 1 year


# ── Verbose & working dir ──────────────────────────────────────────────────
VERBOSE: bool = _bool("VERBOSE", False)           # default: False (quiet)
WORKING_DIR: str = os.environ.get("WORKING_DIR", str(Path.cwd()))
# ── Windows / platform hints ────────────────────────────────────────────────
SYSTEM_PROMPT_EXTRA: str = "You are running on Windows. Use 'ping -n 4 8.8.8.8' instead of 'ping -c 4 8.8.8.8'. Avoid commands that run forever (like 'ping -t')."
