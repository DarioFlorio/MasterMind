"""
autoinstall.py — Silently install missing dependencies on first run.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

REQUIRED = [
    "httpx", "dotenv", "duckduckgo_search", "bs4",
    "requests", "rich", "pathspec", "chardet",
]

INSTALL_MAP = {
    "dotenv":            "python-dotenv",
    "duckduckgo_search": "duckduckgo-search",
    "bs4":               "beautifulsoup4",
}

def _importable(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False

def ensure_dependencies() -> None:
    missing = [p for p in REQUIRED if not _importable(p)]
    if not missing:
        return
    pip_names = [INSTALL_MAP.get(m, m) for m in missing]
    print(f"[autoinstall] Installing: {', '.join(pip_names)}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + pip_names
    )
    print("[autoinstall] Done.")

if __name__ == "__main__":
    ensure_dependencies()
