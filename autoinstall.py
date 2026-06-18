"""
autoinstall.py — Silently install missing dependencies on first run.
MasterMind agentic harness.
"""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

# ── Required packages ─────────────────────────────────────────────────────────
# Import name → pip package name (when they differ)
INSTALL_MAP = {
    "dotenv":            "python-dotenv",
    "duckduckgo_search": "duckduckgo-search",
    "bs4":               "beautifulsoup4",
    "cv2":               "opencv-python",
    "sklearn":           "scikit-learn",
    "PIL":               "Pillow",
    "yaml":              "PyYAML",
    "Crypto":            "pycryptodome",
}

# Packages to ensure are present (use import name here)
REQUIRED = [
    # HTTP / web
    "httpx",
    "requests",
    "bs4",
    # config / env
    "dotenv",
    # search
    "duckduckgo_search",
    # terminal UI
    "rich",
    # file / path utilities
    "pathspec",
    "chardet",
    # audio (voice feature — optional but pre-installed for smooth /voice on)
    "sounddevice",
    # async / concurrency (stdlib — no install needed, listed for doc purposes)
    # "asyncio",  # stdlib
    # "concurrent",  # stdlib
]


def _importable(pkg: str) -> bool:
    try:
        __import__(pkg)
        return True
    except ImportError:
        return False


def ensure_dependencies(extra: list[str] | None = None) -> None:
    """Install any missing packages from REQUIRED + optional `extra` list."""
    all_required = REQUIRED + (extra or [])
    missing = [p for p in all_required if not _importable(p)]
    if not missing:
        return

    pip_names = [INSTALL_MAP.get(m, m) for m in missing]
    print(f"[MasterMind autoinstall] Installing: {', '.join(pip_names)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + pip_names
        )
        print("[MasterMind autoinstall] Done.")
    except subprocess.CalledProcessError as exc:
        print(
            f"[MasterMind autoinstall] WARNING: some packages failed to install: {exc}\n"
            f"  You may need to install them manually: pip install {' '.join(pip_names)}"
        )


if __name__ == "__main__":
    ensure_dependencies()
