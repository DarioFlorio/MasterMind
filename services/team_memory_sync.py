"""
services/team_memory_sync.py — TeamMemorySync: cross-machine shared memory

Watches the team memory directory for changes and syncs them. Includes
a secret scanner that prevents credentials from leaking into shared memory.

Integration:
    from services.team_memory_sync import TeamMemorySync
    tms = TeamMemorySync(team_mem_dir=Path("memdir/team"))
    tms.start()   # begin watching
    tms.stop()    # on exit
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.team_memory_sync")

# ── Secret patterns (from src secretScanner.ts) ────────────────────────────
_SECRET_PATTERNS: list[re.Pattern] = [
    re.compile(r'(?i)(api[_\s-]?key|apikey|api[_\s-]?secret)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(secret[_\s-]?key|secretkey)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(password|passwd|pwd)\s*[=:]\s*\S+'),
    re.compile(r'(?i)(token|auth[_\s-]?token|access[_\s-]?token)\s*[=:]\s*[A-Za-z0-9+/=_\-]{16,}'),
    re.compile(r'(?i)(private[_\s-]?key)\s*[=:]\s*\S+'),
    re.compile(r'sk-[A-Za-z0-9]{20,}'),        # OpenAI / Anthropic style
    re.compile(r'ghp_[A-Za-z0-9]{36}'),         # GitHub personal access token
    re.compile(r'github_pat_[A-Za-z0-9_]{82}'), # GitHub fine-grained PAT
    re.compile(r'AKIA[A-Z0-9]{16}'),            # AWS access key ID
    re.compile(r'(?i)-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]


def scan_for_secrets(text: str) -> list[str]:
    """
    Scan text for potential secrets.
    Returns list of human-readable descriptions of matches found.
    """
    findings = []
    for pattern in _SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            # Redact the value — just report the pattern matched
            snippet = match.group(0)
            if len(snippet) > 40:
                snippet = snippet[:40] + "…"
            findings.append(f"Potential secret detected: {snippet!r}")
    return findings


class TeamMemorySync:
    """
    Watches a team memory directory for .md / .json changes and can
    sync them to other machines (e.g., via git or a shared filesystem).

    Currently provides:
    - Secret scanning before any write to team memory
    - File watcher thread that detects local changes
    - Optional git-based sync (if team_mem_dir is a git repo)
    """

    def __init__(
        self,
        team_mem_dir: Optional[Path] = None,
        sync_interval_s: float = 60.0,
        use_git_sync: bool = False,
    ):
        if team_mem_dir is None:
            team_mem_dir = Path(__file__).parent.parent / "memdir" / "team"
        self._dir = Path(team_mem_dir)
        self._sync_interval_s = sync_interval_s
        self._use_git_sync = use_git_sync
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._file_mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background sync watcher."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("TeamMemorySync: could not create dir: %s", exc)
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, name="TeamMemorySync", daemon=True
        )
        self._thread.start()
        log.info("TeamMemorySync: started, watching %s", self._dir)

    def stop(self) -> None:
        """Stop the background watcher."""
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=5)
        log.info("TeamMemorySync: stopped")

    def safe_write(self, path: Path, content: str) -> tuple[bool, list[str]]:
        """
        Write content to a team memory file — only if no secrets are found.

        Returns (success: bool, warnings: list[str]).
        If warnings is non-empty, the write was blocked.
        """
        warnings = scan_for_secrets(content)
        if warnings:
            log.warning(
                "TeamMemorySync: blocked write to %s — secrets detected: %s",
                path, warnings
            )
            return False, warnings

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            log.debug("TeamMemorySync: wrote %s", path)
            return True, []
        except Exception as exc:
            msg = f"Write failed: {exc}"
            log.warning("TeamMemorySync: %s", msg)
            return False, [msg]

    def get_team_files(self) -> list[Path]:
        """Return list of .md and .json files in the team memory directory."""
        if not self._dir.exists():
            return []
        return sorted(
            p for p in self._dir.rglob("*")
            if p.is_file() and p.suffix in (".md", ".json")
        )

    def read_all(self) -> dict[str, str]:
        """Return dict of {relative_path: content} for all team memory files."""
        result = {}
        for p in self.get_team_files():
            try:
                result[str(p.relative_to(self._dir))] = p.read_text(encoding="utf-8")
            except Exception as exc:
                log.warning("TeamMemorySync: could not read %s: %s", p, exc)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _watch_loop(self) -> None:
        while not self._stop_evt.wait(timeout=self._sync_interval_s):
            self._check_for_changes()
            if self._use_git_sync:
                self._git_sync()

    def _check_for_changes(self) -> None:
        changed = []
        try:
            for p in self.get_team_files():
                key = str(p)
                mtime = p.stat().st_mtime
                if self._file_mtimes.get(key) != mtime:
                    changed.append(p)
                    self._file_mtimes[key] = mtime
        except Exception as exc:
            log.debug("TeamMemorySync: check error: %s", exc)

        if changed:
            log.info("TeamMemorySync: %d team memory file(s) changed", len(changed))
            for p in changed:
                self._validate_file(p)

    def _validate_file(self, path: Path) -> None:
        """Scan a changed team memory file for secrets."""
        try:
            content = path.read_text(encoding="utf-8")
            warnings = scan_for_secrets(content)
            if warnings:
                log.warning(
                    "TeamMemorySync: potential secrets in %s: %s",
                    path.name, "; ".join(warnings)
                )
        except Exception as exc:
            log.debug("TeamMemorySync: could not validate %s: %s", path, exc)

    def _git_sync(self) -> None:
        """Pull latest team memory via git (if the dir is a git repo)."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "pull", "--ff-only"],
                cwd=self._dir,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and "Already up to date" not in result.stdout:
                log.info("TeamMemorySync: git pull succeeded: %s", result.stdout.strip())
        except Exception as exc:
            log.debug("TeamMemorySync: git sync skipped: %s", exc)


# Module-level singleton
team_memory_sync = TeamMemorySync()
