"""
services/consolidation_lock.py — ConsolidationLock for AutoDream

Lock file whose mtime IS the lastConsolidatedAt timestamp. Allows
multiple processes on the same machine to coordinate so only one
runs a dream consolidation at a time.

The lock file lives inside the memdir directory (.consolidate-lock).
Its body is the holder's PID. Stale locks (>1h old with a dead PID)
are automatically reclaimed.

Ported from src/src/services/autoDream/consolidationLock.ts
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.consolidation_lock")

LOCK_FILE = ".consolidate-lock"
HOLDER_STALE_S = 60 * 60  # 1 hour — stale even if PID is live (PID reuse guard)


def _lock_path(mem_dir: Path) -> Path:
    return mem_dir / LOCK_FILE


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we can't signal it


def read_last_consolidated_at(mem_dir: Path) -> float:
    """
    Return the mtime of the lock file (= last consolidation time).
    Returns 0.0 if the file doesn't exist.
    """
    path = _lock_path(mem_dir)
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0
    except Exception as exc:
        log.debug("consolidation_lock: could not stat %s: %s", path, exc)
        return 0.0


def try_acquire_consolidation_lock(mem_dir: Path) -> Optional[float]:
    """
    Try to acquire the consolidation lock.

    Returns:
        float — the prior mtime (for rollback) if acquired.
                 0.0 means the file didn't exist before.
        None  — if blocked (another live process holds the lock).

    On success, the lock file contains our PID and its mtime = now.
    On failure (returns None), the lock is untouched.
    """
    path = _lock_path(mem_dir)

    prior_mtime: float = 0.0
    holder_pid: Optional[int] = None

    try:
        st = path.stat()
        prior_mtime = st.st_mtime
        body = path.read_text(encoding="utf-8").strip()
        parsed = int(body) if body.isdigit() else None
        holder_pid = parsed
    except FileNotFoundError:
        pass  # No prior lock — fine to acquire
    except Exception as exc:
        log.debug("consolidation_lock: read error: %s", exc)

    # Check if another process holds a live, non-stale lock
    if prior_mtime > 0 and (time.time() - prior_mtime) < HOLDER_STALE_S:
        if holder_pid is not None and _is_process_running(holder_pid):
            log.debug(
                "consolidation_lock: lock held by live PID %d (%.0fs ago)",
                holder_pid, time.time() - prior_mtime
            )
            return None
        # Dead PID or unparseable body — reclaim

    # Acquire: write our PID
    try:
        mem_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
    except Exception as exc:
        log.warning("consolidation_lock: could not write lock: %s", exc)
        return None

    # Race check: two reclaiming processes both write — last wins
    try:
        verify = path.read_text(encoding="utf-8").strip()
        if not verify.isdigit() or int(verify) != os.getpid():
            log.debug("consolidation_lock: lost race to PID %s", verify)
            return None
    except Exception:
        return None

    log.debug("consolidation_lock: acquired (prior_mtime=%.3f)", prior_mtime)
    return prior_mtime


def rollback_consolidation_lock(mem_dir: Path, prior_mtime: float) -> None:
    """
    Roll back the lock after a failed consolidation.

    If prior_mtime == 0.0, remove the file entirely (restore no-file state).
    Otherwise restore the file's mtime to prior_mtime and clear the PID body
    (so our still-running process doesn't look like it's holding).
    """
    path = _lock_path(mem_dir)
    try:
        if prior_mtime == 0.0:
            path.unlink(missing_ok=True)
            log.debug("consolidation_lock: rolled back (deleted)")
            return

        path.write_text("", encoding="utf-8")
        os.utime(path, (prior_mtime, prior_mtime))
        log.debug("consolidation_lock: rolled back to mtime %.3f", prior_mtime)
    except Exception as exc:
        log.warning(
            "consolidation_lock: rollback failed: %s — next trigger delayed", exc
        )


def record_consolidation(mem_dir: Path) -> None:
    """
    Record that a consolidation just completed (stamp mtime = now).
    Called by /dream or manual consolidation triggers. Best-effort.
    """
    path = _lock_path(mem_dir)
    try:
        mem_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()), encoding="utf-8")
        log.debug("consolidation_lock: recorded consolidation")
    except Exception as exc:
        log.debug("consolidation_lock: record failed (best-effort): %s", exc)
