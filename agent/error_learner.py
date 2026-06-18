"""
agent/error_learner.py — Persistent error learning for EVE.

GAP IMPLEMENTED: Cross-session error DB — EVE relearns the same errors
each time. Now errors are persisted across sessions with frequency tracking,
"what fixed it" recording, and proactive pattern warnings.

Tracks every tool failure per session, prevents repeating identical broken
calls, and generates a "try differently" hint by looking at the error type
and building a concrete alternative suggestion.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ── Alternative strategy tables ──────────────────────────────────────────────

_WRITE_FILE_ALTS = [
    'Use bash/powershell: open(r"{path}", "w").write(content)',
    'Use bash: echo content > file (powershell Set-Content)',
    'Check path exists first with list_dir, then retry write_file',
    'Write to working directory only (no absolute paths)',
    'Use edit_file instead of write_file for existing files',
]

_BASH_ALTS = [
    'Use powershell instead of bash for Windows commands',
    'Check command syntax: use dir not ls, type not cat',
    'Add error handling: try/except or 2>nul',
    'Use python -c "..." for cross-platform operations',
    'Break command into smaller steps',
]

_POWERSHELL_ALTS = [
    'Simplify PowerShell: avoid unknown parameters',
    'Use Get-Date -Format "yyyy-MM-dd HH:mm:ss" (no extra params)',
    'Try bash tool instead for simple commands',
    'Use python -c "import datetime; print(datetime.datetime.now())"',
    'Check PowerShell version compatibility',
]

_WEB_ALTS = [
    'Broaden the search query (fewer words)',
    'Try web_fetch with a direct URL instead',
    'Search with different keywords',
    'Try site-specific search: site:docs.python.org topic',
    'Break query into simpler parts',
]

_GENERIC_ALTS = [
    'Try a completely different tool to achieve the same result',
    'Break the task into smaller steps',
    'Search for the correct syntax/approach first',
    'Use scratchpad to plan before acting',
    'Check available tools with tool_search',
]

_TOOL_ALT_MAP = {
    'write_file': _WRITE_FILE_ALTS,
    'bash':       _BASH_ALTS,
    'powershell': _POWERSHELL_ALTS,
    'web_search': _WEB_ALTS,
    'web_fetch':  _WEB_ALTS,
}


class ErrorLearner:
    """
    Tracks tool failures within a session and across sessions (via JSON file).
    Provides concrete alternative suggestions based on error type.

    Cross-session error DB features:
      - Frequency tracking: how many times each error pattern has occurred
      - Fix tracking: which approach fixed a previously failed call
      - Proactive warnings: alert before attempting a known bad pattern
      - Error clustering: group similar errors by pattern
    """

    _PERSIST_PATH = Path("memdir/error_knowledge.json")

    def __init__(self, working_dir: Optional[str] = None) -> None:
        # session-level: tool -> list of {input, error, ts}
        self._session_failures: dict[str, list[dict]] = defaultdict(list)
        # cross-session knowledge loaded from disk
        self._known_failures: dict[str, list[str]] = defaultdict(list)
        # NEW: frequency counter — tool:sig -> {count, first_seen, last_seen, fixed_by}
        self._failure_freq: dict[str, dict] = {}
        # NEW: success registry — what approach fixed each error pattern
        self._fixes: dict[str, str] = {}
        self._working_dir = working_dir
        self._load_knowledge()

    # ── Public API ────────────────────────────────────────────────────────────

    def record_failure(self, tool: str, inp: dict, error: str) -> None:
        """Call this every time a tool returns an error."""
        entry = {
            "input": inp,
            "error": error[:300],
            "ts":    time.time(),
        }
        self._session_failures[tool].append(entry)

        # persist error patterns (input signatures → what went wrong)
        sig     = self._inp_sig(inp)
        pattern = f"{sig}: {error[:150]}"
        if pattern not in self._known_failures[tool]:
            self._known_failures[tool].append(pattern)
            if len(self._known_failures[tool]) > 20:
                self._known_failures[tool] = self._known_failures[tool][-20:]

        # NEW: track frequency
        freq_key = f"{tool}:{sig[:60]}"
        if freq_key not in self._failure_freq:
            self._failure_freq[freq_key] = {
                "count":      0,
                "first_seen": time.time(),
                "last_seen":  0.0,
                "fixed_by":   "",
            }
        rec = self._failure_freq[freq_key]
        rec["count"] += 1
        rec["last_seen"] = time.time()

        self._save_knowledge()

    def record_success(self, tool: str, inp: dict, approach: str = "") -> None:
        """
        Call when a previously-failing tool call succeeds.
        Records what approach fixed it so future sessions benefit.
        GAP: Cross-session error DB — fixes are now persisted.
        """
        sig      = self._inp_sig(inp)
        freq_key = f"{tool}:{sig[:60]}"
        if freq_key in self._failure_freq and approach:
            self._failure_freq[freq_key]["fixed_by"] = approach[:200]
        if approach:
            pattern_key = f"{tool}:{sig[:60]}"
            self._fixes[pattern_key] = approach[:200]
        self._save_knowledge()

    def is_repeated(self, tool: str, inp: dict) -> bool:
        """Return True if we've already tried this exact call and it failed."""
        sig = self._inp_sig(inp)
        for entry in self._session_failures.get(tool, []):
            if self._inp_sig(entry["input"]) == sig:
                return True
        return False

    def get_hint(self, tool: str, inp: dict, error: str, attempt: int) -> str:
        """
        Return a concrete, actionable hint for the next attempt.
        Rotates through alternatives so each retry gets a different suggestion.
        Now also surfaces cross-session fix if one is recorded.
        """
        alts = _TOOL_ALT_MAP.get(tool, _GENERIC_ALTS)
        # Pick alternative based on attempt number (cycling)
        alt = alts[(attempt - 1) % len(alts)]

        # NEW: Check if we have a recorded fix for this exact pattern
        sig      = self._inp_sig(inp)
        freq_key = f"{tool}:{sig[:60]}"
        fix      = self._failure_freq.get(freq_key, {}).get("fixed_by", "")
        if not fix:
            fix = self._fixes.get(freq_key, "")

        # Enrich with error-specific context
        err_low = error.lower()
        if "no path provided" in err_low or "path" in err_low:
            specific = "The path argument is missing or wrong. Use the full absolute path."
        elif "permission" in err_low:
            specific = "Permission denied — try a different directory or run as admin."
        elif "not found" in err_low or "no such file" in err_low:
            specific = "File/command not found — verify the path exists first using list_dir."
        elif "syntax" in err_low:
            specific = "Syntax error — simplify the command or use a different approach."
        elif "timed out" in err_low:
            specific = "Timeout — use a faster/shorter command variant."
        elif "not recognized" in err_low:
            specific = "Command not recognized on Windows — use PowerShell equivalent or Python."
        else:
            specific = ""

        # NEW: frequency warning
        freq  = self._failure_freq.get(freq_key, {}).get("count", 0)
        freq_note = f" [Failed {freq}x across sessions]" if freq > 1 else ""

        lines = [
            f"[ErrorLearner] Attempt {attempt} failed for `{tool}`{freq_note}.",
            f"Error: {error[:200]}",
        ]
        if fix:
            lines.append(f"✓ Previously fixed by: {fix}")
        lines.append(f"Alternative approach: {alt}")
        if specific:
            lines.append(f"Specific fix: {specific}")
        lines.append("DO NOT repeat the same call. Try the alternative above.")
        return "\n".join(lines)

    def session_summary(self) -> str:
        """Human-readable summary of what failed this session."""
        if not self._session_failures:
            return ""
        parts = []
        for tool, entries in self._session_failures.items():
            errors = list({e["error"][:80] for e in entries})
            parts.append(f"  {tool}: {len(entries)} failure(s) — {errors[0]}")
        return "Failures this session:\n" + "\n".join(parts)

    def cross_session_warnings(self, tool: str, inp: dict) -> str:
        """
        Return warnings if this exact pattern has failed in past sessions.
        GAP: Cross-session error DB — proactive warnings on known bad patterns.
        """
        sig = self._inp_sig(inp)
        known = self._known_failures.get(tool, [])
        matching = [k for k in known if k.startswith(sig)]

        if not matching:
            return ""

        # NEW: include frequency and fix info
        freq_key = f"{tool}:{sig[:60]}"
        freq     = self._failure_freq.get(freq_key, {}).get("count", 0)
        fix      = self._failure_freq.get(freq_key, {}).get("fixed_by", "")

        lines = [
            f"[ErrorLearner] WARNING: This {tool} call pattern has failed {freq}x before:",
        ]
        lines.extend(f"  • {m}" for m in matching[:3])
        if fix:
            lines.append(f"  ✓ Known fix: {fix}")
        lines.append("Consider a different approach before trying.")
        return "\n".join(lines)

    def get_known_failure_count(self, tool: str, inp: dict) -> int:
        """Return how many times this exact call has failed historically."""
        sig      = self._inp_sig(inp)
        freq_key = f"{tool}:{sig[:60]}"
        return self._failure_freq.get(freq_key, {}).get("count", 0)

    def most_failed_tools(self, top_k: int = 5) -> list[tuple[str, int]]:
        """Return the top-k most-failed tools across all sessions."""
        counts: dict[str, int] = defaultdict(int)
        for key, rec in self._failure_freq.items():
            tool = key.split(":")[0]
            counts[tool] += rec.get("count", 0)
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _inp_sig(inp: dict) -> str:
        """Stable signature for a tool input dict."""
        try:
            return json.dumps(inp, sort_keys=True, default=str)[:200]
        except Exception:
            return str(inp)[:200]

    def _persist_path(self) -> Path:
        if self._working_dir:
            return Path(self._working_dir) / self._PERSIST_PATH
        return self._PERSIST_PATH

    def _load_knowledge(self) -> None:
        try:
            p = self._persist_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                # Support old format (list) and new format (dict)
                if isinstance(data, dict):
                    for k, v in data.get("failures", {}).items():
                        self._known_failures[k] = v
                    self._failure_freq = data.get("frequency", {})
                    self._fixes        = data.get("fixes", {})
                else:
                    # Old format: dict of lists
                    for k, v in data.items():
                        self._known_failures[k] = v
        except Exception:
            pass

    def _save_knowledge(self) -> None:
        try:
            p = self._persist_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "failures":  dict(self._known_failures),
                "frequency": self._failure_freq,
                "fixes":     self._fixes,
                "updated":   time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass
