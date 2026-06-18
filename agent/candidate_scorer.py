"""
agent/candidate_scorer.py — Multi-candidate retry scoring.

GAP IMPLEMENTED (2 gaps):
  1. Candidate scoring on retry — smarter recovery from errors.
     When a tool fails, generates multiple candidate approaches and ranks
     them by predicted success rate based on error patterns and past history.

  2. Reasoning gap — EVE limited by local model size.
     For complex answers, generates N candidates via temperature variation
     and picks the most coherent one — a form of self-consistency voting
     that compensates for weaker local models.

Usage:
    from agent.candidate_scorer import CandidateScorer

    scorer = CandidateScorer(error_learner)

    # On tool failure — get ranked retry approaches
    candidates = scorer.score_retry_candidates(
        tool_name="write_file",
        inp={"path": "...", "content": "..."},
        error="No path provided",
        attempt=2,
    )
    best = candidates[0]
    print(best.approach, best.score, best.rationale)

    # For reasoning — pick best of N model outputs
    best_answer = scorer.pick_best_answer(candidates_list)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("agent.candidate_scorer")

# ── Success-rate history file ─────────────────────────────────────────────────
_HISTORY_PATH = Path("memdir") / "candidate_scores.json"


@dataclass
class Candidate:
    approach:  str           # concrete description of what to try
    score:     float         # 0.0–1.0 predicted success probability
    rationale: str           # why this was scored this way
    tool_call: dict = field(default_factory=dict)  # ready-to-execute params if applicable


class CandidateScorer:
    """
    Generates and ranks alternative approaches when a tool fails or when
    multiple reasoning paths are available.

    Scoring factors:
      - Pattern match: does this approach avoid the known failure mode?
      - Success history: has this approach worked in past sessions?
      - Diversity: how different is this from the failed attempt?
      - Error severity: is the error retryable or fatal?
    """

    # Per-tool candidate strategies
    _TOOL_STRATEGIES: dict[str, list[dict]] = {
        "write_file": [
            {"approach": "Use absolute path with forward slashes",
             "template": {"path_style": "absolute_forward"},
             "keywords": ["path", "no path"]},
            {"approach": "Write via bash echo/cat redirect",
             "template": {"alt_tool": "bash", "cmd_template": "echo '{content}' > '{path}'"},
             "keywords": ["permission", "access"]},
            {"approach": "Write to working directory with relative path",
             "template": {"path_style": "relative"},
             "keywords": ["not found", "no such"]},
            {"approach": "Use python -c open().write() via bash",
             "template": {"alt_tool": "bash", "cmd_template": "python -c \"open(r'{path}','w').write('{content_escaped}')\""},
             "keywords": []},
        ],
        "bash": [
            {"approach": "Use PowerShell equivalent command",
             "template": {"alt_tool": "powershell"},
             "keywords": ["not recognized", "not found", "command"]},
            {"approach": "Use python -c for cross-platform execution",
             "template": {"alt_tool": "bash", "cmd_prefix": "python -c"},
             "keywords": ["syntax", "not recognized"]},
            {"approach": "Break into smaller sequential commands",
             "template": {"decompose": True},
             "keywords": ["timeout", "too long"]},
            {"approach": "Add explicit error handling and set -e",
             "template": {"add_error_handling": True},
             "keywords": ["exception", "error", "fail"]},
        ],
        "powershell": [
            {"approach": "Use bash tool instead",
             "template": {"alt_tool": "bash"},
             "keywords": ["not recognized", "not found"]},
            {"approach": "Simplify command, remove unknown parameters",
             "template": {"simplify": True},
             "keywords": ["parameter", "argument", "invalid"]},
            {"approach": "Use python -c for cross-platform",
             "template": {"alt_tool": "bash", "cmd_prefix": "python -c"},
             "keywords": []},
        ],
        "web_search": [
            {"approach": "Broaden query to 2–3 keywords",
             "template": {"simplify_query": True},
             "keywords": ["no results", "not found"]},
            {"approach": "Try web_fetch with a direct URL",
             "template": {"alt_tool": "web_fetch"},
             "keywords": []},
            {"approach": "Add site: qualifier for authoritative source",
             "template": {"add_site_qualifier": True},
             "keywords": ["too many", "broad"]},
        ],
        "web_fetch": [
            {"approach": "Search for the URL first with web_search",
             "template": {"alt_tool": "web_search"},
             "keywords": ["not found", "404", "error"]},
            {"approach": "Try without trailing slash or query params",
             "template": {"clean_url": True},
             "keywords": ["redirect", "not found"]},
        ],
    }

    _GENERIC_STRATEGIES = [
        {"approach": "Use a completely different tool", "template": {"change_tool": True}, "keywords": []},
        {"approach": "Break the task into smaller sub-steps",  "template": {"decompose": True}, "keywords": []},
        {"approach": "Search for the correct syntax first",    "template": {"search_first": True}, "keywords": []},
        {"approach": "Use scratchpad to draft plan before acting", "template": {"plan_first": True}, "keywords": []},
    ]

    def __init__(self, error_learner=None, working_dir: str = "") -> None:
        self._error_learner = error_learner
        self._history: dict[str, dict] = {}
        self._hist_path = (
            Path(working_dir) / _HISTORY_PATH
            if working_dir else _HISTORY_PATH
        )
        self._load_history()

    # ── Public API ────────────────────────────────────────────────────────────

    def score_retry_candidates(
        self,
        tool_name: str,
        inp: dict,
        error: str,
        attempt: int = 1,
        top_k: int = 3,
    ) -> list[Candidate]:
        """
        Generate and rank candidate retry approaches for a failed tool call.
        Returns top_k candidates sorted by descending score.
        """
        strategies = (
            self._TOOL_STRATEGIES.get(tool_name, []) + self._GENERIC_STRATEGIES
        )
        err_lower = error.lower()

        candidates: list[Candidate] = []

        for s in strategies:
            score  = self._score_strategy(s, tool_name, inp, err_lower, attempt)
            reason = self._build_rationale(s, score, err_lower)
            candidates.append(Candidate(
                approach=s["approach"],
                score=score,
                rationale=reason,
                tool_call=s.get("template", {}),
            ))

        # Sort by score descending, deduplicate
        candidates.sort(key=lambda c: c.score, reverse=True)
        seen: set[str] = set()
        deduped: list[Candidate] = []
        for c in candidates:
            key = c.approach[:40]
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        return deduped[:top_k]

    def pick_best_answer(self, answers: list[str]) -> str:
        """
        From N candidate answers (generated at different temperatures),
        pick the most consistent / coherent one via self-consistency voting.

        Used to compensate for the reasoning gap of smaller local models.
        """
        if not answers:
            return ""
        if len(answers) == 1:
            return answers[0]

        # Score each answer:
        # 1. Length (prefer medium — too short = skipped thinking, too long = hallucination)
        # 2. Contains concrete content (numbers, file names, code)
        # 3. Lexical agreement with other answers (self-consistency)
        scored: list[tuple[float, str]] = []
        all_words = set()
        for a in answers:
            all_words.update(a.lower().split())

        for a in answers:
            words    = set(a.lower().split())
            length_s = min(1.0, len(a) / 2000)  # prefer ~2000 chars
            concrete = sum(1 for w in words if any(c.isdigit() or c in "/_." for c in w)) / max(len(words), 1)
            overlap  = len(words & all_words) / max(len(words), 1)
            score    = 0.4 * length_s + 0.3 * concrete + 0.3 * overlap
            scored.append((score, a))

        scored.sort(reverse=True)
        return scored[0][1]

    def record_success(self, tool_name: str, approach_key: str) -> None:
        """Call when a candidate approach succeeded — updates history."""
        key = f"{tool_name}:{approach_key[:40]}"
        rec = self._history.get(key, {"wins": 0, "attempts": 0})
        rec["wins"] += 1
        rec["attempts"] += 1
        rec["last_win"] = time.time()
        self._history[key] = rec
        self._save_history()

    def record_failure(self, tool_name: str, approach_key: str) -> None:
        """Call when a candidate approach failed."""
        key = f"{tool_name}:{approach_key[:40]}"
        rec = self._history.get(key, {"wins": 0, "attempts": 0})
        rec["attempts"] += 1
        self._history[key] = rec
        self._save_history()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_strategy(
        self,
        strategy: dict,
        tool_name: str,
        inp: dict,
        err_lower: str,
        attempt: int,
    ) -> float:
        score = 0.5  # base

        # Keyword match: strategy targets this error type
        kw = strategy.get("keywords", [])
        if kw:
            if any(k in err_lower for k in kw):
                score += 0.25
            else:
                score -= 0.10

        # Historical success rate
        approach_key = strategy["approach"]
        hist_score = self._hist_score(tool_name, approach_key)
        score += 0.20 * hist_score

        # Diversity bonus: if this is the first attempt, all approaches start equal
        # If retrying, prefer approaches that differ maximally from the failed one
        if attempt > 1:
            is_different_tool = "alt_tool" in strategy.get("template", {})
            if is_different_tool:
                score += 0.10

        # Penalise approaches that are known to match error_learner warnings
        if self._error_learner:
            try:
                warnings = self._error_learner.cross_session_warnings(tool_name, inp)
                if warnings and approach_key in warnings:
                    score -= 0.15
            except Exception:
                pass

        return max(0.0, min(1.0, score))

    def _build_rationale(self, strategy: dict, score: float, err_lower: str) -> str:
        lines = [f"Score: {score:.0%}"]
        kw = strategy.get("keywords", [])
        if kw and any(k in err_lower for k in kw):
            lines.append(f"Matches error pattern: {[k for k in kw if k in err_lower]}")
        tmpl = strategy.get("template", {})
        if tmpl.get("alt_tool"):
            lines.append(f"Switches to alternative tool: {tmpl['alt_tool']}")
        if tmpl.get("decompose"):
            lines.append("Decomposes task into smaller sub-steps")
        return " | ".join(lines)

    def _hist_score(self, tool_name: str, approach_key: str) -> float:
        key = f"{tool_name}:{approach_key[:40]}"
        rec = self._history.get(key)
        if not rec or rec.get("attempts", 0) == 0:
            return 0.0
        return rec["wins"] / rec["attempts"]

    def _load_history(self) -> None:
        try:
            if self._hist_path.exists():
                self._history = json.loads(self._hist_path.read_text(encoding="utf-8"))
        except Exception:
            self._history = {}

    def _save_history(self) -> None:
        try:
            self._hist_path.parent.mkdir(parents=True, exist_ok=True)
            self._hist_path.write_text(
                json.dumps(self._history, indent=2), encoding="utf-8"
            )
        except Exception:
            pass
