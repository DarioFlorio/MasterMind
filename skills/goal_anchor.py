"""
skills/goal_anchor.py — Goal Anchor for MasterMind agentic harness.

Answers the question at every level: "Am I still doing what I was asked to do?"

Architecture
============
This skill operates at TWO levels simultaneously:

  1. **Orchestrator-level** (automatic, zero model-call cost)
     QueryEngine calls GoalAnchorSkill.check() every _GOAL_ANCHOR_INTERVAL
     inner turns. Five pure-Python detectors run in parallel via
     ThreadPoolExecutor and inject a redirect hint into the session on drift.

  2. **Skill-tool level** (on demand via `skill goal_anchor`)
     Returns a structured self-alignment checklist the model uses to
     recalibrate. Called as:
       <tool_use>
         <n>skill</n>
         <input>{"skill": "goal_anchor", "args": {"problem": "current objective"}}</input>
       </tool_use>

Recursive decomposition (hierarchical)
=======================================
    Root Goal
    ├── Sub-goal A  (agent-1 scope)
    │   ├── Task A1
    │   └── Task A2
    └── Sub-goal B  (agent-2 scope)
        ├── Task B1
        └── Task B2

Each sub-agent's QueryEngine holds its own GoalAnchorSkill instance scoped
to its sub-goal.  The root orchestrator's instance watches the root goal.
All instances run their five detectors concurrently — drift is caught at
every level, independently, in parallel.

Parallel operation
==================
All five detectors run concurrently via concurrent.futures.ThreadPoolExecutor
so wall-clock cost = max(detector_time), not sum(detector_time).
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict

from skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from agent.session import Session

# ── Tunables ─────────────────────────────────────────────────────────────────

_TOOL_LOOP_THRESHOLD    = 0.50   # fraction of recent calls dominated by one tool
_SUBAGENT_ABUSE_THRESHOLD = 3    # consecutive agent-tool calls → warn
_MEMORY_SPAM_THRESHOLD  = 5      # memory_write calls in last 20 → warn
_COSINE_DRIFT_THRESHOLD = 0.12   # char-ngram Jaccard below this → goal drift
_EMPTY_TURN_WINDOW      = 6      # last N calls all in _EMPTY_TOOLS → no progress
_EMPTY_TOOLS            = {"memory_write", "memory_read"}
_DRIFT_MIN_GOAL_LEN     = 40     # goals shorter than this skip drift detection
                                 # (greetings / one-liners produce false positives)
_DRIFT_GRACE_TURNS      = 4      # don't fire drift until this many turns in


# ─────────────────────────────────────────────────────────────────────────────
# GoalAnchorSkill
# ─────────────────────────────────────────────────────────────────────────────

class GoalAnchorSkill(BaseSkill):
    """
    MasterMind GoalAnchor — keeps the agent on-task at every level.

    Skill-tool interface  →  execute_impl(problem, **kwargs)
    Orchestrator interface →  check(goal, tool_history, session, turn)
    """

    # ── BaseSkill contract ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "goal_anchor"

    @property
    def description(self) -> str:
        return (
            "Check whether current actions still serve the original goal. "
            "Returns a structured self-alignment checklist with drift analysis. "
            "Also runs automatically every few turns at the orchestrator level "
            "(zero model-call cost, parallel detectors)."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The current objective or goal to check alignment against.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: recent work / session context to diff against.",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                },
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        """Skill-tool entry point — returns a self-alignment checklist."""
        context = kwargs.get("context", "")
        checks  = self._run_checklist(problem, context)

        lines = [
            "═" * 62,
            "  🎯  MASTERMIND GOAL ANCHOR — Self-Alignment Check",
            "═" * 62,
            f"  Goal: {problem[:120]}",
            "─" * 62,
        ]
        for label, status, detail in checks:
            icon = "✅" if status == "OK" else ("⚠️ " if status == "WARN" else "🔴")
            lines.append(f"  {icon}  {label}")
            if detail:
                lines.append(f"       └─ {detail}")

        lines.append("─" * 62)
        has_issue = any(s != "OK" for _, s, _ in checks)
        if has_issue:
            lines.append(
                "  → REDIRECT: Stop current activity. Re-read the goal above.\n"
                "    Identify the NEXT SINGLE ACTION that moves toward it.\n"
                "    Do NOT spawn sub-agents unless the task genuinely cannot\n"
                "    be done in one tool call."
            )
        else:
            lines.append("  → ON TRACK: Current direction aligns with the goal.")
        lines.append("═" * 62)
        return "\n".join(lines)

    # ── Checklist (used by skill-tool) ────────────────────────────────────────

    def _run_checklist(self, goal: str, context: str) -> list[tuple[str, str, str]]:
        """Return list of (label, status, detail) — runs all checks in parallel."""
        checks_input = [
            ("Goal visible in working context",
             lambda: self._check_goal_visible(goal, context)),
            ("Goal decomposed into sub-steps",
             lambda: self._check_decomposed(goal)),
            ("No unresolved blockers",
             lambda: self._check_blockers(context)),
        ]
        results: list[tuple[str, str, str]] = [None] * len(checks_input)  # type: ignore

        with ThreadPoolExecutor(max_workers=len(checks_input)) as pool:
            futures = {
                pool.submit(fn): (i, label)
                for i, (label, fn) in enumerate(checks_input)
            }
            for future in as_completed(futures):
                i, label = futures[future]
                try:
                    status, detail = future.result()
                except Exception as exc:
                    status, detail = "WARN", f"Check failed: {exc}"
                results[i] = (label, status, detail)

        return results

    def _check_goal_visible(self, goal: str, context: str) -> tuple[str, str]:
        if not context:
            return "OK", ""
        overlap = _ngram_overlap(goal, context, n=3)
        if overlap < _COSINE_DRIFT_THRESHOLD:
            return "WARN", f"Similarity {overlap:.2f} — goal may have drifted out of focus."
        return "OK", f"Similarity {overlap:.2f} — goal still in scope."

    def _check_decomposed(self, goal: str) -> tuple[str, str]:
        steps = _extract_sub_steps(goal)
        if steps:
            preview = ", ".join(steps[:3])
            tail    = "…" if len(steps) > 3 else ""
            return "OK", f"{len(steps)} sub-step(s): {preview}{tail}"
        return "WARN", "No sub-steps found — consider using recursive_decompose."

    def _check_blockers(self, context: str) -> tuple[str, str]:
        blockers = _detect_blockers(context)
        if blockers:
            return "WARN", f"Possible blockers: {'; '.join(blockers[:2])}"
        return "OK", ""

    # ── Orchestrator-level API ────────────────────────────────────────────────

    def check(
        self,
        goal: str,
        tool_history: list[str],
        session: "Session",
        turn: int,
    ) -> str:
        """
        Run all five drift detectors in parallel (zero model-call cost).

        Called by QueryEngine every _GOAL_ANCHOR_INTERVAL inner turns.
        Returns a redirect hint string if drift is detected, else "".

        Supports parallel multi-agent use: each sub-agent's QueryEngine
        calls this independently on its own instance with its own goal scope.
        """
        if not goal or not tool_history:
            return ""

        # Grace period: don't fire on the first few turns — the agent is
        # still orienting itself and tool history is too short to be meaningful.
        if len(tool_history) < _DRIFT_GRACE_TURNS:
            return ""

        recent_text = _recent_session_text(session, chars=2000)

        detectors = [
            ("tool_loop",      lambda: _detect_tool_loop(tool_history)),
            ("subagent_abuse", lambda: _detect_subagent_abuse(tool_history)),
            ("memory_spam",    lambda: _detect_memory_spam(tool_history)),
            ("goal_drift",     lambda: _detect_goal_drift(goal, recent_text)),
            ("empty_turns",    lambda: _detect_empty_turns(tool_history)),
        ]

        issues: list[str] = []
        with ThreadPoolExecutor(max_workers=len(detectors)) as pool:
            futures = {pool.submit(fn): name for name, fn in detectors}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    issue = future.result()
                    if issue:
                        issues.append(f"[{name}] {issue}")
                except Exception:
                    pass  # never let a detector crash the loop

        if not issues:
            return ""

        return (
            f"Drift detected at turn {turn}. Issues: {'; '.join(issues)}. "
            f'Original goal: "{goal[:100]}". '
            f"Stop. Re-read the goal. Take the next single action that directly "
            f"moves toward it. Do not spawn sub-agents unless the task genuinely "
            f"cannot be done in one tool call."
        )


# ── Drift detectors (module-level, reusable) ─────────────────────────────────

def _detect_tool_loop(history: list[str]) -> str:
    """Flag if one tool dominates ≥50% of recent calls."""
    window = history[-10:]
    if len(window) < 4:
        return ""
    counts: dict[str, int] = {}
    for t in window:
        counts[t] = counts.get(t, 0) + 1
    dominant = max(counts, key=counts.__getitem__)
    ratio = counts[dominant] / len(window)
    if ratio > _TOOL_LOOP_THRESHOLD:
        return (
            f"Tool '{dominant}' used {counts[dominant]}/{len(window)} recent turns "
            f"({ratio:.0%}) — possible loop."
        )
    return ""


def _detect_subagent_abuse(history: list[str]) -> str:
    """Flag consecutive agent-tool calls."""
    consecutive = 0
    for t in reversed(history):
        if t == "agent":
            consecutive += 1
        else:
            break
    if consecutive >= _SUBAGENT_ABUSE_THRESHOLD:
        return f"{consecutive} consecutive sub-agent calls — handle this inline instead."
    return ""


def _detect_memory_spam(history: list[str]) -> str:
    """Flag excessive memory_write calls."""
    recent = history[-20:]
    count  = sum(1 for t in recent if t == "memory_write")
    if count >= _MEMORY_SPAM_THRESHOLD:
        return (
            f"memory_write called {count}× in last {len(recent)} turns — "
            f"batch writes into one call per turn."
        )
    return ""


def _detect_goal_drift(goal: str, recent_text: str) -> str:
    """Flag if recent session text has diverged from the original goal."""
    if not recent_text or not goal:
        return ""
    # Short goals (greetings, single sentences) produce near-zero overlap with
    # tool output by design — not drift. Skip detection to avoid false positives.
    if len(goal.strip()) < _DRIFT_MIN_GOAL_LEN:
        return ""
    sim = _ngram_overlap(goal, recent_text, n=3)
    if sim < _COSINE_DRIFT_THRESHOLD:
        return f"Goal-context similarity {sim:.2f} — activity may have drifted."
    return ""


def _detect_empty_turns(history: list[str]) -> str:
    """Flag if recent turns are all read/write with no progress."""
    if len(history) < _EMPTY_TURN_WINDOW:
        return ""
    window = history[-_EMPTY_TURN_WINDOW:]
    if all(t in _EMPTY_TOOLS for t in window):
        return f"Last {_EMPTY_TURN_WINDOW} calls are only memory ops — no visible progress."
    return ""


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ngram_overlap(a: str, b: str, n: int = 3) -> float:
    """Character n-gram Jaccard similarity — zero-dependency cosine proxy."""
    def ngrams(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()
    na, nb = ngrams(a), ngrams(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _extract_sub_steps(goal: str) -> list[str]:
    """Identify numbered / bulleted sub-steps or connective phrases."""
    patterns = [
        re.compile(r"^\s*(?:\d+[\.\)]\s+|[-•*]\s+)(.+)$", re.MULTILINE),
        re.compile(r"(?:first|second|third|then|next|finally)[,:]?\s+(.+?)(?:\.|$)",
                   re.IGNORECASE),
    ]
    steps: list[str] = []
    for pat in patterns:
        for m in pat.finditer(goal):
            step = m.group(1).strip()[:60]
            if step and step not in steps:
                steps.append(step)
    return steps


def _detect_blockers(text: str) -> list[str]:
    """Heuristically detect unresolved blocker phrases."""
    if not text:
        return []
    patterns = [
        r"(?:error|exception|failed|cannot|could not|unable to|permission denied)"
        r"[:\s]+([^\n\.]{10,80})",
        r"(?:TODO|FIXME|HACK|BUG)[:\s]+([^\n\.]{5,60})",
        r"(?:waiting for|blocked by|depends on)[:\s]+([^\n\.]{5,60})",
    ]
    found: list[str] = []
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            candidate = m.group(1).strip()
            if candidate not in found:
                found.append(candidate)
    return found


def _recent_session_text(session: "Session", chars: int = 2000) -> str:
    """Extract the last `chars` characters of session messages as plain text."""
    try:
        msgs = session.to_api_messages()
        all_text = " ".join(
            m.get("content", "") for m in msgs
            if isinstance(m.get("content"), str)
        )
        return all_text[-chars:]
    except Exception:
        return ""
