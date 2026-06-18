"""
skills/wakefulness.py — Wakefulness Monitor for MasterMind.

Detects hallucination-precursor signals in the model's own output by
analysing *epistemic sentiment* — the confidence/grounding posture of
the text — rather than its topic or meaning.

Conceptual basis
================
A hallucinating LLM mirrors a sleeping human brain: the generative
machinery keeps running but the supervisory reality-check goes offline.
The output remains fluent and internally coherent, but loses its
anchoring to external fact.  This produces six measurable signals:

  1. Hedging reversal     — flat assertions replace appropriate uncertainty
  2. Grounding loss       — specific anchors (tool results, user statements,
                            dates, sources) disappear from the text
  3. Confabulation style  — smooth, elaborate prose about things that
                            should trigger caution
  4. Semantic drift       — recent output diverges from established context
                            (char n-gram Jaccard as embedding proxy)
  5. Consistency erosion  — subtle contradictions accumulate across turns
  6. Dream-pattern words  — specific lexical markers statistically
                            over-represented in hallucinated text

Each detector returns a score 0.0 – 1.0 (1.0 = maximally asleep).
The composite *wakefulness score* is 1 − weighted_mean(detector_scores).
A score below _ALERT_THRESHOLD triggers an intervention injected into
the session, forcing the model to re-anchor before generating further.

Architecture
============
Two interfaces (mirrors GoalAnchorSkill):

  • Orchestrator-level: check(text, session, turn)
    Called by QueryEngine after every assistant turn, zero model-call
    cost, all detectors run in parallel via ThreadPoolExecutor.

  • Skill-tool level:  execute_impl(problem, **kwargs)
    Called via `skill wakefulness` for a manual diagnostic report.
    Returns a scored breakdown the model reads and acts on.

Zero dependencies beyond stdlib.
"""
from __future__ import annotations

import re
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict

from skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from agent.session import Session

# ── Thresholds ────────────────────────────────────────────────────────────────

_ALERT_THRESHOLD    = 0.40   # wakefulness below this → inject re-anchor
_DRIFT_THRESHOLD    = 0.12   # ngram similarity below this → semantic drift
_HEDGE_RATIO_MIN    = 0.015  # hedges per word below this → over-confident
_GROUND_RATIO_MIN   = 0.008  # grounding anchors per word below this → unmoored
_CONTRADICT_LIMIT   = 2      # contradiction pairs before flagging
_MIN_TEXT_WORDS     = 30     # ignore very short outputs


# ── Epistemic lexicons ────────────────────────────────────────────────────────

# Appropriate uncertainty markers — should be PRESENT in well-calibrated output
_HEDGE_WORDS = {
    "approximately", "roughly", "around", "about", "maybe", "perhaps",
    "possibly", "probably", "likely", "unlikely", "might", "may", "could",
    "seem", "seems", "appear", "appears", "suggest", "suggests",
    "i think", "i believe", "i'm not certain", "i'm not sure",
    "to my knowledge", "as far as i know", "if i recall",
    "you may want to verify", "worth checking", "i'd recommend confirming",
    "approximately", "not entirely sure", "uncertain", "unclear",
    "it's possible", "one possibility", "another possibility",
}

# Grounding anchors — evidence of external reality-check
_GROUND_ANCHORS = {
    "according to", "based on", "the file shows", "search results",
    "you mentioned", "you said", "you told me", "the output shows",
    "the error says", "the log shows", "retrieved", "from the document",
    "the tool returned", "as stated", "as noted", "the data shows",
    "per the", "citing", "source:", "reference:", "from your",
    "the result is", "running the command", "executing",
}

# Flat-assertion markers — RED FLAG when dense without hedges
_CERTAINTY_WORDS = {
    "definitely", "certainly", "absolutely", "undoubtedly", "clearly",
    "obviously", "of course", "without question", "it is a fact",
    "it is well known", "everyone knows", "it is proven", "proven",
    "it is established", "unquestionably", "undeniably", "always",
    "never", "impossible", "guaranteed",
}

# Dream-pattern words — lexical markers over-represented in hallucinated text
# (empirically: smooth elaboration without substance)
_DREAM_PATTERNS = {
    "furthermore", "moreover", "in addition", "it is worth noting",
    "it should be noted", "notably", "importantly", "significantly",
    "interestingly", "as we can see", "as mentioned", "as previously",
    "it is clear that", "it becomes evident", "one can observe",
    "it is important to understand", "in essence", "fundamentally",
    "at its core", "the key insight", "the crucial point",
    "it is fascinating", "remarkably", "strikingly",
}

# Contradiction signal pairs — if both sides appear close together, flag it
_CONTRADICT_PAIRS = [
    ({"always", "never fails", "guaranteed"},    {"sometimes", "can fail", "may not"}),
    ({"does not exist", "no such"},               {"exists", "is available", "there is"}),
    ({"is deprecated", "was removed"},            {"is available", "you can use", "use this"}),
    ({"before", "earlier", "previously"},         {"after", "later", "subsequently"}),
    ({"increase", "rises", "grows"},              {"decrease", "falls", "shrinks"}),
    ({"impossible", "cannot"},                    {"possible", "can", "is able to"}),
]


# ─────────────────────────────────────────────────────────────────────────────
# WakefulnessSkill
# ─────────────────────────────────────────────────────────────────────────────

class WakefulnessSkill(BaseSkill):
    """
    MasterMind Wakefulness Monitor.

    Measures epistemic sentiment of model output to detect hallucination
    precursors before they compound into full confabulation.
    """

    @property
    def name(self) -> str:
        return "wakefulness"

    @property
    def description(self) -> str:
        return (
            "Analyse model output for hallucination-precursor signals using "
            "epistemic sentiment: hedging reversal, grounding loss, confabulation "
            "style, semantic drift, consistency erosion, and dream-pattern lexicon. "
            "Returns a wakefulness score (0=asleep/hallucinating, 1=fully grounded). "
            "Also runs automatically at the orchestrator level after every assistant turn."
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The text to analyse (paste the model output to check).",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: earlier session text to check drift against.",
                },
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["problem"],
        }

    # ── Skill-tool interface ──────────────────────────────────────────────────

    def execute_impl(self, problem: str, **kwargs) -> str:
        context = kwargs.get("context", "")
        report  = self._full_report(problem, context)
        return report

    # ── Orchestrator-level interface ──────────────────────────────────────────

    def check(
        self,
        text: str,
        session: "Session",
        turn: int,
    ) -> str:
        """
        Analyse latest assistant output. Returns an intervention string
        if wakefulness is below threshold, else "".
        Called by QueryEngine after every assistant turn (zero model-call cost).
        """
        if not text or len(text.split()) < _MIN_TEXT_WORDS:
            return ""

        context = _recent_session_text(session, chars=3000)
        scores  = self._run_detectors(text, context)
        w_score = _wakefulness_score(scores)

        if w_score >= _ALERT_THRESHOLD:
            return ""

        # Build a targeted intervention based on which detectors fired worst
        worst = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        causes = ", ".join(f"{k} ({v:.2f})" for k, v in worst)

        return (
            f"[Wakefulness alert at turn {turn}] Score {w_score:.2f}/1.00 — "
            f"hallucination precursors detected: {causes}. "
            f"STOP generating. Before continuing: (1) identify which specific "
            f"claim you are least certain about, (2) use a tool to verify it or "
            f"explicitly flag it as uncertain, (3) add grounding anchors "
            f"('based on X', 'the tool returned Y') to assertions that lack them. "
            f"Do not proceed with confident prose until at least one external "
            f"verification has been performed."
        )

    # ── Detectors ─────────────────────────────────────────────────────────────

    def _run_detectors(self, text: str, context: str) -> dict[str, float]:
        """Run all detectors in parallel, return {name: drift_score 0-1}."""
        jobs = [
            ("hedging_reversal",   lambda: _score_hedging_reversal(text)),
            ("grounding_loss",     lambda: _score_grounding_loss(text)),
            ("certainty_spike",    lambda: _score_certainty_spike(text)),
            ("dream_pattern",      lambda: _score_dream_pattern(text)),
            ("semantic_drift",     lambda: _score_semantic_drift(text, context)),
            ("consistency",        lambda: _score_consistency(text)),
        ]
        results: dict[str, float] = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = {pool.submit(fn): name for name, fn in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = 0.0
        return results

    def _full_report(self, text: str, context: str) -> str:
        """Generate a human-readable diagnostic report."""
        words = text.split()
        if len(words) < _MIN_TEXT_WORDS:
            return (
                f"[Wakefulness] Text too short ({len(words)} words) for reliable analysis. "
                f"Provide at least {_MIN_TEXT_WORDS} words."
            )

        scores  = self._run_detectors(text, context)
        w_score = _wakefulness_score(scores)

        # Grade
        if w_score >= 0.80:
            grade, icon = "FULLY AWAKE",   "🟢"
        elif w_score >= 0.60:
            grade, icon = "MILDLY DROWSY", "🟡"
        elif w_score >= 0.40:
            grade, icon = "DRIFTING",      "🟠"
        else:
            grade, icon = "ASLEEP",        "🔴"

        lines = [
            "═" * 64,
            "  🧠  MASTERMIND WAKEFULNESS MONITOR",
            "═" * 64,
            f"  Wakefulness Score : {w_score:.3f} / 1.000  {icon}  {grade}",
            f"  Words analysed    : {len(words)}",
            "─" * 64,
            "  DETECTOR BREAKDOWN  (higher = more hallucination signal)",
            "─" * 64,
        ]

        detector_labels = {
            "hedging_reversal": "Hedging reversal    (model stopped expressing uncertainty)",
            "grounding_loss":   "Grounding loss      (no anchors to tool results / user input)",
            "certainty_spike":  "Certainty spike     (flat assertions without evidence)",
            "dream_pattern":    "Dream-pattern words (smooth elaboration without substance)",
            "semantic_drift":   "Semantic drift      (text diverging from session context)",
            "consistency":      "Consistency erosion (contradictory statements detected)",
        }

        for key, label in detector_labels.items():
            score = scores.get(key, 0.0)
            bar   = _bar(score)
            flag  = "⚠️ " if score > 0.5 else "  "
            lines.append(f"  {flag}{bar}  {score:.2f}  {label}")

        lines += [
            "─" * 64,
            "",
            "  INTERPRETATION",
        ]

        if w_score >= 0.80:
            lines.append(
                "  Output appears well-grounded. Epistemic posture is calibrated:\n"
                "  appropriate hedging, grounded claims, no confabulation markers."
            )
        elif w_score >= 0.60:
            top = max(scores, key=scores.__getitem__)
            lines.append(
                f"  Mild drift detected. Weakest signal: {top}.\n"
                f"  Recommend: add one explicit verification or uncertainty marker\n"
                f"  before concluding."
            )
        elif w_score >= 0.40:
            worsts = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
            wnames = " and ".join(n for n, _ in worsts)
            lines.append(
                f"  Significant drift. Signals firing: {wnames}.\n"
                f"  Recommend: pause, use a tool to verify the most uncertain claim,\n"
                f"  and rewrite with explicit grounding anchors."
            )
        else:
            lines.append(
                "  ⛔  HALLUCINATION LIKELY. Multiple signals co-firing.\n"
                "  Action required:\n"
                "    1. Identify the claim you are least certain about.\n"
                "    2. Use web_search, read_file, or bash to verify it.\n"
                "    3. If you cannot verify it, state explicitly that you are\n"
                "       uncertain and do not present it as fact.\n"
                "    4. Rewrite the response with grounding anchors throughout."
            )

        lines += ["═" * 64]
        return "\n".join(lines)


# ── Detector implementations ──────────────────────────────────────────────────

def _score_hedging_reversal(text: str) -> float:
    """High score = model is NOT hedging when it should."""
    words = text.lower().split()
    n     = max(len(words), 1)
    hedge_hits = sum(
        1 for phrase in _HEDGE_WORDS
        if phrase in text.lower()
    )
    ratio = hedge_hits / n
    # Expected minimum ratio for calibrated output
    if ratio >= _HEDGE_RATIO_MIN:
        return 0.0
    # Linear scale from 0 (at threshold) to 1 (zero hedges)
    return 1.0 - (ratio / _HEDGE_RATIO_MIN)


def _score_grounding_loss(text: str) -> float:
    """High score = no grounding anchors (tool results, user statements, sources)."""
    words  = text.lower().split()
    n      = max(len(words), 1)
    ground_hits = sum(
        1 for phrase in _GROUND_ANCHORS
        if phrase in text.lower()
    )
    ratio = ground_hits / n
    if ratio >= _GROUND_RATIO_MIN:
        return 0.0
    return 1.0 - (ratio / _GROUND_RATIO_MIN)


def _score_certainty_spike(text: str) -> float:
    """High score = dense flat assertions, especially without offsetting hedges."""
    words   = text.lower().split()
    n       = max(len(words), 1)
    certain = sum(1 for w in _CERTAINTY_WORDS if w in text.lower())
    hedges  = sum(1 for w in _HEDGE_WORDS if w in text.lower())
    # Net certainty: certainty words minus hedges (negative = well-calibrated)
    net = (certain - hedges) / n
    return max(0.0, min(1.0, net * 20))   # scale: 0.05 net ratio → score 1.0


def _score_dream_pattern(text: str) -> float:
    """High score = text full of smooth-sounding elaboration without substance."""
    words = text.lower().split()
    n     = max(len(words), 1)
    hits  = sum(1 for phrase in _DREAM_PATTERNS if phrase in text.lower())
    # ~1 dream-phrase per 50 words is borderline; more is a strong signal
    ratio = hits / (n / 50)
    return max(0.0, min(1.0, (ratio - 0.5) / 2.0))


def _score_semantic_drift(text: str, context: str) -> float:
    """High score = text has diverged from established session context."""
    if not context or len(context.split()) < 20:
        return 0.0   # not enough context to judge
    sim = _ngram_overlap(text, context, n=4)
    if sim >= _DRIFT_THRESHOLD:
        return 0.0
    return 1.0 - (sim / _DRIFT_THRESHOLD)


def _score_consistency(text: str) -> float:
    """High score = contradictory statement pairs detected in text."""
    text_lower = text.lower()
    hits = 0
    for set_a, set_b in _CONTRADICT_PAIRS:
        has_a = any(w in text_lower for w in set_a)
        has_b = any(w in text_lower for w in set_b)
        if has_a and has_b:
            hits += 1
    # _CONTRADICT_LIMIT pairs or more → score 1.0
    return min(1.0, hits / max(_CONTRADICT_LIMIT, 1))


# ── Composite score ───────────────────────────────────────────────────────────

# Detector weights — grounding loss and hedging reversal matter most
_WEIGHTS = {
    "hedging_reversal": 0.25,
    "grounding_loss":   0.25,
    "certainty_spike":  0.20,
    "dream_pattern":    0.15,
    "semantic_drift":   0.10,
    "consistency":      0.05,
}

def _wakefulness_score(scores: dict[str, float]) -> float:
    """Composite wakefulness 0-1 (1=fully awake, 0=fully asleep)."""
    total_weight = sum(_WEIGHTS.values())
    drift = sum(
        _WEIGHTS.get(k, 0.0) * v
        for k, v in scores.items()
    ) / total_weight
    return max(0.0, min(1.0, 1.0 - drift))


# ── Utilities ─────────────────────────────────────────────────────────────────

def _ngram_overlap(a: str, b: str, n: int = 4) -> float:
    """Character n-gram Jaccard similarity."""
    def ng(s: str) -> set[str]:
        s = s.lower()
        return {s[i:i+n] for i in range(len(s) - n + 1)} if len(s) >= n else set()
    na, nb = ng(a), ng(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _bar(score: float, width: int = 10) -> str:
    """ASCII progress bar for a 0-1 score."""
    filled = round(score * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _recent_session_text(session: "Session", chars: int = 3000) -> str:
    try:
        msgs = session.to_api_messages()
        all_text = " ".join(
            m.get("content", "") for m in msgs
            if isinstance(m.get("content"), str)
        )
        return all_text[-chars:]
    except Exception:
        return ""
