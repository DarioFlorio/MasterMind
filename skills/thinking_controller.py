"""
Skill: thinking_controller
Zero-token, zero-latency classifier that decides how much reasoning the model
should show for a given user query.

Called internally by QueryEngine before each user turn — not intended for
direct model invocation, but registered in SKILL_REGISTRY for transparency.

Returns one of three SYSTEM instruction strings:
  concise  — no visible reasoning (greetings, trivial lookups)
  auto     — brief reasoning only when necessary
  full     — thorough step-by-step reasoning with all working shown
"""
from __future__ import annotations

import re

from skills.base_skill import BaseSkill

DESCRIPTION = (
    "Auto-thinking controller: classifies query complexity and returns a "
    "system-level instruction telling the model how much reasoning to show. "
    "Zero token cost — runs locally before each user turn."
)

# ── Compiled patterns (built once at import time) ─────────────────────────────

_GREETING_RE = re.compile(
    r"^(hi|hello|hey|sup|yo|howdy|good\s*(morning|afternoon|evening|day)|"
    r"you\s+there|are\s+you\s+(there|awake)|what'?s\s+up|ping)\b",
    re.I,
)

_TRIVIAL_RE = re.compile(
    r"^(yes|no|ok|okay|sure|thanks|thank\s+you|ty|thx|cool|got\s+it|"
    r"got\s+that|understood|noted|cheers|np|no\s+problem|sounds\s+good|"
    r"sounds\s+great|perfect|great|nice|awesome|good|fine|alright|ack)\b\s*[!.]?$",
    re.I,
)

_FACTUAL_RE = re.compile(
    r"^(what\s+(is|are|was|were)\s+(the\s+)?(capital|population|year|date|name|"
    r"version|latest|current|default)\b|"
    r"who\s+(is|was|invented|created|founded|wrote|made)\b|"
    r"when\s+(was|is|did|were)\b|"
    r"where\s+is\b|"
    r"how\s+(many|much|old|tall|long|far|big|small)\b)",
    re.I,
)

# Keywords that demand full chain-of-thought
_COMPLEX_KW = frozenset([
    # Analysis / reasoning
    "why", "how does", "how do", "how would", "how should",
    "explain", "reason", "cause", "because", "therefore",
    "analyse", "analyze", "evaluate", "assess", "critique",
    # Planning / design
    "plan", "design", "architect", "structure", "strategy",
    "roadmap", "milestone", "priorit", "trade-off", "tradeoff",
    # Debugging / engineering
    "debug", "error", "bug", "crash", "traceback", "exception",
    "fix", "broken", "not working", "failing", "issue",
    "refactor", "optimise", "optimize", "performance",
    # Math / logic
    "calculate", "compute", "solve", "proof", "derive", "formula",
    "equation", "probability", "statistics", "integral", "derivative",
    # Comparison / decision
    "compare", "versus", "vs ", "difference between", "pros and cons",
    "should i", "which is better", "best option", "recommend",
    "decision", "choose", "pick between",
    # Forecasting / future
    "predict", "forecast", "what if", "scenario", "future",
    "will happen", "consequence", "impact", "effect of",
    # Multi-step
    "step by step", "walk me through", "in detail",
    "comprehensive", "thorough", "deep dive",
])

# Code signals → always full
_CODE_RE = re.compile(
    r"(```|\bdef\b|\bclass\b|\bimport\b|\bfunction\b|\bcallback\b|"
    r"\bSQL\b|\bregex\b|\bapi\b|\bendpoint\b|\bschema\b)",
    re.I,
)


class ThinkingControllerSkill(BaseSkill):
    """Classifies query complexity → returns a system instruction string."""

    @property
    def name(self) -> str:
        return "thinking_controller"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The raw user query to classify.",
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

    # ── Public entry point ────────────────────────────────────────────────────

    def execute_impl(self, problem: str, **kwargs) -> str:
        mode = self._classify(problem)
        return _MODE_INSTRUCTIONS[mode]

    # ── Classifier ────────────────────────────────────────────────────────────

    def _classify(self, text: str) -> str:
        stripped = text.strip()
        low = stripped.lower()

        # ── Hard concise triggers ─────────────────────────────────────────────
        if _GREETING_RE.match(low):
            return "concise"
        if _TRIVIAL_RE.match(low):
            return "concise"
        if len(stripped.split()) <= 4:
            return "concise"

        # ── Hard full triggers ────────────────────────────────────────────────
        if _CODE_RE.search(stripped):
            return "full"
        if any(kw in low for kw in _COMPLEX_KW):
            return "full"
        # Long messages are usually complex
        if len(stripped.split()) >= 30:
            return "full"
        # Multi-sentence → likely needs reasoning
        sentence_count = len(re.findall(r"[.!?]+", stripped))
        if sentence_count >= 3:
            return "full"

        # ── Factual one-liners → concise ─────────────────────────────────────
        if _FACTUAL_RE.match(low):
            return "concise"

        # ── Default: light auto mode ──────────────────────────────────────────
        return "auto"


# ── Instruction strings ───────────────────────────────────────────────────────

_MODE_INSTRUCTIONS: dict[str, str] = {
    "concise": (
        "[THINKING MODE: CONCISE] "
        "Answer directly and briefly. "
        "Do NOT show internal reasoning steps or lengthy preamble. "
        "One to three sentences maximum unless code is needed."
    ),
    "auto": (
        "[THINKING MODE: AUTO] "
        "Show brief reasoning only when it genuinely aids clarity "
        "(e.g. a short derivation or single disambiguation step). "
        "Skip preamble; get to the answer quickly."
    ),
    "full": (
        "[THINKING MODE: FULL] "
        "Think step-by-step and show all reasoning before the final answer. "
        "Use structured sections, numbered steps, or code blocks as appropriate. "
        "Do not skip intermediate logic — thoroughness is valued here."
    ),
}
