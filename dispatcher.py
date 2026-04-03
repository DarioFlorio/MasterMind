"""
agent/dispatcher.py — Query routing and intent classification.
Routes to one of: RECALL, SKILL, COT, BACKGROUND, NORMAL.
Covers all 24 reasoning skills.
"""
from __future__ import annotations
import re
import difflib
import logging
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("agent.dispatcher")


class Route(str, Enum):
    RECALL     = "recall"
    SKILL      = "skill"
    COT        = "cot"
    BACKGROUND = "background"
    NORMAL     = "normal"


# ── Pattern banks ──────────────────────────────────────────────────────────────

_BG = re.compile(
    r"\b(research|investigate|compile|gather|find everything|find all|"
    r"deep dive|full report|analyse everything|monitor|track|"
    r"in the background|while i|keep an? eye)\b",
    re.IGNORECASE,
)

_COT = re.compile(
    r"\b(why|how does|explain|compare|difference between|analyse|analyze|"
    r"pros and cons|should i|best way|step by step|plan|strategy|"
    r"calculate|solve|reason|think through|predict|evaluate|critique|"
    r"compute|convert.*unit|how fast|speed of)\b",
    re.IGNORECASE,
)

# ── Core reasoning skill patterns ──────────────────────────────────────────────
_SKILL_CORE = re.compile(
    r"\b("
    # constraint_solve
    r"who (owns|drinks|lives|keeps|smokes|has a)|"
    r"which (house|person|nationality|pet|drink|colour|cigarette)|"
    r"zebra puzzle|einstein riddle|logic grid|"
    r"knight.*knave|knave.*knight|truth.teller|always (lies|tells truth)|"
    # game_solve
    r"who wins|optimal (strategy|play)|game tree|minimax|nim|"
    r"prisoner.s dilemma|nash equilibrium|auction strategy|"
    # bayes_reason
    r"monty hall|base rate|prior.*posterior|bayesian|"
    r"false positive|sensitivity.*specificity|"
    # abduct — expanded
    r"who (did it|killed|stole)|murder mystery|whodunit|suspect|alibi|"
    r"best explanation|diagnos\w*|abduct|what explains|"
    r"what would explain|infer.*from|"
    # recursive_decompose
    r"tower of hanoi|hanoi|recursive(ly)?|divide.and.conquer|"
    # causal_reason (backward) — expanded
    r"root cause|what caused|5 why|fishbone|counterfactual|"
    r"what if .* hadn.t|cause.*effect|why did (this|it|the)|"
    # lateral_thinking
    r"lateral thinking|trick question|word puzzle|riddle|"
    # timeline_reason
    r"(in what|which) order|what (came|happened) (first|before|after)|"
    r"timeline.*consistent|sequence of events|"
    # game puzzles
    r"river crossing|farmer.*wolf|water jug|jug problem|wolf.*sheep|"
    r"solve.*(puzzle|riddle)|logic puzzle|"
    # multi_objective
    r"trade.off|pareto|decision matrix|multiple (goals?|criteria|objectives?)|"
    # inductive_reason (EVE import)
    r"find the rule|what is the rule|what.s the pattern|"
    r"next number|predict the sequence|number sequence|"
    r"what comes next in|inductive reasoning|"
    r"generalise from examples|generalize from examples|what rule generates"
    r")\b",
    re.IGNORECASE,
)

# ── Forward / predictive skill patterns ───────────────────────────────────────
_SKILL_FORWARD = re.compile(
    r"\b("
    # causal_forward_reason
    r"what (happens|will happen) (if|when)|butterfly effect|cascade|"
    r"downstream effect|ripple effect|second.order|unintended consequence|"
    r"consequence of|impact of.*on|"
    # timeline_projection_reason
    r"when will|roadmap|milestone|project.*forward|trajectory|"
    r"what comes next|future sequence|"
    # scenario_whatif_simulation
    r"what if|scenario plan|best case|worst case|stress.test|"
    r"alternative future|hypothetical.*future|branching|"
    # probabilistic_forecasting
    r"probability (that|of)|how likely|forecast|superforecast|"
    r"chance of|predict.*likelihood|calibrat|brier score|"
    # game_theoretic_forward_simulation
    r"how will (they|competitor|rival|opponent) react|"
    r"move.*counter.move|arms race|negotiation dynamic|"
    r"what will (player|actor|company|country) do next|"
    # multi_objective_future_optimization
    r"robust (strategy|plan)|optimise.*future|"
    r"balance.*conflicting.*goal|adaptive (plan|strategy)|"
    # recursive_future_decomposition
    r"fermi (estimate|forecast|decompos)|component.*prediction|sub.forecast|"
    r"break.*(prediction|forecast)|chained prediction|"
    # deep_multi_layer_prediction
    r"deep.*prediction|layer.*prediction|emergent.*future|"
    r"stacked forecast|civilisational|long.arc|"
    # lateral_forward_thinking
    r"wild card|black swan|unexpected future|non.obvious future|"
    r"assumption.*break|creative future|what am i missing.*future|"
    # epistemic_future_reasoning
    r"what will people know|belief.*change.*future|consensus will|"
    r"opinion.*evolve|information.*spread|knowledge.*diffus|"
    r"what will be (accepted|known|believed)"
    r")\b",
    re.IGNORECASE,
)

_RECALL_KW = (
    "remember", "recall", "last session", "last time", "last conversation",
    "last chat", "what were we", "do you remember", "previously", "remind me",
    "catch me up", "bring me up", "what did i", "what was i", "forgot",
    "what we discussed", "what we talked about", "our conversation",
    "you were meant", "i told you", "what did you find", "in your memory",
    "show me what you found", "where we left",
)

_RECALL_ANCHORS = [
    "what were we talking about", "what did we discuss",
    "what were we working on", "do you remember what we",
    "what was the last thing we", "remind me what we did",
    "catch me up on what we did", "what have we been working on",
    "recall our conversation", "what did we talk about",
    "what did i tell you about", "what were we just discussing",
]


@dataclass
class DispatchResult:
    route:      Route
    skill_hint: str = ""      # suggested skill name if route == SKILL
    cot:        bool = False  # inject chain-of-thought


def classify(text: str) -> DispatchResult:
    low = text.lower().strip().rstrip("?.!")
    log.debug("dispatcher.classify: route analysis for %r", low[:60])

    # ── 1. Hard recall keywords (exact substring match) ───────────────────
    if any(k in low for k in _RECALL_KW):
        log.debug("dispatcher → RECALL (keyword match)")
        return DispatchResult(route=Route.RECALL)

    # ── 2. Background ──────────────────────────────────────────────────────
    if _BG.search(text):
        log.debug("dispatcher → BACKGROUND")
        return DispatchResult(route=Route.BACKGROUND, cot=False)

    # ── 3. Core skill — checked BEFORE fuzzy recall to avoid false positives
    if _SKILL_CORE.search(text):
        hint = _pick_core_skill(text)
        log.debug("dispatcher → SKILL (core) hint=%s", hint)
        return DispatchResult(route=Route.SKILL, skill_hint=hint, cot=True)

    # ── 4. Forward / predictive skill ────────────────────────────────────
    if _SKILL_FORWARD.search(text):
        hint = _pick_forward_skill(text)
        log.debug("dispatcher → SKILL (forward) hint=%s", hint)
        return DispatchResult(route=Route.SKILL, skill_hint=hint, cot=True)

    # ── 5. Fuzzy recall anchors (only after ruling out skills) ────────────
    if len(low.split()) >= 3:
        if difflib.get_close_matches(low, _RECALL_ANCHORS, n=1, cutoff=0.72):
            log.debug("dispatcher → RECALL (fuzzy anchor match)")
            return DispatchResult(route=Route.RECALL)

    # ── 6. CoT ─────────────────────────────────────────────────────────────
    if _COT.search(text):
        log.debug("dispatcher → COT")
        return DispatchResult(route=Route.COT, cot=True)

    # ── 7. Normal ──────────────────────────────────────────────────────────
    log.debug("dispatcher → NORMAL")
    return DispatchResult(route=Route.NORMAL)


def _pick_core_skill(text: str) -> str:
    """Map a core-skill-route query to the most appropriate skill module."""
    low = text.lower()

    if any(k in low for k in ("bayesian", "prior", "posterior", "monty hall", "base rate",
                               "false positive", "sensitivity")):
        return "bayes_reason"

    if any(k in low for k in ("root cause", "what caused", "5 why", "fishbone")):
        return "causal_reason"

    if any(k in low for k in ("counterfactual", "what if.*hadn", "hadn't", "would have")):
        return "causal_reason"

    if any(k in low for k in ("zebra", "einstein riddle", "logic grid",
                               "who owns", "who drinks", "who lives",
                               "river crossing", "farmer", "wolf", "water jug",
                               "jug problem", "logic puzzle")):
        return "constraint_solve"

    if any(k in low for k in ("riddle", "lateral", "trick question", "word puzzle")):
        return "lateral_thinking"

    if any(k in low for k in ("constrain", "logic grid", "zebra", "einstein",
                               "knight", "knave", "who owns", "who drinks")):
        return "constraint_solve"

    if any(k in low for k in ("hanoi", "recursive", "divide and conquer", "subgoal",
                               "decompose")):
        return "recursive_decompose"

    if any(k in low for k in ("who wins", "optimal play", "minimax", "nim",
                               "prisoner", "dilemma", "nash", "auction")):
        return "game_solve"

    if any(k in low for k in ("timeline", "schedule", "order", "before", "after",
                               "overlap", "sequence", "chronolog")):
        return "timeline_reason"

    if any(k in low for k in ("analogous", "analogical", "similar to", "like a",
                               "metaphor", "mapping")):
        return "analogical_reason"

    if any(k in low for k in ("uncertain", "epistemic", "know that", "believe",
                               "justified", "evidence quality")):
        return "epistemic_reason"

    if any(k in low for k in ("trade-off", "pareto", "criteria", "decision matrix",
                               "multiple goal", "optimis")):
        return "multi_objective"

    if any(k in low for k in ("best explanation", "diagnos", "diagnosis", "diagnostic", "abduct", "suspect",
                               "whodunit", "detective", "murder mystery")):
        return "abduct"

    if any(k in low for k in (
        "find the rule", "number sequence", "next number",
        "what comes next in", "inductive", "generalise", "generalize",
        "what rule", "predict the sequence", "what is the rule",
    )):
        return "inductive_reason"

    return "deep_reason"


def _pick_forward_skill(text: str) -> str:
    """Map a forward-skill-route query to the most appropriate skill module."""
    low = text.lower()

    if any(k in low for k in ("butterfly", "cascade", "ripple", "second-order",
                               "unintended consequence", "downstream")):
        return "causal_forward_reason"

    if any(k in low for k in ("deep prediction", "deep multi-layer", "stacked forecast",
                               "layer prediction", "emergent future",
                               "civilisational", "long-arc")):
        return "deep_multi_layer_prediction"

    if any(k in low for k in ("fermi", "sub-forecast", "sub forecast",
                               "component prediction", "chained prediction",
                               "break the prediction")):
        return "recursive_future_decomposition"

    if any(k in low for k in ("what if", "scenario", "best case", "worst case",
                               "stress test", "hypothetical", "branching")):
        return "scenario_whatif_simulation"

    if any(k in low for k in ("probability that", "how likely", "forecast",
                               "superforecast", "chance of", "brier", "calibrat")):
        return "probabilistic_forecasting"

    if any(k in low for k in ("how will they react", "competitor", "rival",
                               "move and counter", "arms race", "negotiation dynamic",
                               "what will player", "what will actor")):
        return "game_theoretic_forward_simulation"

    if any(k in low for k in ("when will", "roadmap", "milestone", "trajectory",
                               "what comes next", "future sequence")):
        return "timeline_projection_reason"

    if any(k in low for k in ("robust strategy", "optimise future",
                               "balance conflicting", "adaptive plan")):
        return "multi_objective_future_optimization"

    if any(k in low for k in ("wild card", "black swan", "unexpected future",
                               "non-obvious", "assumption break", "what am i missing")):
        return "lateral_forward_thinking"

    if any(k in low for k in ("what will people know", "belief change future",
                               "consensus will", "opinion evolve",
                               "knowledge diffus", "information spread")):
        return "epistemic_future_reasoning"

    # Generic: consequence / what happens if → causal forward
    if any(k in low for k in ("what happens if", "consequence of",
                               "impact of", "what will happen")):
        return "causal_forward_reason"

    return "causal_forward_reason"
