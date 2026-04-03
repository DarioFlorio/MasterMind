"""
Skill: skill_router
Intelligent meta-router: analyses a problem and recommends the best skill(s)
to apply, with reasoning. Can chain multiple skills.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("skill.skill_router")

DESCRIPTION = (
    "Meta-router: analyses any problem and recommends which reasoning skill(s) "
    "to apply, in what order, and why. Use when unsure which skill fits, "
    "or when a problem spans multiple skill domains."
)

# Registry of all skills with keywords and descriptions
_SKILL_REGISTRY: list[dict] = [
    {
        "name": "reason_chain",
        "keywords": ["chain", "combine", "multi-skill", "compound", "pipeline",
                     "sequence of reasoning", "layer", "multi-step reasoning"],
        "description": "Auto-selects and chains multiple reasoning skills in sequence.",
        "best_for": "Hard multi-faceted problems needing more than one reasoning lens.",
    },
    {
        "name": "inductive_reason",
        "keywords": ["sequence", "pattern", "rule", "next number", "series",
                     "inductive", "generalise", "generalize", "number series",
                     "what comes next", "find the rule"],
        "description": "Inductive reasoning: find rules from sequences or examples.",
        "best_for": "Number sequences, pattern series, rule extraction, scientific laws.",
    },
    {
        "name": "deep_reason",
        "keywords": ["complex", "analyse", "analyze", "multi-step", "comprehensive", "thorough"],
        "description": "Deep multi-level recursive reasoning for complex analytical problems.",
        "best_for": "General complex questions needing structured decomposition.",
    },
    {
        "name": "cot_reason",
        "keywords": ["step by step", "calculate", "compute", "logic", "math", "derive"],
        "description": "Step-by-step chain-of-thought for maths, logic, multi-step problems.",
        "best_for": "Calculation, logic chains, sequential reasoning.",
    },
    {
        "name": "constraint_solve",
        "keywords": ["puzzle", "riddle", "logic grid", "zebra", "einstein", "knight", "knave",
                     "constraint", "satisf", "rules", "who owns", "who drinks"],
        "description": "Constraint satisfaction for logic puzzles and truth puzzles.",
        "best_for": "Logic grids, Einstein riddles, knight/knave, CSP problems.",
    },
    {
        "name": "game_solve",
        "keywords": ["game", "nim", "minimax", "chess", "strategy", "who wins", "optimal play",
                     "prisoner", "dilemma", "nash", "auction", "equilibrium"],
        "description": "Game theory: minimax, Nash equilibria, optimal strategies.",
        "best_for": "Two-player games, strategic interactions, game trees.",
    },
    {
        "name": "bayes_reason",
        "keywords": ["probability", "bayesian", "prior", "posterior", "monty hall",
                     "base rate", "likelihood", "conditional probability"],
        "description": "Bayesian inference, base rates, conditional probability.",
        "best_for": "Probability puzzles, Bayesian updates, medical tests, Monty Hall.",
    },
    {
        "name": "abduct",
        "keywords": ["explain", "diagnosis", "best explanation", "why did", "what caused",
                     "detective", "whodunit", "bug", "debug", "mystery", "infer"],
        "description": "Abductive reasoning: inference to the best explanation.",
        "best_for": "Diagnosis, debugging, detective problems, anomaly explanation.",
    },
    {
        "name": "analogical_reason",
        "keywords": ["analogy", "analogous", "similar to", "like a", "metaphor",
                     "mapping", "domain transfer", "structure"],
        "description": "Analogical reasoning: structural mappings between domains.",
        "best_for": "Understanding via analogy, domain transfer, structural comparison.",
    },
    {
        "name": "timeline_reason",
        "keywords": ["timeline", "order", "sequence", "before", "after", "schedule",
                     "dependency", "when", "history", "chronology"],
        "description": "Temporal sequencing, timeline consistency, scheduling.",
        "best_for": "Ordering events, detecting timeline conflicts, project scheduling.",
    },
    {
        "name": "causal_reason",
        "keywords": ["root cause", "what caused", "why did", "because", "5 why",
                     "fishbone", "counterfactual", "effect of"],
        "description": "Causal reasoning: root cause analysis, counterfactuals.",
        "best_for": "Root cause analysis, explaining past events, counterfactuals.",
    },
    {
        "name": "recursive_decompose",
        "keywords": ["decompose", "break down", "subproblem", "divide", "hanoi",
                     "recursive", "hierarchical", "sub-goal", "step-wise"],
        "description": "Recursive problem decomposition into sub-problems.",
        "best_for": "Large problems needing hierarchical breakdown, recursive structures.",
    },
    {
        "name": "epistemic_reason",
        "keywords": ["know", "believe", "justified", "evidence quality", "certain",
                     "confidence", "uncertainty", "how sure", "source reliability"],
        "description": "Epistemic reasoning: knowledge, belief, justification, evidence.",
        "best_for": "Evaluating evidence, knowledge vs belief, calibration.",
    },
    {
        "name": "lateral_thinking",
        "keywords": ["creative", "lateral", "trick", "unexpected", "unusual", "outside the box",
                     "original", "non-obvious", "different approach"],
        "description": "Lateral / creative out-of-the-box thinking.",
        "best_for": "Creative problems, riddles, finding unexpected solutions.",
    },
    {
        "name": "multi_objective",
        "keywords": ["trade-off", "balance", "optimise", "pareto", "criteria", "decision matrix",
                     "priority", "multiple goals", "best option given"],
        "description": "Multi-objective optimisation and trade-off balancing.",
        "best_for": "Decisions with conflicting criteria, prioritisation, Pareto analysis.",
    },
    # Forward / predictive skills
    {
        "name": "causal_forward_reason",
        "keywords": ["what happens if", "consequence", "effect", "butterfly", "cascade",
                     "downstream", "ripple", "impact of", "what will happen"],
        "description": "Causal forward reasoning: trace cascading consequences.",
        "best_for": "What happens next, butterfly effects, second-order consequences.",
    },
    {
        "name": "timeline_projection_reason",
        "keywords": ["when will", "roadmap", "milestone", "project forward", "by when",
                     "timeline for", "trajectory", "what comes next"],
        "description": "Timeline projection: map future milestones and sequences.",
        "best_for": "Future timelines, roadmaps, milestone sequencing.",
    },
    {
        "name": "scenario_whatif_simulation",
        "keywords": ["what if", "scenario", "alternative future", "best case", "worst case",
                     "stress test", "if instead", "hypothetical"],
        "description": "Scenario / what-if simulation with branching futures.",
        "best_for": "Scenario planning, what-if branches, stress-testing assumptions.",
    },
    {
        "name": "probabilistic_forecasting",
        "keywords": ["probability that", "how likely", "forecast", "predict", "chance of",
                     "superforecasting", "calibrate", "brier", "odds"],
        "description": "Probabilistic forecasting with Bayesian updating.",
        "best_for": "Assigning probabilities to future events, calibrated forecasts.",
    },
    {
        "name": "game_theoretic_forward_simulation",
        "keywords": ["competitor", "how will they react", "move and counter-move", "arms race",
                     "negotiation", "market competition", "what will player", "strategic response"],
        "description": "Game-theoretic forward simulation of moves and counter-moves.",
        "best_for": "Predicting how strategic actors will respond over time.",
    },
    {
        "name": "multi_objective_future_optimization",
        "keywords": ["optimise for both", "robust strategy", "conflicting future goals",
                     "adaptive plan", "multiple objectives over time"],
        "description": "Multi-objective future optimisation across scenarios.",
        "best_for": "Strategies robust across multiple future scenarios and objectives.",
    },
    {
        "name": "recursive_future_decomposition",
        "keywords": ["fermi", "bottom-up forecast", "component prediction", "sub-forecast",
                     "break the prediction", "chained prediction"],
        "description": "Recursive future decomposition: chain sub-predictions.",
        "best_for": "Complex forecasts that can be broken into tractable sub-questions.",
    },
    {
        "name": "deep_multi_layer_prediction",
        "keywords": ["deep prediction", "layers", "emergent", "civilisational", "meta-level",
                     "stacked forecast", "social", "institutional change"],
        "description": "Deep multi-layer prediction stacking physical to cultural layers.",
        "best_for": "Long-arc societal predictions, emergent future phenomena.",
    },
    {
        "name": "lateral_forward_thinking",
        "keywords": ["wild card", "black swan", "unexpected future", "non-obvious",
                     "assumption breaking", "creative future", "what am i missing"],
        "description": "Lateral forward thinking: non-obvious future paths.",
        "best_for": "Finding surprising futures, assumption-busting, wild cards.",
    },
    {
        "name": "epistemic_future_reasoning",
        "keywords": ["what will people know", "belief change", "consensus will", "opinion",
                     "information spread", "knowledge diffusion", "what will be accepted"],
        "description": "Epistemic future reasoning: predict future knowledge states.",
        "best_for": "How beliefs/knowledge will evolve, information diffusion.",
    },
]


def _rank_skills(problem: str, top_n: int) -> list[dict]:
    low = problem.lower()
    scored = []
    for skill in _SKILL_REGISTRY:
        score = sum(1 for kw in skill["keywords"] if kw in low)
        if score > 0:
            scored.append((score, skill))

    # Sort by score desc; break ties by name for determinism
    scored.sort(key=lambda x: (-x[0], x[1]["name"]))

    # Return top N, or all matches if fewer than N
    top = [s for _, s in scored[:top_n]]

    # If no matches, return the generic deep_reason
    if not top:
        generic = next(s for s in _SKILL_REGISTRY if s["name"] == "deep_reason")
        top = [generic]

    return top


def _format_recommendation(problem: str, ranked: list[dict]) -> str:
    if not ranked:
        return "No matching skill found. Use 'deep_reason' as a general fallback."

    primary = ranked[0]
    alts    = ranked[1:]

    alt_block = ""
    if alts:
        alt_block = "\n\n**Also consider:**\n" + "\n".join(
            f"  - `{s['name']}`: {s['best_for']}" for s in alts
        )

    chain_block = ""
    if len(ranked) >= 2:
        chain_block = f"""
**Skill Chaining Suggestion:**
  For maximum depth, chain the top skills:
  1. First run: `{ranked[0]['name']}` — to establish the core reasoning framework
  2. Then run: `{ranked[1]['name']}` — to deepen with specialised analysis
  {f"3. Then run: `{ranked[2]['name']}` — to add the third dimension" if len(ranked) >= 3 else ""}

  Pass the output of each skill as the 'context' argument to the next."""

    return f"""**Skill Router — Recommendation**
Problem analysed: {problem[:120]}{"..." if len(problem) > 120 else ""}

**Primary recommendation: `{primary['name']}`**
  {primary['description']}
  Best for: {primary['best_for']}

  To use:
  ```
  skill: {primary['name']}
  args: {{
    "problem": "<your problem here>",
    ... (skill-specific args)
  }}
  ```
{alt_block}
{chain_block}

**All 24 available skills:**
{chr(10).join(f'  {i+1:2d}. `{s["name"]}` — {s["best_for"]}' for i, s in enumerate(_SKILL_REGISTRY))}
"""


def _list_all_skills() -> str:
    categories = {
        "Core Reasoning": [
            "deep_reason", "cot_reason", "constraint_solve", "game_solve",
            "bayes_reason", "abduct", "analogical_reason", "timeline_reason",
            "causal_reason", "recursive_decompose", "epistemic_reason",
            "lateral_thinking", "multi_objective", "skill_router"
        ],
        "Forward / Predictive": [
            "causal_forward_reason", "timeline_projection_reason",
            "scenario_whatif_simulation", "probabilistic_forecasting",
            "game_theoretic_forward_simulation", "multi_objective_future_optimization",
            "recursive_future_decomposition", "deep_multi_layer_prediction",
            "lateral_forward_thinking", "epistemic_future_reasoning"
        ]
    }

    blocks = []
    for cat, names in categories.items():
        blocks.append(f"\n**{cat} Skills:**")
        for name in names:
            skill = next((s for s in _SKILL_REGISTRY if s["name"] == name), None)
            if skill:
                blocks.append(f"  - `{name}`: {skill['best_for']}")
            else:
                blocks.append(f"  - `{name}`")

    return (
        "**Skill Router — All Available Skills (24 total)**\n"
        + "\n".join(blocks)
        + "\n\nCall skill_router with a 'problem' argument to get a recommendation."
    )

from skills.base_skill import BaseSkill


class SkillRouterSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "skill_router"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth":   {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "top_n": {"type": "integer"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        top_n = int(kwargs.get("top_n", 3))
        ranked = _rank_skills(problem, top_n)
        return _format_recommendation(problem, ranked)
