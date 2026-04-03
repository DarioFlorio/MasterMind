"""
Skill: multi_objective
Multi-objective optimization: balance conflicting goals, trade-off analysis,
Pareto fronts, weighted scoring, decision matrices.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.multi_objective")

DESCRIPTION = (
    "Multi-objective optimization: balance conflicting goals, find Pareto-optimal "
    "solutions, decision matrix analysis, weighted trade-offs. Use for 'best option "
    "given multiple criteria', 'what should I prioritise?', 'trade-off analysis'."
)


def _decision_matrix(problem: str, objectives: list, options: list, weights: list) -> str:
    n_obj = len(objectives)
    n_opt = len(options)
    w = weights if len(weights) == n_obj else [1.0 / n_obj] * n_obj

    header = "| Option | " + " | ".join(objectives) + " | **Weighted Score** |"
    sep    = "|--------|" + "--------|" * n_obj + "-----------------|"

    return f"""**Multi-Objective Decision Matrix**
Problem: {problem}

**Objectives:** {', '.join(objectives)}
**Weights:** {', '.join(f'{o}={w[i]:.2f}' for i, o in enumerate(objectives))}
**Options:** {', '.join(options)}

**Decision Matrix Template:**
{header}
{sep}
{chr(10).join(f'| {opt} | ' + ' | '.join(['_/10_'] * n_obj) + ' | _compute_ |' for opt in options)}

**How to Fill In:**
  Score each option on each objective from 1–10 (10 = best satisfies that objective).
  Weighted Score = Σ (score_i × weight_i) for all objectives.

**Weighted Scoring Formula:**
  WS(option) = score₁×w₁ + score₂×w₂ + ... + scoreₙ×wₙ

**Sensitivity Analysis:**
  After scoring, vary the weights by ±20% each.
  If the top option changes: your choice is weight-sensitive → discuss priorities more carefully.
  If it stays stable: robust recommendation.

**Pareto Check:**
  Also check if any option is dominated (another option scores ≥ on ALL objectives).
  Dominated options can be eliminated before weighting.
"""


def _pareto_analysis(problem: str) -> str:
    return f"""**Multi-Objective: Pareto Analysis**
Problem: {problem}

**Pareto Dominance:**
  Solution A dominates solution B if:
    - A is at least as good as B on ALL objectives, AND
    - A is strictly better than B on at least one objective.
  Dominated solutions can be safely eliminated.

**Pareto Front:**
  The set of non-dominated solutions.
  No solution in the Pareto front can be improved on one objective
  without getting worse on another.
  These are the "efficient" choices — the true trade-off frontier.

**How to Construct the Pareto Front:**
  Step 1: List all candidate solutions with their objective values.
  Step 2: For each solution, check if any other solution dominates it.
  Step 3: Non-dominated solutions form the Pareto front.
  Step 4: Visualise (for 2 objectives): plot objective₁ vs objective₂.
          The front is the upper-right "staircase" of non-dominated points.

**Choosing From the Pareto Front:**
  All Pareto-optimal solutions are equally valid from a pure efficiency standpoint.
  To pick ONE, you must introduce preferences:
    a) Weighted sum (linear scalarisation): combine objectives with weights.
    b) Lexicographic: prioritise objectives in order; break ties with next.
    c) Goal programming: minimise distance from an ideal target point.
    d) Reference point: pick the solution closest to an aspiration vector.

**Apply to this problem:**
  Enumerate solutions, eliminate dominated ones, map the Pareto front,
  then apply stakeholder preferences to select.
"""


def _priority_framework(problem: str) -> str:
    return f"""**Multi-Objective: Priority Framework**
Problem: {problem}

**Frameworks for Prioritisation:**

**1. RICE Score** (Product Management):
  RICE = (Reach × Impact × Confidence) / Effort
  Rank by descending RICE. High reach + impact + confidence + low effort = do first.

**2. Eisenhower Matrix** (Urgency × Importance):
  Q1: Urgent + Important    → Do immediately
  Q2: Not urgent + Important → Schedule (highest long-term value)
  Q3: Urgent + Not important → Delegate
  Q4: Not urgent + Not important → Eliminate

**3. MoSCoW** (Requirement prioritisation):
  Must have / Should have / Could have / Won't have this time.

**4. Weighted Criteria Matrix** (when multiple dimensions matter):
  Step 1: List criteria. Step 2: Weight them (sum to 1). Step 3: Score each option.
  Step 4: Rank by weighted score.

**5. Opportunity Cost Framework:**
  For each option: what must you give up to pursue it?
  The best option has the highest value AND lowest opportunity cost.

**Apply to this problem:**
  Choose the framework most suited to the decision type.
  Make weights and scores explicit — hidden assumptions lead to poor decisions.
  Perform sensitivity check: does the ranking change if weights shift ±30%?
"""


def _general_multiobjective(problem: str, objectives: list, options: list, weights: list) -> str:
    obj_block = ("\nObjectives: " + ", ".join(objectives)) if objectives else ""
    opt_block = ("\nOptions: " + ", ".join(options)) if options else ""
    return f"""**Multi-Objective Optimization**
Problem: {problem}{obj_block}{opt_block}

**Step 1 — Identify and Define Objectives**
  List every goal that matters. For each:
  - Is it something to MAXIMISE or MINIMISE?
  - How is it measured? (Quantitative preferred; qualitative needs a scale.)
  - What is the acceptable range? (Hard constraint vs soft preference?)

**Step 2 — Separate Constraints from Objectives**
  Hard constraints: must be satisfied (non-negotiable). Eliminate options that fail.
  Soft objectives: want to optimise (can trade off). These go in the matrix.

**Step 3 — Generate Options**
  Ensure the option space is broad enough.
  Include: status quo, extreme/pure options, hybrid/compromise options.

**Step 4 — Eliminate Dominated Options**
  If option A ≥ B on every objective and A > B on at least one: eliminate B.
  Only non-dominated options are worth detailed analysis.

**Step 5 — Apply Weights and Score**
  Elicit stakeholder weights (must sum to 1).
  Score each remaining option on each objective (1–10 scale).
  Compute weighted score = Σ (score × weight).

**Step 6 — Sensitivity Analysis**
  Does the top-ranked option remain top if weights shift by ±25%?
  If yes: robust recommendation. If no: the decision is weight-sensitive — surface this explicitly.

**Step 7 — Recommendation**
  State: recommended option, runner-up, key trade-offs made, and conditions
  under which a different option would be preferred.
"""

from skills.base_skill import BaseSkill


class MultiObjectiveSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "multi_objective"

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
                "objectives": {"type": "array", "items": {"type": "string"}},
                "options":    {"type": "array", "items": {"type": "string"}},
                "weights":    {"type": "array", "items": {"type": "number"}},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        objectives = kwargs.get("objectives", [])
        options    = kwargs.get("options", [])
        weights    = kwargs.get("weights", [])
        low = problem.lower()
        if objectives and options:
            return _decision_matrix(problem, objectives, options, weights)
        if any(k in low for k in ("pareto", "frontier", "front", "dominant")):
            return _pareto_analysis(problem)
        if any(k in low for k in ("priority", "priorit", "rank", "what first", "which first")):
            return _priority_framework(problem)
        return _general_multiobjective(problem, objectives, options, weights)
