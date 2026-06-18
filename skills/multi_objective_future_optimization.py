"""
Skill: multi_objective_future_optimization
Multi-objective future optimization: balance conflicting future outcomes,
find strategies that are robust across multiple goals and scenarios.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.multi_objective_future_optimization")

DESCRIPTION = (
    "Multi-objective future optimization: find strategies that best balance "
    "conflicting future goals across multiple scenarios. Use for 'how do I "
    "optimise for both X and Y?', 'what strategy works across different futures?', "
    "'robust planning under uncertainty with multiple objectives'."
)


def _optimize(problem: str, objectives: list, horizon: str) -> str:
    obj_block = ("\nObjectives: " + ", ".join(objectives)) if objectives else ""
    return f"""**Multi-Objective Future Optimization**
Problem: {problem}{obj_block}
Horizon: {horizon}

**Framework: Robust Multi-Objective Strategy Design**

**Step 1 — Define Future Objectives**
  List all goals that matter over the {horizon} horizon.
  For each objective:
    - Direction: maximise / minimise / achieve threshold
    - Time sensitivity: when does it matter most?
    - Measurement: how will success be assessed?
    - Decay rate: does delayed achievement reduce value?

**Step 2 — Map Objective Conflicts**
  Identify pairs of objectives that are in tension:
    High speed ↔ high quality
    Short-term profit ↔ long-term sustainability
    Individual gain ↔ collective welfare
    Risk reduction ↔ expected return

  For each conflict:
    - Is there a Pareto frontier? (No solution dominates all others)
    - Is there a dominant solution? (One option is best on ALL objectives)
    - Is there a threshold below which sacrifice is unacceptable?

**Step 3 — Scenario Matrix**
  Construct 3–4 future scenarios (combinations of key uncertainties).
  For each scenario, evaluate how each candidate strategy performs on each objective.

  | Strategy | Scenario A | Scenario B | Scenario C | Weighted Avg |
  |----------|-----------|-----------|-----------|--------------|
  | Strategy 1 | [score]  | [score]   | [score]   | [compute]   |
  | Strategy 2 | [score]  | [score]   | [score]   | [compute]   |
  | Strategy 3 | [score]  | [score]   | [score]   | [compute]   |

**Step 4 — Identify Robust Strategies**
  A strategy is ROBUST if it performs acceptably across ALL scenarios on ALL objectives.
  Not necessarily optimal in any scenario — but never catastrophic.

  Robustness score = min(performance across all scenario × objective combinations)
  (Maximax = optimistic; Maximin = pessimistic; Robust = avoids worst cases)

**Step 5 — Adaptive Strategy Design**
  Instead of a fixed strategy, design a decision tree:
  - Start with robust actions that work in all scenarios.
  - Define trigger conditions that indicate which scenario is unfolding.
  - Plan pivot actions for each branch:
    "If we observe [X] by [time], we shift from [Strategy A] to [Strategy B]."

**Step 6 — Option Value**
  Which actions preserve optionality (keep future choices open)?
  Which actions foreclose options (lock in one path)?
  In high-uncertainty environments: prefer option-preserving actions initially.
  Exercise options (commit) only when uncertainty has resolved sufficiently.

**Step 7 — Recommendation**
  Robust baseline strategy: [actions to take now regardless of scenario]
  Key decision point 1: [when, what trigger, what branches]
  Key decision point 2: [when, what trigger, what branches]

  Expected performance:
    Best-case scenario: [outcomes on each objective]
    Base-case scenario: [outcomes on each objective]
    Worst-case scenario: [outcomes on each objective]
"""

from skills.base_skill import BaseSkill


class MultiObjectiveFutureOptimizationSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "multi_objective_future_optimization"

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
                "horizon":    {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        objectives = kwargs.get("objectives", [])
        horizon    = kwargs.get("horizon", "medium-term")
        return _optimize(problem, objectives, horizon)
