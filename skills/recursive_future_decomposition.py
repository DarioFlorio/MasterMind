"""
Skill: recursive_future_decomposition
Recursive future decomposition: break big future problems into chained
sub-predictions, solve each level, synthesise upward.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.recursive_future_decomposition")

DESCRIPTION = (
    "Recursive future decomposition: break big future questions into smaller "
    "sub-predictions, solve each, and chain them into an overall forecast. "
    "Use for complex multi-part futures, 'how do we get from here to X?', "
    "large uncertain predictions that can be broken into components."
)


def _decompose(problem: str, depth: int) -> str:
    return f"""**Recursive Future Decomposition**
Problem: {problem}
Decomposition depth: {depth} levels

**Philosophy:**
  Complex future questions are hard to forecast directly.
  Break them into sub-questions, each of which is easier to estimate.
  The overall forecast is then derived from the sub-forecasts.
  (Fermi estimation applied to futures.)

**Level 0 — The Master Question**
  Q: {problem}
  Direct estimate: [X% — your gut/holistic estimate]
  Uncertainty: [very high / high / medium / low]

  Why is this hard to estimate directly?
  → [Identify the key driver of uncertainty]

**Level 1 — First Decomposition**
  Break the master question into 3–5 necessary and sufficient sub-questions.
  "If I knew the answers to all these sub-questions, I could answer the master question."

  Sub-question 1: [...]  → Estimate: [p1%]
  Sub-question 2: [...]  → Estimate: [p2%]
  Sub-question 3: [...]  → Estimate: [p3%]
  {f'Sub-question 4: [...]  → Estimate: [p4%]' if depth >= 2 else ''}
  {f'Sub-question 5: [...]  → Estimate: [p5%]' if depth >= 3 else ''}

**Level 2 — Decompose the Hardest Sub-questions**
  Take whichever sub-question above is most uncertain.
  Decompose IT further:

  Sub-question X → Sub-sub-questions:
    2.1: [...]  → Estimate: [p2.1%]
    2.2: [...]  → Estimate: [p2.2%]
    2.3: [...]  → Estimate: [p2.3%]

  Now estimate sub-question X from its components:
  P(X) = f(p2.1, p2.2, p2.3)  [show the functional relationship]

{f"""**Level 3 — Third-Level Decomposition (if needed)**
  Apply the same process to any remaining highly uncertain sub-sub-questions.
  Stop when sub-questions are answerable from known data or strong base rates.
""" if depth >= 3 else ""}

**Synthesis — Bottom Up**
  Starting from the deepest level, propagate estimates upward:

  Level {depth} estimates → feed into → Level {depth-1} estimates
  Level {depth-1} estimates → feed into → Level {depth-2} estimates
  ...
  Level 1 estimates → combine to → Master Question estimate

  **Combination rules:**
  - AND (all must happen): multiply probabilities: P = p1 × p2 × p3
  - OR (any can happen): P = 1 − (1−p1)(1−p2)(1−p3)
  - Weighted average: P = Σ(wi × pi) when outcomes are not independent chains

**Uncertainty Propagation**
  Each level adds uncertainty. Track ranges, not just point estimates.
  Final estimate: P(master question) = [X%]
  90% confidence interval: [[low%] – [high%]]

**Sanity Check**
  Compare the bottom-up estimate to the Level 0 holistic estimate.
  If they differ significantly: examine which level introduced the most uncertainty.
  Which sub-question matters most? (Run sensitivity analysis: vary each by ±20%,
  see which moves the final estimate most.)

**Key Leverage Sub-question:**
  The sub-question whose estimate most affects the final answer.
  This is where to focus research effort, scenario planning, or risk mitigation.
"""

from skills.base_skill import BaseSkill


class RecursiveFutureDecompositionSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "recursive_future_decomposition"

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
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        depth = int(kwargs.get("depth", 3))
        return _decompose(problem, depth)
