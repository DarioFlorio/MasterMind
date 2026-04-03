"""Skill: constraint_solve — Logic grids, Einstein riddles, knight/knave puzzles."""
from __future__ import annotations

DESCRIPTION = "Constraint satisfaction for logic puzzles: grids, Einstein riddles, truth-tellers."


def _knight_knave(problem: str) -> str:
    return f"""**Constraint Solve: Knight/Knave Puzzle**
Problem: {problem}

**Method — Truth Table Enumeration:**

1. **Enumerate hypotheses:**
   - H1: Person A is a Knight (truth-teller), B is a Knave (liar)
   - H2: Person A is a Knave, B is a Knight
   - H3: Both Knights (only possible if no contradiction)
   - H4: Both Knaves (only possible if no contradiction)

2. **Test each hypothesis:**
   For each hypothesis, evaluate every statement:
   - If the speaker is a Knight → the statement must be TRUE
   - If the speaker is a Knave  → the statement must be FALSE
   Reject any hypothesis that produces a contradiction.

3. **Find the consistent hypothesis:**
   Exactly one hypothesis should survive without contradiction.

4. **Key insight for self-referential statements:**
   - "I am a Knave" — A Knight cannot say this (false). A Knave cannot say this (would be true). → Impossible statement.
   - "We are both the same" — Check under each hypothesis.
   - "At least one of us is a Knave" — Always true (Knight can say it); Knave must say something false (both Knights) — so speaker is Knight and there is indeed a Knave.

**Apply this framework to the specific statements in your puzzle to find the answer.**"""


def _einstein_grid(problem: str) -> str:
    return f"""**Constraint Solve: Logic Grid Puzzle**
Problem: {problem}

**Method — Constraint Propagation:**

1. **Set up the grid:**
   List all categories (colour, nationality, drink, pet, cigarette, position…)
   and all possible values for each.

2. **Record definite clues first** (direct assignments):
   e.g. "The Englishman lives in the red house" → English ↔ Red

3. **Propagate constraints:**
   Each definite assignment eliminates values from other cells in the same category.
   Repeat until no new eliminations are possible.

4. **Apply relational clues:**
   "Next to" → positions differ by 1
   "Left of"  → position is strictly smaller
   Try each valid position and eliminate contradictions.

5. **Iterate:**
   After each new deduction, re-apply all clues — a new deduction often unlocks others.

6. **If stuck — hypothesise:**
   Pick the most constrained remaining cell, try one value, propagate.
   If a contradiction arises, the value is wrong → assign the other.

**Work through the clues in your puzzle systematically using this method.**"""


def _generic_csp(problem: str) -> str:
    return f"""**Constraint Solve: General CSP**
Problem: {problem}

**Framework:**

1. **Variables:** Identify each unknown (what are we solving for?).
2. **Domains:** List the possible values for each variable.
3. **Constraints:** List every rule/clue as a formal constraint between variables.
4. **Arc consistency:** For each constraint, eliminate variable values that can't satisfy it.
5. **Backtracking search:** If pure propagation doesn't solve it, pick the most-constrained variable, try a value, propagate, and backtrack if contradiction arises.

**Tip:** Always start with the most constrained variable (fewest remaining values) — this minimises the search space.
"""

from skills.base_skill import BaseSkill


class ConstraintSolveSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "constraint_solve"

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
        low = problem.lower()
        if any(k in low for k in ("knight", "knave", "truth", "lies", "always")):
            return _knight_knave(problem)
        if any(k in low for k in ("zebra", "einstein", "house", "who owns", "who drinks")):
            return _einstein_grid(problem)
        return _generic_csp(problem)
