"""Skill: recursive_decompose — Recursive problem decomposition (Tower of Hanoi style)."""
from __future__ import annotations

DESCRIPTION = "Recursive decomposition: break complex problems into self-similar sub-problems."


def _hanoi(n: int) -> str:
    steps = []
    _hanoi_solve(n, "A", "C", "B", steps)
    moves = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps[:30]))
    note  = f"\n  ... ({len(steps) - 30} more moves)" if len(steps) > 30 else ""
    return f"""**Tower of Hanoi — {n} disks**

**Recursive structure:**
  T(n, from, to, aux):
    if n == 1: move disk 1 from → to
    else:
      T(n-1, from, aux, to)   # move top n-1 to auxiliary
      move disk n from → to   # move largest disk
      T(n-1, aux, to, from)   # move n-1 from auxiliary to target

**Total moves required:** 2^{n} − 1 = {2**n - 1}

**Move sequence (A→C via B):**
{moves}{note}"""


def _hanoi_solve(n: int, src: str, dst: str, aux: str, steps: list) -> None:
    if n == 1:
        steps.append(f"Move disk 1: {src} → {dst}")
        return
    _hanoi_solve(n - 1, src, aux, dst, steps)
    steps.append(f"Move disk {n}: {src} → {dst}")
    _hanoi_solve(n - 1, aux, dst, src, steps)


def _general_decompose(problem: str) -> str:
    return f"""**Recursive Decomposition**
Problem: {problem}

**Method: Divide and Conquer**

1. **Base case — when is the problem trivially solved?**
   Identify the smallest version of the problem with an obvious answer.

2. **Recursive case — how does size-N reduce to size-(N-1)?**
   Express the solution to the N-case in terms of solutions to smaller cases.
   Template: solve(N) = combine(solve(sub1), solve(sub2), ...)

3. **Decompose this problem:**
   a. What are the sub-problems? (same structure, smaller scale)
   b. What is the combining step? (merge, concatenate, add, select-best)
   c. Are sub-problems independent? (divide & conquer) or overlapping? (dynamic programming)

4. **If sub-problems overlap → use memoisation:**
   Cache results keyed by problem parameters.
   This converts exponential time to polynomial.

5. **Trace the recursion:**
   Draw the recursion tree. Identify the depth and branching factor.
   Total work ≈ (nodes in tree) × (work per node).

6. **Write the recurrence:**
   e.g. T(n) = 2T(n/2) + O(n) → O(n log n) by Master Theorem.

**Apply this structure to your specific problem to find the decomposition.**"""

from skills.base_skill import BaseSkill


class RecursiveDecomposeSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "recursive_decompose"

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
                "n": {"type": "integer"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        n = kwargs.get("n")
        low = problem.lower()
        if "hanoi" in low:
            return _hanoi(int(n) if n else 3)
        return _general_decompose(problem)
