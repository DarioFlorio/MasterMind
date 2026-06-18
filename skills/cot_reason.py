"""Skill: cot_reason — Step-by-step chain-of-thought reasoning."""
from __future__ import annotations

DESCRIPTION = "Step-by-step chain-of-thought reasoning. Good for maths, logic, multi-step problems."


def _math_chain(problem: str) -> list[str]:
    return [
        "1. **Identify what is being asked:** Extract the target quantity or result.",
        "2. **List known values:** Identify all given numbers and their units.",
        "3. **Choose the operation:** Determine which formula or arithmetic applies.",
        "4. **Execute step by step:** Perform calculations in the correct order.",
        "5. **Check units and magnitude:** Verify the result is dimensionally correct.",
        "6. **State the answer clearly.**",
    ]


def _logic_chain(problem: str) -> list[str]:
    return [
        "1. **Identify premises:** List all given statements as P1, P2, ...",
        "2. **Identify the conclusion:** What needs to be proven or evaluated?",
        "3. **Check validity:** Does the conclusion follow necessarily from the premises?",
        "4. **Check soundness:** Are the premises themselves true/plausible?",
        "5. **Identify any fallacies or gaps** in the argument.",
        "6. **State the logical verdict.**",
    ]


def _code_chain(problem: str) -> list[str]:
    return [
        "1. **Understand the goal:** What should the code do?",
        "2. **Reproduce the issue:** Trace through the logic or error message.",
        "3. **Identify the root cause:** Find the exact line or assumption that fails.",
        "4. **Consider the fix:** What minimal change resolves the root cause?",
        "5. **Check for side effects:** Will the fix break other behaviour?",
        "6. **Write and verify the solution.**",
    ]


def _general_chain(problem: str) -> list[str]:
    return [
        "1. **Clarify the question:** State exactly what is being asked.",
        "2. **Gather relevant facts:** What do we know that bears on this?",
        "3. **Identify assumptions:** What must be true for each possible answer?",
        "4. **Evaluate each option:** Score each against the facts and assumptions.",
        "5. **Select the best answer:** Choose the option best supported by evidence.",
        "6. **State the conclusion** with appropriate confidence.",
    ]

from skills.base_skill import BaseSkill


class CotReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "cot_reason"

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
        steps = [f"**Problem:** {problem}\n", "**Step-by-step reasoning:**\n"]
        low = problem.lower()
        if any(k in low for k in ("calculate", "compute", "+", "-", "*", "/", "=", "percent", "%")):
            steps += _math_chain(problem)
        elif any(k in low for k in ("if", "then", "all", "some", "none", "every", "must")):
            steps += _logic_chain(problem)
        elif any(k in low for k in ("code", "bug", "error", "function", "class", "import")):
            steps += _code_chain(problem)
        else:
            steps += _general_chain(problem)
        return "\n".join(steps)
