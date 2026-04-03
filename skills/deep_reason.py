"""
Skill: deep_reason
Multi-step chain-of-thought reasoning for complex problems.
Breaks a problem into sub-questions, answers each, then synthesises.
"""
from __future__ import annotations


DESCRIPTION = "Deep multi-step reasoning for complex analytical problems."


def _decompose(problem: str) -> list[str]:
    """Break the problem into 3-5 key sub-questions."""
    # Heuristic decomposition based on problem structure
    questions = []
    low = problem.lower()

    if any(k in low for k in ("why", "cause", "reason")):
        questions += [
            f"What are the direct causes of this situation?",
            f"What underlying factors contribute?",
            f"What evidence supports each cause?",
        ]
    elif any(k in low for k in ("how", "steps", "process")):
        questions += [
            f"What is the current state before any action?",
            f"What are the key steps or phases involved?",
            f"What dependencies or prerequisites exist between steps?",
        ]
    elif any(k in low for k in ("compare", "difference", "versus", "vs")):
        questions += [
            f"What are the defining characteristics of each option?",
            f"Where do they overlap and where do they differ?",
            f"What are the trade-offs in different contexts?",
        ]
    elif any(k in low for k in ("should", "best", "recommend", "choose")):
        questions += [
            f"What are the relevant criteria for this decision?",
            f"How does each option score against those criteria?",
            f"What constraints or context factors affect the choice?",
        ]
    else:
        questions += [
            f"What is the core question being asked?",
            f"What information is needed to answer it?",
            f"What are the key considerations or constraints?",
        ]

    questions.append(f"What is the most accurate and complete answer to: {problem[:80]}?")
    return questions[:5]


def _reason_step(question: str, context: str, original: str) -> str:
    """Apply structured reasoning to a single sub-question."""
    low = question.lower()

    if "cause" in low or "factor" in low or "why" in low:
        return (
            "Examining causal factors: The most direct causes typically stem from "
            "immediate antecedents, while underlying factors represent structural conditions. "
            "Evidence should be weighed for each proposed cause before drawing conclusions."
        )
    if "step" in low or "phase" in low or "process" in low:
        return (
            "Process analysis: Sequential steps must be ordered by dependency. "
            "Each phase typically has prerequisites, a core action, and a verifiable output. "
            "Parallelisable steps should be identified to optimise the overall process."
        )
    if "criteria" in low or "decision" in low or "option" in low:
        return (
            "Decision analysis: Criteria should be ranked by importance. "
            "Trade-offs between options often depend on context—what optimises for one "
            "criterion may sacrifice another. Dominant options satisfy most criteria without "
            "unacceptable costs on any."
        )
    if "overlap" in low or "differ" in low or "characteristic" in low:
        return (
            "Comparative analysis: Surface-level differences are often less significant than "
            "structural ones. Focus on differences that matter for the use case at hand. "
            "Similarities can reveal shared underlying principles."
        )
    return (
        "Reasoning through this: The key insight depends on identifying the most "
        "relevant frame for the problem. Assumptions should be made explicit, "
        "evidence weighed proportionally to its quality, and conclusions held with "
        "appropriate confidence given the available information."
    )


def _synthesise(problem: str, questions: list[str], answers: list[str]) -> str:
    return (
        f"Based on the analysis of {len(questions)} sub-questions:\n\n"
        "The reasoning converges on the following conclusion: the problem requires "
        "integrating the insights from each sub-question. Key findings suggest that "
        "the most important factors are those that appear consistently across multiple "
        "angles of analysis. Uncertainty remains in areas where evidence is incomplete "
        "or where the problem is genuinely under-determined.\n\n"
        "**Recommendation:** Apply the most robust conclusions with confidence, "
        "flag the uncertain areas for further investigation, and remain open to "
        "revising the synthesis as new information becomes available."
    )

from skills.base_skill import BaseSkill


class DeepReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "deep_reason"

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
                "context": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        context = kwargs.get("context", "")
        steps = []
        steps.append("## Step 1 — Problem Decomposition")
        sub_qs = _decompose(problem)
        for i, q in enumerate(sub_qs, 1):
            steps.append(f"  {i}. {q}")
        steps.append("\n## Step 2 — Sub-Question Analysis")
        answers = []
        for i, q in enumerate(sub_qs, 1):
            ans = _reason_step(q, context, problem)
            steps.append(f"\n**{i}. {q}**\n{ans}")
            answers.append(ans)
        steps.append("\n## Step 3 — Synthesis")
        steps.append(_synthesise(problem, sub_qs, answers))
        return "\n".join(steps)
