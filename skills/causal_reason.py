"""Skill: causal_reason — Causal inference, root cause analysis, do-calculus concepts."""
from __future__ import annotations

DESCRIPTION = "Causal reasoning: root cause analysis, counterfactuals, causal chains."


def _root_cause(problem: str, depth: int) -> str:
    return f"""**Root Cause Analysis**
Problem: {problem}

**Method: 5-Whys + Fishbone**

**5-Whys chain (ask 'why?' {depth} times):**
  Symptom → Why? → Cause 1 → Why? → Cause 2 → Why? → ... → Root Cause

  Start with the observable symptom. Each answer becomes the next "why" question.
  Stop when you reach a cause you can actually act on.

**Fishbone (Ishikawa) categories to check:**
  - People:    Training gaps, human error, staffing?
  - Process:   Missing steps, ambiguous procedures, workarounds?
  - Technology:Tool failures, version mismatches, config drift?
  - Environment: External conditions, dependencies, timing?
  - Materials: Bad inputs, corrupted data, wrong versions?

**Distinguishing root cause from proximate cause:**
  Proximate cause = the immediate trigger (the last domino)
  Root cause      = the systemic condition that made the chain possible
  Fix the root cause to prevent recurrence; fix the proximate cause for immediate relief.

**Counterfactual test:**
  Would removing this cause have prevented the problem? If yes, it's causal.
  If the problem would have occurred anyway, it's correlational."""


def _counterfactual(problem: str) -> str:
    return f"""**Counterfactual Analysis**
Problem: {problem}

**Framework: Potential Outcomes (Rubin Causal Model)**

1. **Define the treatment T** (the action that did/didn't happen)
   and the outcome Y (what we're measuring).

2. **Actual world:** T occurred → Y = y_observed
3. **Counterfactual world:** T did NOT occur → Y = y_counterfactual (unobserved)

4. **Causal effect** = y_observed − y_counterfactual

**Challenges:**
  - The fundamental problem: we can never observe both worlds for the same unit.
  - We must estimate y_counterfactual using:
    a) A comparable control group
    b) Historical baseline (before treatment)
    c) Structural model of the system

**Do-calculus shortcut (Pearl):**
  P(Y | do(T=t)) ≠ P(Y | T=t)
  The left side removes confounders; the right side includes selection bias.
  To estimate the causal effect, we must "block" backdoor paths through confounders.

**Practical question:** What assumptions are needed for the counterfactual to be valid?
  List them — they are the weak points of the causal argument."""


def _causal_chain(problem: str, depth: int) -> str:
    return f"""**Causal Chain Analysis**
Problem: {problem}

**Step 1 — Map the causal graph:**
  Identify variables: A → B → C → ... → Outcome
  Draw arrows only where A directly causes B (not just correlates).

**Step 2 — Identify confounders:**
  A confounder Z causes both A and the outcome.
  This creates a spurious correlation: A ↔ Outcome even if A doesn't cause it.

**Step 3 — Check mediators vs moderators:**
  Mediator M: A → M → Outcome (M is on the causal path)
  Moderator W: affects the strength of A → Outcome (interaction effect)

**Step 4 — Test the causal claim:**
  Does manipulating A change the outcome (holding other variables fixed)?
  If the correlation disappears when conditioning on a confounder, the link was spurious.

**Step 5 — Strength and direction:**
  Is the effect deterministic or probabilistic?
  What is the magnitude? Necessary and/or sufficient?

**Key principle:** Correlation is symmetric; causation is directional.
  "A causes B" is a very different claim from "B causes A" even when A and B are equally correlated."""

from skills.base_skill import BaseSkill


class CausalReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "causal_reason"

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
        low   = problem.lower()
        if any(k in low for k in ("root cause", "why did", "what caused", "reason for")):
            return _root_cause(problem, depth)
        if any(k in low for k in ("what if", "counterfactual", "hadn't", "would have")):
            return _counterfactual(problem)
        return _causal_chain(problem, depth)
