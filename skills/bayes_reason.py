"""Skill: bayes_reason — Bayesian inference, base rates, Monty Hall."""
from __future__ import annotations
import re

DESCRIPTION = "Bayesian reasoning: prior/posterior updates, base rates, conditional probability."


def _monty_hall() -> str:
    return """**Monty Hall Problem — Bayesian Analysis**

**Setup:** 3 doors. Prize behind 1. You pick door 1. Host opens a losing door. Should you switch?

**Prior (before host opens):**
  P(prize = door 1) = 1/3
  P(prize = door 2) = 1/3
  P(prize = door 3) = 1/3

**Likelihood (host opens door 3, given prize location):**
  If prize at door 1: host picks door 2 or 3 with prob 1/2 each → P(host opens 3 | prize=1) = 1/2
  If prize at door 2: host MUST open door 3 → P(host opens 3 | prize=2) = 1
  If prize at door 3: host can't open door 3 → P(host opens 3 | prize=3) = 0

**Posterior via Bayes' theorem:**
  P(prize=1 | host opens 3) = (1/2 × 1/3) / Z = 1/6 / Z
  P(prize=2 | host opens 3) = (1   × 1/3) / Z = 2/6 / Z
  where Z = 1/6 + 2/6 = 3/6 = 1/2

  P(prize=1 | host opens 3) = 1/3
  P(prize=2 | host opens 3) = 2/3

**Answer: Always switch. Switching wins 2/3 of the time.**

**Intuition:** The host's action is informative. By opening a losing door, all the
probability that was on door 3 collapses onto door 2 (not door 1)."""


def _medical_test(problem: str) -> str:
    # Try to extract numbers
    nums = re.findall(r"(\d+(?:\.\d+)?)\s*%?", problem)
    return f"""**Bayesian Medical Test Analysis**
Problem: {problem}

**Framework (fill in your numbers):**

Let:
  P(disease)         = prevalence (base rate) — e.g. 0.01 for 1%
  P(+ | disease)     = sensitivity (true positive rate) — e.g. 0.99
  P(+ | no disease)  = 1 − specificity (false positive rate) — e.g. 0.05

**Bayes' theorem:**
  P(disease | +) = P(+ | disease) × P(disease)
                   ─────────────────────────────────────────────────
                   P(+ | disease) × P(disease) + P(+ | no disease) × P(no disease)

**Key insight:** When prevalence is low, even a very accurate test yields many
false positives. A positive result may still mean the patient is more likely healthy.

**Example with 1% prevalence, 99% sensitivity, 5% FPR:**
  Numerator   = 0.99 × 0.01 = 0.0099
  Denominator = 0.0099 + 0.05 × 0.99 = 0.0099 + 0.0495 = 0.0594
  P(disease | +) ≈ 0.0099 / 0.0594 ≈ 16.7%

Despite a 99% accurate test, only ~17% of positives actually have the disease."""


def _general_bayes(problem: str) -> str:
    return f"""**Bayesian Reasoning Framework**
Problem: {problem}

1. **Define the hypothesis H** and its complement ¬H.

2. **Set your prior:** P(H) — what probability did you assign before seeing evidence?

3. **Identify the evidence E** — what observation are you updating on?

4. **Estimate likelihoods:**
   - P(E | H)  — how likely is this evidence if H is true?
   - P(E | ¬H) — how likely is this evidence if H is false?

5. **Apply Bayes:**
   P(H | E) = P(E | H) × P(H)
              ─────────────────────────────────
              P(E | H) × P(H) + P(E | ¬H) × P(¬H)

6. **Interpret:** Is the posterior meaningfully different from the prior?
   The ratio P(E|H)/P(E|¬H) is the **Bayes factor** — values >10 are strong evidence.

**Common errors to avoid:**
  - Base rate neglect: ignoring P(H) and focusing only on likelihoods
  - Prosecutor's fallacy: confusing P(E|¬H) with P(¬H|E)
  - Confirmation bias: only seeking evidence that increases P(H)
"""

from skills.base_skill import BaseSkill


class BayesReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "bayes_reason"

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
        if "monty hall" in low:
            return _monty_hall()
        if any(k in low for k in ("test", "disease", "positive", "false positive", "sensitivity")):
            return _medical_test(problem)
        return _general_bayes(problem)
