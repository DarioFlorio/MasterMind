"""
Skill: probabilistic_forecasting
Probabilistic forecasting: assign calibrated probabilities to future events,
update via Bayesian reasoning as new evidence arrives.
"""
from __future__ import annotations
import logging
import math

log = logging.getLogger("skill.probabilistic_forecasting")

DESCRIPTION = (
    "Probabilistic forecasting: assign calibrated probabilities to future events, "
    "update beliefs as evidence arrives, superforecasting methodology. "
    "Use for 'what is the probability that X?', 'how likely is Y?', "
    "'forecast the likelihood of Z'."
)


def _general_forecast(problem: str, event: str, evidence: str) -> str:
    ev_block = f"\nEvidence: {evidence}" if evidence else ""
    event_str = event or problem
    return f"""**Probabilistic Forecasting**
Question: {problem}
Event to forecast: {event_str}{ev_block}

**Superforecasting Methodology (Tetlock & Gardner):**

**Step 1 — Clarify the Question**
  Make the event OPERATIONALLY precise:
  - What exactly would count as this event occurring?
  - By what date / under what conditions?
  - Who is the adjudicator (how will we know if it happened)?
  Vague questions produce poorly-calibrated forecasts.

**Step 2 — Find the Reference Class (Outside View)**
  Ask: "What category of events does this belong to?"
  Look up the base rate: how often do events of this type occur?
  Start your forecast at the base rate — resist anchoring on surface features.

  Examples:
  - "Will this startup succeed?" → ~10% of funded startups reach Series B
  - "Will this project deliver on time?" → 85%+ of large software projects are late
  - "Will this policy pass?" → depends on legislative context, average success rate

**Step 3 — Adjust for Specific Factors (Inside View)**
  List specific factors that make this case ABOVE or BELOW the base rate:

  Factors pushing probability UP:
    [List 2–4 specific reasons this is more likely than average]

  Factors pushing probability DOWN:
    [List 2–4 specific reasons this is less likely than average]

  Apply each factor as a multiplicative update:
  Starting from base rate P₀:
  P₁ = P₀ × LR₁  (where LR = likelihood ratio for that factor)
  P₂ = P₁ × LR₂, etc.

**Step 4 — Consider Multiple Scenarios**
  Don't just estimate a point probability — think about the distribution of outcomes.
  Scenario weighting:
    Best case (P = [x%]): [describe]
    Base case (P = [y%]): [describe]
    Worst case (P = [z%]): [describe]
  Weighted probability = Σ(P(scenario) × P(event | scenario))

**Step 5 — State the Forecast**
  Point estimate: P(event) = [X%]
  90% confidence interval: [[low%], [high%]]
  (Your interval should be wide enough that you're genuinely surprised 10% of the time.)

**Step 6 — Commit and Track**
  Record the forecast with a timestamp.
  Define what evidence would cause you to update UP or DOWN.
  Revisit when new evidence arrives — update, don't rationalize.

**Calibration Reminder:**
  Good forecasters are neither overconfident nor underconfident.
  If you say 70%: the event should occur about 70% of the time.
  Track your Brier score over time to improve calibration.
  Brier score = (forecast − outcome)²; lower is better; 0.25 = coin flip.
"""


def _bayesian_update(problem: str, event: str, prior: float, evidence: str) -> str:
    p = max(0.001, min(0.999, prior))
    log_odds_prior = math.log(p / (1 - p))

    return f"""**Probabilistic Forecasting: Bayesian Update**
Question: {problem}
Event: {event}
Prior probability: {prior:.1%}
New evidence: {evidence}

**Bayesian Update Framework:**

Prior odds = P / (1−P) = {p:.3f} / {1-p:.3f} = {p/(1-p):.3f}
Log-prior odds = ln(prior odds) = {log_odds_prior:.3f}

**To update, estimate the Likelihood Ratio for the evidence:**
  LR = P(evidence | event occurs) / P(evidence | event does NOT occur)

  LR > 1 → evidence supports the event (update upward)
  LR < 1 → evidence argues against the event (update downward)
  LR = 1 → evidence is uninformative (no update needed)

**Typical likelihood ratios:**
  Highly diagnostic evidence: LR 10–100 (or 0.01–0.1 against)
  Moderately diagnostic: LR 3–10
  Weakly diagnostic: LR 1.5–3

**Posterior calculation:**
  Posterior odds = Prior odds × LR
  Posterior P = Posterior odds / (1 + Posterior odds)

**Example updates from prior {prior:.1%}:**
  If LR = 3 (moderately supporting):
    Posterior odds = {p/(1-p):.3f} × 3 = {p/(1-p)*3:.3f}
    Posterior P ≈ {(p/(1-p)*3) / (1 + p/(1-p)*3):.1%}

  If LR = 0.33 (moderately against):
    Posterior odds = {p/(1-p):.3f} × 0.33 = {p/(1-p)*0.33:.3f}
    Posterior P ≈ {(p/(1-p)*0.33) / (1 + p/(1-p)*0.33):.1%}

**Apply to '{evidence}':**
  Estimate the LR for this specific piece of evidence.
  Compute the posterior.
  State your updated forecast with confidence interval.

**Note on sequential updates:**
  Each new piece of evidence can be applied in turn.
  Order doesn't matter (conditional independence assumed).
  Watch for: evidence that is not independent (double-counting the same signal).
"""

from skills.base_skill import BaseSkill


class ProbabilisticForecastingSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "probabilistic_forecasting"

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
                "event":    {"type": "string"},
                "prior":    {"type": "number"},
                "evidence": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        event    = kwargs.get("event", "")
        prior    = kwargs.get("prior", None)
        evidence = kwargs.get("evidence", "")
        if prior is not None and evidence:
            try:
                p = float(prior)
                return _bayesian_update(problem, event or problem, p, evidence)
            except (ValueError, TypeError):
                pass
        return _general_forecast(problem, event, evidence)
