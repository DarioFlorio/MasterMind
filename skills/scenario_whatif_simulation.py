"""
Skill: scenario_whatif_simulation
Scenario and what-if simulation: explore branching future predictions,
stress-test assumptions, map alternative futures.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.scenario_whatif_simulation")

DESCRIPTION = (
    "Scenario simulation: explore 'what if' branches, stress-test assumptions, "
    "map alternative futures. Use for 'what if X happened instead?', "
    "'what are the possible outcomes?', 'scenario planning', 'best/worst case'."
)


def _simulate(problem: str, intervention: str, n_scenarios: int) -> str:
    interv_block = f"\nIntervention / Change: {intervention}" if intervention else ""
    scenarios = ["Best Case", "Base Case", "Worst Case"] if n_scenarios == 3 else \
                [f"Scenario {i+1}" for i in range(n_scenarios)]

    scenario_blocks = "\n\n".join([f"""**Scenario {i+1}: {name}**
  Triggering conditions: [What assumptions or events define this scenario?]
  Key difference from base: [What is different here vs the default path?]
  Chain of events:
    T+0: [Initial state]
    T+1: [First consequence]
    T+2: [Second-order effect]
    T+N: [Final state / outcome]
  Probability estimate: [X%]
  Key risks in this scenario: [What could go wrong even here?]
  Key opportunities: [What upside exists?]
  Early warning signals: [What would you observe if this scenario is unfolding?]""" 
    for i, name in enumerate(scenarios)])

    return f"""**Scenario / What-If Simulation**
Problem: {problem}{interv_block}

**Step 1 — Define the Baseline**
  What is the current trajectory if NOTHING changes?
  This is your reference point. All scenarios deviate from here.
  Be explicit about the key assumptions baked into the baseline.

**Step 2 — Identify Key Uncertainties**
  List the 2–3 variables whose values MOST affect the outcome.
  These are your scenario axes — not everything uncertain, just the
  most impactful uncertainties.

  Examples: market adoption rate, regulatory response, competitor reaction,
  technology maturation, public behaviour change.

**Step 3 — Construct Scenarios**
  Combine extreme values of your key uncertainties into coherent scenarios.
  Each scenario must be: internally consistent, meaningfully different, plausible.

{scenario_blocks}

**Step 4 — Cross-Impact Analysis**
  For each pair of key uncertainties: are they correlated?
  (e.g., high regulation AND high adoption is internally inconsistent in some domains.)
  Remove internally inconsistent scenario combinations.

**Step 5 — Robust Strategy Identification**
  What actions perform well (or at least not catastrophically) across ALL scenarios?
  These are your robust moves — take them regardless of which scenario unfolds.

  What actions are high-reward in one scenario but catastrophic in another?
  These are your bets — take them only if you have conviction on the scenario.

**Step 6 — Monitoring Plan**
  Which early indicators will tell you WHICH scenario is unfolding?
  Set tripwires: "If we observe [X] by [date], we shift to [strategy]."

**Summary Table:**
| Scenario | Probability | Key Outcome | Best Action |
|----------|-------------|-------------|-------------|
{chr(10).join(f'| {name} | [%] | [outcome] | [action] |' for name in scenarios)}
"""

from skills.base_skill import BaseSkill


class ScenarioWhatifSimulationSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "scenario_whatif_simulation"

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
                "intervention": {"type": "string"},
                "n_scenarios":  {"type": "integer"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        intervention = kwargs.get("intervention", "")
        n_scenarios  = int(kwargs.get("n_scenarios", 3))
        return _simulate(problem, intervention, n_scenarios)
