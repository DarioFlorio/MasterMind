"""
Skill: timeline_projection_reason
Timeline projection: sequence future events step-by-step, map milestones,
estimate when things will happen, track conditional branches.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.timeline_projection_reason")

DESCRIPTION = (
    "Timeline projection: map future events in sequence, estimate when things "
    "will happen, identify milestones and dependencies, project current trends "
    "forward. Use for 'when will X happen?', 'what is the sequence of events?', "
    "'roadmap', 'what comes next?'"
)


def _project_timeline(problem: str, horizon: str, domain: str, start: str) -> str:
    dom_block   = f"\nDomain: {domain}" if domain else ""
    start_block = f"\nStarting state: {start}" if start else ""
    return f"""**Timeline Projection Reasoning**
Problem: {problem}
Projection horizon: {horizon}{dom_block}{start_block}

**Step 1 — Anchor the Present State**
  Define clearly: what is the current state of affairs?
  What trends are already in motion?
  What forces are currently active (technological, economic, social, political)?
  Confidence: [HIGH / MEDIUM / LOW] in present-state characterisation.

**Step 2 — Identify Driving Forces**
  List the forces that will shape the future trajectory:

  Structural drivers (slow-moving, predictable):
  - Demographic trends
  - Long-cycle technological trajectories
  - Physical/biological constraints
  - Accumulated capital or debt

  Volatile drivers (fast-moving, uncertain):
  - Policy decisions
  - Market dynamics
  - Breakthrough technologies
  - Conflict or cooperation between key actors

**Step 3 — Define Milestones**
  Working forward from now, identify key events/thresholds:

  Near-term [{horizon} = near < 1yr, medium 1–5yr, long > 5yr]:

  Milestone 1: [Event] → Expected when: [timeframe] → Confidence: [%]
  Milestone 2: [Event] → Expected when: [timeframe] → Confidence: [%]
  Milestone 3: [Event] → Expected when: [timeframe] → Confidence: [%]
  ...

  Each milestone should be:
  - Observable (you'll know it when it happens)
  - Causally connected to the next
  - Assigned a probability and timeframe range

**Step 4 — Identify Conditional Branches**
  At each major milestone, what are the two or three possible outcomes?
  Map the branching tree:

  Now → [M1 outcome A] → [M2a] → ...
      → [M1 outcome B] → [M2b] → ...

  For each branch: probability, what triggers it, what it enables next.

**Step 5 — Critical Path and Bottlenecks**
  Which milestone, if delayed, delays everything after it?
  (This is the critical path of the future.)
  What could accelerate or derail the critical path?

**Step 6 — Projection Summary**
  Most likely scenario (highest probability path):
    Now → [M1] → [M2] → [M3] → [Final state at {horizon}]

  Key uncertainties that most affect the outcome:
    1. [Uncertainty A] — resolves by [time]
    2. [Uncertainty B] — resolves by [time]

  Early warning signals to monitor:
    - [Signal 1]: indicates [branch A] is occurring
    - [Signal 2]: indicates [branch B] is occurring

**Calibration Note:**
  Compound probabilities shrink fast. If each of 5 milestones has 80% confidence:
  P(all correct) = 0.8⁵ ≈ 33%. Keep projection ranges wide enough to be honest.
"""

from skills.base_skill import BaseSkill


class TimelineProjectionReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "timeline_projection_reason"

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
                "horizon":     {"type": "string"},
                "domain":      {"type": "string"},
                "start_state": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        horizon = kwargs.get("horizon", "medium-term")
        domain  = kwargs.get("domain", "")
        start   = kwargs.get("start_state", "")
        return _project_timeline(problem, horizon, domain, start)
