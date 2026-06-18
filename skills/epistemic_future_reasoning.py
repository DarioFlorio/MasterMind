"""
Skill: epistemic_future_reasoning
Epistemic future reasoning: predict what others will know, believe, or assume
in the future; model knowledge propagation and belief evolution.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.epistemic_future_reasoning")

DESCRIPTION = (
    "Epistemic future reasoning: predict what people will know or believe later, "
    "model how knowledge and beliefs spread and change over time. Use for "
    "'how will opinions change?', 'when will people realise X?', "
    "'what will the consensus be?', 'information diffusion', 'belief evolution'."
)


def _epistemic_future(problem: str, audience: str, horizon: str) -> str:
    aud_block = f"\nTarget audience: {audience}" if audience else ""
    return f"""**Epistemic Future Reasoning**
Problem: {problem}{aud_block}
Horizon: {horizon}

**Core Question:**
  What will people (individually or collectively) KNOW, BELIEVE, or ASSUME differently
  in the future compared to now?

**Step 1 — Current Epistemic State**
  What does the relevant audience currently know / believe / assume?
  What are the dominant narratives, consensus views, and blind spots?

  Known (consensus): [...]
  Believed but uncertain: [...]
  Assumed but unexamined: [...]
  Unknown to most: [...]

**Step 2 — Knowledge Generation Pipeline**
  New knowledge enters public awareness through:

  Research → Publication → Expert uptake → Media coverage → Public awareness → Policy change

  For each stage, estimate:
  - Time lag: how long between "discovered" and "publicly known"?
  - Filtering: what percentage of findings survive peer review? Media coverage?
  - Distortion: how much does the message change at each stage?

  Current research trajectory predicts that by [{horizon}]:
  - Topic A will be well-established
  - Topic B will be in public debate
  - Topic C is being researched now but won't reach public for [X] years

**Step 3 — Belief Diffusion Modelling**
  New information spreads through social networks.
  Rogers' Diffusion of Innovations curve:
    Innovators (2.5%) → Early Adopters (13.5%) → Early Majority (34%) →
    Late Majority (34%) → Laggards (16%)

  For this topic:
  - Where is public belief currently on this curve?
  - What accelerates diffusion? (Trusted messenger, simple narrative, social proof)
  - What slows it? (Cognitive dissonance, motivated reasoning, vested interests)

  Prediction: by [{horizon}], belief will have diffused to the [Early Adopter / Early Majority /
  Late Majority] stage, implying [X%] of the relevant population will hold this view.

**Step 4 — Motivated Epistemic Resistance**
  Some knowledge threatens identities, interests, or existing beliefs.
  Motivated reasoning causes people to reject evidence that challenges them.

  For this topic, identify:
  - Who has strong incentives to NOT update their beliefs?
  - What cognitive biases will reinforce resistance? (Confirmation bias, sunk cost, etc.)
  - How strong is the motivated resistance likely to be?
  - What would break through it? (Personal experience, peer pressure, authority figure)

**Step 5 — Information Environment Evolution**
  The information ecosystem itself is changing.
  How will these changes affect what people know by [{horizon}]?

  - Media consolidation / fragmentation: [impact on this topic]
  - AI-generated content: [impact on information quality and reach]
  - Filter bubbles and personalisation: [risk of divergent belief clusters]
  - Institutional trust trends: [how this affects which sources people credit]

**Step 6 — Epistemic Forecast**
  By [{horizon}], predict:

  a) What will be widely KNOWN and accepted as fact?
     [List 2–3 things currently uncertain that will be settled]

  b) What will be in active DEBATE (contested but prominent)?
     [List 1–2 things that will be on the agenda but unresolved]

  c) What will remain UNKNOWN even then?
     [List 1–2 genuinely hard epistemic problems]

  d) What MISINFORMATION / false beliefs are likely to persist?
     [Identify 1–2 stubborn false beliefs and why they resist correction]

**Step 7 — Strategic Implications**
  If you know what people will believe in the future:

  Information asymmetry opportunity: what do you know now that they don't know yet?
  → How does this change your strategy today?

  Belief shift risks: what currently stable beliefs are likely to shatter?
  → What are you currently assuming that you should stress-test?

  First-mover advantage: what positions benefit from the future belief state?
  → What should you build, invest in, or prepare for now?
"""

from skills.base_skill import BaseSkill


class EpistemicFutureReasoningSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "epistemic_future_reasoning"

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
                "audience": {"type": "string"},
                "horizon":  {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        audience = kwargs.get("audience", "")
        horizon  = kwargs.get("horizon", "medium-term")
        return _epistemic_future(problem, audience, horizon)
