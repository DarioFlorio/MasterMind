"""
Skill: lateral_forward_thinking
Lateral forward thinking: find creative, non-obvious future paths,
challenge assumptions about what future is possible.
"""
from __future__ import annotations
import logging
import random

log = logging.getLogger("skill.lateral_forward_thinking")

DESCRIPTION = (
    "Lateral forward thinking: find creative, non-obvious future possibilities, "
    "challenge assumptions about what future is possible. Use for "
    "'what unexpected futures could emerge?', 'what am I missing?', "
    "'think outside the box about what comes next', 'wild card scenarios'."
)


def _lateral_forward(problem: str, domain: str) -> str:
    dom_block = f"\nDomain: {domain}" if domain else ""
    return f"""**Lateral Forward Thinking — Non-Obvious Future Paths**
Problem: {problem}{dom_block}

**Philosophy:**
  Most forecasting is extrapolation from the present.
  Lateral forward thinking deliberately breaks from that — it finds futures
  that are genuinely surprising, require assumption-breaking, or emerge from
  combinations nobody is currently considering.

**Step 1 — Map and Reverse Assumptions**
  List 5–7 assumptions embedded in the conventional forecast of this situation.
  For each assumption, ask: "What if this is exactly wrong?"

  Assumption 1: [state it] → Reversed: [what if the opposite is true in the future?]
  Assumption 2: [state it] → Reversed: [...]
  Assumption 3: [state it] → Reversed: [...]
  Assumption 4: [state it] → Reversed: [...]
  Assumption 5: [state it] → Reversed: [...]

  Now: which reversals, while counterintuitive, are actually plausible?
  Build futures from those.

**Step 2 — Cross-Domain Analogy Borrowing**
  What has happened in OTHER domains that is structurally similar to this situation?
  Import the "solution" from that domain into this one.

  Domain analogy 1: [What happened in domain X?] → Applied here: [...]
  Domain analogy 2: [What happened in domain Y?] → Applied here: [...]

  History often rhymes across domains; analogical futures can be hidden in plain sight.

**Step 3 — Combinatorial Futures**
  Combine two or three trends that are currently separate:
  "Trend A × Trend B = Unexpected Future C"

  Combination 1: [Trend A] + [Trend B] → [Emerging Future]
  Combination 2: [Trend C] + [Trend D] → [Emerging Future]
  Combination 3: [Trend E] + [Trend A] + [Trend C] → [Emerging Future]

  The most powerful future scenarios often emerge from combinations nobody planned.

**Step 4 — Constraint Removal**
  List the constraints that currently limit what's possible.
  Imagine each constraint disappears:

  Constraint 1: [e.g., energy cost] → If removed: [what becomes possible?]
  Constraint 2: [e.g., regulatory barrier] → If removed: [...]
  Constraint 3: [e.g., human cognitive limit] → If removed: [...]

  Which constraints are likely to weaken or disappear? → High-value future to plan for.

**Step 5 — Second-Order Adoption Effects**
  Most forecasters predict a technology/change being adopted.
  Fewer predict what that adoption UNLOCKS second-order:

  "Once X becomes ubiquitous, people will use it to do Y, which was previously impossible."

  Apply this: once [technology/trend] is universal in this domain:
  → What NEW BEHAVIOURS become possible that weren't before?
  → What OLD INSTITUTIONS become obsolete overnight?
  → What NEW INSTITUTIONS will emerge to fill the gap?

**Step 6 — Wild Card Inventory**
  Low-probability, high-impact events that could radically change the picture:

  Wild card 1: [Event] — probability: low; impact if it occurs: enormous
    → What would this world look like?
  Wild card 2: [Event] — probability: low; impact: enormous
  Wild card 3: [Event]

**Step 7 — Synthesis: The Non-Obvious Future**
  Select the 2–3 most creative but genuinely plausible futures generated above.
  For each, state:
  - The key assumption it challenges
  - The triggering event or threshold
  - What signals would indicate it's emerging NOW

**Most Non-Obvious Future Worth Taking Seriously:**
  [Describe in concrete terms]
  Why it's likely to be missed by conventional forecasters:
  How to position to benefit from (or protect against) it:
"""

from skills.base_skill import BaseSkill


class LateralForwardThinkingSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "lateral_forward_thinking"

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
                "domain": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        domain = kwargs.get("domain", "")
        return _lateral_forward(problem, domain)
