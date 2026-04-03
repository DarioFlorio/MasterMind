"""
Skill: causal_forward_reason
Causal forward reasoning with butterfly effect: trace how small causes
cascade into large future consequences through causal chains.
"""
from __future__ import annotations
import logging

log = logging.getLogger("skill.causal_forward_reason")

DESCRIPTION = (
    "Causal forward reasoning: trace how causes ripple forward into consequences, "
    "butterfly effect, cascade analysis, 'what happens if X occurs?', "
    "second- and third-order effects, unintended consequences."
)


def _causal_forward(cause: str, depth: int, domain: str, problem: str) -> str:
    dom_block = f"\nDomain context: {domain}" if domain else ""
    return f"""**Causal Forward Reasoning — Butterfly Effect Analysis**
Cause / Trigger: {cause}{dom_block}
Original question: {problem}

**Framework: N-Order Causal Chain (depth = {depth})**

**Order 1 — Direct / Immediate Effects**
  What happens directly and immediately as a result of the cause?
  These are the obvious, first-level consequences that everyone sees.
  Typical timeframe: hours to days.

  Identify:
  - Who/what is directly affected?
  - What processes or systems does this immediately alter?
  - What is the magnitude of the direct effect?

**Order 2 — Indirect / Reactive Effects**
  What happens in response TO the first-order effects?
  Other actors, systems, and feedback loops respond.
  Typical timeframe: days to weeks.

  Key patterns:
  - Substitution: affected parties find workarounds
  - Amplification: the effect is scaled up by a multiplier (e.g., viral spread)
  - Dampening: negative feedback reduces the initial effect
  - Displacement: the problem moves rather than resolves

**Order 3 — Systemic / Structural Effects**
  How do accumulated second-order effects change underlying structures?
  Typical timeframe: weeks to months/years.

  Key patterns:
  - Structural lock-in: new equilibrium becomes hard to reverse
  - Norm change: behaviour patterns shift permanently
  - Systemic risk: stress accumulates until a threshold is crossed
  - Cross-domain spillover: effects jump into adjacent systems

{"".join([f'''
**Order {i} — Extended Cascade**
  Effects from previous order now trigger their own downstream consequences.
  Ask: what has changed structurally? Who now acts differently because of it?
  What new actors or systems are now implicated?
''' for i in range(4, depth + 1)])}

**Butterfly Effect Assessment**
  Small causes → large effects when:
  - System is near a tipping point / bifurcation
  - Positive feedback loops dominate
  - Network effects amplify small signals
  - Path dependence locks in early outcomes

  Rate the sensitivity: LOW / MEDIUM / HIGH / CRITICAL
  State what conditions determine which.

**Unintended Consequences Scan**
  For each order, explicitly ask:
  - What beneficial side effects might occur? (Positive unintended)
  - What harmful side effects might occur? (Negative unintended)
  - What intended effects might be undermined? (Backfire)
  - Who gains and who loses that wasn't in the original plan?

**Summary**
  Trace the full chain:
  [Cause] → [1st order] → [2nd order] → [3rd order] → ... → [Final state]

  Key leverage points: where in the chain can intervention prevent harm or amplify benefit?
  Most impactful point: [identify the single node in the causal chain most worth targeting]
"""

from skills.base_skill import BaseSkill


class CausalForwardReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "causal_forward_reason"

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
                "cause": {"type": "string"},
                "domain": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        cause  = kwargs.get("cause", "")
        depth  = int(kwargs.get("depth", 3))
        domain = kwargs.get("domain", "")
        cause_str = cause or problem
        return _causal_forward(cause_str, depth, domain, problem)
