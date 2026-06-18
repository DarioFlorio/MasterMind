"""Skill: analogical_reason — Analogical mapping and reasoning by analogy."""
from __future__ import annotations

from skills.base_skill import BaseSkill

DESCRIPTION = "Reasoning by analogy: find structural mappings between domains."


class AnalogicalReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "analogical_reason"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem":       {"type": "string", "description": "The problem to solve"},
                "depth":         {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "source_domain": {"type": "string"},
                "target_domain": {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        source  = kwargs.get("source_domain", "")
        target  = kwargs.get("target_domain", "")
        src_str = source or "(the familiar domain used as the analogy)"
        tgt_str = target or "(the unfamiliar domain being understood)"
        return (
            f"**Analogical Reasoning**\n"
            f"Problem: {problem}\n\n"
            f"**Framework: Structure-Mapping Theory**\n\n"
            f"1. **Identify source and target domains:**\n"
            f"   Source: {src_str}\n"
            f"   Target: {tgt_str}\n\n"
            f"2. **Map structural relations (not just surface features):**\n"
            f"   Good analogies share relational structure, not just appearance.\n"
            f"   e.g. \"The atom is like a solar system\" maps:\n"
            f"     nucleus <-> sun | electron <-> planet | gravity <-> electrostatic force\n\n"
            f"3. **Identify where the analogy holds and where it breaks:**\n"
            f"   Every analogy has a boundary. Find it.\n\n"
            f"4. **Use the analogy to generate predictions:**\n"
            f"   If X is true in the source domain, what would be the analogous Y in the target?\n"
            f"   Test whether Y is actually true.\n\n"
            f"5. **Evaluate analogy strength:**\n"
            f"   Strong: deep relational structure, many mapped relations, few disanalogies\n"
            f"   Weak:   surface similarity only, many important disanalogies\n\n"
            f"6. **Common analogical pitfalls:**\n"
            f"   - Extending the analogy beyond its valid range\n"
            f"   - Confusing the map for the territory\n"
            f"   - Using analogy as proof rather than as a hypothesis generator\n\n"
            f"**Conclusion:** Analogies are powerful for generating hypotheses and building "
            f"intuition, but they must be tested against the actual properties of the target domain."
        )
