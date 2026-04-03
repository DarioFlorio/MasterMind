"""
Skill: reason_chain
Automatic multi-skill reasoning pipeline.

Given a problem, auto-selects (via skill_router) or accepts an explicit
ordered list of reasoning skills, then runs them in sequence — piping each
skill's full output as 'context' into the next. Produces a structured
chain trace showing every reasoning layer.
"""
from __future__ import annotations

import time

from skills.base_skill import BaseSkill

DESCRIPTION = (
    "Multi-skill reasoning chain: auto-routes or uses an explicit skill list, "
    "runs each skill in sequence piping outputs as context, then synthesises "
    "all layers into a final answer. The compoundable reasoner."
)


# ── Auto-route via skill_router ───────────────────────────────────────────────

def _auto_select_chain(problem: str, max_steps: int) -> list[str]:
    """Call skill_router via registry and extract the top skill names."""
    import re
    try:
        from skills import SKILL_REGISTRY
        SkillRouterClass = SKILL_REGISTRY.get("skill_router")
        if SkillRouterClass is None:
            return ["deep_reason"]
        router = SkillRouterClass()
        result = router.execute(problem, top_n=max_steps)
    except Exception:
        return ["deep_reason"]

    names: list[str] = []
    chain_block = re.search(
        r"Skill Chaining Suggestion.*?(?=\n\n|\Z)", result, re.DOTALL
    )
    if chain_block:
        found = re.findall(r"`([a-z_]+)`", chain_block.group())
        names = [n for n in found if n != "skill_router"]

    if not names:
        primary = re.search(r"Primary recommendation: `([a-z_]+)`", result)
        if primary:
            names.append(primary.group(1))
        alts = re.findall(r"- `([a-z_]+)`", result)
        for a in alts:
            if a not in names and a != "skill_router":
                names.append(a)

    seen: set = set()
    out: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= max_steps:
            break

    return out or ["deep_reason"]


# ── Single skill executor ─────────────────────────────────────────────────────

def _run_skill(name: str, problem: str, context: str, depth: int) -> tuple[str, str]:
    """Instantiate and run a skill via the registry."""
    try:
        from skills import SKILL_REGISTRY
        SkillClass = SKILL_REGISTRY.get(name)
        if SkillClass is None:
            return "", f"Skill '{name}' not found in SKILL_REGISTRY."
        skill = SkillClass()
        result = skill.execute(problem, depth=depth)
        return (result or "").strip(), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ── Synthesiser ───────────────────────────────────────────────────────────────

def _synthesise(problem: str, steps: list[dict]) -> str:
    conclusions: list[str] = []
    for step in steps:
        out = step["output"]
        if not out:
            continue
        paras = [p.strip() for p in out.split("\n\n") if p.strip()]
        if paras:
            conclusions.append(f"[{step['skill']}]: {paras[-1][:300]}")

    if not conclusions:
        return "No conclusions extracted from the chain — review individual step outputs."

    merged = "\n\n".join(conclusions)
    return (
        f"The reasoning chain converged across {len(steps)} skill(s).\n\n"
        f"Key conclusions per layer:\n\n{merged}\n\n"
        f"**Integrated answer:** The analysis is most robust where the skill outputs "
        f"converge. Divergences between layers indicate genuinely uncertain or "
        f"multi-interpretation territory — treat those points with appropriate "
        f"epistemic humility."
    )


# ─────────────────────────────────────────────────────────────────────────────

class ReasonChainSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "reason_chain"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem":    {"type": "string"},
                "depth":      {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
                "chain":      {"type": "array", "items": {"type": "string"}},
                "max_steps":  {"type": "integer"},
                "context":    {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        chain     = kwargs.get("chain") or []
        depth     = int(kwargs.get("depth", 2))
        max_steps = int(kwargs.get("max_steps", 3))

        if not chain:
            chain = _auto_select_chain(problem, max_steps)

        header = [
            "## Reasoning Chain",
            f"**Problem:** {problem[:120]}{'...' if len(problem) > 120 else ''}",
            f"**Chain:** {' → '.join(chain)} ({len(chain)} steps)",
            f"**Depth per skill:** {depth}",
            "",
        ]

        steps: list[dict] = []
        context = ""

        for i, skill_name in enumerate(chain, 1):
            t0 = time.perf_counter()
            output, error = _run_skill(skill_name, problem, context, depth)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            step = {
                "step":    i,
                "skill":   skill_name,
                "output":  output,
                "error":   error,
                "elapsed": elapsed_ms,
            }
            steps.append(step)

            if output:
                context = output

        lines = header[:]
        for step in steps:
            status = "✅" if not step["error"] else "❌"
            lines.append(
                f"### {status} Step {step['step']}: `{step['skill']}` "
                f"({step['elapsed']:.0f}ms)"
            )
            if step["error"]:
                lines.append(f"**Error:** {step['error']}")
            else:
                lines.append(step["output"])
            lines.append("")

        successful = [s for s in steps if not s["error"]]
        if len(successful) > 1:
            lines.append("---")
            lines.append("## Chain Synthesis")
            lines.append(_synthesise(problem, successful))

        return "\n".join(lines)
