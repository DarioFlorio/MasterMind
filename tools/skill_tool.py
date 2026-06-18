"""
tools/skill_tool.py — Executes reasoning skills via SKILL_REGISTRY.
Replaced the old importlib/run() version; now uses the BaseSkill system.
"""
from __future__ import annotations
import logging
import time

from tools.base_tool import BaseTool, ToolResult

log = logging.getLogger("tools.skill_tool")


class SkillTool(BaseTool):
    name = "skill"
    description = (
        "Execute a structured reasoning skill. Use skills for complex reasoning tasks "
        "instead of reasoning inline — they provide rigorous, structured frameworks. "
        "For multi-step problems, use 'reason_chain' which auto-selects and chains "
        "multiple skills in sequence. "
        "ALWAYS prefer a skill over freeform reasoning for: puzzles, predictions, "
        "causal analysis, game theory, Bayesian problems, what-if scenarios, "
        "trade-off decisions, and timeline reasoning.\n"
        "Input: {\"skill\": \"skill_name\", \"args\": {\"problem\": \"...\", ...}}.\n"
        "Leave skill empty or omit to list all available skills with descriptions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "skill": {
                "type": "string",
                "description": (
                    "Skill name. Examples: reason_chain, deep_reason, cot_reason, "
                    "constraint_solve, game_solve, bayes_reason, abduct, "
                    "analogical_reason, timeline_reason, causal_reason, "
                    "recursive_decompose, epistemic_reason, lateral_thinking, "
                    "multi_objective, skill_router, causal_forward_reason, "
                    "timeline_projection_reason, scenario_whatif_simulation, "
                    "probabilistic_forecasting, game_theoretic_forward_simulation, "
                    "multi_objective_future_optimization, recursive_future_decomposition, "
                    "deep_multi_layer_prediction, lateral_forward_thinking, "
                    "epistemic_future_reasoning, web_search, inductive_reason, goal_anchor, wakefulness. "
                    "Use skill_router if unsure which skill to use."
                ),
            },
            "args": {
                "type": "object",
                "description": (
                    "Arguments for the skill. Always include 'problem' with the full "
                    "question or task. Additional args vary by skill (e.g. 'depth', "
                    "'domain', 'events', 'prior', 'rounds')."
                ),
            },
        },
        "required": [],
    }

    def execute(self, inp: dict) -> ToolResult:
        from skills import SKILL_REGISTRY, list_skills

        skill_name = (inp.get("skill") or "").strip()
        args       = inp.get("args") or {}

        # ── Listing mode ──────────────────────────────────────────────────────
        if not skill_name:
            names = list_skills()
            lines = ["Available Reasoning Skills:\n"]
            for n in names:
                SkillClass = SKILL_REGISTRY.get(n)
                try:
                    desc = SkillClass().description if SkillClass else "(unavailable)"
                except Exception:
                    desc = "(unavailable)"
                lines.append(f"  {n} — {desc}")
            lines.append(f"\nTotal: {len(names)} skills")
            lines.append("Use skill_router to get a recommendation for your problem.")
            return ToolResult("\n".join(lines))

        # Normalise
        skill_name = skill_name.replace("skills/", "").replace(".py", "").strip()
        log.debug("skill_tool: executing '%s' args=%s", skill_name, list(args.keys()))
        t0 = time.perf_counter()

        # ── Look up in registry ───────────────────────────────────────────────
        SkillClass = SKILL_REGISTRY.get(skill_name)
        if SkillClass is None:
            available = list_skills()
            import difflib
            close = difflib.get_close_matches(skill_name, available, n=1, cutoff=0.5)
            suggestion = f" Did you mean: '{close[0]}'?" if close else ""
            return ToolResult(
                f"Skill '{skill_name}' not found.{suggestion}\n"
                f"Available: {', '.join(sorted(available))}",
                is_error=True,
            )

        # ── Extract problem; pass rest as kwargs ──────────────────────────────
        problem = (
            args.get("problem") or args.get("query") or
            args.get("input") or args.get("examples") or ""
        ).strip()

        if not problem:
            return ToolResult(
                f"Skill '{skill_name}' requires a 'problem' key in args.",
                is_error=True,
            )

        extra = {k: v for k, v in args.items()
                 if k not in ("problem", "query", "input")}

        # ── Execute ───────────────────────────────────────────────────────────
        try:
            skill  = SkillClass()
            result = skill.execute(problem, **extra)
        except Exception as exc:
            import traceback
            msg = (
                f"Skill '{skill_name}' raised {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc(limit=6)}"
            )
            log.error("skill_tool exception in '%s': %s", skill_name, exc, exc_info=True)
            return ToolResult(msg, is_error=True)

        elapsed = (time.perf_counter() - t0) * 1000
        log.debug("skill_tool: '%s' completed in %.1f ms", skill_name, elapsed)

        if not isinstance(result, str):
            result = str(result)
        if not result.strip():
            return ToolResult(
                f"Skill '{skill_name}' returned empty output.", is_error=True
            )

        return ToolResult(result)
