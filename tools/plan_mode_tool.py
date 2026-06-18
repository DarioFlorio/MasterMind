# -*- coding: utf-8 -*-
"""tools/plan_mode_tool.py — Plan mode: propose-only execution mode."""
from __future__ import annotations
from tools.base_tool import BaseTool, ToolResult

_plan_mode_active = False
_plan_steps: list[str] = []


def is_plan_mode() -> bool:
    return _plan_mode_active


def get_plan() -> list[str]:
    return list(_plan_steps)


class EnterPlanModeTool(BaseTool):
    name = "enter_plan_mode"
    description = (
        "Enter plan mode: the agent will only propose steps, not execute them. "
        "Use this to draft a plan for user review before taking action."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "plan": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of proposed steps",
            },
        },
    }

    def execute(self, inp: dict) -> ToolResult:
        global _plan_mode_active, _plan_steps
        inp = self.safe_parse(inp)
        _plan_mode_active = True
        _plan_steps = inp.get("plan", [])

        from hooks.manager import hook_manager
        hook_manager.fire("plan_enter", steps=_plan_steps)

        if _plan_steps:
            steps_str = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(_plan_steps))
            return ToolResult(output=f"Plan mode active. Proposed steps:\n{steps_str}")
        return ToolResult(output="Plan mode active. Agent will propose steps only.")


class ExitPlanModeTool(BaseTool):
    name = "exit_plan_mode"
    description = "Exit plan mode and resume normal (execution-enabled) mode."
    input_schema = {
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "Whether the plan was approved by the user",
            },
        },
    }

    def execute(self, inp: dict) -> ToolResult:
        global _plan_mode_active, _plan_steps
        inp = self.safe_parse(inp)
        approved = inp.get("approved", True)
        old_steps = list(_plan_steps)
        _plan_mode_active = False
        _plan_steps = []

        from hooks.manager import hook_manager
        hook_manager.fire("plan_exit", approved=approved, steps=old_steps)

        status = "approved" if approved else "cancelled"
        return ToolResult(output=f"Plan mode exited ({status}). Execution mode resumed.")
