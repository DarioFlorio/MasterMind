"""
tools/agent_tool.py — Sub-agent spawner for MasterMind.

Sub-agent results are cleaned of internal tags (<think>, <tool_use>, dangling
fragments) before being returned, preventing bleed into the parent session.

The `agent` tool should ONLY be used for truly parallelisable or isolated
multi-step subtasks — NOT for greetings, clarification, simple summaries, or
any task the orchestrator can handle in a single turn.
"""
from __future__ import annotations
import re
from tools.base_tool import BaseTool, ToolResult

_factory = None  # set by main.py via AgentTool.set_factory()

# ── Output cleaning (mirrors query_engine._clean_output) ─────────────────────
_THINK_RE       = re.compile(r"<think>.*?</think>|<think>.*$|^\s*</think>", re.DOTALL | re.MULTILINE)
_TOOL_USE_RE    = re.compile(r"<tool_use>.*?</tool_use>", re.DOTALL)
_TOOL_RESULT_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)
_DANGLING_RE    = re.compile(r"</?(?:tool_use|tool_result|think|n|input)>")
_BLEED_RE       = re.compile(r"\[Sub-agent result\]\s*<think>.*", re.DOTALL)


def _clean_subagent_output(text: str) -> str:
    text = _THINK_RE.sub("", text)
    text = _TOOL_USE_RE.sub("", text)
    text = _TOOL_RESULT_RE.sub("", text)
    text = _BLEED_RE.sub("", text)
    text = _DANGLING_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class AgentTool(BaseTool):
    name = "agent"
    description = (
        "Spawn a sub-agent to handle a self-contained, parallelisable task. "
        "The sub-agent has its own tool access and returns a final result. "
        "DO NOT use for simple questions, greetings, clarification, or single-step tasks "
        "the orchestrator can handle directly — that wastes turns."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "task":      {"type": "string", "description": "Task description for the sub-agent"},
            "max_turns": {"type": "integer", "description": "Max turns (default 10)"},
        },
        "required": ["task"],
    }

    @staticmethod
    def set_factory(factory) -> None:
        global _factory
        _factory = factory

    def execute(self, inp: dict) -> ToolResult:
        task      = inp.get("task", "").strip()
        max_turns = int(inp.get("max_turns", 10))

        if not task:
            return ToolResult("No task provided.", is_error=True)
        if _factory is None:
            return ToolResult("Agent factory not configured.", is_error=True)

        try:
            engine = _factory(max_turns=max_turns, is_subagent=True)
            raw_result = engine.submit_message(task)
            cleaned    = _clean_subagent_output(raw_result)
            return ToolResult(f"[Sub-agent result]\n{cleaned}")
        except Exception as e:
            return ToolResult(f"Sub-agent error: {e}", is_error=True)
