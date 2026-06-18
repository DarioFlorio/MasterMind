# -*- coding: utf-8 -*-
"""tools/ask_user_tool.py — Interactive user question tool (MasterMind built-in)."""
from __future__ import annotations
from tools.base_tool import BaseTool, ToolResult


class AskUserTool(BaseTool):
    name = "ask_user"
    description = (
        "Ask the user a clarifying question and wait for their response. "
        "Use when you need information that cannot be inferred or found in the codebase."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question to ask the user"},
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested answers to show the user",
            },
        },
        "required": ["question"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        question = inp.get("question", "")
        options = inp.get("options", [])

        if not question:
            return ToolResult(output="question is required", is_error=True)

        print(f"\n  ❓ {question}")
        if options:
            for i, opt in enumerate(options, 1):
                print(f"     {i}. {opt}")
            print()

        try:
            answer = input("  Your answer: ").strip()
            # If user typed a number and options exist, resolve it
            if options and answer.isdigit():
                idx = int(answer) - 1
                if 0 <= idx < len(options):
                    answer = options[idx]
            return ToolResult(output=answer)
        except (EOFError, KeyboardInterrupt):
            return ToolResult(output="(no answer provided)", is_error=True)
