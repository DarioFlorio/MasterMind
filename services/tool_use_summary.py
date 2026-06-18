"""
services/tool_use_summary.py — ToolUseSummary: human-readable tool batch summaries

Generates short (≤30 char) git-commit-style labels for completed tool
batches. Used by the SDK / SDK-style clients to show progress in mobile
or compact UIs.

Usage:
    from services.tool_use_summary import generate_tool_use_summary
    label = generate_tool_use_summary(tools, model_client)
    # → "Fixed NPE in UserService"
"""
from __future__ import annotations

import json
import logging
from typing import Optional

log = logging.getLogger("services.tool_use_summary")

_SYSTEM_PROMPT = """\
Write a short summary label describing what these tool calls accomplished. \
It appears as a single-line row in a compact UI and truncates around 30 characters, \
so think git-commit-subject, not sentence.

Keep the verb in past tense and the most distinctive noun. \
Drop articles, connectors, and long location context first.

Examples:
- Searched in auth/
- Fixed NPE in UserService
- Created signup endpoint
- Read config.json
- Ran failing tests
- Updated requirements.txt
- Wrote unit tests"""


def _truncate_json(value, max_length: int = 300) -> str:
    try:
        s = json.dumps(value, default=str)
        if len(s) <= max_length:
            return s
        return s[:max_length - 3] + "..."
    except Exception:
        return "[unable to serialize]"


def generate_tool_use_summary(
    tools: list[dict],
    model_client=None,
    last_assistant_text: Optional[str] = None,
) -> Optional[str]:
    """
    Generate a ≤30-char label for a batch of completed tool calls.

    `tools` is a list of dicts with keys: name, input, output.
    `model_client` should have a `.complete_simple(system, user) -> str` method.

    Returns None on failure or empty tool list.
    """
    if not tools:
        return None

    # Build tool summary string
    tool_lines = []
    for t in tools:
        name = t.get("name", "unknown")
        inp = _truncate_json(t.get("input", {}))
        out = _truncate_json(t.get("output", ""))
        tool_lines.append(f"Tool: {name}\nInput: {inp}\nOutput: {out}")
    tool_text = "\n\n".join(tool_lines)

    context_prefix = ""
    if last_assistant_text:
        context_prefix = (
            f"User's intent (from assistant's last message): "
            f"{last_assistant_text[:200]}\n\n"
        )

    user_prompt = f"{context_prefix}Tools completed:\n\n{tool_text}\n\nLabel:"

    if model_client is not None:
        try:
            summary = model_client.complete_simple(
                system=_SYSTEM_PROMPT,
                user=user_prompt,
                max_tokens=20,
            )
            text = (summary or "").strip()
            return text[:50] if text else None
        except Exception as exc:
            log.debug("ToolUseSummary: generation failed: %s", exc)
            return None

    # Fallback: heuristic label from tool names
    return _heuristic_label(tools)


def _heuristic_label(tools: list[dict]) -> str:
    """Simple fallback when no model client is available."""
    if not tools:
        return "Ran tools"

    # Map common tool names to verbs
    verb_map = {
        "bash_tool": "Ran command",
        "read_file_tool": "Read file",
        "write_file_tool": "Wrote file",
        "edit_file_tool": "Edited file",
        "glob_tool": "Searched files",
        "grep_tool": "Searched code",
        "web_search_tool": "Searched web",
        "web_fetch_tool": "Fetched URL",
        "git_tool": "Git operation",
        "agent_tool": "Ran subagent",
    }

    if len(tools) == 1:
        t = tools[0]
        name = t.get("name", "")
        label = verb_map.get(name, f"Used {name}")
        # Try to add a subject from the input
        inp = t.get("input", {})
        if isinstance(inp, dict):
            for key in ("file_path", "command", "query", "path"):
                val = inp.get(key)
                if val and isinstance(val, str):
                    short = val.split("/")[-1][:20]
                    return f"{label}: {short}"
        return label

    # Multiple tools — list unique names
    names = list(dict.fromkeys(t.get("name", "") for t in tools))[:3]
    if len(names) == 1:
        return f"Ran {len(tools)}x {names[0]}"
    return f"Ran {len(tools)} tools ({', '.join(names[:2])})"
