"""
agent/context_budget.py — Context Budget Manager (leaked Claude Code feature).

Implements the "context budget" pattern from the Claude Code leak:
  - Tool descriptions are trimmed to MAX_DESC_CHARS (default 250) to prevent
    context bloat from schema overhead.
  - Tracks total prompt tokens vs a configurable budget ceiling.
  - Detects "context rot" — when tool+schema overhead consumes >ROT_THRESHOLD
    of the available context, EVE switches to a compressed system prompt and
    strips non-essential tool params from the schema.
  - Provides get_slim_tools(tools) → tools with trimmed schemas ready for
    injection into the prompt.

Usage in query_engine.py:
    from agent.context_budget import ContextBudget
    budget = ContextBudget(context_size=CONTEXT_SIZE)
    slim_tools = budget.get_slim_tools(all_tools)
    # pass slim_tools to prompt builder instead of all_tools
    budget.record_turn(prompt_tokens, completion_tokens)
    if budget.rot_detected():
        # switch to compressed system prompt
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field

log = logging.getLogger("agent.context_budget")

# ── Constants (mirroring Claude Code leak values) ─────────────────────────────
MAX_DESC_CHARS   = 250      # hard cap on tool description length in schema
ROT_THRESHOLD    = 0.45     # if tool overhead > 45% of context → rot detected
BUDGET_WARN      = 0.75     # warn at 75% context usage
BUDGET_CRITICAL  = 0.90     # critical at 90% — drop oldest messages


@dataclass
class TurnRecord:
    prompt_tokens:     int
    completion_tokens: int
    tool_overhead:     int   # estimated tokens from tool schemas


class ContextBudget:
    """
    Tracks token usage across turns and trims tool descriptions to prevent
    context rot — the performance degradation Claude Code identified when tool
    schemas consume too large a fraction of the available context window.
    """

    def __init__(self, context_size: int = 8192, max_desc_chars: int = MAX_DESC_CHARS):
        self.context_size  = context_size
        self.max_desc      = max_desc_chars
        self._turns: list[TurnRecord] = []
        self._total_prompt = 0
        self._total_compl  = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def get_slim_tools(self, tools: list) -> list:
        """
        Return tools with descriptions trimmed to MAX_DESC_CHARS.
        Non-required parameters are stripped when rot is detected.
        This is the core leak feature — keep tool schemas lean so they
        don't eat into the reasoning budget.
        """
        rot = self.rot_detected()
        slim = []
        for tool in tools:
            slim.append(_SlimTool(tool, self.max_desc, aggressive=rot))
        overhead = self._estimate_tool_overhead(slim)
        log.debug(
            "context_budget: %d tools, overhead≈%d tokens, rot=%s",
            len(slim), overhead, rot
        )
        return slim

    def record_turn(self, prompt_tokens: int, completion_tokens: int,
                    tool_overhead: int = 0) -> None:
        """Call after each model turn to track cumulative usage."""
        rec = TurnRecord(prompt_tokens, completion_tokens, tool_overhead)
        self._turns.append(rec)
        self._total_prompt += prompt_tokens
        self._total_compl  += completion_tokens

    def rot_detected(self) -> bool:
        """
        Returns True if tool schema overhead is consuming an unhealthy share
        of the context window — Claude Code's "context rot" signal.
        """
        if not self._turns:
            return False
        last = self._turns[-1]
        if last.tool_overhead <= 0:
            return False
        ratio = last.tool_overhead / max(self.context_size, 1)
        if ratio > ROT_THRESHOLD:
            log.warning(
                "context_budget: rot detected — tool overhead %.1f%% of context",
                ratio * 100
            )
            return True
        return False

    def usage_fraction(self) -> float:
        """Current turn's prompt usage as fraction of context window."""
        if not self._turns:
            return 0.0
        return self._turns[-1].prompt_tokens / max(self.context_size, 1)

    def should_compress(self) -> bool:
        return self.usage_fraction() >= BUDGET_WARN

    def is_critical(self) -> bool:
        return self.usage_fraction() >= BUDGET_CRITICAL

    def summary(self) -> str:
        frac = self.usage_fraction() * 100
        rot  = self.rot_detected()
        return (
            f"Context: {frac:.1f}% used | "
            f"turns={len(self._turns)} | "
            f"rot={'YES ⚠' if rot else 'no'} | "
            f"budget={'CRITICAL' if self.is_critical() else 'WARN' if self.should_compress() else 'ok'}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _estimate_tool_overhead(tools: list) -> int:
        """Rough token estimate: 1 token ≈ 4 chars of schema XML."""
        total_chars = sum(len(t.to_xml_schema()) for t in tools)
        return total_chars // 4


class _SlimTool:
    """
    Wrapper that presents a trimmed view of a BaseTool for prompt injection.
    Delegates all real execution to the wrapped tool.
    """

    def __init__(self, tool, max_desc: int, aggressive: bool = False):
        self._tool      = tool
        self._max_desc  = max_desc
        self._aggressive = aggressive

        # Trim description
        desc = getattr(tool, "description", "") or ""
        self.description = desc[:max_desc] + ("…" if len(desc) > max_desc else "")
        self.name        = tool.name

        # Build trimmed schema
        orig_schema = getattr(tool, "input_schema", {}) or {}
        self.input_schema = self._trim_schema(orig_schema, aggressive)

    def _trim_schema(self, schema: dict, aggressive: bool) -> dict:
        """In aggressive (rot) mode, strip optional params to save tokens."""
        if not aggressive:
            return schema
        props    = schema.get("properties", {})
        required = set(schema.get("required", []))
        slim_props = {}
        for k, v in props.items():
            if k in required:
                # Keep required params but strip their description
                slim_props[k] = {"type": v.get("type", "string")}
            # Drop optional params entirely in rot mode
        return {**schema, "properties": slim_props, "required": list(required)}

    def to_xml_schema(self) -> str:
        return self._tool.to_xml_schema().__class__  # delegate to real impl
        # Actually rebuild slim version:
        props = self.input_schema.get("properties", {})
        req   = self.input_schema.get("required", [])
        lines = []
        for p, s in props.items():
            r    = " (required)" if p in req else ""
            desc = s.get("description", "")
            if desc:
                desc = desc[:120]  # also trim param descriptions
            lines.append(
                f"    <param name='{p}' type='{s.get('type','string')}'{r}>{desc}</param>"
            )
        return (
            f"<tool>\n  <n>{self.name}</n>\n"
            f"  <description>{self.description}</description>\n"
            f"  <params>\n" + "\n".join(lines) + "\n  </params>\n</tool>"
        )

    def execute(self, inp: dict):
        return self._tool.execute(inp)

    def safe_parse(self, raw):
        return self._tool.safe_parse(raw)
