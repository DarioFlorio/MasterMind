"""
agent/context_budget.py — Context Budget Manager (MasterMind Harness feature).

GAP IMPLEMENTED: Context quality — EVE degrades as context fills.
Enhanced with:
  - Active context pruning: removes low-value tool results before rot sets in
  - Content scoring: prioritises recent, error-free, decision-relevant messages
  - Skill overhead tracking: monitors skill vs direct-tool cost ratio
  - Quality-aware compression: drops redundant content first, then summaries

Implements the "context budget" pattern natively in MasterMind:
  - Tool descriptions are trimmed to MAX_DESC_CHARS (default 250)
  - Detects "context rot" — tool schema overhead >ROT_THRESHOLD
  - Active pruning of low-value messages when budget is tight
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

log = logging.getLogger("agent.context_budget")

# ── Constants ────────────────────────────────────────────────────────────────
MAX_DESC_CHARS   = 250      # hard cap on tool description length in schema
ROT_THRESHOLD    = 0.45     # tool overhead > 45% of context → rot
BUDGET_WARN      = 0.75     # warn at 75% context usage
BUDGET_CRITICAL  = 0.90     # critical at 90% — drop oldest messages

# Context quality constants
# Messages scoring below this are candidates for pruning
PRUNE_SCORE_THRESHOLD = 0.25
# Maximum tool-result length to keep in quality messages
MAX_TOOL_RESULT_CHARS = 2_000
# Skill call patterns (for overhead tracking)
_SKILL_CALL_RE = re.compile(r'<n>skill</n>', re.DOTALL)
_TOOL_CALL_RE  = re.compile(r'<n>(\w+)</n>', re.DOTALL)


@dataclass
class TurnRecord:
    prompt_tokens:     int
    completion_tokens: int
    tool_overhead:     int   # estimated tokens from tool schemas
    skill_calls:       int = 0
    direct_calls:      int = 0


class ContextBudget:
    """
    Tracks token usage across turns and trims tool descriptions to prevent
    context rot. Now also performs active context quality management.
    """

    def __init__(self, context_size: int = 8192, max_desc_chars: int = MAX_DESC_CHARS):
        self.context_size  = context_size
        self.max_desc      = max_desc_chars
        self._turns: list[TurnRecord] = []
        self._total_prompt = 0
        self._total_compl  = 0
        self._total_skill_calls  = 0
        self._total_direct_calls = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def get_slim_tools(self, tools: list) -> list:
        """
        Return tools with descriptions trimmed to MAX_DESC_CHARS.
        Non-required parameters are stripped when rot is detected.
        """
        rot  = self.rot_detected()
        slim = [_SlimTool(t, self.max_desc, aggressive=rot) for t in tools]
        overhead = self._estimate_tool_overhead(slim)
        log.debug(
            "context_budget: %d tools, overhead≈%d tokens, rot=%s",
            len(slim), overhead, rot
        )
        return slim

    def record_turn(self, prompt_tokens: int, completion_tokens: int,
                    tool_overhead: int = 0, raw_text: str = "") -> None:
        """Call after each model turn to track cumulative usage."""
        skill_calls  = len(_SKILL_CALL_RE.findall(raw_text))
        direct_calls = len(_TOOL_CALL_RE.findall(raw_text)) - skill_calls
        rec = TurnRecord(prompt_tokens, completion_tokens, tool_overhead,
                         skill_calls, max(0, direct_calls))
        self._turns.append(rec)
        self._total_prompt       += prompt_tokens
        self._total_compl        += completion_tokens
        self._total_skill_calls  += skill_calls
        self._total_direct_calls += direct_calls

    def rot_detected(self) -> bool:
        """Returns True if tool schema overhead exceeds ROT_THRESHOLD."""
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
        if not self._turns:
            return 0.0
        return self._turns[-1].prompt_tokens / max(self.context_size, 1)

    def should_compress(self) -> bool:
        return self.usage_fraction() >= BUDGET_WARN

    def is_critical(self) -> bool:
        return self.usage_fraction() >= BUDGET_CRITICAL

    def skill_overhead_ratio(self) -> float:
        """
        GAP: Skill overhead vs quality.
        Returns fraction of tool calls that went through the skill router
        rather than direct tool calls. High ratio = skill overhead is significant.
        """
        total = self._total_skill_calls + self._total_direct_calls
        if total == 0:
            return 0.0
        return self._total_skill_calls / total

    def prune_messages(self, messages: list[dict]) -> list[dict]:
        """
        GAP: Context quality — EVE degrades as context fills.
        Prune low-value messages when context is tight.

        Scoring rubric (0.0–1.0):
          - Recent messages score higher
          - User messages always kept
          - Tool results > MAX_TOOL_RESULT_CHARS are truncated
          - Redundant cached results are dropped
          - Error messages are kept (learning value)

        Only prunes when context usage > BUDGET_WARN.
        """
        if not self.should_compress():
            return messages
        if len(messages) <= 6:
            return messages

        scored: list[tuple[float, dict, int]] = []
        n = len(messages)

        for i, msg in enumerate(messages):
            score = self._score_message(msg, i, n)
            scored.append((score, msg, i))

        # Always keep: first 2, last 4, and all user messages
        must_keep: set[int] = set()
        must_keep.update([0, 1, n-4, n-3, n-2, n-1])
        for i, msg in enumerate(messages):
            if msg.get("role") == "user" and not msg.get("meta", {}).get("tool_response"):
                must_keep.add(i)

        # Drop messages scoring below threshold (except must-keep)
        pruned: list[dict] = []
        dropped = 0
        for score, msg, idx in sorted(scored, key=lambda x: x[2]):  # restore order
            if idx in must_keep or score >= PRUNE_SCORE_THRESHOLD:
                # Truncate large tool results even when keeping
                m = dict(msg)
                if m.get("meta", {}).get("tool_response") and len(m.get("content", "")) > MAX_TOOL_RESULT_CHARS:
                    content = m["content"]
                    m = {**m, "content": content[:MAX_TOOL_RESULT_CHARS] + "\n...[truncated by context_budget]"}
                pruned.append(m)
            else:
                dropped += 1

        if dropped > 0:
            log.info("context_budget: pruned %d low-value messages", dropped)
        return pruned

    def summary(self) -> str:
        frac     = self.usage_fraction() * 100
        rot      = self.rot_detected()
        skill_r  = self.skill_overhead_ratio() * 100
        return (
            f"Context: {frac:.1f}% used | "
            f"turns={len(self._turns)} | "
            f"rot={'YES ⚠' if rot else 'no'} | "
            f"skill_overhead={skill_r:.0f}% | "
            f"budget={'CRITICAL' if self.is_critical() else 'WARN' if self.should_compress() else 'ok'}"
        )

    # ── Internal helpers ───────────────────────────────────────────────────

    def _score_message(self, msg: dict, idx: int, total: int) -> float:
        """Score a message for quality / retention value (0.0–1.0)."""
        score = 0.5

        # Recency bonus: more recent = more valuable
        recency = idx / max(total - 1, 1)
        score  += 0.3 * recency

        # Role: user messages are precious
        role = msg.get("role", "")
        if role == "user":
            if msg.get("meta", {}).get("tool_response"):
                score -= 0.10  # tool results are less valuable than user queries
            else:
                score += 0.20

        content = msg.get("content", "")
        clen = len(content)

        # Error messages: retain for learning
        if any(w in content.lower() for w in ["error", "fail", "exception", "traceback"]):
            score += 0.10

        # Cached / [cached] tag: already compressed
        if content.startswith("[cached]"):
            score -= 0.20

        # Very short: probably not useful
        if clen < 30:
            score -= 0.15

        # Very long tool results: diminishing value
        if clen > 10_000:
            score -= 0.20
        elif clen > 5_000:
            score -= 0.10

        # Goal-anchor injections: high retention
        if "[GoalAnchor]" in content or "GOAL:" in content:
            score += 0.15

        return max(0.0, min(1.0, score))

    @staticmethod
    def _estimate_tool_overhead(tools: list) -> int:
        """Rough token estimate: 1 token ≈ 4 chars of schema XML."""
        total_chars = 0
        for t in tools:
            desc = getattr(t, "description", "") or ""
            schema = getattr(t, "input_schema", {}) or {}
            props = schema.get("properties", {})
            total_chars += len(desc) + sum(len(str(v)) for v in props.values())
        return total_chars // 4


class _SlimTool:
    """
    Wrapper that presents a trimmed view of a BaseTool for prompt injection.
    Delegates all real execution to the wrapped tool.
    """

    def __init__(self, tool, max_desc: int, aggressive: bool = False):
        self._tool       = tool
        self._max_desc   = max_desc
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
                slim_props[k] = {"type": v.get("type", "string")}
        return {**schema, "properties": slim_props, "required": list(required)}

    def to_xml_schema(self) -> str:
        props = self.input_schema.get("properties", {})
        req   = self.input_schema.get("required", [])
        lines = []
        for p, s in props.items():
            r    = " (required)" if p in req else ""
            desc = s.get("description", "")
            if desc:
                desc = desc[:120]
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
