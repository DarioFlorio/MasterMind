"""
Session Compaction — LLM-based context compression.
MasterMind session compaction engine.

When a session approaches the context limit, older messages are summarised
by an LLM and replaced with a compact summary block. The full history stays
on disk; compaction only changes what the model sees.

Features:
  - Token estimation (approx. chars/4)
  - Configurable threshold (default: 80% of limit)
  - Tool call pairs stay together (not split mid-pair)
  - Configurable compaction model (can differ from primary model)
  - Compaction summary saved to session transcript
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── Types ──────────────────────────────────────────────────────────────────────

Message = dict[str, Any]  # {"role": str, "content": str | list}


@dataclass
class CompactionConfig:
    """Configuration for session compaction."""
    # Fraction of context limit at which to trigger compaction
    threshold: float = 0.80
    # Model to use for compaction (None = same as primary)
    model: Optional[str] = None
    # Minimum turns to keep in full form after compaction
    min_tail_turns: int = 6
    # Notify the user when compaction runs
    notify_user: bool = False
    # Maximum tokens to keep as "tail" after compaction
    max_tail_tokens: int = 8192


@dataclass
class CompactionResult:
    """Result of a compaction run."""
    original_messages: list[Message]
    compacted_messages: list[Message]   # summary + tail
    summary: str
    tokens_before: int
    tokens_after: int
    timestamp: float = field(default_factory=time.time)

    @property
    def reduction_pct(self) -> float:
        if self.tokens_before == 0:
            return 0.0
        return (1 - self.tokens_after / self.tokens_before) * 100


# ── Token estimation ───────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)

def _message_tokens(msg: Message) -> int:
    content = msg.get("content", "")
    if isinstance(content, str):
        return _estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, dict):
                total += _estimate_tokens(block.get("text", "") or json.dumps(block))
        return total
    return 0

def _messages_tokens(messages: list[Message]) -> int:
    return sum(_message_tokens(m) for m in messages)


# ── Split helpers ──────────────────────────────────────────────────────────────

def _find_safe_split_point(messages: list[Message], target_idx: int) -> int:
    """
    Find a safe split point that doesn't break tool call / tool result pairs.
    Searches backward from target_idx.
    """
    idx = min(target_idx, len(messages) - 1)
    # Walk backward until we find an assistant message not followed by a tool result
    while idx > 0:
        msg = messages[idx]
        role = msg.get("role", "")
        # If this is a tool_result, skip back past its tool_use
        if role == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                if any(b.get("type") == "tool_result" for b in content if isinstance(b, dict)):
                    idx -= 1
                    continue
        break
    return max(0, idx)


def _extract_text(message: Message) -> str:
    """Extract readable text from a message for summarisation."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    parts.append(f"[tool: {block.get('name', '')}({json.dumps(block.get('input', {}))[:200]})]")
                elif btype == "tool_result":
                    parts.append(f"[result: {str(block.get('content', ''))[:300]}]")
        return "\n".join(parts)
    return ""


# ── Compactor ─────────────────────────────────────────────────────────────────

def _build_compaction_prompt(messages: list[Message], instruction: str = "") -> str:
    """Build a prompt asking the LLM to summarise conversation history."""
    lines = [
        "Summarise the following conversation history into a concise, information-dense",
        "summary that preserves all important context, decisions, and facts. Be thorough",
        "enough that continuing the conversation with only this summary (plus recent messages)",
        "would allow seamless continuation.",
    ]
    if instruction:
        lines.append(f"\nAdditional instruction: {instruction}")
    lines.append("\n---\nConversation history to summarise:\n")

    for msg in messages:
        role = msg.get("role", "unknown").upper()
        text = _extract_text(msg)[:1000]  # Cap per-message for the prompt
        lines.append(f"[{role}]: {text}")

    return "\n".join(lines)


class SessionCompactor:
    """
    Compacts a list of messages by summarising the older portion with an LLM.
    
    Usage:
        compactor = SessionCompactor(llm_fn=my_llm, config=CompactionConfig())
        result = compactor.compact(messages, context_limit=32000)
        # Use result.compacted_messages for the next API call
    """

    def __init__(
        self,
        llm_fn: Callable[[str, str | None], str],
        config: Optional[CompactionConfig] = None,
        on_compact: Optional[Callable[[CompactionResult], None]] = None,
    ):
        self._llm = llm_fn  # fn(prompt, model=None) -> str
        self._config = config or CompactionConfig()
        self._on_compact = on_compact

    def needs_compaction(self, messages: list[Message], context_limit: int) -> bool:
        """Return True if messages exceed the compaction threshold."""
        tokens = _messages_tokens(messages)
        return tokens >= context_limit * self._config.threshold

    def compact(
        self,
        messages: list[Message],
        context_limit: int,
        instruction: str = "",
        system: str = "",
    ) -> CompactionResult:
        """
        Compact messages. Returns the CompactionResult containing the new message list.
        The new list is: [optional system] + [summary_message] + [tail_messages]
        """
        tokens_before = _messages_tokens(messages)

        # Determine split: keep last min_tail_turns exchanges as "tail"
        tail_start = max(0, len(messages) - self._config.min_tail_turns * 2)
        tail_start = _find_safe_split_point(messages, tail_start)

        # If there's not much to compact, compact more aggressively
        if tail_start == 0:
            tail_start = max(0, len(messages) // 2)
            tail_start = _find_safe_split_point(messages, tail_start)

        to_compact = messages[:tail_start]
        tail = messages[tail_start:]

        # Build summary via LLM
        if to_compact:
            prompt = _build_compaction_prompt(to_compact, instruction)
            summary = self._llm(prompt, self._config.model)
        else:
            summary = "[No prior history to summarise.]"

        # Build summary message (injected as system-like context)
        summary_msg: Message = {
            "role": "user",
            "content": (
                f"[COMPACTED HISTORY]\n{summary}\n[END COMPACTED HISTORY]\n\n"
                f"The above is a summary of the conversation so far. Continue naturally."
            ),
        }

        compacted = [summary_msg] + tail
        tokens_after = _messages_tokens(compacted)

        result = CompactionResult(
            original_messages=messages,
            compacted_messages=compacted,
            summary=summary,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

        if self._on_compact:
            self._on_compact(result)

        return result

    def auto_compact_if_needed(
        self,
        messages: list[Message],
        context_limit: int,
        instruction: str = "",
    ) -> tuple[list[Message], Optional[CompactionResult]]:
        """
        Auto-compact messages if they exceed the threshold.
        Returns (messages_to_use, compaction_result_or_None).
        """
        if self.needs_compaction(messages, context_limit):
            result = self.compact(messages, context_limit, instruction)
            if self._config.notify_user:
                print(
                    f"[compaction] Context reduced from ~{result.tokens_before:,} to "
                    f"~{result.tokens_after:,} tokens ({result.reduction_pct:.0f}% reduction)"
                )
            return result.compacted_messages, result
        return messages, None


def compact_messages(
    messages: list[Message],
    llm_fn: Callable,
    context_limit: int = 32000,
    config: Optional[CompactionConfig] = None,
    instruction: str = "",
) -> CompactionResult:
    """Convenience function: compact a list of messages."""
    compactor = SessionCompactor(llm_fn=llm_fn, config=config)
    return compactor.compact(messages, context_limit, instruction)
