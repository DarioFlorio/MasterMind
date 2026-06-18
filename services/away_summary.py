"""
services/away_summary.py — AwaySummary: "while you were away" recap

When the user returns to an idle EVE session, generates a 1–3 sentence
summary of where things stand — the high-level task and concrete next step.
Reads the latest session memory for context.

Usage in main.py:
    from services.away_summary import AwaySummary
    away = AwaySummary(session_memory)

    # Track idle time
    away.on_user_message()          # reset idle timer on every user message
    away.on_assistant_turn_done()   # note that the model finished

    # On next user input, check if we should show a recap
    if away.should_show_recap():
        summary = await away.generate(messages, model_client)
        if summary:
            print(f"[While you were away: {summary}]")
"""
from __future__ import annotations

import logging
import time
from typing import Optional

log = logging.getLogger("services.away_summary")

# Show recap if user was idle for at least this long (seconds)
IDLE_THRESHOLD_S = 300   # 5 minutes
RECENT_MESSAGE_WINDOW = 30  # only feed last N messages to the model


class AwaySummary:
    """
    Generates a short "where we left off" recap when the user returns after
    a period of inactivity.
    """

    def __init__(self, session_memory=None):
        self._session_memory = session_memory  # optional SessionMemory instance
        self._last_user_message_ts: float = time.time()
        self._last_turn_done_ts: float = 0.0
        self._idle_start_ts: float = 0.0
        self._turn_complete = False

    # ------------------------------------------------------------------
    # Event hooks (call these from main.py)
    # ------------------------------------------------------------------

    def on_user_message(self) -> None:
        """Reset idle timer. Call on every incoming user message."""
        self._last_user_message_ts = time.time()
        self._turn_complete = False

    def on_assistant_turn_done(self) -> None:
        """Note that the assistant finished a turn. Idle period starts now."""
        self._last_turn_done_ts = time.time()
        self._idle_start_ts = time.time()
        self._turn_complete = True

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    def should_show_recap(self) -> bool:
        """
        Returns True if the user has been idle long enough that a recap
        makes sense on their next message.
        """
        if not self._turn_complete:
            return False
        idle_s = time.time() - self._idle_start_ts
        return idle_s >= IDLE_THRESHOLD_S

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, messages: list, model_client=None) -> Optional[str]:
        """
        Generate and return a 1–3 sentence away recap.
        Returns None if there's nothing meaningful to say, or on error.

        `model_client` should have a `.complete(messages, system) -> str` method,
        or similar. If None, uses a simple heuristic fallback.
        """
        if not messages:
            return None

        memory_text = ""
        if self._session_memory is not None:
            try:
                memory_text = self._session_memory.get_current_summary()
            except Exception:
                pass

        prompt = self._build_prompt(memory_text)
        recent = messages[-RECENT_MESSAGE_WINDOW:]

        if model_client is not None:
            try:
                response = model_client.complete(
                    messages=recent + [{"role": "user", "content": prompt}],
                    system="",
                    max_tokens=150,
                )
                text = (response or "").strip()
                if text:
                    log.debug("AwaySummary: generated recap (%d chars)", len(text))
                    return text
            except Exception as exc:
                log.warning("AwaySummary: generation failed: %s", exc)
                return None
        else:
            # Heuristic fallback when no model client is available
            return self._heuristic_recap(messages)

        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(memory: str) -> str:
        memory_block = f"Session memory (broader context):\n{memory}\n\n" if memory else ""
        return (
            f"{memory_block}"
            "The user stepped away and is coming back. Write exactly 1-3 short sentences. "
            "Start by stating the high-level task — what they are building or debugging, "
            "not implementation details. Next: the concrete next step. "
            "Skip status reports and commit recaps."
        )

    @staticmethod
    def _heuristic_recap(messages: list) -> Optional[str]:
        """Simple fallback: find last assistant message and truncate it."""
        for msg in reversed(messages):
            role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
            if role == "assistant":
                content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                if isinstance(content, list):
                    # Extract text blocks
                    text = " ".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    )
                elif isinstance(content, str):
                    text = content
                else:
                    text = str(content)
                text = text.strip()
                if text:
                    return text[:200] + ("..." if len(text) > 200 else "")
        return None
