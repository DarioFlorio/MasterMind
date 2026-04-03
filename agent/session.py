"""
agent/session.py — Conversation history with sliding-window context compression.

Two modes:
  - UNLIMITED_CONTEXT=0 (default): fixed window, keep last N messages
  - UNLIMITED_CONTEXT=1          : sliding window with LLM summarisation

The sliding window means the effective context is unlimited: old messages are
compressed into a rolling summary while recent messages stay verbatim.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.model_client import ModelClient

from config.settings import CONTEXT_SIZE, UNLIMITED_CONTEXT, VERBOSE


def _count_tokens_approx(text: str) -> int:
    """~1 token per 3.5 chars (conservative estimate for code/prose mix)."""
    return max(1, len(text) // 3)


@dataclass
class Message:
    role:      str
    content:   str
    timestamp: float = field(default_factory=time.time)
    meta:      dict  = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class Session:
    """
    Manages conversation history for one agent session.
    Supports both fixed-window and sliding-window (unlimited) context modes.
    """

    # Reserve tokens for system prompt + current response
    _RESERVE = 2048
    # Sliding: summarise when context exceeds this fraction of CONTEXT_SIZE
    _COMPRESS_AT = 0.75

    def __init__(
        self,
        session_id: str | None = None,
        model_client: "ModelClient | None" = None,
    ) -> None:
        import uuid
        self.session_id: str = session_id or str(uuid.uuid4())[:8]
        self.created_at: float = time.time()
        self._messages: list[Message] = []
        self._summary: str = ""          # compressed summary of old turns
        self._client = model_client
        self._unlimited = UNLIMITED_CONTEXT
        self._budget = CONTEXT_SIZE - self._RESERVE

    # ── Mutation ─────────────────────────────────────────────────────────────

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))
        self._maybe_compress()

    def add_assistant(self, content: str, meta: dict | None = None) -> None:
        self._messages.append(Message(role="assistant", content=content, meta=meta or {}))
        self._maybe_compress()

    def add_tool_result(self, xml: str) -> None:
        self._messages.append(Message(role="user", content=xml, meta={"tool_result": True}))
        self._maybe_compress()

    def clear(self) -> None:
        self._messages.clear()
        self._summary = ""

    # ── Query ─────────────────────────────────────────────────────────────────

    def to_api_messages(self) -> list[dict]:
        """Return the context window as a list of {role, content} dicts."""
        if self._unlimited and self._summary:
            # Prepend summary as a system-style user message
            summary_msg = {
                "role": "user",
                "content": f"[Context summary of earlier conversation]\n{self._summary}",
            }
            recent = [m.to_dict() for m in self._messages]
            # Ensure alternating roles (llama.cpp requirement)
            return self._fix_alternation([summary_msg] + recent)
        return self._fix_alternation([m.to_dict() for m in self._messages])

    def __len__(self) -> int:
        return len(self._messages)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "summary":    self._summary,
            "messages":   [
                {"role": m.role, "content": m.content, "ts": m.timestamp}
                for m in self._messages
            ],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, model_client: "ModelClient | None" = None) -> "Session":
        data = json.loads(path.read_text(encoding="utf-8"))
        s = cls(session_id=data.get("session_id"), model_client=model_client)
        s.created_at = data.get("created_at", time.time())
        s._summary = data.get("summary", "")
        for m in data.get("messages", []):
            s._messages.append(
                Message(role=m["role"], content=m["content"], timestamp=m.get("ts", 0))
            )
        return s

    # ── Internal ──────────────────────────────────────────────────────────────

    def _token_count(self) -> int:
        total = _count_tokens_approx(self._summary)
        for m in self._messages:
            total += _count_tokens_approx(m.content)
        return total

    def _maybe_compress(self) -> None:
        if not self._unlimited:
            # Fixed window: just trim oldest non-tool messages
            while self._token_count() > self._budget and len(self._messages) > 4:
                self._messages.pop(0)
            return

        if self._token_count() < self._budget * self._COMPRESS_AT:
            return
        if len(self._messages) < 6:
            return

        # Compress oldest half of messages into summary
        cut = max(2, len(self._messages) // 2)
        to_compress = self._messages[:cut]
        self._messages = self._messages[cut:]

        new_chunk = "\n".join(f"{m.role}: {m.content[:300]}" for m in to_compress)

        if self._client:
            try:
                prompt = (
                    "Produce a concise bullet-point summary of the following "
                    "conversation segment. Preserve all technical details, "
                    "file names, decisions made, and tool results. Be brief.\n\n"
                    f"{new_chunk}"
                )
                compressed = self._client.complete(
                    [{"role": "user", "content": prompt}],
                    max_tokens=512,
                    stream=False,
                )
                if isinstance(compressed, str):
                    new_summary = compressed.strip()
                else:
                    new_summary = new_chunk[:800]
            except Exception:
                new_summary = new_chunk[:800]
        else:
            new_summary = new_chunk[:800]

        if self._summary:
            self._summary = f"{self._summary}\n\n--- Later ---\n{new_summary}"
        else:
            self._summary = new_summary

        if VERBOSE:
            print(f"[session] Compressed {cut} messages → summary ({len(self._summary)} chars)")

    @staticmethod
    def _fix_alternation(messages: list[dict]) -> list[dict]:
        """
        llama.cpp requires strictly alternating user/assistant roles.
        Merge consecutive same-role messages.
        """
        if not messages:
            return messages
        out = [messages[0].copy()]
        for m in messages[1:]:
            if m["role"] == out[-1]["role"]:
                out[-1]["content"] += "\n" + m["content"]
            else:
                out.append(m.copy())
        return out
