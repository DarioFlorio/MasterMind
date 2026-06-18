"""
agent/context_offloader.py — Automatic context offloading for long tasks.

When EVE is close to her context limit, this module:
  1. Detects the approaching limit
  2. Writes completed work to a temp file
  3. Injects a compact summary back into the prompt
  4. Allows the task to continue across turns

Works with the query engine's context budget system.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional


_WARN_RATIO  = 0.72   # Start warning at 72% full
_SAVE_RATIO  = 0.80   # Auto-save at 80% full
_CRITICAL    = 0.92   # Hard stop at 92%


class ContextOffloader:
    """
    Monitors context usage and offloads completed work to disk
    so long tasks can continue without hitting the context ceiling.
    """

    def __init__(self, working_dir: str, context_size: int) -> None:
        self._working_dir = Path(working_dir)
        self._context_size = max(context_size, 1)
        self._temp_dir = self._working_dir / "temp" / "context_offload"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._offload_index: list[dict] = []
        self._task_id: str = f"task_{int(time.time())}"

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, messages: list[dict], goal: str = "") -> Optional[str]:
        """
        Call after every tool result. Returns an injection string if action
        is needed, or None if context is fine.
        """
        tokens = self._est_tokens(messages)
        ratio = tokens / self._context_size

        if ratio >= _CRITICAL:
            return self._critical_message(ratio, goal)
        if ratio >= _SAVE_RATIO:
            return self._save_and_continue_message(messages, ratio, goal)
        if ratio >= _WARN_RATIO:
            return self._warn_message(ratio, goal)
        return None

    def save_progress(self, content: str, label: str = "") -> Path:
        """
        Explicitly save content to a temp file. Returns the path.
        Call this when the model decides to checkpoint.
        """
        idx = len(self._offload_index) + 1
        fname = f"{self._task_id}_part{idx:03d}.md"
        path = self._temp_dir / fname
        path.write_text(content, encoding="utf-8")

        entry = {
            "index": idx,
            "file": str(path),
            "label": label or f"Part {idx}",
            "ts": time.time(),
            "chars": len(content),
        }
        self._offload_index.append(entry)
        self._save_index()
        return path

    def collate(self) -> str:
        """Merge all saved parts into one string. Call at task end."""
        if not self._offload_index:
            return ""
        parts = []
        for entry in self._offload_index:
            p = Path(entry["file"])
            if p.exists():
                parts.append(f"# {entry['label']}\n\n{p.read_text(encoding='utf-8')}")
        return "\n\n---\n\n".join(parts)

    def collate_to_file(self, output_name: str = "") -> Path:
        """Merge all parts to a single output file."""
        content = self.collate()
        if not output_name:
            output_name = f"{self._task_id}_complete.md"
        out = self._working_dir / output_name
        out.write_text(content, encoding="utf-8")
        return out

    def list_parts(self) -> list[dict]:
        return list(self._offload_index)

    def load_index(self) -> list[dict]:
        """Load any existing index for this task."""
        index_file = self._temp_dir / f"{self._task_id}_index.json"
        if index_file.exists():
            try:
                self._offload_index = json.loads(index_file.read_text())
            except Exception:
                pass
        return self._offload_index

    # ── Internal ──────────────────────────────────────────────────────────────

    def _warn_message(self, ratio: float, goal: str) -> str:
        return (
            f"\n⚠️ [ContextOffloader] Context is {ratio:.0%} full.\n"
            f"Consider saving progress with write_file before continuing.\n"
            f"Temp directory: {self._temp_dir}\n"
        )

    def _save_and_continue_message(
        self, messages: list[dict], ratio: float, goal: str
    ) -> str:
        # Build a summary of what's in context
        assistant_msgs = [
            m["content"] for m in messages
            if m.get("role") == "assistant" and len(m.get("content", "")) > 50
        ]
        summary = "\n".join(assistant_msgs[-3:])[:1000]

        return (
            f"\n🚨 [ContextOffloader] Context at {ratio:.0%} — MUST save progress now.\n\n"
            f"REQUIRED ACTIONS (do these before anything else):\n"
            f"1. Call write_file to save all completed work so far to:\n"
            f"   {self._temp_dir}/{self._task_id}_partNNN.md\n"
            f"2. The next turn will start fresh and load from that file.\n"
            f"3. Current goal to resume: {goal[:200] if goal else '(same as before)'}\n\n"
            f"Recent progress snapshot:\n{summary}\n"
        )

    def _critical_message(self, ratio: float, goal: str) -> str:
        return (
            f"\n🛑 [ContextOffloader] CRITICAL: Context at {ratio:.0%}.\n"
            f"STOP current action. Save everything to disk NOW.\n"
            f"write_file to: {self._temp_dir}/{self._task_id}_emergency.md\n"
            f"Then use /compact to free context before continuing.\n"
            f"Goal to resume: {goal[:200] if goal else '(check memory)'}\n"
        )

    def _save_index(self) -> None:
        index_file = self._temp_dir / f"{self._task_id}_index.json"
        try:
            index_file.write_text(
                json.dumps(self._offload_index, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    @staticmethod
    def _est_tokens(messages: list[dict]) -> int:
        return sum(max(1, len(m.get("content", "")) // 3) for m in messages)
