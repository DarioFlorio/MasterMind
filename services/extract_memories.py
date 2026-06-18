"""
services/extract_memories.py — ExtractMemories: end-of-turn memory extraction

Runs after every complete query loop (when the model produces a final
response with no pending tool calls). A background subagent analyses the
new messages since the last extraction and writes durable memories to the
memdir (facts.json or individual .md files).

Integration:
    from services.extract_memories import ExtractMemories
    em = ExtractMemories(mem_dir=Path("memdir"))
    em.init()

    # After each complete turn:
    em.on_turn_complete(messages, run_subagent_fn)

    # On session end:
    em.drain()
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("services.extract_memories")

_EXTRACT_PROMPT_TEMPLATE = """\
You are now acting as the memory extraction subagent. Analyse the most recent ~{new_message_count} \
messages above and use them to update persistent memory.

Available tools: read_file, grep_tool, glob_tool, read-only bash (ls/find/cat/stat/wc/head/tail), \
and edit_file/write_file for paths inside the memory directory only. \
Bash rm is not permitted. All other tools will be denied.

You have a limited turn budget. edit_file requires a prior read_file of the same file, so the \
efficient strategy is:
  turn 1 — issue all read_file calls in parallel for every file you might update
  turn 2 — issue all write_file/edit_file calls in parallel

You MUST only use content from the last ~{new_message_count} messages. Do not waste turns \
investigating or verifying that content further.

{existing_memories_block}

## What to Save

Save memories that are:
- **Facts about the user**: preferences, role, skills, constraints (save to facts.json)
- **Project conventions**: naming, tooling, test commands, architecture patterns (save to memdir/*.md)
- **Error patterns**: bugs encountered and how they were fixed (save to memdir/errors.md)
- **Workflow knowledge**: commands, sequences, shortcuts that recur (save to memdir/workflow.md)

## What NOT to Save

- Ephemeral context: what file was edited just now, which line has a bug today
- Conversational meta: "the user said", "I suggested"  
- Already-known information already in facts.json or existing memory files
- Sensitive data: API keys, passwords, personal information

## How to Save

Write each memory to its own file in `{memory_dir}` using this format:

```
---
title: Short descriptive title
tags: [relevant, tags]
created: {today}
---

Memory content here.
```

Or update `facts.json` for user facts:
```json
{{
  "preferences": {{"language": "Python", "style": "concise"}},
  "role": "backend engineer",
  ...
}}
```

If the user explicitly asked you to remember something, save it immediately.
If they asked you to forget something, find and remove the relevant entry.
"""


class ExtractMemories:
    """
    Post-turn memory extraction. Runs a background subagent to distil
    durable memories from new conversation turns.
    """

    def __init__(self, mem_dir: Optional[Path] = None):
        if mem_dir is None:
            mem_dir = Path(__file__).parent.parent / "memdir"
        self._mem_dir = Path(mem_dir)
        self._last_processed_idx = 0   # index into messages list
        self._in_progress = False
        self._lock = threading.Lock()
        self._pending: Optional[tuple] = None   # stashed (messages, fn) for trailing run

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Ensure memory directory exists."""
        try:
            self._mem_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            log.warning("ExtractMemories: could not create mem_dir: %s", exc)

    def on_turn_complete(
        self,
        messages: list,
        run_subagent_fn: Optional[Callable] = None,
        *,
        has_pending_tool_calls: bool = False,
    ) -> None:
        """
        Call after each complete assistant turn (no pending tool calls).
        `run_subagent_fn(prompt)` should run a subagent and return.
        """
        if has_pending_tool_calls:
            return

        with self._lock:
            if self._in_progress:
                # Stash latest context for trailing run
                self._pending = (messages, run_subagent_fn)
                log.debug("ExtractMemories: extraction in progress — stashing")
                return

            self._in_progress = True

        self._run_extraction(messages, run_subagent_fn)

    def drain(self, timeout_s: float = 30.0) -> None:
        """Wait for any in-progress extraction to finish."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if not self._in_progress:
                    return
            time.sleep(0.2)
        log.warning("ExtractMemories: drain timed out after %.1fs", timeout_s)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_extraction(self, messages: list, run_subagent_fn) -> None:
        try:
            new_messages = messages[self._last_processed_idx:]
            if not new_messages:
                return

            existing = self._scan_existing_memories()
            prompt = self._build_prompt(len(new_messages), existing)

            if run_subagent_fn is not None:
                try:
                    run_subagent_fn(prompt)
                    self._last_processed_idx = len(messages)
                    log.debug("ExtractMemories: extraction done, cursor at %d", self._last_processed_idx)
                except Exception as exc:
                    log.warning("ExtractMemories: extraction failed: %s", exc)
            else:
                log.debug("ExtractMemories: no run_subagent_fn — skipping")
                self._last_processed_idx = len(messages)

        finally:
            with self._lock:
                self._in_progress = False
                pending = self._pending
                self._pending = None

            if pending:
                log.debug("ExtractMemories: running trailing extraction")
                msgs, fn = pending
                with self._lock:
                    self._in_progress = True
                self._run_extraction(msgs, fn)

    def _scan_existing_memories(self) -> str:
        """Return a brief manifest of existing memory files."""
        lines = []
        try:
            for p in sorted(self._mem_dir.iterdir()):
                if p.suffix in (".md", ".json") and p.name != "facts.json":
                    lines.append(f"- {p.name}")
            if (self._mem_dir / "facts.json").exists():
                lines.insert(0, "- facts.json (user facts store)")
        except Exception:
            pass
        return "\n".join(lines) if lines else ""

    def _build_prompt(self, new_message_count: int, existing: str) -> str:
        import datetime
        existing_block = (
            f"\n## Existing memory files\n\n{existing}\n\n"
            "Check this list before writing — update an existing file rather than creating a duplicate."
            if existing else ""
        )
        return _EXTRACT_PROMPT_TEMPLATE.format(
            new_message_count=new_message_count,
            existing_memories_block=existing_block,
            memory_dir=str(self._mem_dir),
            today=datetime.date.today().isoformat(),
        )


# Module-level singleton
extract_memories = ExtractMemories()
