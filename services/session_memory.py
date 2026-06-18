"""
services/session_memory.py — SessionMemory: rolling working-memory summary

Maintains a structured markdown file (~/.eve/session_memory.md) that is
updated in the background after every N tool calls or K tokens of
conversation. The summary is injected into the system prompt so EVE
always has "what we were doing" even after context compaction.

Integration:
    from services.session_memory import SessionMemory
    sm = SessionMemory()
    sm.init()                              # call once at startup
    sm.on_turn(messages, tool_call_count)  # call after each assistant turn
    summary = sm.get_current_summary()     # inject into system prompt
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.session_memory")

# ── Config ────────────────────────────────────────────────────────────────────
MIN_TOOL_CALLS_BETWEEN_UPDATES = 5   # trigger update after N tool calls
MIN_MESSAGES_TO_INIT = 10            # don't start until conversation has depth
UPDATE_COOLDOWN_S = 60               # don't update more than once per minute

_DEFAULT_TEMPLATE = """\
# Session Title
_A short and distinctive 5-10 word descriptive title for the session. Super info dense, no filler_

# Current State
_What is actively being worked on right now? Pending tasks not yet completed. Immediate next steps._

# Task Specification
_What did the user ask to build? Any design decisions or other explanatory context_

# Files and Functions
_What are the important files? In short, what do they contain and why are they relevant?_

# Workflow
_What bash commands are usually run and in what order? How to interpret their output?_

# Errors and Corrections
_Errors encountered and how they were fixed. What did the user correct? What approaches failed?_

# Codebase and System Documentation
_What are the important system components? How do they work/fit together?_

# Learnings
_What has worked well? What has not? What to avoid? Do not duplicate items from other sections_

# Key Results
_If the user asked for specific output such as an answer, table, or document, repeat the exact result here_

# Worklog
_Step by step, what was attempted, done? Very terse summary for each step_
"""

_UPDATE_PROMPT_TEMPLATE = """\
IMPORTANT: This message and these instructions are NOT part of the actual user conversation. \
Do NOT include any references to "note-taking", "session notes extraction", or these update \
instructions in the notes content.

Based on the user conversation above (EXCLUDING this instruction message), update the session \
notes file.

The file {notes_path} has already been read for you. Here are its current contents:
<current_notes_content>
{current_notes}
</current_notes_content>

Your ONLY task is to use the edit_file tool to update the notes file, then stop. \
You can make multiple edits (update every section as needed) — make all edits in parallel \
in a single message. Do not call any other tools.

CRITICAL RULES FOR EDITING:
- The file must maintain its exact structure with all sections, headers, and italic descriptions intact
- NEVER modify, delete, or add section headers (lines starting with '#')
- NEVER modify or delete the italic _section description_ lines (start/end with underscores)
- ONLY update the actual content that appears BELOW the italic _section descriptions_
- Do NOT add any new sections outside the existing structure
- Write DETAILED, INFO-DENSE content — include specifics like file paths, function names, \
error messages, exact commands, technical details
- Keep each section under ~2000 tokens — condense by cycling out less important details
- IMPORTANT: Always update "Current State" to reflect the most recent work

Use the edit_file tool with file_path: {notes_path}

REMEMBER: Use the edit_file tool and stop. Do not continue after the edits.
"""


class SessionMemory:
    """
    Rolling working-memory summary updated in the background each turn.
    """

    def __init__(self, memory_dir: Optional[Path] = None):
        if memory_dir is None:
            memory_dir = Path.home() / ".eve" / "session_memory"
        self._dir = Path(memory_dir)
        self._path = self._dir / "session_memory.md"
        self._lock = threading.Lock()
        self._tool_calls_since_update = 0
        self._last_update_ts = 0.0
        self._initialized = False
        self._message_count_at_init = 0
        self._current_summary: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create the memory directory and initialise the file."""
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            if not self._path.exists():
                self._path.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
                log.info("SessionMemory: created %s", self._path)
            else:
                self._current_summary = self._path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("SessionMemory: init failed: %s", exc)

    def get_current_summary(self) -> str:
        """Return the latest session memory content (for injecting into system prompt)."""
        return self._current_summary

    def get_path(self) -> Path:
        return self._path

    def on_turn(
        self,
        messages: list,
        tool_call_count: int,
        run_extraction_fn=None,
    ) -> None:
        """
        Called after each assistant turn. Triggers an update if thresholds are met.

        `run_extraction_fn(prompt, file_path)` should spawn a background subagent.
        """
        with self._lock:
            self._tool_calls_since_update += tool_call_count

            # Wait until conversation has enough depth
            if len(messages) < MIN_MESSAGES_TO_INIT:
                return

            # Cooldown
            now = time.time()
            if now - self._last_update_ts < UPDATE_COOLDOWN_S:
                return

            # Check threshold
            if self._tool_calls_since_update < MIN_TOOL_CALLS_BETWEEN_UPDATES:
                return

            # Reset counters
            self._tool_calls_since_update = 0
            self._last_update_ts = now

        self._trigger_update(run_extraction_fn)

    def force_update(self, run_extraction_fn=None) -> None:
        """Manually trigger a session memory update (e.g., from /summary command)."""
        self._trigger_update(run_extraction_fn)

    def reload(self) -> str:
        """Reload the summary from disk and return it."""
        try:
            self._current_summary = self._path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("SessionMemory: reload failed: %s", exc)
        return self._current_summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _trigger_update(self, run_extraction_fn) -> None:
        try:
            current_notes = self._path.read_text(encoding="utf-8") if self._path.exists() else _DEFAULT_TEMPLATE
        except Exception as exc:
            log.warning("SessionMemory: could not read notes: %s", exc)
            return

        prompt = _UPDATE_PROMPT_TEMPLATE.format(
            notes_path=str(self._path),
            current_notes=current_notes,
        )

        if run_extraction_fn is not None:
            try:
                log.debug("SessionMemory: triggering background update")
                run_extraction_fn(prompt, str(self._path))
            except Exception as exc:
                log.warning("SessionMemory: extraction failed: %s", exc)
        else:
            log.debug("SessionMemory: no run_extraction_fn — skipping update")

    def build_update_prompt(self) -> str:
        """Build the update prompt (for use when running inline)."""
        try:
            current_notes = self._path.read_text(encoding="utf-8") if self._path.exists() else _DEFAULT_TEMPLATE
        except Exception:
            current_notes = _DEFAULT_TEMPLATE

        return _UPDATE_PROMPT_TEMPLATE.format(
            notes_path=str(self._path),
            current_notes=current_notes,
        )


# Module-level singleton
session_memory = SessionMemory()
