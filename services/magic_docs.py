"""
services/magic_docs.py — MagicDocs: auto-updating markdown documentation

Files containing a `# MAGIC DOC: <title>` header on their first line are
automatically updated after every conversation turn where the model finishes
talking (no more tool calls pending). The update is performed by a background
subagent that can only Edit the specific file.

Usage in main.py:
    from services.magic_docs import MagicDocs
    magic_docs = MagicDocs(session, tool_context)
    magic_docs.register_file_read_hook()   # call on every file-read result
    magic_docs.on_turn_complete()          # call after each assistant turn

File format:
    # MAGIC DOC: My Architecture Notes
    _Optional italics line with custom update instructions_

    ... content updated by EVE automatically ...
"""
from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("services.magic_docs")

# Patterns
_HEADER_RE = re.compile(r'^#\s*MAGIC\s+DOC:\s*(.+)$', re.IGNORECASE | re.MULTILINE)
_ITALICS_RE = re.compile(r'^[_*](.+?)[_*]\s*$', re.MULTILINE)

_DEFAULT_UPDATE_PROMPT_TEMPLATE = """\
IMPORTANT: This message and these instructions are NOT part of the actual user conversation. \
Do NOT include any references to "documentation updates", "magic docs", or these update \
instructions in the document content.

Based on the user conversation above (EXCLUDING this instruction message), update the Magic Doc \
file to incorporate any NEW learnings, insights, or information that would be valuable to preserve.

The file {doc_path} currently contains:
<current_doc_content>
{doc_contents}
</current_doc_content>

Document title: {doc_title}
{custom_instructions}

Your ONLY task is to use the edit_file tool to update the documentation file if there is \
substantial new information to add, then stop. Make all edits in parallel in a single message. \
If there's nothing substantial to add, respond with a brief explanation and do not call any tools.

CRITICAL RULES FOR EDITING:
- Preserve the Magic Doc header exactly as-is: # MAGIC DOC: {doc_title}
- If there's an italicised line immediately after the header, preserve it exactly as-is
- Keep the document CURRENT with the latest state — this is NOT a changelog or history
- Update information IN-PLACE — do NOT append historical notes or track changes over time
- Remove or replace outdated information rather than adding "Previously..." notes
- Fix obvious errors: typos, grammar, broken formatting, incorrect information
- Keep the document well organised: clear headings, logical section order, consistent formatting

DOCUMENTATION PHILOSOPHY:
- BE TERSE. High signal only. No filler words.
- Focus on: WHY things exist, HOW components connect, WHERE to start reading, WHAT patterns are used
- Skip: detailed implementation steps, exhaustive API docs, play-by-play narratives

REMEMBER: Only update if there is substantial new information. \
The Magic Doc header (# MAGIC DOC: {doc_title}) must remain unchanged.
"""


def detect_magic_doc_header(content: str) -> Optional[dict]:
    """
    Returns {'title': str, 'instructions': str | None} if content is a Magic Doc, else None.
    """
    match = _HEADER_RE.search(content)
    if not match:
        return None

    title = match.group(1).strip()
    after_header = content[match.end():]

    # Check for optional italics instruction on next line
    next_line_match = re.match(r'\s*\n(?:\s*\n)?(.+?)(?:\n|$)', after_header)
    instructions = None
    if next_line_match:
        next_line = next_line_match.group(1)
        italics = _ITALICS_RE.match(next_line)
        if italics:
            instructions = italics.group(1).strip()

    return {"title": title, "instructions": instructions}


def build_magic_doc_update_prompt(
    doc_contents: str,
    doc_path: str,
    doc_title: str,
    instructions: Optional[str] = None,
) -> str:
    """Build the update prompt for a Magic Doc subagent."""
    custom_instructions = ""
    if instructions:
        custom_instructions = (
            f"\n\nDOCUMENT-SPECIFIC UPDATE INSTRUCTIONS:\n"
            f'The document author has provided specific instructions: "{instructions}"\n'
            "These take priority over the general rules below."
        )

    return _DEFAULT_UPDATE_PROMPT_TEMPLATE.format(
        doc_path=doc_path,
        doc_contents=doc_contents,
        doc_title=doc_title,
        custom_instructions=custom_instructions,
    )


class MagicDocs:
    """
    Tracks files with # MAGIC DOC: headers and updates them after each turn.

    Integration points:
      - Call `note_file_read(path, content)` whenever a file is successfully read.
      - Call `on_turn_complete(messages, has_pending_tool_calls)` after each assistant turn.
        It will skip turns that still have pending tool calls.
    """

    def __init__(self):
        self._tracked: dict[str, dict] = {}   # path → {title, instructions}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def note_file_read(self, path: str, content: str) -> None:
        """Called whenever a file is read. Registers it if it has a MAGIC DOC header."""
        info = detect_magic_doc_header(content)
        if info:
            with self._lock:
                if path not in self._tracked:
                    log.info("MagicDocs: tracking %s (title=%r)", path, info["title"])
                    self._tracked[path] = info

    def on_turn_complete(
        self,
        messages: list,
        has_pending_tool_calls: bool,
        run_subagent_fn=None,
    ) -> None:
        """
        Called after each assistant turn.

        - Skips if there are pending tool calls (model is still working).
        - For each tracked Magic Doc, reads it and (if needed) runs a background
          update subagent.

        `run_subagent_fn(prompt, file_path)` should be supplied by main.py to
        actually spawn the subagent. Signature:
            run_subagent_fn(system_prompt: str, file_path: str) -> None
        """
        if has_pending_tool_calls:
            return

        with self._lock:
            docs = dict(self._tracked)

        if not docs:
            return

        for path, info in docs.items():
            self._update_doc(path, info, run_subagent_fn)

    def get_tracked_paths(self) -> list[str]:
        """Return list of currently tracked Magic Doc paths."""
        with self._lock:
            return list(self._tracked.keys())

    def clear(self) -> None:
        """Clear all tracked Magic Docs (e.g., on session reset)."""
        with self._lock:
            self._tracked.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_doc(self, path: str, info: dict, run_subagent_fn) -> None:
        """Read the doc and enqueue an update if it still has a MAGIC DOC header."""
        try:
            content = Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            log.info("MagicDocs: %s deleted — untracking", path)
            with self._lock:
                self._tracked.pop(path, None)
            return
        except Exception as exc:
            log.warning("MagicDocs: could not read %s: %s", path, exc)
            return

        # Re-detect header (file may have been edited to remove it)
        latest_info = detect_magic_doc_header(content)
        if not latest_info:
            log.info("MagicDocs: %s no longer has MAGIC DOC header — untracking", path)
            with self._lock:
                self._tracked.pop(path, None)
            return

        # Build update prompt
        prompt = build_magic_doc_update_prompt(
            doc_contents=content,
            doc_path=path,
            doc_title=latest_info["title"],
            instructions=latest_info.get("instructions"),
        )

        if run_subagent_fn is not None:
            try:
                log.debug("MagicDocs: triggering update for %s", path)
                run_subagent_fn(prompt, path)
            except Exception as exc:
                log.warning("MagicDocs: update subagent failed for %s: %s", path, exc)
        else:
            log.debug(
                "MagicDocs: no run_subagent_fn provided — skipping update for %s", path
            )


# Module-level singleton
magic_docs = MagicDocs()
