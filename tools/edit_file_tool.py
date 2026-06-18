from __future__ import annotations
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR


class EditFileTool(BaseTool):
    name = "edit_file"
    description = (
        "Make a surgical edit to a file by replacing an exact string. "
        "old_str must match the file EXACTLY (including whitespace/indentation). "
        "Use read_file first to confirm the exact text to replace. "
        "For inserting at a specific location, include surrounding context in old_str."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path":     {"type": "string", "description": "File path to edit"},
            "old_str":  {"type": "string", "description": "Exact string to find and replace"},
            "new_str":  {"type": "string", "description": "Replacement string (empty = delete)"},
        },
        "required": ["path", "old_str"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        # ── Accept alternate param names from different model conventions ──
        # Some models emit "content_original"/"content" instead of "old_str"/"new_str"
        # Others use "search"/"replace", "find"/"replacement", etc.
        if not inp.get("old_str"):
            for alias in ("content_original", "search", "find", "text", "original"):
                if inp.get(alias):
                    inp["old_str"] = inp.pop(alias); break
        if not inp.get("new_str"):
            for alias in ("content", "replace", "replacement", "new_text", "updated"):
                if alias in inp:
                    inp["new_str"] = inp.pop(alias); break
        # ──────────────────────────────────────────────────────────────────
        raw     = inp.get("path", "")
        old_str = inp.get("old_str", "")
        new_str = inp.get("new_str", "")

        if not raw:
            return ToolResult("No path provided.", is_error=True)
        if not old_str:
            return ToolResult("old_str is required.", is_error=True)

        path = Path(raw) if Path(raw).is_absolute() else self._cwd / raw

        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)

        try:
            original = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

        count = original.count(old_str)
        if count == 0:
            # Give a helpful diff hint
            snippet = old_str[:100].replace("\n", "↵")
            return ToolResult(
                f"old_str not found in {path.name}.\n"
                f"Looking for: {snippet!r}\n"
                f"Tip: Use read_file to confirm exact whitespace/indentation.",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                f"old_str matched {count} locations in {path.name}. "
                f"Add more context to make it unique.",
                is_error=True,
            )

        updated = original.replace(old_str, new_str, 1)
        try:
            path.write_text(updated, encoding="utf-8")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

        added   = new_str.count("\n") - old_str.count("\n")
        sign    = "+" if added >= 0 else ""
        return ToolResult(f"Edited {path.name} ({sign}{added} lines)")
