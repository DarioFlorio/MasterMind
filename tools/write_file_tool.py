from __future__ import annotations
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR


class WriteFileTool(BaseTool):
    name = "write_file"
    description = (
        "Write content to a file, creating parent directories as needed. "
        "Overwrites existing files. For surgical edits use edit_file instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path":    {"type": "string", "description": "File path to write"},
            "content": {"type": "string", "description": "Content to write"},
        },
        "required": ["path", "content"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        # Accept common alternative key names that small models produce
        raw = (
            inp.get("path") or inp.get("file") or
            inp.get("filename") or inp.get("filepath") or
            inp.get("file_path") or ""
        )
        content = (
            inp.get("content") or inp.get("text") or
            inp.get("data") or inp.get("body") or ""
        )
        if not raw:
            return ToolResult("No path provided.", is_error=True)

        path = Path(raw) if Path(raw).is_absolute() else self._cwd / raw
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            lines = content.count("\n") + 1
            return ToolResult(f"Written {len(content)} bytes ({lines} lines) → {path}")
        except Exception as e:
            return ToolResult(str(e), is_error=True)
