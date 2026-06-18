# -*- coding: utf-8 -*-
"""tools/brief_tool.py — Attach files to the session as context (MasterMind built-in BriefTool)."""
from __future__ import annotations
import base64
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult

_MAX_SIZE = 512 * 1024  # 512KB text cap
_BINARY_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".zip",
                ".tar", ".gz", ".exe", ".bin", ".pyc"}


class BriefTool(BaseTool):
    name = "brief"
    description = (
        "Attach a file to the session context. For text files, reads content directly. "
        "For images/PDFs, provides base64 reference. Use to give the agent context from files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path to attach"},
            "note": {"type": "string", "description": "Optional note about the file"},
            "max_lines": {"type": "integer", "description": "Max lines to read for text files (default: 500)"},
        },
        "required": ["path"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        path_str = inp.get("path", "")
        note = inp.get("note", "")
        max_lines = int(inp.get("max_lines", 500))

        if not path_str:
            return ToolResult(output="path is required", is_error=True)

        path = Path(path_str)
        if not path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        stat = path.stat()
        ext = path.suffix.lower()
        header = f"📎 {path.name} ({stat.st_size:,} bytes)"
        if note:
            header += f"\nNote: {note}"

        if ext in _BINARY_EXTS:
            try:
                data = path.read_bytes()
                b64 = base64.b64encode(data).decode()[:200]
                return ToolResult(
                    output=f"{header}\nBinary file ({ext}). Base64 prefix: {b64}...\n"
                           f"(Use read_file for text content)"
                )
            except Exception as e:
                return ToolResult(output=f"Cannot read binary file: {e}", is_error=True)

        # Text file
        try:
            if stat.st_size > _MAX_SIZE:
                # Read only first max_lines lines
                lines = []
                with open(path, encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                content = "".join(lines)
                truncated = f"\n[Truncated — {stat.st_size:,} bytes total, showing first {max_lines} lines]"
            else:
                content = path.read_text(encoding="utf-8", errors="replace")
                lines_count = content.count("\n")
                if lines_count > max_lines:
                    content = "\n".join(content.splitlines()[:max_lines])
                    truncated = f"\n[Showing first {max_lines} of {lines_count} lines]"
                else:
                    truncated = ""

            return ToolResult(output=f"{header}\n{'─'*40}\n{content}{truncated}")
        except Exception as e:
            return ToolResult(output=f"Cannot read file: {e}", is_error=True)
