from __future__ import annotations
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR

_MAX_BYTES = 200_000  # 200 KB hard cap


class ReadFileTool(BaseTool):
    name = "read_file"
    description = (
        "Read a file and return its contents. "
        "Supports optional line range. Use offset/limit for large files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path":   {"type": "string", "description": "File path (absolute or relative to cwd)"},
            "offset": {"type": "integer", "description": "Start line (1-based, optional)"},
            "limit":  {"type": "integer", "description": "Max lines to return (optional)"},
        },
        "required": ["path"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        raw  = inp.get("path", "")
        off  = inp.get("offset")
        lim  = inp.get("limit")

        path = self._resolve(raw)
        if not path:
            return ToolResult(f"Path not provided.", is_error=True)
        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if path.is_dir():
            return ToolResult(f"Path is a directory. Use list_dir instead: {path}", is_error=True)
        if path.stat().st_size > _MAX_BYTES:
            return ToolResult(
                f"File too large ({path.stat().st_size/1024:.0f} KB). "
                f"Use offset/limit parameters or bash 'head'/'tail'.", is_error=True
            )

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

        if off is not None or lim is not None:
            lines = text.splitlines(keepends=True)
            start = max(0, (int(off) - 1) if off else 0)
            end   = (start + int(lim)) if lim else len(lines)
            text  = "".join(lines[start:end])

        if not text:
            return ToolResult("(empty file)")

        # Prefix with line numbers for code files
        if path.suffix in (".py", ".ts", ".js", ".tsx", ".jsx", ".java", ".c", ".cpp",
                           ".go", ".rs", ".rb", ".php", ".cs", ".kt", ".swift"):
            numbered = []
            for i, line in enumerate(text.splitlines(), start=(int(off) if off else 1)):
                numbered.append(f"{i:4d}  {line}")
            text = "\n".join(numbered)

        return ToolResult(text[:_MAX_BYTES])

    def _resolve(self, raw: str) -> Path | None:
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else self._cwd / p
