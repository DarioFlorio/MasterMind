from __future__ import annotations
import re
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR

_SKIP_DIRS  = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".tox", "dist", "build"}
_SKIP_EXTS  = {".pyc", ".pyo", ".exe", ".dll", ".so", ".dylib", ".bin", ".png",
               ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".woff", ".ttf"}
_MAX_OUTPUT = 8000


class GrepTool(BaseTool):
    name = "grep"
    description = (
        "Search file contents for a regex pattern. "
        "Returns matching lines with file:line context. "
        "Use glob to find files first if needed."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern":   {"type": "string",  "description": "Regex pattern to search for"},
            "path":      {"type": "string",  "description": "File or directory to search"},
            "glob":      {"type": "string",  "description": "File glob filter (e.g. '*.py')"},
            "ignorecase":{"type": "boolean", "description": "Case-insensitive search (default true)"},
            "context":   {"type": "integer", "description": "Lines of context around matches (default 0)"},
        },
        "required": ["pattern"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        pattern    = inp.get("pattern", "")
        root_raw   = inp.get("path", "")
        glob_pat   = inp.get("glob", "*")
        ignorecase = inp.get("ignorecase", True)
        ctx_lines  = int(inp.get("context", 0))

        if not pattern:
            return ToolResult("No pattern provided.", is_error=True)

        root = Path(root_raw) if root_raw else self._cwd
        if not root.is_absolute():
            root = self._cwd / root

        flags = re.IGNORECASE if ignorecase else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return ToolResult(f"Invalid regex: {e}", is_error=True)

        files: list[Path] = []
        if root.is_file():
            files = [root]
        else:
            files = [
                f for f in root.rglob(glob_pat)
                if f.is_file()
                and f.suffix not in _SKIP_EXTS
                and not any(p in _SKIP_DIRS for p in f.parts)
            ]

        results: list[str] = []
        total_matches = 0

        for fpath in sorted(files)[:200]:
            try:
                lines = fpath.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue

            for i, line in enumerate(lines):
                if regex.search(line):
                    total_matches += 1
                    rel = str(fpath.relative_to(self._cwd) if fpath.is_absolute() else fpath)
                    if ctx_lines:
                        start = max(0, i - ctx_lines)
                        end   = min(len(lines), i + ctx_lines + 1)
                        block = [f"{rel}:{j+1}: {lines[j]}" for j in range(start, end)]
                        results.append("\n".join(block))
                    else:
                        results.append(f"{rel}:{i+1}: {line}")

        if not results:
            return ToolResult(f"No matches for '{pattern}'")

        out = "\n".join(results)
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + f"\n... (truncated, {total_matches} total matches)"
        return ToolResult(f"{total_matches} match(es) for '{pattern}'\n{out}")
