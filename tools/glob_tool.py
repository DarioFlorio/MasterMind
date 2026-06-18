from __future__ import annotations
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR


class GlobTool(BaseTool):
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Examples: '**/*.py', 'src/**/*.ts', '*.json'. "
        "Returns sorted list of matching paths."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')"},
            "path":    {"type": "string", "description": "Root directory to search (default: cwd)"},
            "limit":   {"type": "integer","description": "Max results (default 200)"},
        },
        "required": ["pattern"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        pattern = inp.get("pattern", "")
        root    = Path(inp["path"]) if inp.get("path") else self._cwd
        limit   = int(inp.get("limit", 200))

        if not pattern:
            return ToolResult("No pattern provided.", is_error=True)
        if not root.exists():
            return ToolResult(f"Directory not found: {root}", is_error=True)

        try:
            matches = sorted(root.glob(pattern))
            # Filter out __pycache__, .git, node_modules
            skip = {"__pycache__", ".git", "node_modules", ".venv", "venv", ".tox"}
            matches = [
                m for m in matches
                if not any(part in skip for part in m.parts)
            ]
            paths = [str(m.relative_to(root)) for m in matches[:limit]]
        except Exception as e:
            return ToolResult(str(e), is_error=True)

        if not paths:
            return ToolResult(f"No files matched '{pattern}' in {root}")

        header = f"{len(paths)} file(s) matched '{pattern}'"
        if len(matches) > limit:
            header += f" (showing first {limit})"
        return ToolResult(header + "\n" + "\n".join(paths))
