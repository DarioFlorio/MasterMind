from __future__ import annotations
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR

_SKIP = {"__pycache__", ".git", "node_modules", ".venv", "venv"}


class ListDirTool(BaseTool):
    name = "list_dir"
    description = "List directory contents as a tree. Shows files and subdirectories."
    input_schema = {
        "type": "object",
        "properties": {
            "path":  {"type": "string",  "description": "Directory path (default: cwd)"},
            "depth": {"type": "integer", "description": "Tree depth (default 2, max 4)"},
        },
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = Path(working_dir)

    def execute(self, inp: dict) -> ToolResult:
        raw   = inp.get("path", "")
        depth = min(int(inp.get("depth", 2)), 4)

        root  = Path(raw) if raw else self._cwd
        if not root.is_absolute():
            root = self._cwd / root

        if not root.exists():
            return ToolResult(f"Not found: {root}", is_error=True)
        if root.is_file():
            stat = root.stat()
            return ToolResult(f"{root.name}  ({stat.st_size} bytes)")

        lines = [str(root)]
        self._walk(root, "", depth, lines)
        return ToolResult("\n".join(lines))

    def _walk(self, directory: Path, prefix: str, depth: int, lines: list) -> None:
        if depth == 0:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return

        entries = [e for e in entries if e.name not in _SKIP and not e.name.startswith(".")]
        for i, entry in enumerate(entries):
            connector = "└── " if i == len(entries) - 1 else "├── "
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if i == len(entries) - 1 else "│   "
                self._walk(entry, prefix + extension, depth - 1, lines)
            else:
                size = entry.stat().st_size
                size_str = f"{size:,}b" if size < 10000 else f"{size//1024}KB"
                lines.append(f"{prefix}{connector}{entry.name}  ({size_str})")
