"""
tools/git_tool.py — MasterMind Git integration tool.

Exposes git operations the model can call: status, diff, add, commit,
log, show, branch, stash. Runs git as a subprocess in the working dir.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from tools.base_tool import BaseTool, ToolResult


class GitTool(BaseTool):
    name        = "git"
    description = (
        "Run git commands in the project repository. "
        "Supported ops: status, diff, add, commit, log, show, branch, stash, restore. "
        "Use this to inspect changes, stage files, and commit work."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "enum": ["status", "diff", "add", "commit", "log",
                         "show", "branch", "stash", "restore", "raw"],
                "description": "Git operation to perform.",
            },
            "args": {
                "type": "string",
                "description": (
                    "Arguments for the git command. "
                    "Examples: 'path/to/file' for add/restore; "
                    "'-m \"fix: typo\"' for commit; "
                    "'--oneline -10' for log; "
                    "'any git flags' for raw."
                ),
            },
            "path": {
                "type": "string",
                "description": "Working directory (defaults to engine CWD).",
            },
        },
        "required": ["op"],
    }

    def __init__(self, working_dir: str = ".") -> None:
        self._cwd = working_dir

    def execute(self, inp: dict) -> ToolResult:
        op   = (inp.get("op") or "status").lower().strip()
        args = (inp.get("args") or "").strip()
        cwd  = inp.get("path") or self._cwd

        if not Path(cwd).is_dir():
            return ToolResult(f"Directory not found: {cwd}", is_error=True)

        cmd = self._build_cmd(op, args)
        if cmd is None:
            return ToolResult(f"Unknown git op: {op}", is_error=True)

        return self._run(cmd, cwd)

    def _build_cmd(self, op: str, args: str) -> list[str] | None:
        base = ["git"]
        if op == "status":
            return base + ["status", "--short", "--branch"] + (args.split() if args else [])
        if op == "diff":
            return base + ["diff"] + (args.split() if args else ["HEAD"])
        if op == "add":
            files = args.split() if args else ["."]
            return base + ["add"] + files
        if op == "commit":
            # args should contain -m "message"
            import shlex
            try:
                return base + ["commit"] + shlex.split(args)
            except ValueError:
                return base + ["commit", "-m", args.strip('"\'') or "checkpoint"]
        if op == "log":
            flags = args.split() if args else ["--oneline", "-15"]
            return base + ["log"] + flags
        if op == "show":
            return base + ["show"] + (args.split() if args else ["HEAD"])
        if op == "branch":
            return base + ["branch"] + (args.split() if args else ["-v"])
        if op == "stash":
            return base + ["stash"] + (args.split() if args else ["list"])
        if op == "restore":
            files = args.split() if args else []
            if not files:
                return None
            return base + ["restore"] + files
        if op == "raw":
            import shlex
            try:
                return base + shlex.split(args)
            except ValueError:
                return None
        return None

    def _run(self, cmd: list[str], cwd: str) -> ToolResult:
        try:
            proc = subprocess.run(
                cmd, cwd=cwd, capture_output=True, text=True,
                timeout=15, encoding="utf-8", errors="replace",
            )
            out = (proc.stdout or "").strip()
            err = (proc.stderr or "").strip()

            if proc.returncode != 0:
                msg = f"git exited {proc.returncode}"
                if err:
                    msg += f"\n{err}"
                if out:
                    msg += f"\n{out}"
                return ToolResult(msg, is_error=True)

            return ToolResult(out or "(no output)")
        except FileNotFoundError:
            return ToolResult("git not found. Install git and ensure it is on PATH.", is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult("git command timed out after 15s.", is_error=True)
        except Exception as exc:
            return ToolResult(f"git error: {exc}", is_error=True)
