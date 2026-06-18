# -*- coding: utf-8 -*-
"""tools/worktree_tool.py — Git worktree management (MasterMind built-in)."""
from __future__ import annotations
import subprocess
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult


def _git(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    r = subprocess.run(["git"] + args, capture_output=True, text=True, cwd=cwd)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


class EnterWorktreeTool(BaseTool):
    name = "worktree_enter"
    description = (
        "Create and switch to a git worktree for isolated branch work. "
        "Creates a worktree at ../<repo>-worktree-<branch>."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "branch": {"type": "string", "description": "Branch name to create/checkout"},
            "new_branch": {"type": "boolean", "description": "Create new branch (default: false)"},
            "path": {"type": "string", "description": "Custom worktree path (optional)"},
        },
        "required": ["branch"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        branch = inp.get("branch", "")
        new_branch = inp.get("new_branch", False)
        custom_path = inp.get("path", "")

        if not branch:
            return ToolResult(output="branch is required", is_error=True)

        # Determine worktree path
        if custom_path:
            wt_path = Path(custom_path)
        else:
            code, repo_root, _ = _git(["rev-parse", "--show-toplevel"])
            if code != 0:
                return ToolResult(output="Not a git repository", is_error=True)
            repo_dir = Path(repo_root)
            wt_path = repo_dir.parent / f"{repo_dir.name}-wt-{branch.replace('/', '-')}"

        args = ["worktree", "add"]
        if new_branch:
            args += ["-b", branch]
        args += [str(wt_path), branch if not new_branch else "HEAD"]

        code, out, err = _git(args)
        if code != 0:
            return ToolResult(output=f"git worktree add failed:\n{err}", is_error=True)

        from hooks.manager import hook_manager
        hook_manager.fire("worktree_enter", branch=branch, path=str(wt_path))

        return ToolResult(output=f"Worktree created at: {wt_path}\nBranch: {branch}")


class ExitWorktreeTool(BaseTool):
    name = "worktree_exit"
    description = "Remove a git worktree after work is done."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Worktree path to remove"},
            "force": {"type": "boolean", "description": "Force removal even with changes (default: false)"},
        },
        "required": ["path"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        path = inp.get("path", "")
        force = inp.get("force", False)

        if not path:
            return ToolResult(output="path is required", is_error=True)

        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(path)

        code, out, err = _git(args)
        if code != 0:
            return ToolResult(output=f"git worktree remove failed:\n{err}", is_error=True)

        from hooks.manager import hook_manager
        hook_manager.fire("worktree_exit", path=path)

        return ToolResult(output=f"Worktree removed: {path}")


class WorktreeListTool(BaseTool):
    name = "worktree_list"
    description = "List all active git worktrees."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, inp: dict) -> ToolResult:
        code, out, err = _git(["worktree", "list", "--porcelain"])
        if code != 0:
            return ToolResult(output=f"git worktree list failed:\n{err}", is_error=True)
        return ToolResult(output=out or "(no worktrees)")
