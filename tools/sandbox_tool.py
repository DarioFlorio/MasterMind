# -*- coding: utf-8 -*-
"""
tools/sandbox_tool.py — Sandboxed code execution.

Runs code in an isolated subprocess with resource limits.
On Linux: uses bubblewrap (bwrap) if available, falls back to subprocess.
On other platforms: subprocess with timeout only.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult


class SandboxTool(BaseTool):
    name = "sandbox"
    description = (
        "Execute code in a sandboxed environment with resource limits. "
        "Supports Python, JavaScript (node), Bash, Ruby."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "language": {"type": "string", "description": "python | javascript | bash | ruby"},
            "code": {"type": "string", "description": "Code to execute"},
            "timeout": {"type": "integer", "description": "Timeout seconds (default: 30, max: 120)"},
            "stdin": {"type": "string", "description": "Optional stdin input"},
        },
        "required": ["language", "code"],
    }

    _RUNNERS = {
        "python": [sys.executable, "-c"],
        "javascript": ["node", "-e"],
        "bash": ["bash", "-c"],
        "sh": ["sh", "-c"],
        "ruby": ["ruby", "-e"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        lang = inp.get("language", "python").lower()
        code = inp.get("code", "")
        timeout = min(int(inp.get("timeout", 30)), 120)
        stdin_data = inp.get("stdin", None)

        if not code:
            return ToolResult(output="code is required", is_error=True)

        runner = self._RUNNERS.get(lang)
        if not runner:
            return ToolResult(
                output=f"Unsupported language: {lang!r}. Use: {', '.join(self._RUNNERS)}",
                is_error=True
            )

        # Check runner is available
        if not shutil.which(runner[0]):
            return ToolResult(
                output=f"Runtime not found: {runner[0]}. Install {lang} to use this.",
                is_error=True
            )

        try:
            cmd = runner + [code]

            # Try bubblewrap sandbox on Linux
            if sys.platform == "linux" and shutil.which("bwrap"):
                cmd = self._bwrap_wrap(cmd)

            result = subprocess.run(
                cmd,
                capture_output=True, text=True,
                timeout=timeout,
                input=stdin_data,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr[:500]}"
            output = output[:5000]  # cap output
            is_error = result.returncode != 0
            if is_error:
                output = f"[exit {result.returncode}]\n{output}"
            return ToolResult(output=output or "(no output)", is_error=is_error)

        except subprocess.TimeoutExpired:
            return ToolResult(output=f"Sandbox timed out after {timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Sandbox error: {e}", is_error=True)

    @staticmethod
    def _bwrap_wrap(cmd: list[str]) -> list[str]:
        """Wrap command in bubblewrap for sandboxing."""
        return [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/bin", "/bin",
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/tmp",
            "--unshare-all",
            "--die-with-parent",
        ] + cmd
