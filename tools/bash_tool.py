from __future__ import annotations
import os, subprocess, sys
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import BASH_TIMEOUT_S, WORKING_DIR

_IS_WIN = sys.platform == "win32"

class BashTool(BaseTool):
    name = "bash"
    description = (
        "Execute a shell command and return stdout+stderr. "
        "Use for running scripts, git, pip, file operations, etc. "
        "Prefer read_file/write_file/glob/grep for file tasks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command":    {"type": "string",  "description": "Shell command to run"},
            "timeout":    {"type": "number",  "description": "Timeout in seconds (default 120)"},
            "workdir":    {"type": "string",  "description": "Working directory override"},
        },
        "required": ["command"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        self._cwd = working_dir

    def _translate_ping(self, cmd: str) -> str:
        """Translate Linux ping flags to Windows on the fly."""
        if not _IS_WIN:
            return cmd
        # Only modify if it's a ping command
        if not cmd.strip().startswith("ping"):
            return cmd

        parts = cmd.split()
        # Find the flags
        new_parts = []
        i = 0
        count = None
        target = None
        while i < len(parts):
            p = parts[i]
            if p == "-c" and i+1 < len(parts):
                count = parts[i+1]
                i += 2
                continue
            elif p == "-t":
                # Infinite ping -> convert to 4 pings
                count = "4"
                i += 1
                continue
            elif p.startswith("-c"):
                # e.g., -c4
                count = p[2:]
                i += 1
                continue
            else:
                new_parts.append(p)
                i += 1

        # Reconstruct command
        if count:
            new_parts = ["ping", "-n", count] + [p for p in new_parts if p != "ping"]
        else:
            # No count specified, default to 4
            new_parts = ["ping", "-n", "4"] + [p for p in new_parts if p != "ping"]
        return " ".join(new_parts)

    def execute(self, inp: dict) -> ToolResult:
        # Accept common alternative key names that small models produce
        cmd = (
            inp.get("command") or inp.get("cmd") or
            inp.get("shell") or inp.get("script") or ""
        ).strip()
        timeout = float(inp.get("timeout", BASH_TIMEOUT_S))
        cwd = inp.get("workdir") or self._cwd

        if not cmd:
            return ToolResult("No command provided.", is_error=True)

        # Apply ping translation on Windows
        original_cmd = cmd
        if _IS_WIN and cmd.startswith("ping"):
            cmd = self._translate_ping(cmd)
            if cmd != original_cmd:
                # Optional: log translation (can be removed)
                pass

        # Safety: block obviously destructive commands without confirmation
        _DANGEROUS = ("rm -rf /", "mkfs", "dd if=", ":(){:|:&};")
        if any(d in cmd for d in _DANGEROUS):
            return ToolResult(f"Blocked dangerous command: {cmd[:80]}", is_error=True)

        shell = True
        executable = None
        if _IS_WIN:
            executable = "powershell.exe" if cmd.strip().startswith("ps:") else None
            cmd = cmd.removeprefix("ps:").strip() if executable else cmd

        try:
            proc = subprocess.run(
                cmd, shell=shell, executable=executable,
                capture_output=True, text=True,
                timeout=timeout, cwd=cwd,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            out = out[:8000]  # cap at 8k chars
            is_err = proc.returncode != 0
            if is_err and not out:
                out = f"Exit code {proc.returncode}"
            return ToolResult(out or "(no output)", is_error=is_err)
        except subprocess.TimeoutExpired:
            return ToolResult(f"Command timed out after {timeout}s", is_error=True)
        except Exception as e:
            return ToolResult(str(e), is_error=True)