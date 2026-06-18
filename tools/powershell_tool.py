"""
tools/powershell_tool.py — PowerShell execution tool (Windows-native preferred).

GAP IMPLEMENTED: PowerShell native support — Windows tasks need workarounds.
EVE now treats PowerShell as a first-class tool with:
  - UTF-8 output encoding (fixes garbled Windows output)
  - Script file execution (not just one-liners)
  - Stdin piping support
  - Windows-native command preference detection
  - Better error messages with PowerShell-specific hints
  - Auto-detection of pwsh vs powershell.exe
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult
from config.settings import BASH_TIMEOUT_S

# Default timeout for PowerShell (shorter than bash for safety)
_PS_DEFAULT_TIMEOUT = min(60, BASH_TIMEOUT_S)

# Common Windows commands that SHOULD go through PowerShell
_WINDOWS_NATIVE_CMDS = {
    "Get-", "Set-", "New-", "Remove-", "Test-", "Out-", "Select-",
    "Where-", "ForEach-", "Write-", "Read-", "Invoke-", "Start-",
    "Stop-", "Restart-", "Enable-", "Disable-", "Install-", "Uninstall-",
    "reg ", "regsvr32", "sc ", "schtasks", "netsh", "ipconfig", "systeminfo",
    "wmic", "bcdedit", "diskpart", "certutil", "cipher",
}


class PowerShellTool(BaseTool):
    name = "powershell"
    description = (
        "Execute PowerShell commands natively on Windows. "
        "Preferred for: file system ops, registry, services, scheduled tasks, "
        "system info, and all Windows-native operations. "
        "Supports script files, stdin piping, and full UTF-8 output."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "PowerShell command, script block, or multi-line script",
            },
            "script_path": {
                "type": "string",
                "description": "Path to a .ps1 script file to execute (alternative to command)",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default: {_PS_DEFAULT_TIMEOUT})",
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory (optional)",
            },
            "stdin_input": {
                "type": "string",
                "description": "Text to pipe into the command via stdin (optional)",
            },
            "no_profile": {
                "type": "boolean",
                "description": "Skip loading PowerShell profile (default: true, faster)",
            },
        },
    }

    def __init__(self) -> None:
        self._pwsh     = self._find_pwsh()
        self._is_win   = sys.platform == "win32"
        self._ps_ver   = self._detect_version()

    # ── Public ────────────────────────────────────────────────────────────────

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)

        command     = inp.get("command", "").strip()
        script_path = inp.get("script_path", "").strip()
        timeout     = int(inp.get("timeout", _PS_DEFAULT_TIMEOUT))
        cwd         = inp.get("working_dir") or None
        stdin_input = inp.get("stdin_input", "")
        no_profile  = bool(inp.get("no_profile", True))

        if not command and not script_path:
            return ToolResult(output="Either 'command' or 'script_path' is required", is_error=True)

        # ── Auto-detect bash heredoc syntax (<<'PY' / <<EOF) ──────────────
        # PowerShell doesn't support bash heredocs. Convert: extract the
        # script body and write it to a temp file, then execute that file.
        if command and ("<<" in command):
            m = re.search(
                r"python\s+-\s+<<['\"]?(\w+)['\"]?\s*\n(.*?)\n\1",
                command, re.DOTALL
            )
            if m:
                script_body = m.group(2)
                return self.write_and_run_script(script_body, timeout=timeout)
            # Generic heredoc: try to extract anything between << markers
            m2 = re.search(r"<<['\"]?\w+['\"]?\s*\n(.*)", command, re.DOTALL)
            if m2:
                return ToolResult(
                    output=(
                        "[PowerShell] bash heredoc syntax (<<) is not supported in PowerShell.\n"
                        "Use the 'script_path' parameter or write the script to a temp file first.\n\n"
                        "Example:\n"
                        "  write_file: {\"path\": \"temp/script.py\", \"content\": \"...\"}\n"
                        "  powershell: {\"command\": \"python temp/script.py\"}"
                    ),
                    is_error=True,
                )

        if not self._pwsh:
            return ToolResult(
                output=(
                    "PowerShell not found on this system.\n"
                    "• Windows: should be available as powershell.exe\n"
                    "• Install PowerShell 7+: https://aka.ms/install-powershell\n"
                    "• Linux/macOS: sudo snap install powershell"
                ),
                is_error=True,
            )

        # Build the argument list
        args = [self._pwsh]
        if no_profile:
            args += ["-NoProfile", "-NonInteractive"]

        # UTF-8 output — critical for correct Windows output
        # Sets both console and output encoding to UTF-8
        encoding_setup = (
            "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
            "$OutputEncoding = [System.Text.Encoding]::UTF8; "
        )

        if script_path:
            # Execute a .ps1 file
            p = Path(script_path)
            if not p.exists():
                return ToolResult(
                    output=f"Script not found: {script_path}", is_error=True
                )
            args += ["-ExecutionPolicy", "Bypass", "-File", str(p)]
        else:
            # Execute inline command with encoding setup prepended
            full_cmd = encoding_setup + command
            args += ["-Command", full_cmd]

        try:
            result = subprocess.run(
                args,
                input=stdin_input if stdin_input else None,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=cwd,
                env={**os.environ, "PYTHONUTF8": "1"},
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            exit_code = result.returncode

            # Build output
            output_parts: list[str] = []
            if stdout:
                output_parts.append(stdout)
            if stderr:
                # Filter PowerShell version banner noise
                stderr_lines = [
                    ln for ln in stderr.splitlines()
                    if not ln.startswith("Copyright") and "PowerShell" not in ln[:15]
                ]
                if stderr_lines:
                    output_parts.append("[stderr]\n" + "\n".join(stderr_lines))

            output = "\n".join(output_parts) if output_parts else "(no output)"

            if exit_code != 0:
                hint = self._error_hint(output, command)
                if hint:
                    output += f"\n\n[PowerShell hint: {hint}]"
                return ToolResult(output=f"[exit {exit_code}]\n{output}", is_error=True)

            return ToolResult(output=output)

        except subprocess.TimeoutExpired:
            return ToolResult(
                output=f"PowerShell timed out after {timeout}s. "
                       "Use a shorter command or increase the timeout parameter.",
                is_error=True,
            )
        except FileNotFoundError:
            return ToolResult(
                output=f"PowerShell binary not found: {self._pwsh}",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(output=f"PowerShell execution error: {exc}", is_error=True)

    def write_and_run_script(self, script_content: str, timeout: int = _PS_DEFAULT_TIMEOUT) -> ToolResult:
        """
        Write content to a temp .ps1 file and execute it.
        Useful for multi-line scripts that are complex as inline commands.
        """
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".ps1", mode="w", encoding="utf-8", delete=False
            ) as f:
                f.write(script_content)
                tmp_path = f.name
            return self.execute({"script_path": tmp_path, "timeout": timeout})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # ── Detection helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _find_pwsh() -> str | None:
        """Find the best available PowerShell binary."""
        # Prefer pwsh (PowerShell 7+) over powershell.exe (5.x)
        for name in ("pwsh", "pwsh.exe", "powershell", "powershell.exe"):
            found = shutil.which(name)
            if found:
                return found
        # Windows fallback: common install paths
        if sys.platform == "win32":
            candidates = [
                r"C:\Program Files\PowerShell\7\pwsh.exe",
                r"C:\Program Files\PowerShell\7-preview\pwsh.exe",
                r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            ]
            for c in candidates:
                if Path(c).exists():
                    return c
        return None

    def _detect_version(self) -> str:
        """Detect PowerShell version string."""
        if not self._pwsh:
            return "not available"
        try:
            result = subprocess.run(
                [self._pwsh, "-NoProfile", "-NonInteractive", "-Command",
                 "$PSVersionTable.PSVersion.ToString()"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() or "unknown"
        except Exception:
            return "unknown"

    @staticmethod
    def _error_hint(output: str, command: str) -> str:
        """Return a Windows-specific hint for common PowerShell errors."""
        out_l = output.lower()
        if "is not recognized" in out_l or "the term" in out_l:
            return (
                "Command not recognized — check PowerShell syntax. "
                "Use Get-Command to verify, or try bash tool for Unix commands."
            )
        if "access is denied" in out_l or "unauthorized" in out_l:
            return "Access denied — may require elevated (Admin) PowerShell."
        if "cannot find path" in out_l or "does not exist" in out_l:
            return "Path not found — verify with Test-Path before using it."
        if "execution policy" in out_l:
            return "Add -ExecutionPolicy Bypass to allow script execution."
        if "parameter" in out_l and ("invalid" in out_l or "unknown" in out_l):
            return "Invalid parameter — run Get-Help <cmdlet> -Full to see valid parameters."
        return ""

    @property
    def available(self) -> bool:
        return self._pwsh is not None

    @property
    def version(self) -> str:
        return self._ps_ver
