"""
tools/test_runner_tool.py — Write, run, and debug Python programs and tests.

Actions:
  run_file     — Execute any Python file and capture stdout/stderr/exit code
  run_code     — Execute an ad-hoc Python snippet (written to a temp file)
  run_tests    — Run pytest on a file, directory, or the whole project
  write_test   — Write a pytest test file to disk
  debug_run    — Run a file with verbose tracebacks and return structured error info
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

from tools.base_tool import BaseTool, ToolResult


class TestRunnerTool(BaseTool):
    name = "test_runner"
    description = (
        "Write, run, and debug Python programs and test suites. "
        "Actions: run_file, run_code, run_tests, write_test, debug_run. "
        "Use run_code for quick snippets, run_tests to execute pytest, "
        "write_test to generate a test file, debug_run to get structured "
        "error info including traceback, locals, and suggested fixes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run_file", "run_code", "run_tests", "write_test", "debug_run"],
                "description": "What to do.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path (for run_file, run_tests, write_test, debug_run).",
            },
            "code": {
                "type": "string",
                "description": "Python source code (for run_code or write_test body).",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Extra CLI args passed to python / pytest.",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 60).",
            },
            "working_dir": {
                "type": "string",
                "description": "CWD for the subprocess. Defaults to tool's working_dir.",
            },
        },
        "required": ["action"],
    }

    def __init__(self, working_dir: str = "."):
        self._cwd = str(Path(working_dir).resolve())

    # ── internal ──────────────────────────────────────────────────────────────

    def _run(self, cmd: list[str], cwd: str, timeout: int) -> dict:
        """Run a subprocess and return structured result."""
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=cwd, timeout=timeout,
            )
            elapsed = time.perf_counter() - t0
            return {
                "exit_code": proc.returncode,
                "stdout":    proc.stdout,
                "stderr":    proc.stderr,
                "elapsed":   round(elapsed, 2),
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout":    "",
                "stderr":    f"Process timed out after {timeout}s",
                "elapsed":   timeout,
                "timed_out": True,
            }
        except Exception as exc:
            return {
                "exit_code": -1,
                "stdout":    "",
                "stderr":    str(exc),
                "elapsed":   time.perf_counter() - t0,
                "timed_out": False,
            }

    def _fmt(self, r: dict, label: str = "") -> str:
        """Format a run result into a readable string."""
        parts = []
        if label:
            parts.append(f"── {label} ──")
        parts.append(f"Exit code : {r['exit_code']}  ({r['elapsed']}s)")
        if r["stdout"].strip():
            parts.append("STDOUT:\n" + r["stdout"].rstrip())
        if r["stderr"].strip():
            parts.append("STDERR:\n" + r["stderr"].rstrip())
        if r["timed_out"]:
            parts.append("⚠️  TIMED OUT")
        status = "✅ PASSED" if r["exit_code"] == 0 else "❌ FAILED"
        parts.append(status)
        return "\n".join(parts)

    # ── actions ───────────────────────────────────────────────────────────────

    def _run_file(self, inp: dict) -> ToolResult:
        path    = inp.get("path", "")
        extra   = inp.get("args", [])
        timeout = int(inp.get("timeout", 60))
        cwd     = inp.get("working_dir", self._cwd)

        if not path:
            return ToolResult("'path' is required for run_file.", is_error=True)

        p = Path(path)
        if not p.is_absolute():
            p = Path(cwd) / p
        if not p.exists():
            return ToolResult(f"File not found: {p}", is_error=True)

        cmd = [sys.executable, str(p)] + extra
        r   = self._run(cmd, cwd, timeout)
        return ToolResult(self._fmt(r, f"python {p.name}"),
                          is_error=(r["exit_code"] != 0))

    def _run_code(self, inp: dict) -> ToolResult:
        code    = inp.get("code", "").strip()
        extra   = inp.get("args", [])
        timeout = int(inp.get("timeout", 60))
        cwd     = inp.get("working_dir", self._cwd)

        if not code:
            return ToolResult("'code' is required for run_code.", is_error=True)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=cwd, prefix="_mm_snippet_"
        ) as f:
            f.write(textwrap.dedent(code))
            tmp = f.name

        try:
            cmd = [sys.executable, tmp] + extra
            r   = self._run(cmd, cwd, timeout)
            return ToolResult(self._fmt(r, "snippet"),
                              is_error=(r["exit_code"] != 0))
        finally:
            os.unlink(tmp)

    def _run_tests(self, inp: dict) -> ToolResult:
        path    = inp.get("path", ".")
        extra   = inp.get("args", [])
        timeout = int(inp.get("timeout", 120))
        cwd     = inp.get("working_dir", self._cwd)

        p = Path(path)
        if not p.is_absolute():
            p = Path(cwd) / p

        # build pytest command; -v for verbose, --tb=short for readable tracebacks
        cmd = [sys.executable, "-m", "pytest", str(p), "-v", "--tb=short",
               "--no-header"] + extra
        r   = self._run(cmd, cwd, timeout)

        output = self._fmt(r, f"pytest {p}")

        # parse summary line from pytest output
        for line in reversed(r["stdout"].splitlines()):
            if "passed" in line or "failed" in line or "error" in line:
                output = f"Summary: {line.strip()}\n\n" + output
                break

        return ToolResult(output, is_error=(r["exit_code"] != 0))

    def _write_test(self, inp: dict) -> ToolResult:
        path = inp.get("path", "")
        code = inp.get("code", "").strip()
        cwd  = inp.get("working_dir", self._cwd)

        if not path:
            return ToolResult("'path' is required for write_test.", is_error=True)
        if not code:
            return ToolResult("'code' is required for write_test.", is_error=True)

        p = Path(path)
        if not p.is_absolute():
            p = Path(cwd) / p

        p.parent.mkdir(parents=True, exist_ok=True)

        # Ensure it starts with a valid pytest header if not already there
        if "import pytest" not in code and "def test_" not in code[:200]:
            header = "import pytest\n\n"
        else:
            header = ""

        p.write_text(header + textwrap.dedent(code), encoding="utf-8")
        return ToolResult(f"✅ Test file written: {p}\n\nTo run it:\n  pytest {p} -v")

    def _debug_run(self, inp: dict) -> ToolResult:
        path    = inp.get("path", "")
        code    = inp.get("code", "")
        timeout = int(inp.get("timeout", 60))
        cwd     = inp.get("working_dir", self._cwd)

        # wrap the target in a debug harness that prints locals on exception
        harness = textwrap.dedent("""\
            import sys, traceback, pprint

            def _debug_exec(target_path):
                import importlib.util
                spec = importlib.util.spec_from_file_location("_target", target_path)
                mod  = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(mod)
                    print("\\n✅ Program completed successfully.")
                except Exception as exc:
                    tb = sys.exc_info()[2]
                    # Walk to innermost frame
                    while tb.tb_next:
                        tb = tb.tb_next
                    local_vars = tb.tb_frame.f_locals
                    print("\\n❌ EXCEPTION:", type(exc).__name__, str(exc))
                    print("\\nFULL TRACEBACK:")
                    traceback.print_exc()
                    print("\\nLOCALS AT CRASH SITE:")
                    pprint.pprint(local_vars)
                    sys.exit(1)

            _debug_exec(sys.argv[1])
        """)

        # write harness to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir=cwd, prefix="_mm_debug_harness_"
        ) as hf:
            hf.write(harness)
            harness_path = hf.name

        try:
            if code:
                # write code snippet to temp file
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".py", delete=False,
                    dir=cwd, prefix="_mm_debug_target_"
                ) as tf:
                    tf.write(textwrap.dedent(code))
                    target_path = tf.name
                cleanup_target = True
            elif path:
                p = Path(path)
                if not p.is_absolute():
                    p = Path(cwd) / p
                if not p.exists():
                    return ToolResult(f"File not found: {p}", is_error=True)
                target_path    = str(p)
                cleanup_target = False
            else:
                return ToolResult("'path' or 'code' is required for debug_run.", is_error=True)

            cmd = [sys.executable, harness_path, target_path]
            r   = self._run(cmd, cwd, timeout)

            output_lines = ["── debug_run ──"]
            output_lines.append(self._fmt(r))

            if r["exit_code"] != 0 and r["stderr"].strip():
                output_lines.append("\n🔍 DIAGNOSIS:")
                # identify common error patterns
                err = r["stderr"] + r["stdout"]
                if "ModuleNotFoundError" in err or "ImportError" in err:
                    output_lines.append("  • Missing dependency — check imports or run: pip install <pkg>")
                if "IndentationError" in err or "SyntaxError" in err:
                    output_lines.append("  • Syntax/indentation error — check the line number in the traceback")
                if "AttributeError" in err:
                    output_lines.append("  • AttributeError — object doesn't have that method/property")
                if "TypeError" in err:
                    output_lines.append("  • TypeError — wrong argument type or count")
                if "KeyError" in err or "IndexError" in err:
                    output_lines.append("  • Key/Index error — check dict keys and list bounds")
                if "NameError" in err:
                    output_lines.append("  • NameError — variable used before assignment")
                if "FileNotFoundError" in err:
                    output_lines.append("  • FileNotFoundError — check that the path exists")

            return ToolResult("\n".join(output_lines),
                              is_error=(r["exit_code"] != 0))

        finally:
            os.unlink(harness_path)
            if "cleanup_target" in dir() and cleanup_target:
                os.unlink(target_path)

    # ── dispatch ──────────────────────────────────────────────────────────────

    def execute(self, inp: dict) -> ToolResult:
        action = inp.get("action", "").strip()
        dispatch = {
            "run_file":  self._run_file,
            "run_code":  self._run_code,
            "run_tests": self._run_tests,
            "write_test": self._write_test,
            "debug_run": self._debug_run,
        }
        fn = dispatch.get(action)
        if fn is None:
            return ToolResult(
                f"Unknown action '{action}'. "
                f"Valid: {', '.join(dispatch)}",
                is_error=True,
            )
        return fn(inp)
