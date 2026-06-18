"""
Skill: debug
Systematic debugging: hypothesis generation, root cause isolation, fix proposals.
Works hand-in-hand with the test_runner tool.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("skill.debug")

DESCRIPTION = (
    "Systematic software debugging: given an error message, traceback, or "
    "unexpected behaviour, generate ranked hypotheses, isolate the root cause, "
    "propose concrete fixes, and suggest regression tests. "
    "Works best when paired with the test_runner tool."
)


# ── pattern library ──────────────────────────────────────────────────────────

_PATTERNS: list[dict] = [
    {
        "match": r"ModuleNotFoundError|ImportError",
        "category": "Import / Dependency",
        "hypotheses": [
            "Package not installed in the active virtualenv / environment",
            "Wrong Python interpreter selected (check `which python`)",
            "Circular import — module A imports B which imports A",
            "Typo in the module name",
            "Package installed globally but script runs in a venv",
        ],
        "fixes": [
            "Run: `pip install <package>` (or `pip install -r requirements.txt`)",
            "Verify with: `python -c 'import <package>'`",
            "Check for circular imports by tracing the import chain",
            "Confirm `sys.path` includes the project root",
        ],
    },
    {
        "match": r"SyntaxError|IndentationError|TabError",
        "category": "Syntax / Indentation",
        "hypotheses": [
            "Mixing tabs and spaces",
            "Missing colon after if/for/def/class",
            "Unclosed bracket, parenthesis, or string literal",
            "Python 2 syntax used in Python 3 code",
            "f-string with nested quotes of the same type",
        ],
        "fixes": [
            "Open the file and check the exact line number in the traceback",
            "Run: `python -m py_compile <file>.py` to check syntax only",
            "Use an editor with syntax highlighting to spot the issue",
            "Run: `autopep8 --in-place <file>.py` to auto-fix indentation",
        ],
    },
    {
        "match": r"AttributeError",
        "category": "AttributeError",
        "hypotheses": [
            "Calling a method on None (object was never assigned)",
            "Typo in the attribute or method name",
            "Wrong object type returned by a previous call",
            "Accessing an attribute before __init__ sets it",
            "Library version change removed or renamed the attribute",
        ],
        "fixes": [
            "Add a None check before accessing the attribute",
            "Print `type(obj)` and `dir(obj)` to see what's actually there",
            "Check the library's changelog for renamed attributes",
            "Add defensive assertion: `assert obj is not None, 'obj is None'`",
        ],
    },
    {
        "match": r"TypeError",
        "category": "TypeError",
        "hypotheses": [
            "Wrong number of arguments passed to a function",
            "Passing a string where an int is expected (or vice versa)",
            "Calling a non-callable (e.g. calling a property as a method)",
            "NoneType passed where an object is required",
            "Unpacking a non-iterable",
        ],
        "fixes": [
            "Check the function signature vs the call site",
            "Add `print(type(arg))` before the failing call",
            "Explicitly cast: `int(x)`, `str(x)`, `list(x)`",
            "Check whether a function that should return a value is returning None",
        ],
    },
    {
        "match": r"KeyError",
        "category": "KeyError",
        "hypotheses": [
            "Key doesn't exist in the dict at that point in execution",
            "Typo in the key string",
            "Dict populated conditionally — key absent in some code paths",
            "API/JSON response structure changed",
        ],
        "fixes": [
            "Use `dict.get(key, default)` instead of `dict[key]`",
            "Add `if key in dict:` guard before access",
            "Print the dict to verify its contents at that line",
            "Use `dict.setdefault(key, default)` to ensure key exists",
        ],
    },
    {
        "match": r"IndexError",
        "category": "IndexError",
        "hypotheses": [
            "List/tuple is shorter than expected",
            "Off-by-one error in a loop index",
            "Empty list accessed with index 0",
            "Hardcoded index that's valid for test data but not production data",
        ],
        "fixes": [
            "Check `len(lst)` before indexing",
            "Use `lst[-1]` for last element safely only if `lst` is non-empty",
            "Replace fixed index with a loop or list comprehension",
            "Add assert: `assert len(lst) > idx, f'Expected >{idx} items, got {len(lst)}'`",
        ],
    },
    {
        "match": r"NameError|UnboundLocalError",
        "category": "Name / Scope",
        "hypotheses": [
            "Variable used before assignment in this scope",
            "Typo in variable name",
            "Variable defined inside an if-block that didn't execute",
            "Local variable shadows a global — modified locally without `global` keyword",
        ],
        "fixes": [
            "Assign a default value before the conditional block",
            "Declare `global x` or `nonlocal x` if modifying outer scope",
            "Check spelling against the definition site",
        ],
    },
    {
        "match": r"RecursionError|maximum recursion depth",
        "category": "Recursion",
        "hypotheses": [
            "Base case missing or unreachable",
            "Base case condition is wrong (off-by-one, wrong type comparison)",
            "Mutual recursion between two functions with no termination",
            "Data structure has a cycle (e.g. circular linked list)",
        ],
        "fixes": [
            "Draw the recursion tree for a small input and identify when it stops",
            "Add a depth counter / guard: `if depth > 1000: raise RuntimeError`",
            "Convert recursion to iteration with an explicit stack",
            "Use `functools.lru_cache` and verify the base case exits cleanly",
        ],
    },
    {
        "match": r"TimeoutError|timeout|timed out",
        "category": "Timeout / Performance",
        "hypotheses": [
            "Infinite loop — loop condition never becomes False",
            "Waiting on I/O that never completes (network, file, lock)",
            "Algorithm complexity is too high for the input size (O(n²) or worse)",
            "Deadlock between two threads",
        ],
        "fixes": [
            "Add print statements inside the loop to see iteration count",
            "Use cProfile: `python -m cProfile -s cumtime <file>.py`",
            "Set explicit timeouts on all I/O calls",
            "Visualise the loop condition and verify it changes each iteration",
        ],
    },
    {
        "match": r"ConnectionError|ConnectionRefusedError|requests\.exceptions|HTTPError",
        "category": "Network / HTTP",
        "hypotheses": [
            "Server is not running or not reachable at that host:port",
            "Wrong URL — typo, wrong port, http vs https",
            "Firewall or proxy blocking the connection",
            "Server returns non-200 status (check response.status_code)",
            "SSL certificate error",
        ],
        "fixes": [
            "Test connectivity: `curl -v <url>` from the same machine",
            "Add `response.raise_for_status()` and catch `requests.HTTPError`",
            "Check `REQUESTS_CA_BUNDLE` or pass `verify=False` for dev (not prod)",
            "Wrap in retry logic with exponential backoff",
        ],
    },
    {
        "match": r"PermissionError|FileNotFoundError|IsADirectoryError",
        "category": "File / Permission",
        "hypotheses": [
            "File path doesn't exist — check spelling and CWD",
            "Process doesn't have read/write permission on that path",
            "Directory missing — parent dirs not created",
            "File locked by another process",
        ],
        "fixes": [
            "Use `Path.exists()` before opening",
            "Use `Path.mkdir(parents=True, exist_ok=True)` to create dirs",
            "Check permissions: `ls -la <path>`",
            "Use `pathlib.Path` instead of string paths to avoid CWD confusion",
        ],
    },
]

_GENERIC: dict = {
    "category": "General / Unknown",
    "hypotheses": [
        "Silent data corruption earlier in the pipeline",
        "Race condition in multithreaded code",
        "Environment difference (dev vs prod, OS, Python version)",
        "Third-party library bug or version mismatch",
        "Incorrect assumption about an external API's response format",
    ],
    "fixes": [
        "Add logging/print at each step to trace data flow",
        "Bisect the code: comment out half, check if error persists",
        "Run with a minimal reproducible example",
        "Check git log for recent changes near the failing code",
        "Compare `pip freeze` between working and broken environments",
    ],
}


def _match_patterns(error_text: str) -> list[dict]:
    matched = []
    for p in _PATTERNS:
        if re.search(p["match"], error_text, re.IGNORECASE):
            matched.append(p)
    return matched if matched else [_GENERIC]


def _format_debug_report(
    problem: str,
    patterns: list[dict],
    context: str = "",
    depth: int = 3,
) -> str:
    sections = []

    sections.append("# 🔍 Debug Report\n")
    sections.append(f"**Problem:** {problem[:300]}\n")

    if context:
        sections.append(f"**Context provided:** {context[:500]}\n")

    # ── Error classification ──────────────────────────────────────────────────
    categories = [p["category"] for p in patterns]
    sections.append(f"**Error category:** {', '.join(categories)}\n")

    # ── Hypotheses ────────────────────────────────────────────────────────────
    sections.append("\n## 🧠 Ranked Hypotheses\n")
    hyp_num = 1
    for pat in patterns:
        for h in pat["hypotheses"][:depth]:
            sections.append(f"{hyp_num}. {h}")
            hyp_num += 1
    sections.append("")

    # ── Fixes ─────────────────────────────────────────────────────────────────
    sections.append("\n## 🔧 Suggested Fixes\n")
    for pat in patterns:
        if len(patterns) > 1:
            sections.append(f"**{pat['category']}:**")
        for f in pat["fixes"]:
            sections.append(f"  - {f}")
    sections.append("")

    # ── Debug steps ───────────────────────────────────────────────────────────
    sections.append("\n## 🪜 Step-by-Step Debug Plan\n")
    sections.append("1. **Reproduce** the error with the smallest possible input")
    sections.append("2. **Isolate** — comment out or stub code until error disappears, then re-add")
    sections.append("3. **Inspect** — add `print()` or use `pdb.set_trace()` just before the crash")
    sections.append("4. **Verify assumptions** — print types, lengths, and values of all inputs")
    sections.append("5. **Apply fix** — change one thing at a time")
    sections.append("6. **Confirm** — re-run the failing case, then run the full test suite")
    sections.append("7. **Add regression test** — use `test_runner` → `write_test` to prevent recurrence")
    sections.append("")

    # ── Test runner hint ──────────────────────────────────────────────────────
    sections.append("\n## 🧪 Suggested Test Approach\n")
    sections.append("Use the `test_runner` tool to verify your fix:")
    sections.append("```")
    sections.append('tool: test_runner')
    sections.append('action: debug_run')
    sections.append('path: <your_file.py>')
    sections.append("```")
    sections.append("Then write a regression test:")
    sections.append("```")
    sections.append('tool: test_runner')
    sections.append('action: write_test')
    sections.append('path: tests/test_<module>.py')
    sections.append('code: |')
    sections.append('  def test_<scenario>():')
    sections.append('      # reproduce the bug, assert the fix holds')
    sections.append("```")

    return "\n".join(sections)


from skills.base_skill import BaseSkill


class DebugSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "debug"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "Error message, traceback, or description of unexpected behaviour",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant code snippet or additional context (optional)",
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                    "description": "Number of hypotheses per category (default 3)",
                },
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        context = str(kwargs.get("context", ""))
        depth   = int(kwargs.get("depth", 3))
        patterns = _match_patterns(problem + " " + context)
        return _format_debug_report(problem, patterns, context, depth)
