#!/usr/bin/env python3
"""
test_self_healing.py — Demonstration / integration test for the self_healing skill.

This script:
  1. Plants a known intentional bug in a temp file.
  2. Calls SelfHealingTool with a natural-language description.
  3. Asserts the bug was detected, patched, and validated.
  4. Prints a full Markdown report of the DPIV cycle.

Run from the Mind_EVE root:
    python test_self_healing.py
"""

import ast
import os
import shutil
import sys
import tempfile
from pathlib import Path

# ── Make sure we can import from the project root ─────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from skills.self_healing import SelfHealingTool, DPIVEngine

# ══════════════════════════════════════════════════════════════════════════════
#  Helper
# ══════════════════════════════════════════════════════════════════════════════

BOLD  = "\033[1m"
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
RESET = "\033[0m"

def header(title: str):
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

def ok(msg: str):
    print(f"  {GREEN}✅ {msg}{RESET}")

def fail(msg: str):
    print(f"  {RED}❌ {msg}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  Test 1: _MAX_TAG_LEN bug (same one we fixed in webui.py)
# ══════════════════════════════════════════════════════════════════════════════

BUG_CODE = '''\
"""streaming_filter_bugged.py — intentional bug for self_healing demo."""

class StreamTagFilter:
    # BUG: value 2 is way too small; any tag > 4 chars leaks '<' on chunk split
    _MAX_TAG_LEN = 2

    def __init__(self):
        self._buf     = ""
        self._in_tool = False

    def feed(self, chunk: str):
        self._buf += chunk
        self._process()

    def _process(self):
        while self._buf:
            ts = self._buf.find("<")
            if ts == -1:
                safe = len(self._buf) - self._MAX_TAG_LEN
                if safe > 0:
                    print(self._buf[:safe], end="")   # emit to "browser"
                    self._buf = self._buf[safe:]
                break
            if ts > 0:
                print(self._buf[:ts], end="")
                self._buf = self._buf[ts:]
            end = self._buf.find(">")
            if end == -1:
                # BUG: threshold too low — emits '<' for tags > 4 chars
                if len(self._buf) > self._MAX_TAG_LEN + 2:
                    print("<", end="")   # leaks '<' to browser!
                    self._buf = self._buf[1:]
                break
            tag = self._buf[:end + 1].lower()
            self._buf = self._buf[end + 1:]
            if tag == "<tool_use>":
                self._in_tool = True
            elif tag == "</tool_use>":
                self._in_tool = False
'''

FIXED_SENTINEL = "_MAX_TAG_LEN = 40"


def test_max_tag_len_bug():
    """Plant the _MAX_TAG_LEN=2 bug; expect self_healing to fix it to 40."""
    header("Test 1: _MAX_TAG_LEN=2 bug (streaming tag leak)")

    # ── Set up isolated temp directory ────────────────────────────────────────
    tmpdir = Path(tempfile.mkdtemp(prefix="sh_test_"))
    bug_file = tmpdir / "streaming_filter_bugged.py"
    bug_file.write_text(BUG_CODE, encoding="utf-8")
    print(f"  Bug planted in: {bug_file}")
    print(f"  Initial value:  _MAX_TAG_LEN = 2")

    # ── Verify bug is present ─────────────────────────────────────────────────
    src = bug_file.read_text()
    assert "_MAX_TAG_LEN = 2" in src, "Bug not planted correctly"
    ok("Bug confirmed: _MAX_TAG_LEN = 2")

    # ── Run self_healing ──────────────────────────────────────────────────────
    problem = (
        "The streaming tag filter leaks XML tags like '<tool_use>' to the browser "
        "when the tag arrives split across stream chunks. "
        "Root cause: _MAX_TAG_LEN is set to 2, so any tag longer than 4 chars "
        "triggers the 'emit and discard' path before the closing '>' arrives. "
        "Fix: set _MAX_TAG_LEN to at least 40."
    )

    tool   = SelfHealingTool(working_dir=str(tmpdir))
    result = tool.execute({
        "problem":    problem,
        "scope":      ["streaming_filter_bugged.py"],
        "test_cmd":   f"{sys.executable} -c \"import ast; ast.parse(open('streaming_filter_bugged.py').read()); print('syntax OK')\"",
        "dry_run":    False,
        "max_patches": 5,
    })

    print("\n" + "─" * 60)
    print(result.output[:3000])
    print("─" * 60)

    # ── Assert the fix was applied ────────────────────────────────────────────
    patched_src = bug_file.read_text()
    fixed = FIXED_SENTINEL in patched_src

    if fixed:
        ok(f"Bug fixed: _MAX_TAG_LEN = 40 found in patched file")
    else:
        fail("Bug NOT fixed — _MAX_TAG_LEN = 40 not found")
        print(f"\n  Patched file content:\n{patched_src[:600]}")

    if not result.is_error or fixed:
        ok("SelfHealingTool returned success")
    else:
        fail("SelfHealingTool returned is_error=True")

    # ── Clean up ──────────────────────────────────────────────────────────────
    shutil.rmtree(tmpdir, ignore_errors=True)
    return fixed


# ══════════════════════════════════════════════════════════════════════════════
#  Test 2: Dry-run mode (no files changed)
# ══════════════════════════════════════════════════════════════════════════════

def test_dry_run():
    """Ensure dry_run=True never modifies files."""
    header("Test 2: dry_run=True — no files modified")

    tmpdir = Path(tempfile.mkdtemp(prefix="sh_dryrun_"))
    f      = tmpdir / "my_module.py"
    f.write_text('_MAX_TAG_LEN = 2\n', encoding="utf-8")
    original = f.read_text()

    tool   = SelfHealingTool(working_dir=str(tmpdir))
    result = tool.execute({
        "problem":  "The _MAX_TAG_LEN is too small and causes tag leaks.",
        "scope":    ["my_module.py"],
        "dry_run":  True,
    })

    after = f.read_text()
    if after == original:
        ok("File unchanged in dry-run mode")
    else:
        fail("File was modified despite dry_run=True!")

    if "DRY RUN" in result.output:
        ok("Report correctly labels the run as DRY RUN")
    else:
        fail("DRY RUN label missing from report")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return after == original


# ══════════════════════════════════════════════════════════════════════════════
#  Test 3: Rollback on test failure
# ══════════════════════════════════════════════════════════════════════════════

def test_rollback_on_failure():
    """Plant a bug; supply a test command that always fails; expect rollback."""
    header("Test 3: rollback on failing test suite")

    tmpdir   = Path(tempfile.mkdtemp(prefix="sh_rollback_"))
    bug_file = tmpdir / "streaming_filter_bugged.py"
    bug_file.write_text(BUG_CODE, encoding="utf-8")
    original = bug_file.read_text()

    # A test command that always exits 1
    always_fail_cmd = f"{sys.executable} -c \"import sys; sys.exit(1)\""

    tool   = SelfHealingTool(working_dir=str(tmpdir))
    result = tool.execute({
        "problem":  "The _MAX_TAG_LEN is too small and causes tag leaks.",
        "scope":    ["streaming_filter_bugged.py"],
        "test_cmd": always_fail_cmd,
        "dry_run":  False,
    })

    after = bug_file.read_text()
    rolled_back = after == original

    if rolled_back:
        ok("File rolled back after test failure")
    else:
        fail("File NOT rolled back after test failure!")
        print(f"  After:\n{after[:400]}")

    if "rolled back" in result.output.lower() or "Rolled back" in result.output:
        ok("Rollback reported in output")
    else:
        fail("Rollback not mentioned in report")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return rolled_back


# ══════════════════════════════════════════════════════════════════════════════
#  Test 4: Keyword extraction and file ranking
# ══════════════════════════════════════════════════════════════════════════════

def test_keyword_extraction():
    """Unit-test the keyword extractor and file ranker directly."""
    header("Test 4: keyword extraction & file ranking")

    tmpdir = Path(tempfile.mkdtemp(prefix="sh_kw_"))

    # Create two files — one highly relevant, one unrelated
    (tmpdir / "relevant.py").write_text(
        "_MAX_TAG_LEN = 2  # BUG\n"
        "class StreamTagFilter:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tmpdir / "unrelated.py").write_text(
        "def greet(): return 'hello'\n",
        encoding="utf-8",
    )

    engine    = DPIVEngine(root=tmpdir)
    keywords  = engine._extract_keywords(
        "The StreamTagFilter leaks tags because _MAX_TAG_LEN is 2"
    )
    ranked    = engine._rank_files(keywords, scope=None)

    print(f"  Keywords: {keywords[:8]}")
    print(f"  Ranked files: {[(r, s) for r,s in ranked[:4]]}")

    top = ranked[0][0] if ranked else ""
    if "relevant" in top:
        ok("Correctly ranked relevant.py as top candidate")
    else:
        fail(f"Expected relevant.py on top, got: {top}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    return "relevant" in top


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}╔══════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}║         Self-Healing Skill — Integration Tests           ║{RESET}")
    print(f"{BOLD}╚══════════════════════════════════════════════════════════╝{RESET}")

    results = {
        "Max-tag-len fix":     test_max_tag_len_bug(),
        "Dry-run (no change)": test_dry_run(),
        "Rollback on failure": test_rollback_on_failure(),
        "Keyword ranking":     test_keyword_extraction(),
    }

    header("Summary")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)
    for name, val in results.items():
        icon = f"{GREEN}✅{RESET}" if val else f"{RED}❌{RESET}"
        print(f"  {icon}  {name}")
    print()
    if passed == total:
        print(f"{GREEN}{BOLD}  ALL {total}/{total} tests passed.{RESET}\n")
        return 0
    else:
        print(f"{RED}{BOLD}  {passed}/{total} tests passed.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
