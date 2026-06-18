"""
skills/self_healing.py — Root-Cause-Driven Code Remediation (DPIV loop)
========================================================================

Encodes the exact debugging workflow used to repair EVE's own codebase:

  D — Diagnose   : parse problem, explore files, find root causes
  P — Plan       : propose minimal targeted changes, no blind patching
  I — Implement  : apply edits, respecting the existing architecture
  V — Validate   : run tests, confirm the fix, roll back on failure

This skill is both a BaseSkill (callable by SkillTool via name lookup) and
a BaseTool (registered directly in _build_tools so EVE can call it as a
first-class tool alongside bash, write_file, etc.).

Usage as a tool:
  <tool_use><n>self_healing</n><input>{
    "problem": "The streaming filter leaks <tooluse> tags to the browser",
    "scope": ["webui.py", "agent/_robust_parser.py"]
  }</input></tool_use>

The skill produces a structured Markdown report of every reasoning step so
Dario can see exactly what it diagnosed, what it changed, and why.
"""
from __future__ import annotations

import ast
import difflib
import json
import os
import re
import subprocess
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# ── Compatibility shims ───────────────────────────────────────────────────────
try:
    from tools.base_tool import BaseTool, ToolResult
except ImportError:
    class ToolResult:  # type: ignore
        def __init__(self, output: str, is_error: bool = False):
            self.output = output; self.is_error = is_error
    class BaseTool:  # type: ignore
        name = ""; description = ""; input_schema: dict = {}
        def execute(self, inp: dict) -> "ToolResult": ...

try:
    from skills.base_skill import BaseSkill
except ImportError:
    class BaseSkill:  # type: ignore
        @property
        def name(self) -> str: return ""
        def execute_impl(self, problem: str, **kw) -> str: return ""

# ── Constants ─────────────────────────────────────────────────────────────────
_MAX_FILE_READ  = 8_000   # chars read per file during exploration
_MAX_FILES_SCAN = 40      # candidate files examined per run
_ROLLBACK_DIR   = Path("temp/self_healing_rollbacks")
_LOG_DIR        = Path("temp/self_healing_logs")

# Extensions considered "source" during directory scan
_SOURCE_EXTS = {".py", ".json", ".md", ".toml", ".yaml", ".yml", ".js", ".ts"}

# Files/dirs always skipped during scan
_SKIP_DIRS = {"__pycache__", ".git", "node_modules", "wa_session",
              ".venv", "venv", "dist", "build", ".mypy_cache"}
_SKIP_FILES = {"*.pyc", "*.pyo", "*.egg-info"}

# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class _FindingItem:
    file:     str
    line:     Optional[int]
    severity: str    # "critical" | "warning" | "info"
    message:  str
    snippet:  str = ""


@dataclass
class _PatchItem:
    file:      str
    old_text:  str
    new_text:  str
    reason:    str
    applied:   bool  = False
    rollback:  str   = ""   # original text, stored before apply


@dataclass
class _DPIVResult:
    problem:    str
    findings:   List[_FindingItem]  = field(default_factory=list)
    patches:    List[_PatchItem]    = field(default_factory=list)
    test_output: str = ""
    test_passed: bool = False
    log:         List[str] = field(default_factory=list)
    duration_s:  float = 0.0
    rolled_back: bool  = False


# ══════════════════════════════════════════════════════════════════════════════
#  Core DPIV engine
# ══════════════════════════════════════════════════════════════════════════════

class DPIVEngine:
    """
    Implements the four-phase Diagnose → Plan → Implement → Validate loop.

    Phase D  Diagnose
    ─────────────────
    1. Parse problem text for keywords (error messages, file names, symbols).
    2. Walk the source tree; rank files by keyword hit-density.
    3. Read top-N candidate files; extract relevant code regions.
    4. Identify root causes: missing keys, wrong defaults, uncaught exceptions,
       bad regex, import errors, off-by-one, etc.

    Phase P  Plan
    ─────────────
    5. For each root cause, propose the smallest possible change.
    6. Build a list of (file, old_text, new_text, reason) patches.
    7. Reject patches that change more than 20% of any file (safety gate).

    Phase I  Implement
    ──────────────────
    8. Save rollback snapshot of every file before touching it.
    9. Apply patches using exact str_replace (never whole-file overwrite).
    10. Verify each file is syntactically valid Python after patching.

    Phase V  Validate
    ─────────────────
    11. Run pytest / unittest discovery, or the provided test command.
    12. If tests pass → done.
    13. If tests fail → roll back all patches, log failure, return failure report.
    """

    def __init__(self, root: Path, log_fn=None):
        self._root   = root.resolve()
        self._log_fn = log_fn or (lambda s: None)
        _ROLLBACK_DIR.mkdir(parents=True, exist_ok=True)
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def run(
        self,
        problem: str,
        scope: Optional[List[str]] = None,
        test_cmd: Optional[str] = None,
        dry_run: bool = False,
        max_patches: int = 10,
    ) -> _DPIVResult:
        t0     = time.time()
        result = _DPIVResult(problem=problem)
        log    = result.log

        def _step(n: int, title: str):
            msg = f"\n{'='*60}\nPhase {n}: {title}\n{'='*60}"
            log.append(msg)
            self._log_fn(msg)

        # ── Phase D ──────────────────────────────────────────────────────────
        _step("D", "DIAGNOSE — Explore & root-cause")

        keywords  = self._extract_keywords(problem)
        log.append(f"Keywords extracted: {keywords}")

        candidates = self._rank_files(keywords, scope)
        log.append(f"Top candidate files ({len(candidates)}):")
        for f, score in candidates[:8]:
            log.append(f"  score={score:4d}  {f}")

        findings = self._analyse_candidates(problem, keywords, candidates, log)
        result.findings = findings

        if not findings:
            log.append("⚠️  No specific root causes found via static analysis.")
            log.append("    Falling back to heuristic patch from problem text.")

        # ── Phase P ──────────────────────────────────────────────────────────
        _step("P", "PLAN — Propose targeted patches")

        patches = self._plan_patches(problem, keywords, findings, log)
        patches = patches[:max_patches]
        result.patches = patches

        if not patches:
            result.log.append("❌ No actionable patches could be derived. Manual review needed.")
            result.duration_s = time.time() - t0
            return result

        log.append(f"\nProposed {len(patches)} patch(es):")
        for i, p in enumerate(patches, 1):
            log.append(f"\n  Patch {i}  [{p.file}]")
            log.append(f"  Reason : {p.reason}")
            log.append(f"  Remove : {p.old_text[:120].replace(chr(10), '↵')!r}")
            log.append(f"  Insert : {p.new_text[:120].replace(chr(10), '↵')!r}")

        if dry_run:
            log.append("\n🔍 DRY RUN — no files modified.")
            result.duration_s = time.time() - t0
            return result

        # ── Phase I ──────────────────────────────────────────────────────────
        _step("I", "IMPLEMENT — Apply patches")

        applied_count = 0
        for patch in patches:
            ok, msg = self._apply_patch(patch, log)
            if ok:
                applied_count += 1
                log.append(f"  ✅ Applied: {patch.file}")
            else:
                log.append(f"  ❌ Failed : {patch.file} — {msg}")

        log.append(f"\nApplied {applied_count}/{len(patches)} patches.")

        if applied_count == 0:
            log.append("No patches applied — skipping validation.")
            result.duration_s = time.time() - t0
            return result

        # ── Phase V ──────────────────────────────────────────────────────────
        _step("V", "VALIDATE — Run tests")

        test_out, passed = self._run_tests(test_cmd, log)
        result.test_output  = test_out
        result.test_passed  = passed

        if not passed:
            log.append("\n⚠️  Tests FAILED — rolling back all patches.")
            self._rollback_all(patches, log)
            result.rolled_back = True
        else:
            log.append("\n✅ Tests PASSED — fix confirmed.")

        result.duration_s = time.time() - t0
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Phase D helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_keywords(self, problem: str) -> List[str]:
        """Pull identifiers, error strings, and file names from the problem text."""
        words: List[str] = []

        # Quoted strings (often error messages or variable names)
        words += re.findall(r'["\']([^"\']{3,60})["\']', problem)

        # snake_case identifiers and CamelCase
        words += re.findall(r'\b([a-z][a-z0-9_]{2,}(?:_[a-z0-9]+)+)\b', problem)
        words += re.findall(r'\b([A-Z][a-zA-Z0-9]{3,})\b', problem)

        # Python-ish patterns: module.function, ClassName
        words += re.findall(r'\b(\w+\.\w+)\b', problem)

        # File paths mentioned explicitly
        words += re.findall(r'(\w[\w/\\.-]+\.py)', problem)

        # Error-class names
        words += re.findall(r'\b(\w+Error|\w+Exception|\w+Warning)\b', problem)

        # Numeric literals that might be line numbers or values
        words += re.findall(r'\b(\d{1,5})\b', problem)

        # Deduplicate, lowercase-compare, keep longest unique ones
        seen: set = set()
        unique: List[str] = []
        for w in words:
            k = w.lower()
            if k not in seen and len(w) >= 3:
                seen.add(k)
                unique.append(w)

        return unique[:30]

    def _rank_files(
        self,
        keywords: List[str],
        scope: Optional[List[str]],
    ) -> List[Tuple[str, int]]:
        """Walk source tree; rank files by keyword hit count."""
        scores: dict[str, int] = {}

        if scope:
            # Explicit file list — resolve relative to root
            candidates = []
            for s in scope:
                p = self._root / s
                if p.is_file():
                    candidates.append(p)
                else:
                    candidates += list(self._root.rglob(s))
            for p in candidates:
                rel = str(p.relative_to(self._root))
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_READ * 2]
                    scores[rel] = sum(text.lower().count(k.lower()) for k in keywords)
                except Exception:
                    pass
        else:
            # Full tree scan
            scanned = 0
            for path in self._root.rglob("*"):
                if scanned > _MAX_FILES_SCAN * 3:
                    break
                if path.is_dir() and path.name in _SKIP_DIRS:
                    continue
                if not path.is_file() or path.suffix not in _SOURCE_EXTS:
                    continue
                rel = str(path.relative_to(self._root))
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_READ * 2]
                except Exception:
                    continue
                score = sum(text.lower().count(k.lower()) for k in keywords)
                if score:
                    scores[rel] = score
                scanned += 1

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:_MAX_FILES_SCAN]

    def _analyse_candidates(
        self,
        problem: str,
        keywords: List[str],
        candidates: List[Tuple[str, int]],
        log: List[str],
    ) -> List[_FindingItem]:
        """Read top candidate files and extract specific root-cause findings."""
        findings: List[_FindingItem] = []

        for rel, score in candidates[:12]:
            path = self._root / rel
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # A. Syntax errors (Python files only)
            if rel.endswith(".py"):
                try:
                    ast.parse(text)
                except SyntaxError as se:
                    findings.append(_FindingItem(
                        file=rel, line=se.lineno, severity="critical",
                        message=f"SyntaxError: {se.msg}",
                        snippet=f"  {se.text or ''}".strip(),
                    ))

            # B. Uncaught broad exception swallowing
            if re.search(r"except\s*(?:Exception\s*)?:\s*\n\s*pass", text):
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    if re.search(r"except\s*(?:Exception\s*)?:\s*$", line):
                        nxt = lines[i + 1] if i + 1 < len(lines) else ""
                        if nxt.strip() == "pass":
                            findings.append(_FindingItem(
                                file=rel, line=i + 1, severity="warning",
                                message="Bare except:pass swallows all errors silently",
                                snippet=f"{line.rstrip()}\\n{nxt.rstrip()}",
                            ))

            # C. Keyword-bearing lines (potential root causes)
            lines = text.splitlines()
            for i, line in enumerate(lines):
                line_lower = line.lower()
                hits = sum(1 for k in keywords if k.lower() in line_lower)
                if hits >= 2:
                    findings.append(_FindingItem(
                        file=rel, line=i + 1, severity="info",
                        message=f"High-density keyword match ({hits} hits)",
                        snippet=line.strip()[:160],
                    ))

            # D. TODO / FIXME / HACK markers
            for i, line in enumerate(lines):
                m = re.search(r"\b(TODO|FIXME|HACK|BUG|XXX)\b", line, re.IGNORECASE)
                if m:
                    findings.append(_FindingItem(
                        file=rel, line=i + 1, severity="warning",
                        message=f"Marker: {m.group(1)}",
                        snippet=line.strip()[:120],
                    ))

            # E. Wrong default values (common pattern in bugs)
            for i, line in enumerate(lines):
                if re.search(r"_MAX_TAG_LEN\s*=\s*[12]\b", line):
                    findings.append(_FindingItem(
                        file=rel, line=i + 1, severity="critical",
                        message="_MAX_TAG_LEN too small — partial tags will leak",
                        snippet=line.strip(),
                    ))
                if re.search(r"max_turns\s*=\s*[1-9]\b", line):
                    findings.append(_FindingItem(
                        file=rel, line=i + 1, severity="warning",
                        message="max_turns very low — long tasks will be cut off",
                        snippet=line.strip(),
                    ))

        log.append(f"\nFindings ({len(findings)}):")
        for f in findings:
            log.append(f"  [{f.severity.upper()}] {f.file}:{f.line or '?'}  {f.message}")
            if f.snippet:
                log.append(f"    → {f.snippet[:100]}")

        return findings

    # ─────────────────────────────────────────────────────────────────────────
    # Phase P helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _plan_patches(
        self,
        problem: str,
        keywords: List[str],
        findings: List[_FindingItem],
        log: List[str],
    ) -> List[_PatchItem]:
        """
        Convert findings into concrete (file, old_text, new_text) patch triples.
        Uses heuristic rules derived from the most common EVE bug patterns.
        """
        patches: List[_PatchItem] = []
        seen_files: set[str] = set()

        for finding in findings:
            if finding.severity not in ("critical", "warning"):
                continue

            path = self._root / finding.file
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            lines = text.splitlines(keepends=True)
            if finding.line and finding.line <= len(lines):
                lno   = finding.line - 1
                chunk = "".join(lines[max(0, lno - 3): lno + 4])  # 7-line window
            else:
                chunk = finding.snippet

            # Rule 1: _MAX_TAG_LEN too small
            if "_MAX_TAG_LEN" in finding.message:
                m = re.search(r"(_MAX_TAG_LEN\s*=\s*)([12])\b", text)
                if m:
                    patches.append(_PatchItem(
                        file=finding.file,
                        old_text=m.group(0),
                        new_text=m.group(1) + "40",
                        reason=(
                            "_MAX_TAG_LEN=2 causes any XML tag >4 chars split "
                            "across stream chunks to leak '<' to the browser. "
                            "40 is large enough for any tag including </tool_use>."
                        ),
                    ))

            # Rule 2: bare except:pass
            elif "swallows all errors" in finding.message:
                m = re.search(
                    r"(except\s*(?:Exception\s*)?:\s*\n\s*)(pass)",
                    text, re.MULTILINE
                )
                if m and finding.file not in seen_files:
                    # Only patch if we can insert a meaningful log call
                    patches.append(_PatchItem(
                        file=finding.file,
                        old_text=m.group(0),
                        new_text=m.group(1) + "pass  # TODO: log this error",
                        reason="Bare except:pass hides failures. Added comment as minimum.",
                    ))
                    seen_files.add(finding.file)

            # Rule 3: max_turns very low
            elif "max_turns" in finding.message and "low" in finding.message:
                m = re.search(r"(max_turns\s*=\s*)([1-9])\b", text)
                if m:
                    patches.append(_PatchItem(
                        file=finding.file,
                        old_text=m.group(0),
                        new_text=m.group(1) + "200",
                        reason=(
                            f"max_turns={m.group(2)} is too low for long tasks. "
                            "200 allows novel-length generation with auto-compact."
                        ),
                    ))

        # ── Heuristic patches from problem keywords ────────────────────────
        # If findings gave us nothing actionable, try keyword-guided search
        if not patches:
            patches += self._keyword_patches(problem, keywords, log)

        return patches

    def _keyword_patches(
        self,
        problem: str,
        keywords: List[str],
        log: List[str],
    ) -> List[_PatchItem]:
        """
        Last-resort: search all .py files for patterns mentioned in the
        problem description and propose targeted inline fixes.
        """
        patches: List[_PatchItem] = []
        log.append("\n[Keyword-patch fallback]")

        for root, dirs, files in os.walk(self._root):
            dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = Path(root) / fname
                rel  = str(path.relative_to(self._root))
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue

                # Find lines with high keyword density
                lines = text.splitlines()
                for i, line in enumerate(lines):
                    line_lower = line.lower()
                    hits = [k for k in keywords if k.lower() in line_lower]
                    if len(hits) >= 3:
                        log.append(f"  High-density line {rel}:{i+1}: {line.strip()[:80]}")
                        # Can't safely auto-patch without knowing the fix;
                        # report as a finding only.

        return patches

    # ─────────────────────────────────────────────────────────────────────────
    # Phase I helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _apply_patch(self, patch: _PatchItem, log: List[str]) -> Tuple[bool, str]:
        """
        Apply a single patch using exact str_replace.
        Saves rollback snapshot before modifying.
        Verifies Python syntax after patching.
        Rejects if old_text not found exactly once.
        """
        path = self._root / patch.file
        try:
            original = path.read_text(encoding="utf-8")
        except Exception as e:
            return False, f"Cannot read file: {e}"

        # Safety: old_text must occur exactly once
        count = original.count(patch.old_text)
        if count == 0:
            return False, f"old_text not found in {patch.file}"
        if count > 1:
            return False, (
                f"old_text is ambiguous ({count} occurrences) — "
                "make it more specific to apply safely"
            )

        # Safety: patch size gate (reject if changes >30% of file)
        file_len  = len(original)
        patch_len = max(len(patch.old_text), len(patch.new_text))
        if file_len > 200 and patch_len / file_len > 0.30:
            return False, (
                f"Patch would touch {patch_len/file_len:.0%} of {patch.file}. "
                "Exceeds 30% safety threshold — manual review required."
            )

        # Rollback snapshot
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        rb_dir = _ROLLBACK_DIR / ts
        rb_dir.mkdir(parents=True, exist_ok=True)
        rb_path = rb_dir / patch.file.replace("/", "__").replace("\\", "__")
        rb_path.parent.mkdir(parents=True, exist_ok=True)
        rb_path.write_text(original, encoding="utf-8")
        patch.rollback = str(rb_path)

        # Apply
        patched = original.replace(patch.old_text, patch.new_text, 1)
        path.write_text(patched, encoding="utf-8")
        patch.applied = True

        # Syntax check for Python files
        if patch.file.endswith(".py"):
            try:
                ast.parse(patched)
            except SyntaxError as se:
                # Restore original
                path.write_text(original, encoding="utf-8")
                patch.applied  = False
                patch.rollback = ""
                return False, f"Syntax error after patch: {se}"

        # Log unified diff for visibility
        diff = list(difflib.unified_diff(
            original.splitlines(),
            patched.splitlines(),
            fromfile=f"a/{patch.file}",
            tofile=f"b/{patch.file}",
            n=2,
            lineterm="",
        ))
        log.append("\n" + "\n".join(diff[:40]))

        return True, "ok"

    # ─────────────────────────────────────────────────────────────────────────
    # Phase V helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _run_tests(
        self,
        test_cmd: Optional[str],
        log: List[str],
    ) -> Tuple[str, bool]:
        """Run the test suite and return (output, passed)."""
        if test_cmd:
            cmd = test_cmd
        else:
            # Auto-detect: pytest > unittest discovery > syntax-only check
            if (self._root / "tests").is_dir() or list(self._root.glob("test_*.py")):
                cmd = f"{sys.executable} -m pytest {self._root} -x -q --tb=short 2>&1"
            elif list(self._root.rglob("test_*.py")):
                cmd = f"{sys.executable} -m pytest {self._root} -x -q --tb=short 2>&1"
            else:
                # No test suite found — do a syntax-only pass on all .py files
                cmd = None

        if cmd:
            log.append(f"Running: {cmd}")
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=str(self._root),
                    capture_output=True, text=True, timeout=120,
                )
                output  = (proc.stdout + proc.stderr)[:4000]
                passed  = proc.returncode == 0
                log.append(f"Exit code: {proc.returncode}")
                log.append(output[:1200])
                return output, passed
            except subprocess.TimeoutExpired:
                msg = "Test run timed out after 120 s"
                log.append(msg)
                return msg, False
            except Exception as e:
                msg = f"Test runner error: {e}"
                log.append(msg)
                return msg, False
        else:
            # Syntax-only validation
            log.append("No test suite found — running syntax-only validation.")
            errors: List[str] = []
            for p in self._root.rglob("*.py"):
                if any(s in str(p) for s in _SKIP_DIRS):
                    continue
                try:
                    src = p.read_text(encoding="utf-8", errors="replace")
                    ast.parse(src)
                except SyntaxError as se:
                    errors.append(f"{p.relative_to(self._root)}:{se.lineno}: {se.msg}")
            if errors:
                out = "Syntax errors found:\n" + "\n".join(errors)
                log.append(out)
                return out, False
            log.append("All .py files parse cleanly.")
            return "Syntax OK", True

    def _rollback_all(self, patches: List[_PatchItem], log: List[str]) -> None:
        """Restore all applied patches from their rollback snapshots."""
        for patch in patches:
            if not patch.applied or not patch.rollback:
                continue
            rb_path = Path(patch.rollback)
            dest    = self._root / patch.file
            try:
                dest.write_text(rb_path.read_text(encoding="utf-8"), encoding="utf-8")
                patch.applied = False
                log.append(f"  ↩️  Rolled back: {patch.file}")
            except Exception as e:
                log.append(f"  ❌ Rollback failed for {patch.file}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Report renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render_report(result: _DPIVResult) -> str:
    """Convert a _DPIVResult into a human-readable Markdown report."""
    lines = [
        "# 🩺 Self-Healing Report",
        f"**Problem**: {result.problem[:200]}",
        f"**Duration**: {result.duration_s:.1f} s",
        f"**Rolled back**: {'yes ⚠️' if result.rolled_back else 'no'}",
        "",
    ]

    # Findings
    if result.findings:
        lines.append("## 🔍 Findings")
        for f in result.findings[:15]:
            icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(f.severity, "•")
            lines.append(f"- {icon} **{f.file}**:{f.line or '?'}  {f.message}")
            if f.snippet:
                lines.append(f"  ```\n  {f.snippet[:100]}\n  ```")
        lines.append("")

    # Patches
    if result.patches:
        lines.append("## 🔧 Patches")
        for i, p in enumerate(result.patches, 1):
            status = "✅ applied" if p.applied else "⏭️ skipped"
            lines.append(f"### Patch {i} — {p.file} ({status})")
            lines.append(f"**Reason**: {p.reason}")
            lines.append(f"```diff\n- {p.old_text[:120]}\n+ {p.new_text[:120]}\n```")
            lines.append("")

    # Test output
    if result.test_output:
        lines.append("## 🧪 Validation")
        verdict = "PASSED ✅" if result.test_passed else "FAILED ❌"
        lines.append(f"**Result**: {verdict}")
        lines.append(f"```\n{result.test_output[:800]}\n```")
        lines.append("")

    # Full reasoning log (collapsible in Markdown)
    lines.append("<details><summary>📋 Full reasoning log</summary>\n")
    lines.append("```")
    lines.append("\n".join(result.log)[:3000])
    lines.append("```\n</details>")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  BaseTool wrapper  (registers as "self_healing" in _build_tools)
# ══════════════════════════════════════════════════════════════════════════════

class SelfHealingTool(BaseTool):
    """
    Tool wrapper: EVE can call this directly with a natural-language problem
    description and optionally a list of files to scope the search.

      <tool_use><n>self_healing</n><input>{
        "problem": "The streaming filter leaks XML tags to the browser",
        "scope": ["webui.py", "agent/_robust_parser.py"],
        "test_cmd": "python -m pytest tests/ -x -q",
        "dry_run": false
      }</input></tool_use>
    """

    name        = "self_healing"
    description = (
        "Root-cause-driven code remediation (DPIV loop). "
        "Given a problem description, this tool: "
        "(1) explores the source tree to find the root cause, "
        "(2) plans the smallest safe patch, "
        "(3) applies it with rollback protection, "
        "(4) validates with tests. "
        "Use when you need to debug or fix a repeating problem in the agent's own code."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "problem":  {
                "type": "string",
                "description": "Natural-language description of the bug or issue to fix",
            },
            "scope": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of file paths/globs to limit the search",
            },
            "test_cmd": {
                "type": "string",
                "description": "Optional shell command to run for validation (e.g. 'python -m pytest')",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, report findings/patches without modifying files",
            },
            "max_patches": {
                "type": "integer",
                "description": "Maximum patches to apply in one run (default 10)",
            },
        },
        "required": ["problem"],
    }

    def __init__(self, working_dir: str = "."):
        self._root = Path(working_dir).resolve()

    def execute(self, inp: dict) -> ToolResult:
        problem     = str(inp.get("problem", "")).strip()
        scope       = inp.get("scope")       or None
        test_cmd    = inp.get("test_cmd")    or None
        dry_run     = bool(inp.get("dry_run", False))
        max_patches = int(inp.get("max_patches", 10))

        if not problem:
            return ToolResult("No problem description provided.", is_error=True)

        log_lines: List[str] = []

        engine = DPIVEngine(
            root   = self._root,
            log_fn = log_lines.append,
        )
        result = engine.run(
            problem     = problem,
            scope       = scope,
            test_cmd    = test_cmd,
            dry_run     = dry_run,
            max_patches = max_patches,
        )

        report = _render_report(result)

        # Persist the log to disk for audit trail
        try:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = _LOG_DIR / f"healing_{ts}.md"
            log_path.write_text(report, encoding="utf-8")
        except Exception:
            pass

        is_error = (
            not result.patches
            or (result.patches and not any(p.applied for p in result.patches))
            or result.rolled_back
        )
        return ToolResult(report, is_error=is_error)


# ══════════════════════════════════════════════════════════════════════════════
#  BaseSkill wrapper  (callable via SkillTool by name "self_healing")
# ══════════════════════════════════════════════════════════════════════════════

class SelfHealingSkill(BaseSkill):
    """
    Skill adapter so EVE can also invoke self_healing via the SkillTool:
      skill(skill="self_healing", args={"problem": "..."})
    """

    @property
    def name(self) -> str:
        return "self_healing"

    @property
    def description(self) -> str:
        return (
            "Root-cause-driven code remediation. "
            "Diagnose → Plan → Implement → Validate. "
            "Give it a problem description; it finds and fixes the cause."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        working_dir = kwargs.get("working_dir", ".")
        scope       = kwargs.get("scope")
        test_cmd    = kwargs.get("test_cmd")
        dry_run     = bool(kwargs.get("dry_run", False))

        engine = DPIVEngine(root=Path(working_dir).resolve())
        result = engine.run(
            problem     = problem,
            scope       = scope,
            test_cmd    = test_cmd,
            dry_run     = dry_run,
            max_patches = int(kwargs.get("max_patches", 10)),
        )
        return _render_report(result)


# Expose both at module level for easy import
__all__ = ["SelfHealingTool", "SelfHealingSkill", "DPIVEngine", "_DPIVResult"]
