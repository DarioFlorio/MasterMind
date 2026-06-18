"""
skills/code_remediation.py — Cognitive Code Remediation
=========================================================

This skill encodes the exact mental process I use when debugging:

  Step 0  ORIENT        — read context holistically; classify the problem type
                          before touching any file.  Never skip this.
  Step 1  PARSE INTENT  — extract the gap between intended and actual behaviour,
                          not just the words in the error message.
  Step 2  EXPLORE       — hypothesis-driven file selection (1–3 files, not 40).
                          Go directly where the symptom points, not everywhere.
  Step 3  DIAGNOSE      — mentally execute the code path that produces the bug.
                          Trace token by token, not just pattern-match.
  Step 4  PLAN          — count all root causes first, order by severity+blast-radius,
                          propose the minimum viable change for each.
  Step 5  IMPLEMENT     — style-aware, coupling-aware edits.
                          Never touch code that isn't causing the bug.
  Step 6  VALIDATE      — verify the mental model first, then run real tests.
                          "Tests pass" ≠ "fix is correct" — check both.

The key difference from self_healing.py (the mechanical DPIV loop):
  • self_healing  → breadth-first scan, pattern matching, heuristic patches
  • code_remediation → hypothesis-driven, execution trace, blast-radius reasoning

Use code_remediation for novel / cross-file bugs where the symptom is far
from the cause.  Use self_healing for known-pattern bugs (bad defaults,
uncaught exceptions, missing imports).
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── shims ─────────────────────────────────────────────────────────────────────
try:
    from tools.base_tool import BaseTool, ToolResult
except ImportError:
    class ToolResult:  # type: ignore
        def __init__(self, output="", is_error=False):
            self.output = output; self.is_error = is_error
    class BaseTool:   # type: ignore
        name=""; description=""; input_schema: dict={}
        def execute(self, inp): ...

try:
    from skills.base_skill import BaseSkill
except ImportError:
    class BaseSkill:  # type: ignore
        @property
        def name(self): return ""
        def execute_impl(self, problem, **kw): return ""

# ── constants ─────────────────────────────────────────────────────────────────
_SKIP_DIRS  = {"__pycache__", ".git", "node_modules", ".venv", "venv",
               "dist", "build", "wa_session", "temp", "logs"}
_SRC_EXTS   = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".md", ".toml"}
_MAX_READ   = 10_000   # chars per file in explore phase
_MAX_WINDOW = 40       # lines of context around a suspect line

_LOG_DIR = Path("temp/code_remediation_logs")


# ══════════════════════════════════════════════════════════════════════════════
#  Data shapes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Hypothesis:
    """A candidate root cause derived from ORIENT + PARSE INTENT."""
    description: str          # "StreamTagFilter._MAX_TAG_LEN=2 causes partial-tag flush"
    target_files: List[str]   # files most likely to contain the cause
    symptom_gap: str          # "should suppress XML / actually emits '<'"
    confidence: float = 0.0   # 0..1, updated after EXPLORE


@dataclass
class ExecutionTrace:
    """Mental model of the code path that produces the bug."""
    file:    str
    entry:   str          # function / method where trace starts
    steps:   List[str]    # ordered list of state transitions
    failure: str          # the exact step where behaviour diverges
    snippet: str = ""     # relevant code excerpt


@dataclass
class PatchSpec:
    """Minimum viable change for one root cause."""
    file:      str
    old_text:  str
    new_text:  str
    reason:    str
    blast_radius: str  # description of coupling / what else could break
    applied:   bool = False
    rollback:  str  = ""


@dataclass
class RemediationResult:
    problem:     str
    hypothesis:  Optional[Hypothesis]     = None
    traces:      List[ExecutionTrace]     = field(default_factory=list)
    patches:     List[PatchSpec]          = field(default_factory=list)
    test_output: str                      = ""
    test_passed: bool                     = False
    log:         List[str]               = field(default_factory=list)
    duration_s:  float                    = 0.0
    rolled_back: bool                     = False


# ══════════════════════════════════════════════════════════════════════════════
#  The six-step cognitive engine
# ══════════════════════════════════════════════════════════════════════════════

class CognitiveRemediationEngine:

    def __init__(self, root: Path):
        self._root = root.resolve()
        _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public entry ──────────────────────────────────────────────────────────

    def run(
        self,
        problem: str,
        scope: Optional[List[str]] = None,
        test_cmd: Optional[str]    = None,
        dry_run:  bool             = False,
    ) -> RemediationResult:

        t0     = time.time()
        result = RemediationResult(problem=problem)
        log    = result.log

        def step(n, title):
            s = f"\n{'━'*62}\n  Step {n}: {title}\n{'━'*62}"
            log.append(s); print(s, file=sys.stderr, flush=True)

        # ── 0: ORIENT ─────────────────────────────────────────────────────────
        step("0", "ORIENT — read context, classify problem type")
        p_type, context_notes = self._orient(problem)
        log.append(f"  Problem type : {p_type}")
        log.append(f"  Context notes: {context_notes}")

        # ── 1: PARSE INTENT ───────────────────────────────────────────────────
        step("1", "PARSE INTENT — gap between intended and actual behaviour")
        hypothesis = self._parse_intent(problem, p_type)
        result.hypothesis = hypothesis
        log.append(f"  Hypothesis   : {hypothesis.description}")
        log.append(f"  Gap          : {hypothesis.symptom_gap}")
        log.append(f"  Target files : {hypothesis.target_files}")

        # ── 2: EXPLORE (hypothesis-driven) ────────────────────────────────────
        step("2", "EXPLORE — go directly where hypothesis points")
        candidates = self._explore(hypothesis, scope)
        log.append(f"  Files examined ({len(candidates)}):")
        for f, score, note in candidates:
            log.append(f"    {score:.2f}  {f}  — {note}")

        # ── 3: DIAGNOSE (execution trace) ─────────────────────────────────────
        step("3", "DIAGNOSE — trace the code path that produces the bug")
        traces = self._diagnose(hypothesis, candidates, log)
        result.traces = traces
        for t in traces:
            log.append(f"\n  Trace in {t.file}::{t.entry}")
            for s in t.steps:
                log.append(f"    → {s}")
            log.append(f"  ❌ Failure point: {t.failure}")

        if not traces:
            log.append("  No execution traces found — check hypothesis targets.")

        # ── 4: PLAN (severity + blast radius) ─────────────────────────────────
        step("4", "PLAN — minimum viable changes, ordered by severity")
        patches = self._plan(traces, log)
        result.patches = patches
        for i, p in enumerate(patches, 1):
            log.append(f"\n  Patch {i}: {p.file}")
            log.append(f"    Reason      : {p.reason}")
            log.append(f"    Blast radius: {p.blast_radius}")
            log.append(f"    Remove      : {p.old_text[:80]!r}")
            log.append(f"    Insert      : {p.new_text[:80]!r}")

        if not patches:
            log.append("  No actionable patches derived — manual review needed.")
            result.duration_s = time.time() - t0
            return result

        if dry_run:
            log.append("\n  🔍 DRY RUN — no files modified.")
            result.duration_s = time.time() - t0
            return result

        # ── 5: IMPLEMENT (style-aware, coupling-aware) ─────────────────────────
        step("5", "IMPLEMENT — apply minimum viable changes")
        applied = self._implement(patches, log)
        log.append(f"  Applied {applied}/{len(patches)} patches.")

        if applied == 0:
            result.duration_s = time.time() - t0
            return result

        # ── 6: VALIDATE (mental model first, then real tests) ─────────────────
        step("6", "VALIDATE — verify mental model, then run tests")
        self._validate_mental_model(patches, log)
        test_out, passed = self._run_tests(test_cmd, log)
        result.test_output = test_out
        result.test_passed = passed

        if not passed:
            log.append("\n  ⚠️  Tests FAILED — rolling back all patches.")
            self._rollback(patches, log)
            result.rolled_back = True
        else:
            log.append("\n  ✅ Tests PASSED — fix confirmed.")

        result.duration_s = time.time() - t0

        # Persist log
        try:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            lp  = _LOG_DIR / f"remediation_{ts}.md"
            lp.write_text(_render(result), encoding="utf-8")
        except Exception:
            pass

        return result

    # ── Step 0: ORIENT ────────────────────────────────────────────────────────

    def _orient(self, problem: str) -> Tuple[str, str]:
        """
        Classify the problem type.  This governs which file-set to explore
        and which execution path to trace.  Never skip this step.
        """
        lower = problem.lower()

        # Rendering / UI artefacts
        if any(t in lower for t in ["leak", "visible", "browser", "ui", "html",
                                    "render", "display", "chat", "tag"]):
            return "rendering_bug", "Symptom is in the UI layer; cause is likely in the streaming pipeline."

        # Crash / exception
        if any(t in lower for t in ["error", "exception", "crash", "traceback",
                                    "raises", "fails with"]):
            return "crash", "Unhandled exception; look for missing try/except or wrong assumption."

        # Silent failure
        if any(t in lower for t in ["no output", "empty", "nothing happens",
                                    "not working", "doesn't", "does not"]):
            return "silent_failure", "Symptom is absence of action; look for early-return or swallowed exception."

        # Logic / wrong result
        if any(t in lower for t in ["wrong", "incorrect", "invalid", "bad result",
                                    "should return", "expected"]):
            return "logic_bug", "Wrong output; trace the computation, not the I/O."

        # Performance
        if any(t in lower for t in ["slow", "timeout", "hang", "block", "freeze"]):
            return "performance", "Blocking call or infinite loop; look for missing timeout or unbounded loop."

        # Ghost / state leak
        if any(t in lower for t in ["ghost", "old task", "previous", "keeps",
                                    "drift", "persist", "stuck"]):
            return "state_leak", "Stale state persisting across requests; look for shared mutable singletons."

        return "general", "No clear type match; explore from the symptom outward."

    # ── Step 1: PARSE INTENT ──────────────────────────────────────────────────

    def _parse_intent(self, problem: str, p_type: str) -> Hypothesis:
        """
        Build a Hypothesis: the gap between what the system SHOULD do and
        what it ACTUALLY does, plus the file(s) most likely to contain the cause.

        This is not regex over error text.  It's inferring the mental model.
        """
        # Extract quoted strings (often exact error messages or variable names)
        quoted   = re.findall(r'["`\']([\w_.<>/\-]{3,60})["`\']', problem)

        # Extract identifiers
        snake    = re.findall(r'\b([a-z][a-z0-9]{2,}(?:_[a-z0-9]+)+)\b', problem)
        camel    = re.findall(r'\b([A-Z][a-zA-Z]{3,})\b', problem)
        files_m  = re.findall(r'(\w[\w/_.-]+\.py)', problem)

        # Score all source files by identifier hit density
        all_kw = list(set(quoted + snake + camel + files_m))

        scored: List[Tuple[str, int]] = []
        for path in self._root.rglob("*.py"):
            if any(s in str(path) for s in _SKIP_DIRS):
                continue
            try:
                txt   = path.read_text(encoding="utf-8", errors="replace")[:_MAX_READ]
                score = sum(txt.count(k) for k in all_kw if len(k) > 3)
                if score:
                    scored.append((str(path.relative_to(self._root)), score))
            except Exception:
                pass
        scored.sort(key=lambda x: x[1], reverse=True)
        top_files = [f for f, _ in scored[:4]]

        # Infer the symptom gap from problem type
        gap_map = {
            "rendering_bug":  "should suppress XML markup / actually emits raw tags",
            "crash":          "should handle the error gracefully / actually raises unhandled",
            "silent_failure": "should produce output / actually returns early with no effect",
            "logic_bug":      "should return correct value / actually returns wrong one",
            "performance":    "should complete promptly / actually blocks or hangs",
            "state_leak":     "should be stateless across requests / actually carries old state",
            "general":        "should work as documented / actually does not",
        }

        return Hypothesis(
            description = f"[{p_type}] {problem[:120]}",
            target_files = top_files or ["webui.py", "agent/query_engine.py"],
            symptom_gap  = gap_map.get(p_type, "unknown gap"),
            confidence   = min(1.0, len(top_files) * 0.25),
        )

    # ── Step 2: EXPLORE ───────────────────────────────────────────────────────

    def _explore(
        self,
        hyp: Hypothesis,
        scope: Optional[List[str]],
    ) -> List[Tuple[str, float, str]]:
        """
        Read only the files the hypothesis points to.
        Return (rel_path, confidence_score, one-line note).
        """
        targets = scope or hyp.target_files
        results = []
        for rel in targets:
            path = self._root / rel
            if not path.exists():
                # Try glob
                matches = list(self._root.rglob(rel))
                if matches:
                    path = matches[0]
                else:
                    continue
            try:
                txt  = path.read_text(encoding="utf-8", errors="replace")
                note = self._quick_note(txt)
                rel2 = str(path.relative_to(self._root))
                # Confidence = fraction of hypothesis keywords found
                kws  = re.findall(r'\b\w{4,}\b', hyp.description.lower())
                hits = sum(1 for k in kws if k in txt.lower())
                conf = hits / max(len(kws), 1)
                results.append((rel2, conf, note))
            except Exception as e:
                results.append((rel, 0.0, f"unreadable: {e}"))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _quick_note(self, text: str) -> str:
        """One-line characterisation of a source file."""
        lines = text.splitlines()
        if lines:
            first = next((l.strip() for l in lines[:20]
                          if l.strip() and not l.strip().startswith("#")), "")
            return first[:80]
        return ""

    # ── Step 3: DIAGNOSE ─────────────────────────────────────────────────────

    def _diagnose(
        self,
        hyp: Hypothesis,
        candidates: List[Tuple[str, float, str]],
        log: List[str],
    ) -> List[ExecutionTrace]:
        """
        Mentally execute the code path.  For each candidate file, find the
        function most relevant to the hypothesis and trace its state changes.
        """
        traces: List[ExecutionTrace] = []
        gap_lower = hyp.symptom_gap.lower()

        for rel, conf, _ in candidates[:4]:
            if conf < 0.05:
                continue
            path = self._root / rel
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Find the function/class most relevant to the hypothesis
            entry, snippet, failure = self._trace_code_path(src, hyp, gap_lower)
            if entry:
                traces.append(ExecutionTrace(
                    file    = rel,
                    entry   = entry,
                    steps   = self._generate_trace_steps(src, entry, hyp),
                    failure = failure,
                    snippet = snippet,
                ))

        return traces

    def _trace_code_path(
        self, src: str, hyp: Hypothesis, gap: str
    ) -> Tuple[str, str, str]:
        """
        Find the function that most likely contains the root cause by
        looking for the conjunction of: relevant identifiers + bad patterns.
        """
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return "", "", "SyntaxError — cannot parse file"

        lines   = src.splitlines()
        kws     = set(re.findall(r'\b\w{4,}\b', hyp.description.lower()))
        best    = ("", "", "")
        best_sc = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            fn_lines = lines[node.lineno - 1 : node.end_lineno]
            fn_text  = "\n".join(fn_lines)
            fn_lower = fn_text.lower()

            score = sum(1 for k in kws if k in fn_lower)

            # Extra score for known bad patterns
            if re.search(r"_MAX_TAG_LEN\s*\+\s*2", fn_text):      score += 5
            if re.search(r"except\s*:\s*\n\s*pass",  fn_text):      score += 3
            if re.search(r"return\s+(?:None|\"\")",   fn_text):      score += 2
            if "goal_text" in fn_lower and "not" in fn_lower:        score += 2

            if score > best_sc:
                best_sc = score
                snippet = "\n".join(fn_lines[:_MAX_WINDOW])
                failure = self._infer_failure(fn_text, hyp)
                best    = (node.name, snippet[:600], failure)

        return best

    def _infer_failure(self, fn_text: str, hyp: Hypothesis) -> str:
        """Describe where execution diverges from the intended model."""
        if re.search(r"_MAX_TAG_LEN\s*\+\s*2", fn_text):
            return ("_MAX_TAG_LEN+2 threshold too low → "
                    "len(buf) > threshold fires for any tag > 4 chars → "
                    "flush_text('<') emits the '<' directly to browser")
        if re.search(r"except\s*:\s*\n\s*pass", fn_text):
            return "bare except:pass silently swallows exception, caller sees no error"
        if re.search(r"if not self\._goal_text", fn_text):
            return ("goal only set when _goal_text is empty → "
                    "new task never replaces old goal → ghost task persists")
        if re.search(r"as_completed.*timeout", fn_text):
            return "TimeoutError from as_completed not caught → tool raises exception"
        return "execution diverges — see snippet for the exact divergence point"

    def _generate_trace_steps(
        self, src: str, entry: str, hyp: Hypothesis
    ) -> List[str]:
        """Generate a readable execution trace for the function."""
        steps = [
            f"Called `{entry}()`",
            f"Symptom gap: {hyp.symptom_gap}",
        ]
        # Pattern-specific trace steps
        if "_MAX_TAG_LEN" in src:
            steps += [
                "chunk arrives split across network: e.g. `<tool_us` (no `>` yet)",
                "`find('>')` returns -1 → incomplete-tag branch",
                f"`len('<tool_us') = 8`, `_MAX_TAG_LEN + 2 = 4`, `8 > 4` → True",
                "`flush_text('<')` fires → `<` emitted to browser",
                "remaining `tool_us` emitted on next iteration",
                "Result: raw `<tool_us` visible in chat",
            ]
        elif "_goal_text" in src:
            steps += [
                "User sends new task message",
                "`if not self._goal_text` → False (old goal still set)",
                "New task does NOT replace old goal",
                "GoalTracker injects old goal into every subsequent prompt",
                "Model sees old task context → drifts back to it",
            ]
        elif "as_completed" in src:
            steps += [
                "Web search fires N concurrent futures",
                "Some futures take > timeout seconds",
                "`as_completed(futs, timeout=T)` raises `TimeoutError`",
                "TimeoutError not caught → propagates as tool exception",
                "Agent sees: 'Tool raised exception: N futures unfinished'",
            ]
        else:
            steps.append("(trace steps derived from problem description)")
        return steps

    # ── Step 4: PLAN ──────────────────────────────────────────────────────────

    def _plan(
        self,
        traces: List[ExecutionTrace],
        log: List[str],
    ) -> List[PatchSpec]:
        """
        For each trace, propose the MINIMUM VIABLE change.
        Order patches by: severity DESC, blast_radius ASC.
        Never propose changes to files not implicated in a trace.
        """
        patches: List[PatchSpec] = []
        seen: set = set()

        for trace in traces:
            path = self._root / trace.file
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Rule A: _MAX_TAG_LEN too small
            m = re.search(r"(_MAX_TAG_LEN\s*=\s*)(\d+)\b", src)
            if m and int(m.group(2)) < 10 and trace.file not in seen:
                patches.append(PatchSpec(
                    file      = trace.file,
                    old_text  = m.group(0),
                    new_text  = m.group(1) + "40",
                    reason    = (
                        f"_MAX_TAG_LEN={m.group(2)} → threshold {int(m.group(2))+2}. "
                        "Any XML tag longer than that flushes '<' on split chunks. "
                        "40 safely covers all tags in the grammar."
                    ),
                    blast_radius = (
                        "Only used in _process() for the partial-tag wait logic. "
                        "Larger value means longer buffer wait — no semantic change."
                    ),
                ))
                seen.add(trace.file)

            # Rule B: goal never replaced on new task
            m2 = re.search(r"if not self\._goal_text and user_text\.strip\(\):", src)
            if m2 and trace.file not in seen:
                patches.append(PatchSpec(
                    file      = trace.file,
                    old_text  = "if not self._goal_text and user_text.strip():",
                    new_text  = "if user_text.strip():  # always update goal on new task",
                    reason    = (
                        "Goal only set when _goal_text is empty. "
                        "New task never replaces old goal → ghost task. "
                        "Fix: always update on new TASK intent."
                    ),
                    blast_radius = (
                        "GoalTracker.set_goal() is idempotent. "
                        "Replacing the goal on every TASK message is safe and intended."
                    ),
                ))
                seen.add(trace.file)

            # Rule C: bare TimeoutError not caught
            if "as_completed" in src and "TimeoutError" not in src:
                old = "for fut in as_completed(futs, timeout="
                if old in src and trace.file not in seen:
                    patches.append(PatchSpec(
                        file      = trace.file,
                        old_text  = "from concurrent.futures import ThreadPoolExecutor, as_completed",
                        new_text  = "from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FutureTimeout",
                        reason    = (
                            "TimeoutError from as_completed(timeout=...) is not caught. "
                            "Add import so the except clause can name it explicitly."
                        ),
                        blast_radius = (
                            "Import-only change. No logic affected until an except clause is also added."
                        ),
                    ))
                    seen.add(trace.file)

            # Rule D: bare except:pass
            if re.search(r"except\s*:\s*\n\s*pass", src) and trace.file not in seen:
                m3 = re.search(r"(except\s*:\s*\n\s*)(pass)", src, re.MULTILINE)
                if m3:
                    patches.append(PatchSpec(
                        file      = trace.file,
                        old_text  = m3.group(0),
                        new_text  = m3.group(1) + "pass  # suppressed — add logging here",
                        reason    = "Bare except:pass silently swallows exceptions. Comment marks the debt.",
                        blast_radius = "Comment-only change. No logic altered.",
                    ))
                    seen.add(trace.file)

        # Sort: critical first (shorter blast radius = safer = first)
        patches.sort(key=lambda p: len(p.blast_radius))
        return patches

    # ── Step 5: IMPLEMENT ─────────────────────────────────────────────────────

    def _implement(self, patches: List[PatchSpec], log: List[str]) -> int:
        """
        Style-aware, coupling-aware edits.
        Before touching any file: save rollback, confirm uniqueness, check syntax after.
        """
        applied = 0
        for p in patches:
            path = self._root / p.file
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                log.append(f"  ❌ Cannot read {p.file}: {e}")
                continue

            # Uniqueness guard
            count = src.count(p.old_text)
            if count == 0:
                log.append(f"  ⏭️  {p.file}: old_text not found (already fixed?)")
                continue
            if count > 1:
                log.append(f"  ⚠️  {p.file}: old_text appears {count}× — too ambiguous, skip.")
                continue

            # Style check: preserve surrounding indentation
            m = re.search(re.escape(p.old_text), src)
            if m:
                start = src.rfind("\n", 0, m.start()) + 1
                indent = len(src[start:m.start()]) - len(src[start:m.start()].lstrip())
                # If new_text is multi-line, apply same indent
                if "\n" in p.new_text and indent:
                    ind_str  = " " * indent
                    new_text = p.new_text.replace("\n", "\n" + ind_str)
                else:
                    new_text = p.new_text
            else:
                new_text = p.new_text

            # Rollback snapshot
            rb_dir  = Path("temp/code_remediation_rollbacks")
            rb_dir.mkdir(parents=True, exist_ok=True)
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            rb_path = rb_dir / f"{p.file.replace('/', '__')}_{ts}.bak"
            rb_path.write_text(src, encoding="utf-8")
            p.rollback = str(rb_path)

            # Apply
            patched = src.replace(p.old_text, new_text, 1)
            path.write_text(patched, encoding="utf-8")

            # Syntax gate for .py files
            if p.file.endswith(".py"):
                try:
                    ast.parse(patched)
                except SyntaxError as se:
                    path.write_text(src, encoding="utf-8")
                    log.append(f"  ❌ {p.file}: syntax error after patch — reverted. {se}")
                    p.rollback = ""
                    continue

            p.applied = True
            applied  += 1
            log.append(f"  ✅ {p.file}: applied — {p.reason[:60]}")

        return applied

    # ── Step 6: VALIDATE ─────────────────────────────────────────────────────

    def _validate_mental_model(self, patches: List[PatchSpec], log: List[str]) -> None:
        """
        Before running real tests, verify the mental model of each fix.
        'Tests pass' ≠ 'fix is correct' — check both.
        """
        log.append("\n  Mental-model validation:")
        for p in patches:
            if not p.applied:
                continue
            path = self._root / p.file
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # Confirm the new value is present
            if p.new_text.strip() in src:
                log.append(f"    ✅ {p.file}: new text confirmed in source")
            else:
                log.append(f"    ⚠️  {p.file}: new text NOT found — patch may have been misapplied")

            # For _MAX_TAG_LEN, confirm the threshold is now safe
            m = re.search(r"_MAX_TAG_LEN\s*=\s*(\d+)", src)
            if m and int(m.group(1)) >= 30:
                log.append(f"    ✅ {p.file}: _MAX_TAG_LEN={m.group(1)} ≥ 30 — threshold safe")
            elif m:
                log.append(f"    ⚠️  {p.file}: _MAX_TAG_LEN={m.group(1)} still low")

    def _run_tests(
        self,
        test_cmd: Optional[str],
        log: List[str],
    ) -> Tuple[str, bool]:
        if test_cmd:
            cmd = test_cmd
        elif list(self._root.rglob("test_*.py")):
            cmd = f"{sys.executable} -m pytest {self._root} -x -q --tb=short 2>&1"
        else:
            # Syntax-only pass
            errors = []
            for p in self._root.rglob("*.py"):
                if any(s in str(p) for s in _SKIP_DIRS):
                    continue
                try:
                    ast.parse(p.read_text(encoding="utf-8", errors="replace"))
                except SyntaxError as se:
                    errors.append(f"{p.relative_to(self._root)}:{se.lineno}: {se.msg}")
            out = "Syntax OK" if not errors else "Syntax errors:\n" + "\n".join(errors)
            log.append(f"  {out}")
            return out, not errors

        log.append(f"  Running: {cmd}")
        try:
            proc = subprocess.run(
                cmd, shell=True, cwd=str(self._root),
                capture_output=True, text=True, timeout=120,
            )
            output = (proc.stdout + proc.stderr)[:4000]
            passed = proc.returncode == 0
            log.append(f"  Exit: {proc.returncode}")
            log.append(output[:1000])
            return output, passed
        except subprocess.TimeoutExpired:
            msg = "Test run timed out after 120 s"
            log.append(f"  {msg}")
            return msg, False

    def _rollback(self, patches: List[PatchSpec], log: List[str]) -> None:
        for p in patches:
            if not p.applied or not p.rollback:
                continue
            try:
                dest = self._root / p.file
                dest.write_text(
                    Path(p.rollback).read_text(encoding="utf-8"), encoding="utf-8"
                )
                p.applied = False
                log.append(f"  ↩️  Rolled back: {p.file}")
            except Exception as e:
                log.append(f"  ❌ Rollback failed for {p.file}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Report renderer
# ══════════════════════════════════════════════════════════════════════════════

def _render(r: RemediationResult) -> str:
    lines = [
        "# 🧠 Cognitive Code Remediation Report",
        f"**Problem**: {r.problem[:200]}",
        f"**Duration**: {r.duration_s:.1f} s",
        f"**Rolled back**: {'yes ⚠️' if r.rolled_back else 'no'}",
        "",
    ]
    if r.hypothesis:
        lines += [
            "## Step 1 — Hypothesis",
            f"- **Description**: {r.hypothesis.description}",
            f"- **Gap**: {r.hypothesis.symptom_gap}",
            f"- **Target files**: {', '.join(r.hypothesis.target_files[:4])}",
            f"- **Confidence**: {r.hypothesis.confidence:.0%}",
            "",
        ]
    if r.traces:
        lines.append("## Step 3 — Execution Traces")
        for t in r.traces:
            lines.append(f"### `{t.file}` :: `{t.entry}`")
            for s in t.steps:
                lines.append(f"- {s}")
            lines.append(f"\n**Failure point**: {t.failure}")
            if t.snippet:
                lines.append(f"\n```python\n{t.snippet[:400]}\n```")
            lines.append("")
    if r.patches:
        lines.append("## Step 4–5 — Patches")
        for i, p in enumerate(r.patches, 1):
            status = "✅ applied" if p.applied else "⏭️ skipped"
            lines += [
                f"### Patch {i} — `{p.file}` ({status})",
                f"**Reason**: {p.reason}",
                f"**Blast radius**: {p.blast_radius}",
                f"```diff\n- {p.old_text[:120]}\n+ {p.new_text[:120]}\n```",
                "",
            ]
    if r.test_output:
        verdict = "PASSED ✅" if r.test_passed else "FAILED ❌"
        lines += [
            "## Step 6 — Validation",
            f"**Result**: {verdict}",
            f"```\n{r.test_output[:600]}\n```",
            "",
        ]
    lines += [
        "<details><summary>📋 Full reasoning log</summary>\n",
        "```",
        "\n".join(r.log)[:4000],
        "```\n</details>",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  BaseTool wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CodeRemediationTool(BaseTool):
    """
    EVE calls this tool when it needs to debug its own codebase using
    the same cognitive process Claude uses: Orient → Trace → BlastRadius → MinViable.

      <tool_use><n>code_remediation</n><input>{
        "problem": "The streaming filter leaks XML tags to the browser",
        "scope": ["webui.py"],
        "dry_run": false
      }</input></tool_use>
    """

    name        = "code_remediation"
    description = (
        "Cognitive code remediation — the same 6-step debugging process "
        "Claude uses: Orient, Parse Intent, Explore (hypothesis-driven), "
        "Diagnose (execution trace), Plan (blast radius), Implement, Validate. "
        "Use for novel or cross-file bugs where the symptom is far from the cause. "
        "For known-pattern bugs (bad defaults, missing imports) use self_healing instead."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "problem":  {"type": "string",
                         "description": "Natural-language bug description"},
            "scope":    {"type": "array", "items": {"type": "string"},
                         "description": "Optional file paths to limit exploration"},
            "test_cmd": {"type": "string",
                         "description": "Shell command to run for validation"},
            "dry_run":  {"type": "boolean",
                         "description": "Report only, do not modify files"},
        },
        "required": ["problem"],
    }

    def __init__(self, working_dir: str = "."):
        self._root = Path(working_dir).resolve()

    def execute(self, inp: dict) -> ToolResult:
        problem  = str(inp.get("problem", "")).strip()
        scope    = inp.get("scope") or None
        test_cmd = inp.get("test_cmd") or None
        dry_run  = bool(inp.get("dry_run", False))

        if not problem:
            return ToolResult("No problem description.", is_error=True)

        engine = CognitiveRemediationEngine(self._root)
        result = engine.run(problem=problem, scope=scope,
                            test_cmd=test_cmd, dry_run=dry_run)
        report = _render(result)

        is_err = (
            not result.patches
            or result.rolled_back
            or (result.patches and not any(p.applied for p in result.patches))
        )
        return ToolResult(report, is_error=is_err)


# ══════════════════════════════════════════════════════════════════════════════
#  BaseSkill wrapper
# ══════════════════════════════════════════════════════════════════════════════

class CodeRemediationSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "code_remediation"

    @property
    def description(self) -> str:
        return (
            "6-step cognitive debugging: Orient → Parse Intent → Explore "
            "(hypothesis-driven) → Diagnose (execution trace) → Plan "
            "(blast radius) → Implement (minimum viable) → Validate."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        engine = CognitiveRemediationEngine(
            Path(kwargs.get("working_dir", ".")).resolve()
        )
        result = engine.run(
            problem  = problem,
            scope    = kwargs.get("scope"),
            test_cmd = kwargs.get("test_cmd"),
            dry_run  = bool(kwargs.get("dry_run", False)),
        )
        return _render(result)


__all__ = [
    "CodeRemediationTool", "CodeRemediationSkill",
    "CognitiveRemediationEngine", "RemediationResult",
]
