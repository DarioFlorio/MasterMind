"""
Skill: surgical_debug
Evidence-first investigation methodology for bugs, unexpected behaviour,
and system questions.

Methodology (exactly as applied when diagnosing this codebase):
  1. Read before reasoning   — never answer from hypothesis alone
  2. Architecture map first  — understand scope before depth
  3. Follow the data         — trace actual execution path, not docs
  4. Separate symptom→cause  — ask what mechanism produces the symptom
  5. Find all bugs, not one  — resist closing at first cause found
  6. Minimum intervention    — change fewest lines that fix the most
  7. Verify by re-tracing    — confirm fix is correct before committing
"""
from __future__ import annotations

DESCRIPTION = (
    "Evidence-first surgical debugging: read source before reasoning, "
    "map architecture, trace data flow, separate symptoms from causes, "
    "find minimum intervention. Use for bugs, unexpected behaviour, "
    "and 'why does X do Y' questions about real systems."
)


# ── Phase helpers ──────────────────────────────────────────────────────────────

def _phase_read_first(problem: str) -> str:
    """Phase 1: suppress hypothesis, demand evidence."""
    return (
        "## Phase 1 — Read Before Reasoning\n\n"
        "STOP. Do not answer from hypothesis.\n\n"
        f"Problem stated: {problem}\n\n"
        "Before forming any conclusion, identify:\n"
        "  • What source files / logs / data are directly relevant?\n"
        "  • What is the smallest set of things to read that could contain the answer?\n"
        "  • What would DISPROVE the most tempting hypothesis?\n\n"
        "Rule: no conclusion is valid until it is grounded in something you have actually read."
    )


def _phase_architecture_map(problem: str) -> str:
    """Phase 2: build a mental map before diving deep."""
    return (
        "## Phase 2 — Architecture Map\n\n"
        "Before reading any single file in depth, answer:\n\n"
        "  1. What are the major components involved in this problem?\n"
        "  2. Which component is most likely to CONTAIN the bug/answer?\n"
        "  3. Which component is most likely to SURFACE the symptom?\n"
        "     (These are often different — the bug lives upstream of where you see it.)\n"
        "  4. What is the boundary between them?\n\n"
        "Draw the map in plain text:\n"
        "  [input] → [component A] → [component B] → [output/symptom]\n\n"
        "This map is your investigation plan. Read in this order — start where the "
        "symptom surfaces, then walk upstream toward the actual cause."
    )


def _phase_trace_data_flow(problem: str) -> str:
    """Phase 3: follow the actual execution path."""
    return (
        "## Phase 3 — Trace the Data Flow\n\n"
        "Follow what actually happens — not what the docs say happens.\n\n"
        "For each step in your architecture map, answer:\n"
        "  • What enters this component? (type, value, format)\n"
        "  • What transformation happens inside?\n"
        "  • What exits? Does it match what the next component expects?\n"
        "  • Where could the path DIVERGE from the happy path?\n\n"
        "Trace rules:\n"
        "  - Follow the token / request / event / data structure, not the function names\n"
        "  - Threads, queues, and async boundaries are where race conditions live — "
        "mark them explicitly\n"
        "  - If two components share state (a buffer, a flag, a file), "
        "that shared state is a suspect\n\n"
        "Write the trace as a numbered path:\n"
        "  1. X enters component A as type T\n"
        "  2. A transforms X → Y\n"
        "  3. Y is written to shared_buffer\n"
        "  4. Component B reads shared_buffer on a different thread ← RACE CONDITION SUSPECT"
    )


def _phase_separate_symptom_cause(problem: str) -> str:
    """Phase 4: separate what you see from what causes it."""
    return (
        "## Phase 4 — Separate Symptom from Cause\n\n"
        "The symptom is what you observe. The cause is the mechanism that produces it.\n"
        "They are almost never in the same place.\n\n"
        "For this problem:\n\n"
        "  SYMPTOM: what is observed? (exact behaviour, error message, timing)\n"
        "  ↓\n"
        "  MECHANISM: what code path produces this observation?\n"
        "  ↓\n"
        "  CAUSE: what condition makes the mechanism behave this way?\n"
        "  ↓\n"
        "  ROOT CAUSE: what triggered that condition?\n\n"
        "Diagnostic questions:\n"
        "  • 'What would have to be TRUE for this symptom to appear?'\n"
        "  • 'What would have to be FALSE for this symptom to disappear?'\n"
        "  • 'Is the symptom deterministic or intermittent?' "
        "(intermittent → timing/concurrency; deterministic → logic)\n"
        "  • 'Where in the stack does the wrong value FIRST appear?'\n\n"
        "Do NOT stop at the first cause found. Ask: "
        "'Are there other independent causes producing the same symptom?'"
    )


def _phase_find_all_bugs(problem: str) -> str:
    """Phase 5: resist closing at first cause found."""
    return (
        "## Phase 5 — Find All Bugs, Not Just One\n\n"
        "The first bug you find is rarely the only one.\n"
        "Closing the investigation at first cause is the most common diagnostic error.\n\n"
        "After finding cause #1:\n"
        "  • Does fixing it fully explain the symptom? Or only part of it?\n"
        "  • Are there other symptoms in the original report that this doesn't explain?\n"
        "  • Are there related paths through the same code that have the same flaw?\n"
        "  • Is the root cause a PATTERN (e.g. 'wrong entry point', 'missing sync') "
        "that appears elsewhere?\n\n"
        "Triage:\n"
        "  BUG A: [description] → [impact] → [fix complexity]\n"
        "  BUG B: [description] → [impact] → [fix complexity]\n\n"
        "Fix in impact order, not discovery order. "
        "A simple fix with high impact beats a complex fix with low impact."
    )


def _phase_minimum_intervention(problem: str) -> str:
    """Phase 6: find the smallest correct fix."""
    return (
        "## Phase 6 — Minimum Effective Intervention\n\n"
        "The best fix changes the fewest lines that correct the most behaviour.\n"
        "Rewrites are not fixes — they introduce new bugs and destroy context.\n\n"
        "For each bug found:\n"
        "  1. What is the EXACT line(s) where the wrong thing happens?\n"
        "  2. What is the correct value/behaviour at that exact point?\n"
        "  3. What is the smallest change that produces it?\n\n"
        "Intervention hierarchy (use the first that applies):\n"
        "  1. Change a constant or config value\n"
        "  2. Change a single comparison or condition\n"
        "  3. Add a guard or synchronisation primitive\n"
        "  4. Replace a function call with a more appropriate one\n"
        "  5. Add a small helper function\n"
        "  6. Refactor a component (last resort)\n\n"
        "Before writing the fix, state:\n"
        "  'I am changing [X] at [location] from [wrong] to [correct] because [reason].'\n\n"
        "If you cannot state that sentence clearly, you do not yet understand the bug."
    )


def _phase_verify(problem: str) -> str:
    """Phase 7: re-trace the corrected flow before committing."""
    return (
        "## Phase 7 — Verify by Re-Tracing\n\n"
        "Do not commit a fix until you have re-traced the corrected execution path.\n\n"
        "Verification checklist:\n"
        "  □ Re-run Phase 3 (data flow trace) with the fix applied — does the path now "
        "produce correct output at every step?\n"
        "  □ Does the fix break any ADJACENT path that was working before?\n"
        "  □ Are there edge cases (empty input, concurrent access, first-run state) "
        "where the fix fails?\n"
        "  □ Does the fix require state reset or re-initialisation somewhere "
        "(e.g. an Event that must be cleared before reuse)?\n"
        "  □ Is the fix symmetric — if setup changed, is teardown also updated?\n\n"
        "Only after this re-trace is complete, state the fix with confidence.\n\n"
        "Final output format:\n"
        "  BUG: [one sentence description]\n"
        "  CAUSE: [mechanism that produced it]\n"
        "  FIX: [exact change, location, reason]\n"
        "  VERIFIED: [what you re-traced to confirm correctness]"
    )


def _synthesise_investigation(problem: str, phases: list[str]) -> str:
    """Combine all phases into a final actionable report."""
    return (
        f"\n{'═' * 64}\n"
        f"## INVESTIGATION COMPLETE\n"
        f"{'═' * 64}\n\n"
        f"Problem: {problem}\n\n"
        "The methodology applied:\n"
        "  Read → Map → Trace → Separate → Find all → Minimum fix → Verify\n\n"
        "This approach guarantees:\n"
        "  • No conclusion without evidence\n"
        "  • No partial fix presented as complete\n"
        "  • No unnecessary code changed\n"
        "  • No fix committed without verification\n\n"
        "Apply the phases above in order. Each phase's output is the input to the next.\n"
        "Do not skip Phase 1 (reading). Do not stop at Phase 4 (first cause). "
        "Do not skip Phase 7 (verification)."
    )


# ── Skill class ────────────────────────────────────────────────────────────────

from skills.base_skill import BaseSkill


class SurgicalDebugSkill(BaseSkill):
    """
    Evidence-first investigation skill.

    Encodes the exact methodology used to diagnose:
      - Why 'reasoning' disappeared and became 'thinking' (race condition)
      - Why cold start took 199 seconds (wrong prewarm entry point)

    Use for: bugs, unexpected behaviour, 'why does X do Y' questions,
    performance problems, race conditions, wrong output.
    """

    @property
    def name(self) -> str:
        return "surgical_debug"

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
                    "description": (
                        "Describe the unexpected behaviour. Include: what you observe, "
                        "what you expected, and any error messages or timing details."
                    ),
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "default": 7,
                    "description": (
                        "Number of investigation phases to run (1–7). "
                        "Default 7 = full investigation. "
                        "Use 3 for quick triage, 5 for most bugs, 7 for deep/concurrent issues."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": "Optional: paste relevant code, logs, or error output here.",
                },
            },
            "required": ["problem"],
        }

    @property
    def cache_results(self) -> bool:
        return False  # bugs are context-specific; never cache

    def execute_impl(self, problem: str, **kwargs) -> str:
        depth   = min(int(kwargs.get("depth", 7)), 7)
        context = kwargs.get("context", "")

        # Augment problem with context if provided
        full_problem = problem
        if context:
            full_problem = f"{problem}\n\nContext provided:\n```\n{context[:3000]}\n```"

        # Run phases up to requested depth
        phase_fns = [
            _phase_read_first,
            _phase_architecture_map,
            _phase_trace_data_flow,
            _phase_separate_symptom_cause,
            _phase_find_all_bugs,
            _phase_minimum_intervention,
            _phase_verify,
        ]

        phase_labels = [
            "Read Before Reasoning",
            "Architecture Map",
            "Trace Data Flow",
            "Separate Symptom from Cause",
            "Find All Bugs",
            "Minimum Intervention",
            "Verify by Re-Tracing",
        ]

        sections = [
            f"# Surgical Debug Investigation\n\n"
            f"**Problem:** {problem}\n\n"
            f"**Phases:** {depth}/7 — {', '.join(phase_labels[:depth])}\n\n"
            f"{'─' * 64}"
        ]

        phases_output = []
        for i, fn in enumerate(phase_fns[:depth]):
            output = fn(full_problem)
            sections.append(output)
            phases_output.append(output)

        sections.append(_synthesise_investigation(problem, phases_output))

        return "\n\n".join(sections)
