"""
Skill: mode_switch
Adaptive cognitive mode orchestrator.

The gap this fills:
  - skill_router    → picks ONE skill, statically, before execution starts
  - reason_chain    → chains skills in a fixed sequence decided upfront
  - mode_switch     → detects mode from question structure, runs primary skill,
                      reads the OUTPUT for mode-shift signals, and hands off
                      to the next appropriate skill automatically with full context.

The 5 cognitive modes:
  INVESTIGATE  "why does X do Y / something is broken / unexpected behaviour"
               → read first, trace, separate symptom/cause, minimum fix
               → skills: surgical_debug, abduct

  CAUSAL       "what causes X / why did this happen / effects of Y"
               → build cause-effect chain, forward AND backward
               → skills: causal_reason, causal_forward_reason

  PLAN         "how should I do X / design this / what are the steps"
               → decompose goal, sequence dependencies, identify failure modes
               → skills: recursive_decompose, timeline_reason, multi_objective

  EVALUATE     "which is better / should I do X or Y / assess this"
               → establish criteria, score options, surface tradeoffs, verdict
               → skills: multi_objective, epistemic_reason, bayes_reason

  GENERATE     "create / write / design from scratch / make"
               → avoid obvious first move, surprise without losing accuracy
               → skills: lateral_thinking, cot_reason

Mode-shift signals (detected mid-execution in skill output):
  Output contains "why does / what causes / broken / unexpected"  → INVESTIGATE
  Output contains "how should / steps / depends on / sequence"    → PLAN
  Output contains "better / trade-off / criteria / which"         → EVALUATE
  Output contains "if X then / consequence / what happens if"     → CAUSAL
  Output contains "generate / create / draft / write"             → GENERATE
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from skills.base_skill import BaseSkill

DESCRIPTION = (
    "Adaptive cognitive mode orchestrator. Detects the reasoning mode required "
    "(investigate / causal / plan / evaluate / generate), runs the primary skill, "
    "then scans the output for mode-shift signals and hands off automatically. "
    "Use when a problem may require switching reasoning styles mid-execution — "
    "most real problems do."
)


# ── Mode taxonomy ──────────────────────────────────────────────────────────────

class Mode(Enum):
    INVESTIGATE = "investigate"
    CAUSAL      = "causal"
    PLAN        = "plan"
    EVALUATE    = "evaluate"
    GENERATE    = "generate"
    UNKNOWN     = "unknown"


@dataclass
class ModeSignal:
    mode:       Mode
    confidence: float          # 0.0–1.0
    triggers:   list[str]      # which keywords fired
    skills:     list[str]      # primary skill(s) for this mode


@dataclass
class ExecutionStep:
    mode:       Mode
    skill:      str
    output:     str
    shift_to:   Optional[Mode]  # None = no shift detected
    shift_why:  str             # reason for shift, or ""


# ── Mode detection ─────────────────────────────────────────────────────────────

_MODE_PATTERNS: list[tuple[Mode, list[str], list[str]]] = [
    # (mode, trigger_phrases, primary_skills)
    (
        Mode.INVESTIGATE,
        [
            "why does", "why is it", "why doesn't", "what's wrong",
            "broken", "unexpected", "not working", "bug", "error",
            "disappears", "fails", "wrong output", "weird behaviour",
            "how do i fix", "debug", "investigate",
        ],
        ["surgical_debug", "abduct"],
    ),
    (
        Mode.CAUSAL,
        [
            "what causes", "what caused", "why did", "root cause", "what led to",
            "effect of", "consequence", "because of", "resulted in",
            "what happens when", "impact of", "5 why", "fishbone",
            "counterfactual", "if we hadn't", "why was", "why is it",
            "what made", "what triggered", "how did it end up",
        ],
        ["causal_reason", "causal_forward_reason"],
    ),
    (
        Mode.PLAN,
        [
            "how should i", "how do i", "what are the steps", "design",
            "plan", "roadmap", "build", "implement", "structure",
            "architecture", "approach", "strategy for", "how to",
            "what order", "sequence of", "dependencies",
        ],
        ["recursive_decompose", "timeline_reason", "multi_objective"],
    ),
    (
        Mode.EVALUATE,
        [
            "which is better", "should i use", "compare", "versus", " vs ",
            "trade-off", "pros and cons", "best option", "recommend",
            "assess", "evaluate", "criteria", "rank", "choose between",
            "is it worth", "should i",
        ],
        ["multi_objective", "epistemic_reason"],
    ),
    (
        Mode.GENERATE,
        [
            "create", "write", "generate", "make", "draft", "design from scratch",
            "come up with", "invent", "produce", "build a new",
            "write me", "give me a", "make me",
        ],
        ["lateral_thinking", "cot_reason"],
    ),
]


def _detect_mode(text: str) -> ModeSignal:
    """Score the text against all mode patterns, return best match."""
    low = text.lower()
    best_mode     = Mode.UNKNOWN
    best_score    = 0.0
    best_triggers = []
    best_skills   = ["deep_reason"]

    for mode, triggers, skills in _MODE_PATTERNS:
        fired = [t for t in triggers if t in low]
        score = len(fired) / max(len(triggers), 1)
        if score > best_score:
            best_score    = score
            best_mode     = mode
            best_triggers = fired
            best_skills   = skills

    # Fallback: if nothing matched, default to PLAN (most general)
    if best_mode == Mode.UNKNOWN:
        best_mode   = Mode.PLAN
        best_skills = ["recursive_decompose"]
        best_score  = 0.1

    return ModeSignal(
        mode       = best_mode,
        confidence = min(best_score * 3.0, 1.0),  # normalise to 0-1
        triggers   = best_triggers,
        skills     = best_skills,
    )


# ── Mid-execution shift detection ──────────────────────────────────────────────

_SHIFT_SIGNALS: list[tuple[Mode, list[str]]] = [
    (Mode.INVESTIGATE, [
        "why does", "broken", "unexpected", "not working", "bug", "error",
        "failed", "wrong", "doesn't work", "incorrect output",
    ]),
    (Mode.CAUSAL, [
        "what caused", "root cause", "because", "led to", "resulted in",
        "consequence", "what happens if", "downstream effect",
    ]),
    (Mode.PLAN, [
        "next step", "how should", "depends on", "prerequisite", "sequence",
        "before we can", "in order to", "first we need", "plan",
    ]),
    (Mode.EVALUATE, [
        "trade-off", "which is better", "criteria", "on the other hand",
        "however", "alternatively", "option a", "option b", "compare",
    ]),
    (Mode.GENERATE, [
        "generate", "create", "write", "draft", "produce", "design",
    ]),
]


def _detect_shift(output: str, current_mode: Mode) -> tuple[Optional[Mode], str]:
    """
    Scan skill output for signals that the problem has changed type.
    Returns (new_mode, reason) or (None, "") if no shift detected.
    """
    low = output.lower()
    scores: dict[Mode, list[str]] = {}

    for mode, signals in _SHIFT_SIGNALS:
        if mode == current_mode:
            continue  # don't shift back to same mode
        fired = [s for s in signals if s in low]
        if fired:
            scores[mode] = fired

    if not scores:
        return None, ""

    # Pick the mode with most signals fired
    best = max(scores.items(), key=lambda x: len(x[1]))
    new_mode, triggers = best

    # Require at least 2 signals to shift (reduces false positives)
    if len(triggers) < 2:
        return None, ""

    return new_mode, f"Output contains mode-shift signals: {triggers}"


# ── Skill runner ───────────────────────────────────────────────────────────────

def _run_skill(skill_name: str, problem: str, context: str, depth: int) -> tuple[str, str]:
    """Run a single skill. Returns (output, error)."""
    try:
        from skills import SKILL_REGISTRY
        SkillClass = SKILL_REGISTRY.get(skill_name)
        if SkillClass is None:
            return "", f"Skill '{skill_name}' not found in registry."
        skill  = SkillClass()
        result = skill.execute(problem, depth=depth, context=context)
        return (result or "").strip(), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


def _mode_label(mode: Mode) -> str:
    labels = {
        Mode.INVESTIGATE: "🔍 INVESTIGATE",
        Mode.CAUSAL:      "⛓  CAUSAL",
        Mode.PLAN:        "📐 PLAN",
        Mode.EVALUATE:    "⚖️  EVALUATE",
        Mode.GENERATE:    "✨ GENERATE",
        Mode.UNKNOWN:     "❓ UNKNOWN",
    }
    return labels.get(mode, str(mode))


# ── Main synthesis ─────────────────────────────────────────────────────────────

def _synthesise(problem: str, steps: list[ExecutionStep]) -> str:
    if not steps:
        return "No steps executed."

    successful = [s for s in steps if s.output]
    if not successful:
        return "All steps failed — no output to synthesise."

    # Build a mode transition summary
    transitions = []
    for i, step in enumerate(steps):
        label = _mode_label(step.mode)
        transitions.append(f"  {i+1}. {label} → `{step.skill}`")
        if step.shift_to:
            transitions.append(
                f"     ↳ shift detected ({step.shift_why[:60]})"
            )

    # Extract final conclusions from each step
    conclusions = []
    for step in successful:
        paras = [p.strip() for p in step.output.split("\n\n") if p.strip()]
        if paras:
            last = paras[-1][:400]
            conclusions.append(f"[{_mode_label(step.mode)} / {step.skill}]\n{last}")

    mode_path = " → ".join(_mode_label(s.mode) for s in steps)

    return "\n\n".join([
        "## Mode-Switch Synthesis",
        f"**Problem:** {problem[:120]}",
        f"**Mode path:** {mode_path}",
        "",
        "**Execution trace:**",
        "\n".join(transitions),
        "",
        "**Conclusions per mode:**",
        "\n\n---\n\n".join(conclusions),
        "",
        "**Integration:** The answer lives at the intersection of the modes above. "
        "Where modes converge on the same point, treat it as high-confidence. "
        "Where they diverge, the problem is genuinely ambiguous — flag it explicitly.",
    ])


# ── Skill class ────────────────────────────────────────────────────────────────

class ModeSwitchSkill(BaseSkill):
    """
    Adaptive cognitive mode orchestrator.

    Unlike skill_router (picks one skill upfront) and reason_chain
    (fixed sequence), mode_switch:
      1. Detects the cognitive mode from question structure
      2. Runs the primary skill for that mode
      3. Scans the output for mode-shift signals
      4. If a shift is detected, hands off to the next skill with full context
      5. Repeats until no shift detected or max_switches reached

    When to use:
      - You don't know what type of problem this is
      - The problem is compound (e.g. "why is this broken AND how do I fix it")
      - A previous skill's answer revealed a new question
      - Any problem longer than ~2 sentences

    When NOT to use:
      - You know exactly which skill you need → call it directly
      - Simple single-type questions → skill_router is faster
    """

    @property
    def name(self) -> str:
        return "mode_switch"

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
                        "The full problem. Include everything — context, constraints, "
                        "what you've tried. The mode detector reads the STRUCTURE of "
                        "the question, not just keywords."
                    ),
                },
                "depth": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 7,
                    "default": 3,
                    "description": "Reasoning depth passed to each skill.",
                },
                "max_switches": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4,
                    "default": 2,
                    "description": (
                        "Max number of mode switches allowed. "
                        "1 = run primary mode only. "
                        "2 = allow one handoff. "
                        "4 = full adaptive chain."
                    ),
                },
                "force_mode": {
                    "type": "string",
                    "enum": ["investigate", "causal", "plan", "evaluate", "generate"],
                    "description": "Override auto-detection and start in a specific mode.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional: paste code, logs, or prior output here.",
                },
            },
            "required": ["problem"],
        }

    @property
    def cache_results(self) -> bool:
        return False

    def execute_impl(self, problem: str, **kwargs) -> str:
        depth        = int(kwargs.get("depth", 3))
        max_switches = int(kwargs.get("max_switches", 2))
        force_mode   = kwargs.get("force_mode", "")
        context      = kwargs.get("context", "")

        # ── Step 1: Detect initial mode ────────────────────────────────────────
        if force_mode:
            try:
                initial_signal = ModeSignal(
                    mode       = Mode(force_mode),
                    confidence = 1.0,
                    triggers   = ["forced"],
                    skills     = next(
                        skills for mode, _, skills in _MODE_PATTERNS
                        if mode == Mode(force_mode)
                    ),
                )
            except (ValueError, StopIteration):
                initial_signal = _detect_mode(problem)
        else:
            initial_signal = _detect_mode(problem)

        lines = [
            "# Mode-Switch Orchestrator\n",
            f"**Problem:** {problem[:150]}{'...' if len(problem) > 150 else ''}",
            f"**Detected mode:** {_mode_label(initial_signal.mode)} "
            f"(confidence: {initial_signal.confidence:.0%})",
            f"**Triggers:** {initial_signal.triggers or ['default']}",
            f"**Max switches:** {max_switches}",
            "",
            "---",
            "",
        ]

        # ── Step 2: Execute loop with mode-shift detection ─────────────────────
        steps:        list[ExecutionStep] = []
        current_mode:  Mode              = initial_signal.mode
        current_skill: str               = initial_signal.skills[0]
        cumulative_context:  str         = context
        switches_used: int               = 0

        while switches_used <= max_switches:
            step_num = len(steps) + 1

            lines.append(
                f"## Step {step_num} — {_mode_label(current_mode)} "
                f"→ `{current_skill}`\n"
            )

            # Build context for this step
            step_context = cumulative_context
            if steps:
                # Pass prior output as context to next skill
                prior_output = steps[-1].output[:2000]
                step_context = (
                    f"Prior reasoning ({_mode_label(steps[-1].mode)}):\n"
                    f"{prior_output}\n\n"
                    f"{cumulative_context}"
                ).strip()

            # Run the skill
            output, error = _run_skill(current_skill, problem, step_context, depth)

            if error:
                lines.append(f"**Error:** {error}")
                lines.append("")
                # Try fallback skill for this mode
                signal = next(
                    (s for m, _, s in _MODE_PATTERNS if m == current_mode),
                    None,
                )
                fallback = signal[1] if signal and len(signal) > 1 else None
                if fallback and fallback != current_skill:
                    lines.append(f"*Retrying with fallback skill: `{fallback}`*\n")
                    output, error = _run_skill(fallback, problem, step_context, depth)
                    if error:
                        lines.append(f"**Fallback also failed:** {error}\n")
                        break
                    current_skill = fallback
                else:
                    break

            lines.append(output)
            lines.append("")

            # ── Step 3: Detect mode shift in output ────────────────────────────
            shift_mode, shift_why = _detect_shift(output, current_mode)

            step = ExecutionStep(
                mode      = current_mode,
                skill     = current_skill,
                output    = output,
                shift_to  = shift_mode,
                shift_why = shift_why,
            )
            steps.append(step)

            # Update cumulative context
            cumulative_context = (
                f"{cumulative_context}\n\n[{_mode_label(current_mode)}] {output[:1500]}"
            ).strip()

            # No shift → we're done
            if not shift_mode:
                lines.append(
                    f"*No mode shift detected — {_mode_label(current_mode)} "
                    f"mode resolved the problem.*\n"
                )
                break

            # Shift detected → hand off
            switches_used += 1
            if switches_used > max_switches:
                lines.append(
                    f"*Mode shift detected ({_mode_label(shift_mode)}) but "
                    f"max_switches={max_switches} reached. Stopping.*\n"
                )
                break

            lines.append(
                f"**⇄ Mode shift detected → {_mode_label(shift_mode)}**  \n"
                f"*Reason: {shift_why}*\n"
            )

            # Find skills for the new mode
            new_skills = next(
                (skills for mode, _, skills in _MODE_PATTERNS if mode == shift_mode),
                ["deep_reason"],
            )
            current_mode  = shift_mode
            current_skill = new_skills[0]

        # ── Step 4: Synthesise ─────────────────────────────────────────────────
        lines.append("---\n")
        lines.append(_synthesise(problem, steps))

        return "\n".join(lines)
