# -*- coding: utf-8 -*-
"""
skills/temporal_cognition.py — Temporal Cognition Skill
════════════════════════════════════════════════════════

Gives EVE a unified "sense of time" across three temporal directions:

  BACKWARD-LOOKING  ← actual episode history from the DB
    hindsight       — understand what just happened, in context of past
    retrospection   — deliberate review of a span of past sessions
    afterthought    — surface what was missed; late regrets

  FORWARD-LOOKING   ← pattern-projection from episode history
    foresight       — anticipate future needs / risks from past patterns
    foreknowledge   — what is already determined / scheduled / certain
    forethought     — deliberate pre-action planning with risk map

  INTUITIVE / SYNTHETIC
    precognition    — weak-signal pattern recognition across episodes
    retrocognition  — reconstruct a session from fragments (after interrupt/crash)

AUTO-FIRE (check() interface, same as goal_anchor):
    QueryEngine calls check() EVERY inner turn via _run_temporal_cognition().
    The method is fast — it only reads lightweight episode pattern counts and
    returns a non-empty string when a genuine alert exists. Silent otherwise.
    Registered in query_engine alongside goal_anchor and wakefulness.

DIRECT USE:
    {"skill": "temporal_cognition", "args": {"problem": "...", "mode": "foresight"}}
    mode is optional — auto-detected from problem keywords when omitted.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Optional

from skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from agent.session import Session

# ── Keyword → mode routing ────────────────────────────────────────────────────
_MODE_KEYWORDS: dict[str, list[str]] = {
    "hindsight":      ["hindsight","in retrospect","looking back","what happened","after all",
                       "now that","turned out","it's clear now","obviously in hindsight",
                       "power cut","crash","interrupted","what went wrong","post mortem"],
    "retrospection":  ["review","recap","what did we do","what have i done","last session",
                       "past week","history of","summarise what","summarize what","what was accomplished",
                       "remember when","what happened during","previous run","look back"],
    "afterthought":   ["should have","forgot to","missed","overlooked","in hindsight i should",
                       "late realisation","afterthought","didn't think of","wish i had",
                       "could have","regret","too late","realised after"],
    "foresight":      ["what could go wrong","anticipate","risk","might happen","before we",
                       "plan ahead","avoid","prevent","watch out","could fail","be careful",
                       "prepare for","heading","trajectory","if we continue","potential issue"],
    "foreknowledge":  ["already know","certain","scheduled","will happen","deterministic",
                       "guaranteed","definitely","known fact","predetermined","inevitable",
                       "committed","confirmed","fixed","deadline"],
    "forethought":    ["before i","before we","planning","think through","step by step plan",
                       "precaution","consider first","think ahead","pre-plan","deliberate",
                       "let me plan","map out","outline the steps","design","architect"],
    "precognition":   ["pattern","feel like","intuition","hunch","something tells me",
                       "usually when","tends to","historically","often","typically",
                       "weak signal","early sign","indicator","warning sign","leading indicator"],
    "retrocognition": ["reconstruct","piece together","figure out what happened","after the crash",
                       "pick up","resume","what was happening","where were we","lost context",
                       "restore","recover state","power came back","just restarted","back online"],
}

# ── Episode type → human label ─────────────────────────────────────────────────
_EP_LABELS = {
    "tool_error":    "🔧 Tool error",
    "token_limit":   "📊 Token limit",
    "crash":         "💥 Crash",
    "interrupted":   "⛔ Interrupted",
    "power_cut":     "⚡ Power cut",
    "error":         "✖ Error",
    "provider_fail": "☁ Provider fail",
    "session_start": "▶ Session start",
    "session_end":   "✓ Session end",
    "note":          "📝 Note",
}


# ── Episode log bridge (graceful — never crash if DB unavailable) ─────────────

def _ep_search(query: str, type_filter: str = "", limit: int = 6) -> str:
    try:
        from utils.episode_log import ep
        return ep.search(query, type_filter=type_filter, limit=limit)
    except Exception as e:
        return f"(episode log unavailable: {e})"


def _ep_recent(n: int = 10) -> list[dict]:
    """Return the n most recent episodes as plain dicts."""
    try:
        from utils.episode_log import ep
        from utils.episode_log import _conn
        with _conn() as c:
            rows = c.execute(
                "SELECT id,ts_human,type,title,severity,project FROM episodes "
                "ORDER BY ts DESC LIMIT ?", (n,)
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _ep_pattern_summary(window: int = 50) -> dict:
    """Count episode types in the last `window` episodes for pattern detection."""
    try:
        from utils.episode_log import _conn
        with _conn() as c:
            rows = c.execute(
                "SELECT type, COUNT(*) as n FROM ("
                "  SELECT type FROM episodes ORDER BY ts DESC LIMIT ?"
                ") GROUP BY type ORDER BY n DESC",
                (window,)
            ).fetchall()
        return {r["type"]: r["n"] for r in rows}
    except Exception:
        return {}


def _ep_last_session_state() -> dict:
    """Return info about the most recent incomplete session."""
    try:
        from utils.episode_log import _conn
        with _conn() as c:
            prev = c.execute(
                "SELECT id, started_at, ended_at, clean_exit, cwd, model "
                "FROM ep_sessions ORDER BY started_at DESC LIMIT 2"
            ).fetchall()
            if len(prev) < 2:
                return {}
            last = dict(prev[1])  # second most recent = previous
            if last.get("clean_exit"):
                return {}
            # Get last few episodes from that session
            eps = c.execute(
                "SELECT ts_human, type, title FROM episodes "
                "WHERE session_id=? ORDER BY ts DESC LIMIT 5",
                (last["id"],)
            ).fetchall()
        last["last_episodes"] = [dict(r) for r in eps]
        return last
    except Exception:
        return {}


# ── Mode reasoning templates ──────────────────────────────────────────────────

def _hindsight(problem: str, history: str) -> str:
    return f"""**HINDSIGHT ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Situation: {problem}

**What Hindsight Means Here**
Hindsight is *post-event clarity* — you now have information you lacked before.
The goal is to extract useful lessons, not assign blame.

**Step 1 — Reconstruct the event faithfully**
  - What was the original intention?
  - What actually happened?
  - At which exact point did the outcome diverge from intention?

**Step 2 — What information was missing beforehand?**
  - Was it unknowable, or merely unnoticed?
  - Would a hindsight-informed version of you have acted differently?

**Step 3 — Hindsight bias check**
  - Am I judging past-self by present knowledge?  (Hindsight bias)
  - Given what was known *then*, was the decision reasonable?
  - Adjust for information asymmetry before concluding.

**Step 4 — Lesson extraction (actionable)**
  - What concrete change would prevent this outcome?
  - Is this a one-off anomaly, or a recurring pattern?

**Relevant Episode History:**
{history}

**Output format:**
  Event:    [what happened]
  Missed:   [what was not seen in advance]
  Bias-adj: [was the past decision reasonable given info at the time?]
  Lesson:   [specific, actionable change]
"""


def _retrospection(problem: str, history: str) -> str:
    return f"""**RETROSPECTION ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Review period / context: {problem}

**What Retrospection Means Here**
Retrospection is *deliberate, structured review* — not casual glancing back.
The aim is a comprehensive, honest account of what was accomplished, attempted, and avoided.

**Review Dimensions:**

1. **Accomplishments** — What was completed? Verifiable outcomes only (tool success ✓).
2. **Attempts without completion** — What was started but not finished? Why?
3. **Errors & recoveries** — What went wrong, and how was it resolved (or not)?
4. **Decisions made** — Key choice points and the reasoning at the time.
5. **Knowledge gained** — What do I know now that I didn't before this period?
6. **Patterns across the period** — Recurring themes, bottlenecks, dependencies.

**Episode Log (most relevant to this review):**
{history}

**Synthesis prompt:**
  Given the above, answer:
  a) What was the single most impactful thing that happened?
  b) What would I do differently if I could repeat this period?
  c) What should carry forward into the next session as priority?
"""


def _afterthought(problem: str, history: str) -> str:
    return f"""**AFTERTHOUGHT ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context: {problem}

**What Afterthought Means Here**
An afterthought is something that *should have been considered* but wasn't —
a late addition, an overlooked dependency, a forgotten edge case.
Recognising afterthoughts prevents them becoming recurring blind spots.

**Afterthought Detection Framework:**

1. **What assumption did I make silently?**
   - Unspoken assumptions are the most common source of afterthoughts.
   - List every assumption made. Challenge each one.

2. **Who or what was not consulted?**
   - Was there a stakeholder, tool, file, or data source I ignored?
   - What would have changed if I had checked it first?

3. **Edge cases not considered:**
   - Empty input / null / zero-length
   - Permissions / sandboxing / path issues
   - Network / API unavailability
   - Token or rate limits
   - Interruption mid-operation

4. **The "two hours later" test:**
   - Pretend two hours have passed. What would I regret not having done?
   - What is now obvious that wasn't visible during execution?

5. **Capture it permanently:**
   - Log this afterthought as an episode so the same blind spot is searchable next time.

**Past afterthoughts / missed steps in similar contexts:**
{history}

**Resolution template:**
  Missed:     [what was overlooked]
  Impact:     [what would have happened / what actually happened because of it]
  Prevention: [what check/step to add before similar tasks in future]
  Logged:     [record as an episode with tag 'afterthought']
"""


def _foresight(problem: str, history: str, pattern: dict) -> str:
    top_errors = [k for k, v in sorted(pattern.items(), key=lambda x: -x[1])
                  if k in ("tool_error","crash","token_limit","interrupted","power_cut","provider_fail")]
    risk_profile = "\n".join(
        f"  {_EP_LABELS.get(t, t):22s}  occurred {pattern[t]}× in recent history"
        for t in top_errors
    ) or "  No significant error patterns detected in recent history."

    return f"""**FORESIGHT ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Situation ahead: {problem}

**What Foresight Means Here**
Foresight is *anticipatory awareness* — seeing around corners before you walk into them.
It combines pattern recognition from the past with forward projection.

**Risk Profile (from episode history):**
{risk_profile}

**Foresight Framework:**

1. **Pre-mortem** (assume failure, work backwards)
   - Imagine this task has failed. What was the most likely cause?
   - List the top 3 failure modes in order of probability.

2. **Dependency map**
   - What does this task depend on that I don't control?
     (Network, API quota, file permissions, model output quality, user input)
   - Which dependency is the weakest link?

3. **Cascade analysis**
   - If step N fails, which downstream steps break automatically?
   - Where are the natural recovery checkpoints?

4. **Resource limits**
   - Will this exceed token budget? → break into phases.
   - Will this make many tool calls? → add checkpointing.
   - Is there a time constraint? → identify the critical path.

5. **Mitigation plan**
   - For each top risk: what is the mitigation?
   - What is the fallback if the mitigation itself fails?

**Similar past situations (episode log):**
{history}

**Foresight output format:**
  Risk 1: [most likely failure] — Mitigation: [...]
  Risk 2: [second likely failure] — Mitigation: [...]
  Risk 3: [third likely failure] — Mitigation: [...]
  Checkpoint: [where to save state before the riskiest step]
"""


def _foreknowledge(problem: str, history: str) -> str:
    return f"""**FOREKNOWLEDGE INVENTORY**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context: {problem}

**What Foreknowledge Means Here**
Foreknowledge is *pre-existing certainty* about what will happen —
not prediction, but inventory of what is already determined or known.

**Categories of Foreknowledge:**

1. **Deterministic facts** (will happen regardless of my actions)
   - Known API limits / rate limits for the providers in use
   - File system constraints (paths, permissions, encoding)
   - Python version / library availability on this machine
   - Windows-specific behaviours vs Linux assumptions

2. **Scheduled / committed** (will happen at a known time)
   - Any cron tasks, background jobs, or scheduled events
   - User-stated deadlines or "by tonight" type constraints
   - Pending tasks recorded in the journal / task list

3. **Pattern-certain** (virtually guaranteed from history)
   - If we've hit Gemini quota 3× in the last hour, we'll hit it again soon
   - If the last 3 bash calls failed on this path, the path is wrong
   - If the model needed 2× the expected tokens last time, plan for that now

4. **User-stated intent** (foreknowledge about user behaviour)
   - What has the user explicitly said they want?
   - What preferences / constraints have been stated across sessions?

**Known relevant episodes:**
{history}

**Output format:**
  Certain:   [list of facts you already know will be true]
  Scheduled: [any time-bound commitments or tasks]
  Pattern:   [near-certain outcomes based on recurrence]
  Unknown:   [what remains genuinely uncertain — be honest]
"""


def _forethought(problem: str, history: str, pattern: dict) -> str:
    top_errors = [k for k, v in sorted(pattern.items(), key=lambda x: -x[1])
                  if k in ("tool_error","crash","token_limit","interrupted","provider_fail")]
    risk_line = (
        "Past error distribution: " + ", ".join(f"{k}×{pattern[k]}" for k in top_errors)
        if top_errors else "No significant error history — still apply precautions."
    )

    return f"""**FORETHOUGHT PLANNING**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Task to plan: {problem}

**What Forethought Means Here**
Forethought is *deliberate anticipatory planning before acting*.
Not just "think before you leap" — a structured pre-flight checklist.

**{risk_line}**

**Forethought Checklist (run this before every significant action):**

□ 1. **Clarify the goal precisely**
     - What is the exact desired end state? (Not "make it work" — what does working look like?)
     - How will I verify success? (Tool output, file existence, test run, user confirmation?)

□ 2. **Identify the reversibility of each step**
     - Reversible actions: safe to try immediately.
     - Irreversible actions (delete, overwrite, send, deploy): require a backup or dry-run first.

□ 3. **Sequence and dependencies**
     - List steps in dependency order (nothing depends on a step before it exists).
     - Which step is the critical path? (Longest or riskiest chain.)

□ 4. **Resource budget**
     - Estimated tool calls: [N]
     - Estimated token cost: [high/medium/low]
     - Risk of hitting quota before completion → if high, checkpoint early.

□ 5. **Interrupt resilience**
     - If stopped at step K, what state is left on disk?
     - Can the task be resumed from that state?
     - Checkpoint: write progress to file before the riskiest step.

□ 6. **Edge case pre-emption**
     - Empty / missing inputs handled?
     - Paths platform-agnostic (Windows backslash)?
     - Encoding (UTF-8 BOM on Windows)?
     - File locks / process conflicts?

**Similar past tasks (for reference):**
{history}

**Plan output format:**
  Goal:         [exact success criterion]
  Steps:        [ordered, each with reversibility flag ✓/✗]
  Riskiest:     [step N — why — mitigation]
  Checkpoint:   [after step N, write state to: ...]
  Verify:       [how to confirm success at end]
"""


def _precognition(problem: str, history: str, pattern: dict) -> str:
    # Surface the most suspicious pattern combinations
    signals = []
    if pattern.get("token_limit", 0) >= 2:
        signals.append(f"⚠ Quota exhaustion pattern: {pattern['token_limit']}× token limits recently — next long inference likely to hit again.")
    if pattern.get("tool_error", 0) >= 3:
        signals.append(f"⚠ Tool failure cluster: {pattern['tool_error']}× tool errors — likely a systemic issue (path, permission, or API).")
    if pattern.get("interrupted", 0) + pattern.get("power_cut", 0) >= 2:
        signals.append(f"⚠ Interruption recurrence: {pattern.get('interrupted',0)+pattern.get('power_cut',0)}× unclean exits — consider more frequent checkpointing.")
    if pattern.get("crash", 0) >= 1:
        signals.append(f"⚠ Recent crash detected — the same code path may crash again without a fix.")
    if not signals:
        signals.append("✓ No strong negative patterns detected in recent episode history.")

    signals_str = "\n".join(signals)

    return f"""**PRECOGNITION — PATTERN RECOGNITION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Context: {problem}

**What Precognition Means Here**
Precognition (in the rational, non-mystical sense) is *weak-signal pattern recognition*:
detecting the early indicators of a likely outcome before it becomes obvious.
It is not magic — it is statistics applied to experience.

**Detected Signals from Episode History:**
{signals_str}

**Pattern Recognition Framework:**

1. **Recurrence check**
   - Has this exact situation occurred before?
   - If yes: what was the outcome every time?
   - Base rate: [outcome A] happened [N/M] times in similar contexts.

2. **Cluster detection**
   - Are multiple small anomalies clustering around the same component/tool/path?
   - Three independent failures near the same thing = not a coincidence.

3. **Temporal patterns**
   - Does this tend to fail at a certain point in execution (early/late/under load)?
   - Is there a time-of-day or usage-intensity correlation?

4. **Weak signals**
   - What small, easy-to-ignore signs precede the main event?
   - (e.g. slower responses before a token limit, a 503 before a full outage)

5. **Confidence calibration**
   - How confident am I in this pattern? (0–100%)
   - What would falsify it? (What outcome would prove me wrong?)

**Relevant past episodes:**
{history}

**Precognitive output format:**
  Signal:     [what weak indicator is visible now]
  Pattern:    [the historical recurrence that supports concern]
  Prediction: [what is likely to happen if unchecked]
  Confidence: [0–100%]
  Action:     [what to do NOW, before it materialises]
"""


def _retrocognition(problem: str, session_state: dict) -> str:
    last_eps = session_state.get("last_episodes", [])
    eps_str = "\n".join(
        f"  {r['ts_human']}  [{r['type']}]  {r['title']}"
        for r in last_eps
    ) or "  (no episode records found for previous session)"

    cwd   = session_state.get("cwd", "unknown")
    model = session_state.get("model", "unknown")
    clean = session_state.get("clean_exit", True)

    status = "UNCLEAN EXIT (crash / power cut / kill)" if not clean else "clean exit"

    return f"""**RETROCOGNITION — SESSION RECONSTRUCTION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reconstructing: {problem}

**Previous Session Summary:**
  Status:   {status}
  Project:  {cwd}
  Model:    {model}

**Last recorded events in that session:**
{eps_str}

**What Retrocognition Means Here**
Retrocognition is *reconstructing the past from available fragments*
when direct memory is absent. Like forensic archaeology — piece together
what happened from traces left behind.

**Reconstruction Protocol:**

1. **Ground truth from episode log** (highest reliability)
   - What episodes were recorded? (tool errors, steps, notes)
   - What was the last successful tool call before the interruption?
   - Was there an in-progress write/download/execution?

2. **File system evidence**
   - Check for partial files, temp files, `.part`, `_incomplete`, `.lock`
   - Check output directories for timestamps
   - Check log files, if any

3. **Process state inference**
   - If it was a power cut: assume nothing was committed after the last journal write.
   - If it was a CTRL+C: the in-flight tool call was aborted; prior steps likely completed.
   - If it was a crash (unhandled exception): check the episode log for the exception type.

4. **Reconstruct intent from context**
   - What was the last user message / goal before the interruption?
   - What tool was in flight (from episode log 'last_tool')?
   - Can the interrupted operation be resumed, or must it be restarted?

5. **Safety check before resuming**
   - Is there a partial write that would corrupt a file if resumed?
   - Were any external side effects already committed (email sent, API called, file deleted)?

**Reconstruction output format:**
  Last known state:  [what was in progress]
  Evidence from:     [episode log / filesystem / process]
  Side effects committed: [yes/no — list]
  Safe to resume:    [yes/no — if not, explain why]
  Next step:         [first action to recover/continue correctly]
"""


# ── Mode selector ─────────────────────────────────────────────────────────────

def _detect_mode(problem: str) -> str:
    low = problem.lower()
    scores: dict[str, int] = {}
    for mode, keywords in _MODE_KEYWORDS.items():
        scores[mode] = sum(1 for kw in keywords if kw in low)
    best = max(scores, key=lambda m: scores[m])
    return best if scores[best] > 0 else "forethought"   # default: plan before acting


# ── Main skill class ──────────────────────────────────────────────────────────

class TemporalCognitionSkill(BaseSkill):
    """
    Unified temporal intelligence: hindsight, retrospection, afterthought,
    foresight, foreknowledge, forethought, precognition, retrocognition.
    Auto-fires EVERY turn via check() — registered in QueryEngine alongside
    goal_anchor and wakefulness. Silent when nothing warrants attention.
    Queries the episode log to ground all reasoning in real history.
    """

    name = "temporal_cognition"
    description = (
        "Temporal cognition — reasons across all time directions using real episode history. "
        "Backward: hindsight / retrospection / afterthought. "
        "Forward: foresight / foreknowledge / forethought. "
        "Intuitive: precognition (pattern signals) / retrocognition (session reconstruction). "
        "Use after any interruption, crash, or quota hit. Use before any risky or long task. "
        "Accepts mode= to target a specific lens, or auto-detects from the problem."
    )

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The situation, event, or task to reason about.",
                },
                "mode": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "hindsight", "retrospection", "afterthought",
                        "foresight", "foreknowledge", "forethought",
                        "precognition", "retrocognition",
                    ],
                    "description": (
                        "Which temporal lens to apply. "
                        "Omit or set 'auto' to detect from the problem text. "
                        "hindsight/retrospection/afterthought = past-facing. "
                        "foresight/foreknowledge/forethought = future-facing. "
                        "precognition = pattern signals. retrocognition = session reconstruction."
                    ),
                },
                "search_query": {
                    "type": "string",
                    "description": "Optional: override the episode log search query.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max episodes to retrieve (default 6).",
                },
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        mode         = (kwargs.get("mode") or "auto").strip().lower()
        search_q     = kwargs.get("search_query") or problem
        limit        = int(kwargs.get("limit") or 6)

        if mode == "auto":
            mode = _detect_mode(problem)

        # Fetch history and patterns
        history = _ep_search(search_q, limit=limit)
        pattern = _ep_pattern_summary(window=50)

        if mode == "hindsight":
            return _hindsight(problem, history)
        elif mode == "retrospection":
            return _retrospection(problem, history)
        elif mode == "afterthought":
            return _afterthought(problem, history)
        elif mode == "foresight":
            return _foresight(problem, history, pattern)
        elif mode == "foreknowledge":
            return _foreknowledge(problem, history)
        elif mode == "forethought":
            return _forethought(problem, history, pattern)
        elif mode == "precognition":
            return _precognition(problem, history, pattern)
        elif mode == "retrocognition":
            state = _ep_last_session_state()
            return _retrocognition(problem, state)
        else:
            return _forethought(problem, history, pattern)  # safe default

    # ── Auto-fire hook (called by QueryEngine EVERY turn via _run_temporal_cognition) ──

    def check(
        self,
        goal:         str,
        tool_history: list[str],
        session:      "Session",
        turn:         int,
    ) -> str:
        """
        Runs automatically on EVERY inner turn (called by _run_temporal_cognition).
        Pattern detection is lightweight — only reads aggregate counts from the
        episode DB; no model calls, no I/O beyond that.
        Returns a non-empty string (temporal alert) only when something genuinely
        warrants attention. Silent otherwise — never noisy.
        """
        pattern = _ep_pattern_summary(window=20)
        alerts:  list[str] = []

        # ── Alert: quota/token pattern ─────────────────────────────────────
        tl = pattern.get("token_limit", 0)
        if tl >= 2:
            alerts.append(
                f"[temporal:precognition] ⚠ Token/quota limit has been hit {tl}× recently. "
                f"Consider checkpointing or splitting the current task before it hits again."
            )

        # ── Alert: recent unclean exit ─────────────────────────────────────
        if pattern.get("power_cut", 0) or pattern.get("interrupted", 0):
            n = pattern.get("power_cut", 0) + pattern.get("interrupted", 0)
            alerts.append(
                f"[temporal:retrocognition] ⚡ {n} unclean exit(s) in recent history. "
                f"Checkpoint progress to disk frequently in this session."
            )

        # ── Alert: crash ──────────────────────────────────────────────────
        if pattern.get("crash", 0):
            alerts.append(
                f"[temporal:hindsight] 💥 {pattern['crash']} crash(es) recorded recently. "
                f"Verify the fix was applied before repeating the same code path."
            )

        # ── Alert: tool error cluster ──────────────────────────────────────
        te = pattern.get("tool_error", 0)
        if te >= 4:
            alerts.append(
                f"[temporal:foresight] 🔧 {te} tool errors in recent history — "
                f"likely a systemic issue. Run skill temporal_cognition (foresight) before next tool call."
            )

        return "\n".join(alerts)