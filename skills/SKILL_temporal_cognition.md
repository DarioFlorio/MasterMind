---
name: temporal_cognition
file: skills/temporal_cognition.py
class: TemporalCognitionSkill
auto_fire: true
registered_in: agent/query_engine.py → _run_temporal_cognition()
fires_every: every inner turn (unconditional)
direct_invoke: '{"skill": "temporal_cognition", "args": {"problem": "...", "mode": "foresight"}}'
episode_log: required (gracefully degrades if unavailable)
---

# Temporal Cognition Skill

Gives EVE a unified **sense of time** — reasoning backwards into history,
forwards into the future, and laterally across patterns, using the real
episode log as its ground truth.

---

## How It Works: Two Entry Points

### 1. Auto-fire (always-on, orchestrator level)

`QueryEngine._run_temporal_cognition(inner)` is called **every inner turn**,
unconditionally, alongside `_run_goal_anchor` and `_run_wakefulness`.

```
main loop (query_engine.py)
  │
  ├─ _run_goal_anchor(inner)          ← drift detection
  ├─ _run_temporal_cognition(inner)   ← temporal awareness  ← THIS SKILL
  ├─ _goal_tracker.tick()
  └─ … (model call, tools, etc.)
```

The method calls `TemporalCognitionSkill().check(...)` which:

1. Reads `_ep_pattern_summary(window=20)` — a lightweight aggregate count
   from the episode DB (no model calls, no network I/O).
2. Checks four alert conditions (see below).
3. Returns a non-empty alert string **only when a genuine pattern warrants
   attention**. Otherwise returns `""` and is completely silent.
4. If non-empty, QueryEngine injects it as a tool result:
   ```xml
   <tool_result>
     <n>temporal_cognition</n>
     <o>[TemporalCognition] ⚠ …alert text…</o>
   </tool_result>
   ```

**Cost:** Near-zero. One SQLite aggregate query per turn. No model call.
**Noise policy:** Silent by default. Only fires when thresholds are crossed.

#### The four auto-fire alert conditions

| Condition | Threshold | Alert tag |
|---|---|---|
| Token/quota exhaustion | ≥ 2 `token_limit` episodes in last 20 | `[temporal:precognition]` |
| Unclean exits (crash / power cut) | ≥ 1 | `[temporal:retrocognition]` |
| Recent crash | ≥ 1 `crash` episode | `[temporal:hindsight]` |
| Tool error cluster | ≥ 4 `tool_error` episodes in last 20 | `[temporal:foresight]` |

To tune these thresholds, edit the constants inside `check()` in
`skills/temporal_cognition.py`.

---

### 2. Direct invocation (on demand, via skill tool)

Call it like any other skill:

```json
{"skill": "temporal_cognition", "args": {"problem": "We're about to run a long download + parse pipeline. What could go wrong?", "mode": "foresight"}}
```

`mode` is optional. When omitted or set to `"auto"`, the mode is
auto-detected from keywords in `problem` (see Mode Reference below).

**Direct invocation returns a full structured reasoning template** — not
just an alert flag. Use this when you want deep analysis, not just a check.

---

## Mode Reference

Eight temporal lenses. Each produces a different reasoning template,
grounded in episode log history fetched via `_ep_search()`.

### BACKWARD-LOOKING (past → present)

#### `hindsight`
**When:** After something went wrong or turned out differently than expected.
Triggered by: "in retrospect", "what happened", "crash", "power cut",
"what went wrong", "post mortem".

**What it does:** Reconstructs the event faithfully, identifies what
information was missing beforehand, applies hindsight bias correction
(was the past decision reasonable given what was known *then*?), and
extracts a specific actionable lesson.

**Output format:**
```
Event:    [what happened]
Missed:   [what was not seen in advance]
Bias-adj: [was the past decision reasonable given info at the time?]
Lesson:   [specific, actionable change]
```

---

#### `retrospection`
**When:** Deliberate review of a span of past work — "what did we do",
"recap", "what was accomplished", "last session", "summarise what".

**What it does:** Structured multi-dimensional review: accomplishments,
incomplete attempts, errors & recoveries, key decisions, knowledge gained,
and recurring patterns across the period.

**Output format:** Answers three synthesis questions:
- What was the single most impactful thing that happened?
- What would I do differently if I could repeat this period?
- What should carry forward into the next session as priority?

---

#### `afterthought`
**When:** Something was missed, forgotten, or only obvious after the fact.
Triggered by: "should have", "forgot to", "missed", "overlooked",
"wish I had", "could have", "too late".

**What it does:** Surfaces silent assumptions, identifies unconsulted
stakeholders/tools/data, enumerates unconsidered edge cases, applies the
"two hours later" test (what would I regret not having done?), and prompts
logging the blind spot as an episode.

**Output format:**
```
Missed:     [what was overlooked]
Impact:     [what happened / would have happened because of it]
Prevention: [check/step to add before similar tasks in future]
Logged:     [record as episode with tag 'afterthought']
```

---

### FORWARD-LOOKING (present → future)

#### `foresight`
**When:** About to do something risky, long, or multi-step.
Triggered by: "what could go wrong", "anticipate", "risk", "might happen",
"prevent", "heading", "trajectory", "potential issue".

**What it does:** Runs a pre-mortem (assume failure, work backwards),
maps dependencies you don't control, traces failure cascades, assesses
resource limits (tokens, tool calls), and builds a ranked mitigation plan.

Uses `_ep_pattern_summary` to populate a live risk profile from recent
error history (quota hits, tool errors, crashes, provider failures).

**Output format:**
```
Risk 1: [most likely failure] — Mitigation: [...]
Risk 2: [second likely failure] — Mitigation: [...]
Risk 3: [third likely failure] — Mitigation: [...]
Checkpoint: [where to save state before the riskiest step]
```

---

#### `foreknowledge`
**When:** Need to inventory what is already certain before acting.
Triggered by: "already know", "certain", "scheduled", "will happen",
"guaranteed", "deadline", "committed", "confirmed".

**What it does:** Catalogues deterministic facts (API limits, filesystem
constraints, platform behaviours), scheduled/committed events, pattern-
certain outcomes (things that are virtually guaranteed from history), and
user-stated constraints.

**Output format:**
```
Certain:   [list of facts you already know will be true]
Scheduled: [any time-bound commitments or tasks]
Pattern:   [near-certain outcomes based on recurrence]
Unknown:   [what remains genuinely uncertain — be honest]
```

---

#### `forethought`
**When:** About to begin a significant task — the default mode when no
keywords match anything else.
Triggered by: "planning", "before we", "step by step", "outline the steps",
"map out", "architect", "design", "precaution".

**What it does:** A full pre-flight checklist: clarify the exact success
criterion, flag irreversible steps (delete/overwrite/send/deploy),
sequence steps in dependency order, budget token and tool-call cost,
assess interrupt resilience (what state is left on disk at step K?),
and pre-empt edge cases.

**Output format:**
```
Goal:       [exact success criterion]
Steps:      [ordered, each with reversibility flag ✓/✗]
Riskiest:   [step N — why — mitigation]
Checkpoint: [after step N, write state to: ...]
Verify:     [how to confirm success at end]
```

---

### INTUITIVE / SYNTHETIC

#### `precognition`
**When:** Pattern recognition — something *feels* like it's about to
go wrong; recurring signals; early indicators.
Triggered by: "pattern", "intuition", "usually when", "tends to",
"historically", "weak signal", "early sign", "warning sign".

**What it does:** Detects recurrence (has this exact situation occurred
before?), clusters (multiple independent anomalies near the same component),
temporal patterns (does it tend to fail at a certain point in execution?),
and weak signals. Requires a confidence estimate and a falsification test.

Uses `_ep_pattern_summary` to surface the strongest signals automatically.

**Output format:**
```
Signal:     [what weak indicator is visible now]
Pattern:    [the historical recurrence that supports concern]
Prediction: [what is likely to happen if unchecked]
Confidence: [0–100%]
Action:     [what to do NOW, before it materialises]
```

---

#### `retrocognition`
**When:** After a crash, power cut, or interrupted session — need to
reconstruct what was happening.
Triggered by: "reconstruct", "piece together", "after the crash",
"resume", "where were we", "lost context", "back online", "just restarted".

**What it does:** Queries `_ep_last_session_state()` to get the previous
session's clean-exit flag, working directory, model, and last 5 episode
records. Provides a forensic reconstruction protocol: ground truth from
episode log, file system evidence (partial files, `.part`, `.lock`, temp
files), process state inference (power cut vs CTRL+C vs exception), intent
reconstruction, and a safety check (were any irreversible side effects
already committed?).

**Output format:**
```
Last known state:       [what was in progress]
Evidence from:          [episode log / filesystem / process]
Side effects committed: [yes/no — list]
Safe to resume:         [yes/no — if not, explain why]
Next step:              [first action to recover/continue correctly]
```

---

## Mode Auto-Detection

When `mode` is omitted or `"auto"`, the skill scores each mode by counting
keyword matches in the lowercased `problem` string. The mode with the
highest score wins. Default (zero matches): `forethought`.

To override auto-detection, pass `mode` explicitly.

---

## Episode Log Integration

All modes query the episode log for grounding. Three functions are used:

| Function | What it returns | Used by |
|---|---|---|
| `_ep_search(query, limit)` | FTS search results for the problem text | All direct-invoke modes |
| `_ep_pattern_summary(window)` | Count of each episode type in last N records | `foresight`, `forethought`, `precognition`, auto-fire `check()` |
| `_ep_last_session_state()` | Previous session's clean-exit flag + last 5 episodes | `retrocognition` |

All three are wrapped in `try/except` and degrade gracefully — if the
episode DB is unavailable the skill still returns a useful reasoning
template, just without historical grounding.

---

## Wiring: Registration in query_engine.py

The skill is registered in two places:

**1. Skip set** (line ~255) — prevents it appearing as a user-callable
skill in the tool list (it's an orchestrator-level concern):
```python
skip = {"thinking_controller", "wakefulness", "goal_anchor", "temporal_cognition"}
```

**2. Main loop call** (line ~467) — fires every turn:
```python
# Temporal cognition fires every turn — always-on temporal awareness
self._run_temporal_cognition(inner)
```

**3. Method** (after `_run_goal_anchor`) — the actual registration:
```python
def _run_temporal_cognition(self, inner: int) -> None:
    try:
        from skills.temporal_cognition import TemporalCognitionSkill
        tc = TemporalCognitionSkill()
        result = tc.check(
            goal=self._goal_text or "",
            tool_history=self._tool_history,
            session=self.session,
            turn=inner,
        )
        if result:
            if self.verbose:
                print(f"[TemporalCognition] alert at turn {inner}: {result[:80]}", file=sys.stderr)
            self.session.add_tool_result(
                f"<tool_result><n>temporal_cognition</n><o>[TemporalCognition] {result}</o></tool_result>"
            )
    except Exception as exc:
        if self.verbose:
            print(f"[TemporalCognition] check failed: {exc}", file=sys.stderr)
```

If you ever want to add this skill to a sub-agent's QueryEngine, add the
same call inside that engine's loop. The skill is stateless — each call
creates a fresh `TemporalCognitionSkill()` instance.

---

## Direct Invocation Reference

```json
// Auto-detect mode from problem text
{"skill": "temporal_cognition", "args": {"problem": "We're about to run a long file conversion pipeline"}}

// Force a specific mode
{"skill": "temporal_cognition", "args": {"problem": "The download crashed halfway through", "mode": "retrocognition"}}

// Override search query (searched against episode log)
{"skill": "temporal_cognition", "args": {"problem": "review last week's work", "mode": "retrospection", "search_query": "file conversion pipeline errors"}}

// Increase episode history depth
{"skill": "temporal_cognition", "args": {"problem": "token quota pattern", "mode": "precognition", "limit": 12}}
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `problem` | string | required | The situation, event, or task to reason about |
| `mode` | enum or "auto" | "auto" | Which temporal lens to apply (see Mode Reference) |
| `search_query` | string | same as `problem` | Override for the episode log FTS search |
| `limit` | integer | 6 | Max episodes to retrieve from log |

---

## When to Use Which Mode

| Situation | Mode |
|---|---|
| Session just crashed / power cut | `retrocognition` |
| About to start a long risky task | `forethought` or `foresight` |
| Something went wrong — what can I learn? | `hindsight` |
| Reviewing work over a period | `retrospection` |
| Just noticed something I forgot to do | `afterthought` |
| Need to inventory what's already certain | `foreknowledge` |
| Noticing a recurring pattern / weak signal | `precognition` |
| Not sure | omit `mode` — auto-detection picks the best fit |

---

## Noise Policy

The auto-fire `check()` method is designed to be **silent by default**.
It only injects an alert when a concrete threshold is crossed in the
episode log. It will never inject an empty or low-signal message.

Direct invocations via the skill tool always return a full template,
because you explicitly asked for the analysis.

---

## File Locations

| File | Role |
|---|---|
| `skills/temporal_cognition.py` | Skill implementation — all 8 modes + check() |
| `skills/SKILL_temporal_cognition.md` | This documentation file |
| `agent/query_engine.py` | Registration: skip set, loop call, _run_temporal_cognition() |
| `utils/episode_log.py` | Episode DB — source of all historical grounding |
