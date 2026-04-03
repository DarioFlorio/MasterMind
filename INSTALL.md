# EVE — Claude Code Leak Feature Integration

## Files in this archive

| File | Destination | Action |
|------|-------------|--------|
| `agent/context_budget.py` | `agent/` | **NEW** — copy in |
| `agent/ultraplan.py`      | `agent/` | **NEW** — copy in |
| `memory/autodream.py`     | `memory/` | **NEW** — copy in |
| `kairos.py`               | project root | **NEW** — copy in |
| `main.py`                 | project root | **REPLACE** |

---

## Feature 1 — Context Budget (context rot prevention)

**What it does:** Trims tool descriptions to 250 chars, detects when tool
schema overhead consumes >45% of your 8192-token context, and switches to a
compressed schema automatically. Directly mirrors the Claude Code leak detail.

**Wire into `agent/query_engine.py`** — add near top of `QueryEngine.run()`:

```python
from agent.context_budget import ContextBudget
_budget = ContextBudget(context_size=CONTEXT_SIZE)
# Replace `all_tools` with:
tools_for_prompt = _budget.get_slim_tools(all_tools)
# After each turn:
_budget.record_turn(prompt_tokens=N, completion_tokens=M, tool_overhead=K)
if _budget.rot_detected():
    system_prompt = SYSTEM_PROMPT_COMPRESSED   # switch to shorter sys prompt
```

---

## Feature 2 — AutoDream (idle memory consolidation)

**What it does:** After 120s of idle time, deduplicates your `journal.json`
and `facts.json` using Jaccard similarity, clusters related entries, writes a
dream summary, and prunes entries older than 30 days if over the 200-entry cap.

**Wire into `main.py`** — already patched in the `main.py` in this archive.
Manually, add to your startup block:

```python
from memory.autodream import AutoDream
autodream = AutoDream()
autodream.start()          # starts background thread
heartbeat.register(30, autodream.ping)   # keep idle clock honest
# In input loop, after each user message:
autodream.ping()
# On exit:
autodream.stop()
```

---

## Feature 3 — Kairos (persistent background daemon)

**What it does:** Spawns a detached OS process (`kairos_daemon.py`) that
survives terminal close. Every 5 minutes it scans your journal/facts for
unresolved TODOs, long idle periods, and stale facts, then writes insights
to `memdir/kairos_insights.json`. On next boot EVE reads and displays them.

**Wire into `main.py`**:

```python
from kairos import Kairos, write_daemon_script
write_daemon_script()       # writes kairos_daemon.py once
kairos = Kairos()
kairos.ensure_running()     # spawns daemon if not running
# On boot, display insights:
insights = kairos.pop_insights()
for ins in insights:
    print(f"[Kairos] {ins}")
# At end of each session:
kairos.push_context(session_summary_string)
```

**Note:** On Windows, the daemon uses `DETACHED_PROCESS`. On Linux/Mac it
uses `start_new_session=True`. Both survive terminal close.

---

## Feature 4 — UltraPlan (10-30 min deep task blueprinting)

**What it does:** Before executing a complex task, generates a full phased
Blueprint — phases, steps, tool hints, risk assessment, success criteria —
saved to `.ultraplan/plan_<timestamp>.json`. Supports mid-task resumption.

**Trigger keywords:** `PLAN:`, `build me a`, `architect`, `implement a full`,
`create a complete`, `end-to-end`, `from scratch`.

**Wire into `main.py` input loop**:

```python
from agent.ultraplan import UltraPlan, should_ultraplan
ultra = UltraPlan(tools=list(tool_registry.values()), working_dir=WORKING_DIR)

# In your input loop, before dispatching:
if should_ultraplan(user_input):
    blueprint = ultra.plan(user_input)
    print(blueprint.render())
    # Then execute phase by phase using query_engine per phase
    phase = ultra.next_actionable(blueprint)
    while phase:
        # run phase.steps through query_engine ...
        ultra.mark_step(blueprint, phase.id, step.id, "done", output)
        phase = ultra.next_actionable(blueprint)
```

**Resume interrupted plan:**

```python
bp = ultra.load_blueprint(".ultraplan/plan_1234567_my_task.json")
print(bp.render())   # shows current progress with ✅/⬜ per step
```

---

## Settings additions (optional — `.env`)

```
KAIROS_TICK_S=300        # Kairos daemon tick interval (seconds)
AUTODREAM_IDLE_S=120     # AutoDream idle trigger (seconds)
CONTEXT_MAX_DESC=250     # ContextBudget max tool description chars
```
