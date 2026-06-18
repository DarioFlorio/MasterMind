---
name: reflector_agent
file: reflector_agent.py
singleton: true
boot: main.py → build_reflector()
system_anchor: REFLECTOR_SYSTEM_ANCHOR → injected at top of system prompt
fires: before every LLM call (query_engine.py → submit_message)
ingest: after every LLM response (query_engine.py → _run_loop no-tool branch)
---

# Reflector Agent — Pre-Conscious Deliberation Buffer

EVE's proprioceptive layer. Sits between raw sensory input and the LLM.
Runs in ≤1.2s before every prompt. Builds a `<chain_of_thought>` XML block
the LLM must reason over before generating any response.

Based on biological sensory neuroscience — six systems implemented.

---

## Architecture

```
user_text
    │
    ▼
[REFLECTOR]  ≤1.2s hard timeout
    │
    ├─ SensoryPrimitive.poll(habituation)
    │     drain queue + polled channels
    │     drop expired signals (>30s)
    │     deduplicate by (tag, key)
    │     habituation attenuation
    │     cross-signal inhibition (phasic flood)
    │
    ├─ AffectiveState.update(signals)
    │     drift curiosity / urgency / fatigue / satisfaction
    │
    ├─ ReflexFlow.evaluate(signals)
    │     tag → [PHASIC / TONIC / OFFSET]
    │     hard block on alert
    │
    ├─ ChainOfThought.build(signals, reflex, affect)
    │     XML: phasic_signals, tonic_background,
    │          offset_signals, affective_state, deliberation
    │
    ▼
augmented_prompt (CoT + original text)
    │
    ▼
dispatcher.classify() → intent gate → _run_loop (ReAct)
    │
    ▼  (after EVE responds)
reflector.ingest_turn(response)  →  three_tier working memory
```

---

## Six Biological Systems

### 1. Phasic / Tonic Receptors
Two receptor classes running in parallel.

**Phasic** (fast-adapting — skin mechanoreceptors, LC-NE bursts):
- Fires on ONSET of a new/changed value, then goes silent.
- Fires on OFFSET (value removed from store).
- Confidence ≥ 0.85 → injected as `[PHASIC:key]`
- Used for: volatile store changes, intent change → three_tier retrieval,
  new journal/facts entries.

**Tonic** (slow-adapting — proprioception, background pain):
- Fires on a ~30s heartbeat regardless of change.
- Confidence 0.45 → injected as `[TONIC:key]`
- Used for: persistent journal/facts — background self-awareness.
- EVE always knows where she is, even when nothing is new.

### 2. Affective State
Four continuous accumulators in [0.0, 1.0]:

| Accumulator  | Rises on                          | Decays on       |
|---|---|---|
| curiosity    | novel phasic signals              | per turn (slow) |
| urgency      | alerts, tool failures             | per turn (fast) |
| fatigue      | session length, tool failures     | task completion |
| satisfaction | task completion, tool success     | per turn (slow) |

Injected into `<affective_state>` in the CoT block.
LLM instruction: let affect shape TONE and INITIATIVE, not facts.
- High fatigue → be concise
- High curiosity → dig deeper unprompted
- High urgency → prioritise and move fast
- High satisfaction → consolidate

### 3. Offset Signals
Tracks previous volatile store snapshot.
When a key disappears → fires `[OFFSET:key] previously '...' — now absent`
into the `<offset_signals>` lane.
Mirrors how humans notice when something goes away.

### 4. Signal Decay
`SensoryTag.decayed_confidence(half_life=8.0)`:
Confidence halves every 8 seconds via exponential decay.
Signals older than 30s are dropped entirely on `poll()`.
A surprise from 2s ago is vivid. One from 30s ago is noise.

### 5. Cross-Signal Inhibition
When ≥4 phasic signals arrive in one `poll()`:
All signals are suppressed to a ceiling of 0.70 confidence.
Mirrors lateral inhibition — simultaneous stimuli compete for attention.
Prevents a full session reload from flooding the context with equal-weight bursts.

### 6. Habituation
`HabituationTracker` tracks fire count per signal key.
`attenuated_conf = base_conf / (1 + (count-1) * 0.1)`
- First fire: always unattenuated (it IS novel).
- After 5 fires: 0.71x
- After 20 fires: 0.34x → effectively tonic
Recovery: if >120s since last fire, count halves.
`cwd`, `model`, `perm_mode` — seen every session → naturally become tonic over time.

---

## System-Prompt Anchor

`REFLECTOR_SYSTEM_ANCHOR` must be injected at the TOP of EVE's system prompt
at boot. Without it the CoT XML is decoration — the LLM has no instruction
telling it what the tags mean or that it must act on them.

**In `main.py`, where the system prompt is assembled:**
```python
from reflector_agent import REFLECTOR_SYSTEM_ANCHOR
system_prompt = REFLECTOR_SYSTEM_ANCHOR + "\n\n" + system_prompt
```

The anchor tells EVE:
1. `<phasic_signals>` — reason about each one BEFORE responding.
2. `<tonic_background>` — read passively, use only if relevant.
3. `<offset_signals>` — acknowledge absence if it affects reasoning.
4. `<affective_state>` — modulate tone and initiative accordingly.
5. `<deliberation>` — complete this with actual step-by-step reasoning. Do not skip.

---

## Memory ↔ Thought Cross-Link

Every EVE response is ingested back into three_tier working memory:
```
query_engine.py → _run_loop (no-tool branch):
    _REFLECTOR.ingest_turn(display_text, role="agent", turn=inner)
```

Three-tier auto-consolidates:
- Every 10 turns: working → episodic summary
- Every 50 turns: high-value episodic → semantic

On the next prompt, three_tier retrieval fires (phasic, on intent change)
and injects those stored thoughts back into the CoT block.
**Past thoughts become future memory hits.**

---

## Wiring Points

| Location | What happens |
|---|---|
| `main.py` boot | `_ra._REFLECTOR = build_reflector()` — starts background thread |
| `main.py` boot | `REFLECTOR_SYSTEM_ANCHOR` prepended to system prompt |
| `main.py` boot | `_REFLECTOR.memory.remember("cwd", cwd)` etc. |
| `main.py` shutdown | `_REFLECTOR.memory.stop()` — clean thread join |
| `query_engine.py submit_message` | `_REFLECTOR.process(user_text)` — augments prompt |
| `query_engine.py submit_message` | `_REFLECTOR.ingest_turn(user_text, role="user")` |
| `query_engine.py _run_loop` | `_REFLECTOR.ingest_turn(display_text, role="agent")` |
| `skills/chain_of_memories.py` | `real_retrieve_memories()` — replaces mock data |

---

## Public API

```python
from reflector_agent import build_reflector, _REFLECTOR, REFLECTOR_SYSTEM_ANCHOR

# Boot (once)
reflector = build_reflector(poll_interval=0.25, timeout=1.2)

# Before every LLM call
augmented, reflex = reflector.process(user_prompt)
if reflex.blocked:
    return "[REFLECTOR] blocked by reflex rule"

# After every LLM response
reflector.ingest_turn(response_text, role="agent", turn=inner)

# After tool results
reflector.on_tool_result(success=True)

# After goal completion
reflector.on_task_complete()

# Add a sensory channel (custom source)
reflector.primitive.register("battery", lambda: SensoryTag("battery", 0.2, confidence=0.99))

# Push an interrupt-style signal
reflector.primitive.push(SensoryTag("alert", "quota at 95%", confidence=0.95))

# Read current affect
state = reflector.affect.as_dict()
# → {'curiosity': 0.64, 'urgency': 0.02, 'fatigue': 0.01, 'satisfaction': 0.52}
```

---

## CoT Block Structure

```xml
<chain_of_thought>
  <phasic_signals>
    [PHASIC:model] qwen2.5-14b          ← model just changed
    [PHASIC:EPISODIC] last task was...  ← three_tier hit on new intent
  </phasic_signals>
  <tonic_background>
    [TONIC:background_context] ...      ← journal heartbeat, low salience
  </tonic_background>
  <offset_signals>
    [OFFSET:task] previously 'reflector integration' — now absent
  </offset_signals>
  <affective_state>curiosity=0.64 urgency=0.02 fatigue=0.01 satisfaction=0.52 | genuinely curious about this</affective_state>
  <deliberation>REQUIRED — reason step by step before responding</deliberation>
</chain_of_thought>

[original user prompt follows]
```

---

## File Locations

| File | Role |
|---|---|
| `reflector_agent.py` | Full implementation — all 6 systems |
| `skills/SKILL_reflector_agent.md` | This documentation |
| `main.py` | Boot, shutdown, system-prompt anchor injection |
| `agent/query_engine.py` | process() call, ingest_turn() calls |
| `skills/chain_of_memories.py` | real_retrieve_memories() bridge |
| `memory/three_tier.py` | Working/episodic/semantic storage backend |
| `memory/manager.py` | journal.json + facts.json persistent store |