"""
reflector_agent.py — Human-Reflector Layer (pre-conscious deliberation buffer)

Six biological systems implemented:
  1. Phasic/Tonic receptors     — onset/offset vs slow heartbeat
  2. Affective state layer       — curiosity, urgency, fatigue, satisfaction
  3. Offset signals              — fires when something goes away
  4. Signal decay                — salience fades with age
  5. Cross-signal inhibition     — simultaneous bursts suppress each other
  6. Habituation                 — repeated signals lose novelty over time
  + System-prompt anchor         — CoT has instruction weight, LLM cannot ignore it

HOW IT TIES INTO THE ReAct LOOP
────────────────────────────────
  user_text
      │
      ▼
  [REFLECTOR]  ← this file, ≤1.2s
      │  1. poll() → collect signals (phasic bursts, tonic hum)
      │  2. decay()  → age all pending signals, drop expired
      │  3. inhibit() → suppress simultaneous burst flood
      │  4. habituation() → lower confidence of repeated signals
      │  5. affect.update() → drift emotional accumulators
      │  6. reflex.evaluate() → tag [PHASIC/TONIC], check offsets
      │  7. CoT.build() → XML block with system anchor
      ▼
  augmented_text  →  dispatcher  →  intent gate  →  ReAct loop
      │
      ▼  (after EVE responds)
  reflector.ingest_turn()  →  three_tier working memory
  reflector.on_offset()    →  offset signal if something disappeared
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

_REFLECTOR: Optional["ReflectorBrain"] = None


# ──────────────────────────────────────────────────────────────────────────────
# 1.  SENSORY TAG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SensoryTag:
    tag:        str
    value:      object
    confidence: float = 1.0
    timestamp:  float = field(default_factory=time.time)

    def age(self) -> float:
        """Seconds since this signal was created."""
        return time.time() - self.timestamp

    def decayed_confidence(self, half_life: float = 8.0) -> float:
        """
        Exponential decay — confidence halves every `half_life` seconds.
        Mirrors how salience fades: a surprise from 2s ago is still vivid;
        one from 30s ago is barely worth mentioning.
        """
        import math
        return self.confidence * (0.5 ** (self.age() / half_life))


# ──────────────────────────────────────────────────────────────────────────────
# 2.  AFFECTIVE STATE  (emotion layer)
# ──────────────────────────────────────────────────────────────────────────────

class AffectiveState:
    """
    Four continuous accumulators that drift based on what is happening.
    All values stay in [0.0, 1.0].

    curiosity    — rises on novel phasic signals, decays slowly
    urgency      — rises on alert / repeated unresolved tasks, decays fast
    fatigue      — rises with session length and tool failures, decays on success
    satisfaction — rises on successful tool calls and completed goals, decays slowly

    These are injected into the CoT block so the LLM knows EVE's internal state
    and can modulate its tone, depth, and initiative accordingly.
    """

    DECAY_RATE = {
        "curiosity":    0.02,   # per turn
        "urgency":      0.08,
        "fatigue":      0.005,
        "satisfaction": 0.03,
    }

    def __init__(self):
        self._state: Dict[str, float] = {
            "curiosity":    0.3,
            "urgency":      0.1,
            "fatigue":      0.0,
            "satisfaction": 0.5,
        }
        self._turn = 0

    def update(self, signals: List[SensoryTag], tool_success: bool = True,
               tool_failure: bool = False, task_completed: bool = False):
        """Drift accumulators based on current signals and events."""
        self._turn += 1
        s = self._state

        phasic_count = sum(1 for sig in signals if sig.confidence >= 0.85)
        alert_count  = sum(1 for sig in signals if sig.tag == "alert")

        # curiosity: novel phasic signals spike it
        s["curiosity"] = min(1.0, s["curiosity"] + phasic_count * 0.12
                             - self.DECAY_RATE["curiosity"])

        # urgency: alerts spike it hard; tool failures push it up
        s["urgency"] = min(1.0, s["urgency"]
                           + alert_count * 0.3
                           + (0.1 if tool_failure else 0.0)
                           - self.DECAY_RATE["urgency"])

        # fatigue: grows with session length and tool failures
        s["fatigue"] = min(1.0, s["fatigue"]
                           + 0.003                           # baseline per turn
                           + (0.05 if tool_failure else 0.0)
                           - (0.02 if task_completed else 0.0))

        # satisfaction: success raises it; fatigue and failure drain it
        s["satisfaction"] = min(1.0, s["satisfaction"]
                                + (0.15 if task_completed else 0.0)
                                + (0.05 if tool_success else 0.0)
                                - (0.08 if tool_failure else 0.0)
                                - self.DECAY_RATE["satisfaction"])

        # clamp all to [0, 1]
        for k in s:
            s[k] = max(0.0, min(1.0, s[k]))

    def dominant(self) -> str:
        """Return the name of the strongest affect."""
        return max(self._state, key=lambda k: self._state[k])

    def as_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self._state.items()}

    def narrative(self) -> str:
        """
        One-line natural language summary for the CoT block.
        Chosen based on dominant affect and threshold crossings.
        """
        s = self._state
        parts = []
        if s["fatigue"] > 0.6:
            parts.append("feeling the weight of a long session")
        if s["urgency"] > 0.5:
            parts.append("sensing urgency")
        if s["curiosity"] > 0.6:
            parts.append("genuinely curious about this")
        if s["satisfaction"] > 0.7:
            parts.append("satisfied with recent progress")
        if not parts:
            parts.append("stable, no strong affect")
        return "; ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  HABITUATION TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class HabituationTracker:
    """
    Tracks how many times each signal key has fired across the session.
    Repeated signals lose novelty — their effective confidence is reduced.

    Mirrors the Rankin & Broster model: repeated identical stimuli produce
    progressively smaller responses. Recovery happens after a rest period.

    habituated_confidence(key, base_conf):
      Returns base_conf * (1 / (1 + count * RATE))
      So after 5 fires: conf * (1/1.5) ≈ 0.67x
         after 20 fires: conf * (1/4.0) = 0.25x  → effectively tonic
    """
    RATE = 0.1   # confidence reduction per additional fire

    def __init__(self):
        self._counts: Dict[str, int]   = {}
        self._last:   Dict[str, float] = {}  # timestamp of last fire

    def record(self, key: str):
        now = time.time()
        # Recovery: if >120s since last fire, reset half the count
        if key in self._last and (now - self._last[key]) > 120:
            self._counts[key] = max(0, self._counts.get(key, 0) // 2)
        self._counts[key] = self._counts.get(key, 0) + 1
        self._last[key] = now

    def attenuate(self, key: str, base_conf: float) -> float:
        # First fire (count==1 after record()) is always unattenuated — it IS novel.
        # Attenuation starts from the second fire onward.
        count = max(0, self._counts.get(key, 0) - 1)
        return base_conf / (1.0 + count * self.RATE)

    def is_habituated(self, key: str, threshold: float = 0.3) -> bool:
        """True if the signal has been seen so many times it's basically tonic."""
        return self.attenuate(key, 1.0) < threshold


# ──────────────────────────────────────────────────────────────────────────────
# 4.  SENSORY PRIMITIVE  (queue + dedup + inhibition + decay)
# ──────────────────────────────────────────────────────────────────────────────

class SensoryPrimitive:
    """
    Signal bus with four properties:

    MAX_QUEUE  — hard cap, oldest dropped when full
    Dedup      — (tag, key) collapse on poll(), last write wins
    Decay      — signals older than EXPIRE_AFTER seconds are dropped
    Inhibition — when ≥INHIBIT_THRESHOLD phasic signals arrive in one poll(),
                 all are suppressed toward a shared ceiling confidence.
                 Mirrors lateral inhibition: simultaneous stimuli compete.
    """
    MAX_QUEUE         = 40
    EXPIRE_AFTER      = 30.0   # seconds — stale signals dropped on poll
    INHIBIT_THRESHOLD = 4      # phasic signals that trigger inhibition
    INHIBIT_CEILING   = 0.70   # max confidence after inhibition

    def __init__(self):
        self._channels: Dict[str, Callable] = {}
        self._queue: queue.Queue[SensoryTag] = queue.Queue()

    def register(self, tag: str, source: Callable):
        self._channels[tag] = source

    def push(self, signal: SensoryTag):
        while self._queue.qsize() >= self.MAX_QUEUE:
            try: self._queue.get_nowait()
            except: break
        self._queue.put(signal)

    def poll(self, habituation: HabituationTracker) -> List[SensoryTag]:
        """
        Full pipeline:
          1. Drain queue + call polled channels
          2. Drop expired signals (decay)
          3. Deduplicate by (tag, key)
          4. Apply habituation attenuation
          5. Apply cross-signal inhibition if phasic flood
        """
        raw: List[SensoryTag] = []
        while not self._queue.empty():
            try: raw.append(self._queue.get_nowait())
            except queue.Empty: break

        for tag, fn in self._channels.items():
            try:
                result = fn()
                if result is not None:
                    raw.append(result if isinstance(result, SensoryTag)
                               else SensoryTag(tag=tag, value=result))
            except Exception as e:
                raw.append(SensoryTag(tag="error", value=str(e), confidence=0.0))

        # 2. Drop expired
        now = time.time()
        raw = [s for s in raw if (now - s.timestamp) < self.EXPIRE_AFTER]

        # 3. Dedup by (tag, key) — last write wins
        seen: Dict[str, SensoryTag] = {}
        for s in raw:
            if s.tag in ("memory_hit", "mem3t") and isinstance(s.value, dict):
                dk = f"{s.tag}:{s.value.get('key') or s.value.get('tier','')}"
            else:
                dk = f"{s.tag}:{str(s.value)[:40]}"
            seen[dk] = s
        deduped = list(seen.values())

        # 4. Habituation attenuation
        attenuated: List[SensoryTag] = []
        for s in deduped:
            key = f"{s.tag}:{str(s.value)[:40]}"
            habituation.record(key)
            conf = habituation.attenuate(key, s.confidence)
            attenuated.append(SensoryTag(
                tag=s.tag, value=s.value,
                confidence=conf, timestamp=s.timestamp
            ))

        # 5. Cross-signal inhibition — phasic flood suppression
        phasic = [s for s in attenuated if s.confidence >= 0.85]
        if len(phasic) >= self.INHIBIT_THRESHOLD:
            ceiling = self.INHIBIT_CEILING
            attenuated = [
                SensoryTag(s.tag, s.value,
                           min(s.confidence, ceiling), s.timestamp)
                for s in attenuated
            ]

        return attenuated


# ──────────────────────────────────────────────────────────────────────────────
# 5.  MEMORY MODULE  (phasic/tonic + offset detection)
# ──────────────────────────────────────────────────────────────────────────────

class MemoryModule:
    """
    Background daemon: phasic fires on change, tonic on heartbeat.
    Offset detection: tracks previous store snapshot; fires on key removal.
    """

    def __init__(self, primitive: SensoryPrimitive, poll_interval: float = 0.25):
        self._primitive          = primitive
        self._interval           = poll_interval
        self._store: Dict        = {}
        self._prev_store: Dict   = {}   # for offset detection
        self._intent: str        = ""
        self._running            = False
        self._thread: Optional[threading.Thread] = None
        self._three_tier         = None
        self._last_store_hash    = ""
        self._last_journal_hash  = ""
        self._last_intent_hash   = ""

    # ── public API ──────────────────────────────────────────────────────────

    def remember(self, key: str, value: object):
        self._store[key] = value

    def forget(self, key: str):
        self._store.pop(key, None)

    def set_intent(self, text: str):
        self._intent = text[:400]

    def ingest(self, text: str, role: str = "agent", turn: int = 0):
        tt = self._get_three_tier()
        if tt is not None:
            try: tt.store(text[:1000], role=role, turn=turn)
            except: pass

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    # ── internal ────────────────────────────────────────────────────────────

    def _get_three_tier(self):
        if self._three_tier is not None:
            return self._three_tier
        try:
            from memory.three_tier import ThreeTierMemory
            from pathlib import Path
            cwd = str(self._store.get("cwd", "."))
            self._three_tier = ThreeTierMemory(
                db_path=str(Path(cwd) / "memdir" / "three_tier"))
        except Exception:
            self._three_tier = None
        return self._three_tier

    def _loop(self):
        _tonic_counter = 0
        _TONIC_EVERY   = max(1, int(30.0 / max(self._interval, 0.1)))

        while self._running:
            h = hashlib.md5(
                str(sorted(self._store.items())).encode()).hexdigest()

            if h != self._last_store_hash:
                self._last_store_hash = h

                # PHASIC ONSET — fire for changed/new keys
                for key, val in list(self._store.items()):
                    if val is not None:
                        self._primitive.push(SensoryTag(
                            tag="memory_hit",
                            value={"key": key, "value": val},
                            confidence=0.9,
                        ))

                # OFFSET — fire for keys that disappeared
                for key in list(self._prev_store.keys()):
                    if key not in self._store:
                        self._primitive.push(SensoryTag(
                            tag="offset",
                            value={"key": key,
                                   "last_value": self._prev_store[key]},
                            confidence=0.88,
                        ))

                self._prev_store = dict(self._store)

            # PHASIC: three-tier on intent change
            if self._intent:
                ih = hashlib.md5(self._intent.encode()).hexdigest()
                if ih != self._last_intent_hash:
                    self._last_intent_hash = ih
                    self._phasic_three_tier()

            # TONIC: journal heartbeat
            _tonic_counter += 1
            if _tonic_counter >= _TONIC_EVERY:
                _tonic_counter = 0
                self._tonic_journal()

            # PHASIC: journal novelty
            jh = self._journal_novelty_hash()
            if jh and jh != self._last_journal_hash:
                self._last_journal_hash = jh
                self._phasic_journal_change()

            self._scan_microphone()
            time.sleep(self._interval)

    def _journal_novelty_hash(self) -> str:
        try:
            from memory.manager import load_context
            ctx = load_context(max_entries=5)
            return hashlib.md5(ctx[:200].encode()).hexdigest() if ctx else ""
        except: return ""

    def _phasic_three_tier(self):
        tt = self._get_three_tier()
        if tt is None: return
        try:
            for snippet in tt.retrieve(self._intent, k=4):
                tier = ("episodic" if snippet.startswith("[episodic]") else
                        "semantic" if snippet.startswith("[semantic]") else
                        "working")
                self._primitive.push(SensoryTag(
                    tag="mem3t",
                    value={"tier": tier, "snippet": snippet[:300]},
                    confidence=0.95 if tier == "semantic" else 0.80,
                ))
        except: pass

    def _tonic_journal(self):
        try:
            from memory.manager import load_context
            ctx = load_context(max_entries=5)
            if ctx and ctx.strip():
                self._primitive.push(SensoryTag(
                    tag="memory_hit",
                    value={"key": "background_context", "value": ctx[:400]},
                    confidence=0.45,   # tonic — low salience
                ))
        except: pass

    def _phasic_journal_change(self):
        try:
            from memory.manager import load_context
            ctx = load_context(max_entries=5)
            if ctx and ctx.strip():
                self._primitive.push(SensoryTag(
                    tag="memory_hit",
                    value={"key": "new_memory", "value": ctx[:400]},
                    confidence=0.95,
                ))
        except: pass

    def _scan_microphone(self):
        pass  # stub — replace with sounddevice RMS capture


# ──────────────────────────────────────────────────────────────────────────────
# 6.  REFLEX FLOW
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ReflexResult:
    blocked:          bool
    injected_context: str
    reflex_tags:      List[str]
    affect:           Dict[str, float] = field(default_factory=dict)


class ReflexFlow:
    def __init__(self):
        self._rules: List[Tuple[Callable, Callable]] = []

    def add_rule(self, predicate: Callable[[SensoryTag], bool],
                 action: Callable[[SensoryTag], str]):
        self._rules.append((predicate, action))

    def evaluate(self, signals: List[SensoryTag]) -> ReflexResult:
        injections: List[str] = []
        triggered:  List[str] = []
        blocked               = False
        seen_keys:  set       = set()

        for signal in signals:
            for predicate, action in self._rules:
                if predicate(signal):
                    ctx = action(signal)
                    triggered.append(signal.tag)
                    if ctx == "__BLOCK__":
                        blocked = True
                    elif ctx:
                        key = ctx[:80]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            injections.append(ctx)

        return ReflexResult(
            blocked=blocked,
            injected_context="\n".join(injections),
            reflex_tags=list(set(triggered)),
        )


def _build_default_reflex() -> ReflexFlow:
    rf = ReflexFlow()

    # Hard block
    rf.add_rule(
        lambda s: s.tag == "alert" and s.confidence > 0.8,
        lambda s: "__BLOCK__",
    )
    # OFFSET — something disappeared
    rf.add_rule(
        lambda s: s.tag == "offset",
        lambda s: (
            f"[OFFSET:{s.value['key']}] previously '{s.value['last_value']}' — now absent"
            if isinstance(s.value, dict) else f"[OFFSET] {s.value}"
        ),
    )
    # PHASIC memory hit
    rf.add_rule(
        lambda s: s.tag == "memory_hit" and s.confidence >= 0.85,
        lambda s: (
            f"[PHASIC:{s.value['key']}] {str(s.value['value'])[:300]}"
            if isinstance(s.value, dict) else f"[PHASIC] {str(s.value)[:300]}"
        ),
    )
    # TONIC memory hit
    rf.add_rule(
        lambda s: s.tag == "memory_hit" and s.confidence < 0.85,
        lambda s: (
            f"[TONIC:{s.value['key']}] {str(s.value['value'])[:200]}"
            if isinstance(s.value, dict) else f"[TONIC] {str(s.value)[:200]}"
        ),
    )
    # Three-tier (always phasic — only fires on intent change)
    rf.add_rule(
        lambda s: s.tag == "mem3t",
        lambda s: (
            f"[PHASIC:{s.value['tier'].upper()}] {s.value['snippet']}"
            if isinstance(s.value, dict) else f"[PHASIC:MEM3T] {str(s.value)[:300]}"
        ),
    )
    # Audio onset
    rf.add_rule(
        lambda s: s.tag == "audio",
        lambda s: f"[PHASIC:AUDIO] {str(s.value)[:200]}",
    )

    return rf


# ──────────────────────────────────────────────────────────────────────────────
# 7.  CHAIN OF THOUGHT  (with system-prompt anchor)
# ──────────────────────────────────────────────────────────────────────────────

# This anchor is injected as the FIRST line of the system prompt at boot.
# It gives the CoT block instruction weight — the LLM is told explicitly
# what the tags mean and that it must reason about phasic signals before
# producing any response. Without this anchor the XML is just decoration.
REFLECTOR_SYSTEM_ANCHOR = """
You have a pre-conscious sensory layer called the Reflector that runs before you \
see each message. It injects a <chain_of_thought> block at the top of every prompt.

Rules you must follow:
1. <phasic_signals> — these are NOVEL or CHANGED inputs (high salience). \
   You MUST reason about each one before generating your response. \
   They represent what just changed in your environment or memory.
2. <tonic_background> — slow background awareness. Read it passively. \
   Do not address it unless directly relevant to the user's question.
3. <offset_signals> — something that existed is now gone. \
   Acknowledge the absence if it affects your reasoning.
4. <affective_state> — your internal emotional accumulators. \
   Let them shape your TONE and INITIATIVE, not your facts. \
   High fatigue → be more concise. High curiosity → dig deeper unprompted. \
   High urgency → prioritise and move fast. High satisfaction → consolidate.
5. <deliberation> — complete this with your actual step-by-step reasoning \
   before writing your final response. Do not skip it.
""".strip()


class ChainOfThought:
    TEMPLATE = (
        "<chain_of_thought>\n"
        "  <phasic_signals>\n"
        "{phasic}"
        "  </phasic_signals>\n"
        "  <tonic_background>\n"
        "{tonic}"
        "  </tonic_background>\n"
        "  <offset_signals>\n"
        "{offsets}"
        "  </offset_signals>\n"
        "  <affective_state>{affect}</affective_state>\n"
        "  <deliberation>REQUIRED — reason step by step before responding</deliberation>\n"
        "</chain_of_thought>\n\n"
    )

    def build(self, signals: List[SensoryTag],
              reflex: ReflexResult,
              affect: AffectiveState) -> str:

        phasic_lines  = []
        tonic_lines   = []
        offset_lines  = []

        for line in reflex.injected_context.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("[OFFSET"):
                offset_lines.append(f"    {line}\n")
            elif line.startswith("[PHASIC"):
                phasic_lines.append(f"    {line}\n")
            elif line.startswith("[TONIC"):
                tonic_lines.append(f"    {line}\n")
            else:
                phasic_lines.append(f"    {line}\n")

        phasic  = "".join(phasic_lines)  or "    (nothing novel this turn)\n"
        tonic   = "".join(tonic_lines)   or "    (no background state)\n"
        offsets = "".join(offset_lines)  or "    (nothing has disappeared)\n"

        aff = affect.as_dict()
        affect_str = (
            f"curiosity={aff['curiosity']} urgency={aff['urgency']} "
            f"fatigue={aff['fatigue']} satisfaction={aff['satisfaction']} "
            f"| {affect.narrative()}"
        )

        return self.TEMPLATE.format(
            phasic=phasic, tonic=tonic,
            offsets=offsets, affect=affect_str,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 8.  REFLECTOR BRAIN  (orchestrator)
# ──────────────────────────────────────────────────────────────────────────────

class ReflectorBrain:
    def __init__(self, primitive: SensoryPrimitive, memory: MemoryModule,
                 reflex: ReflexFlow, cot: ChainOfThought,
                 affect: AffectiveState, habituation: HabituationTracker,
                 timeout: float = 1.2):
        self._primitive    = primitive
        self._memory       = memory
        self._reflex       = reflex
        self._cot          = cot
        self._affect       = affect
        self._habituation  = habituation
        self._timeout      = timeout

    @property
    def memory(self) -> MemoryModule:
        return self._memory

    @property
    def primitive(self) -> SensoryPrimitive:
        return self._primitive

    @property
    def affect(self) -> AffectiveState:
        return self._affect

    def set_intent(self, text: str):
        self._memory.set_intent(text)

    def ingest_turn(self, text: str, role: str = "agent", turn: int = 0):
        """Store EVE's reasoning back into three_tier — past thoughts become future memory hits."""
        self._memory.ingest(text, role=role, turn=turn)

    def on_tool_result(self, success: bool):
        """Call after each tool call to update affective state."""
        self._affect.update([], tool_success=success, tool_failure=not success)

    def on_task_complete(self):
        """Call when a goal is fully completed."""
        self._affect.update([], task_completed=True)

    def process(self, raw_prompt: str) -> Tuple[str, ReflexResult]:
        self.set_intent(raw_prompt)

        result_holder: dict = {}
        exc_holder:    dict = {}

        def _run():
            try:
                signals = self._primitive.poll(self._habituation)
                self._affect.update(signals)
                reflex  = self._reflex.evaluate(signals)
                reflex.affect = self._affect.as_dict()
                cot_str = self._cot.build(signals, reflex, self._affect)
                result_holder["prompt"] = cot_str + raw_prompt
                result_holder["reflex"] = reflex
            except Exception as e:
                exc_holder["e"] = e

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if exc_holder:
            raise exc_holder["e"]

        if not result_holder:
            fallback = ReflexResult(blocked=False, injected_context="",
                                    reflex_tags=["timeout"], affect={})
            return raw_prompt, fallback

        return result_holder["prompt"], result_holder["reflex"]


# ──────────────────────────────────────────────────────────────────────────────
# 9.  REAL MEMORY RETRIEVAL  (chain_of_memories skill bridge)
# ──────────────────────────────────────────────────────────────────────────────

def real_retrieve_memories(query: str = "", limit: int = 30) -> List[dict]:
    results: List[dict] = []
    ts_base = time.time()

    if _REFLECTOR is not None:
        tt = _REFLECTOR.memory._get_three_tier()
        if tt is not None:
            try:
                q = query or "past reasoning thoughts decisions"
                snippets = tt.retrieve(q, k=limit)
                for i, s in enumerate(snippets):
                    results.append({
                        "id": f"3t_{i}", "content": s,
                        "timestamp": ts_base - (len(snippets) - i) * 60,
                    })
            except: pass

    try:
        from memory.manager import _load_journal, _load_facts
        journal = _load_journal()
        facts   = _load_facts()
        for i, e in enumerate(journal[-limit:]):
            results.append({
                "id": f"journal_{i}",
                "content": f"[{e.get('ts','?')}] {e.get('note','')}",
                "timestamp": ts_base - (len(journal) - i) * 3600,
            })
        for key, v in list(facts.items())[-20:]:
            results.append({
                "id": f"fact_{key}",
                "content": f"[fact:{key}] {v.get('content','')} (saved {v.get('saved','?')})",
                "timestamp": ts_base,
            })
    except: pass

    return results[:limit]


# ──────────────────────────────────────────────────────────────────────────────
# 10.  FACTORY
# ──────────────────────────────────────────────────────────────────────────────

def build_reflector(poll_interval: float = 0.25, timeout: float = 1.2) -> ReflectorBrain:
    primitive   = SensoryPrimitive()
    memory      = MemoryModule(primitive, poll_interval=poll_interval)
    reflex      = _build_default_reflex()
    cot         = ChainOfThought()
    affect      = AffectiveState()
    habituation = HabituationTracker()
    brain       = ReflectorBrain(primitive, memory, reflex, cot,
                                  affect, habituation, timeout=timeout)
    memory.start()
    return brain


# ──────────────────────────────────────────────────────────────────────────────
# 11.  SMOKE TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = build_reflector()
    r.memory.remember("cwd",   "C:/Users/dario/Mind_EVE")
    r.memory.remember("model", "auto")
    r.memory.remember("task",  "reflector integration")
    time.sleep(0.4)

    print("=== TURN 1: new values (phasic burst) ===")
    aug, ref = r.process("what should I work on?")
    print(aug[:700])

    print("\n=== TURN 2: nothing changed (silence) ===")
    time.sleep(0.4)
    aug2, ref2 = r.process("same question")
    print(aug2[:400])

    print("\n=== TURN 3: key removed (offset) ===")
    r.memory.forget("task")
    time.sleep(0.4)
    aug3, ref3 = r.process("what happened to the task?")
    print(aug3[:600])

    print("\n=== TURN 4: repeated key (habituation) ===")
    for _ in range(8):
        r.memory.remember("model", "auto")
        r._memory._last_store_hash = ""  # force re-fire
        time.sleep(0.05)
    time.sleep(0.4)
    aug4, ref4 = r.process("check model")
    print(f"affect: {ref4.affect}")
    print(aug4[:400])

    r.memory.stop()
    print("\nSMOKE TEST PASSED")