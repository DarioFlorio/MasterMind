"""
Skill: timeline_reason
Temporal sequencing and timeline reasoning: ordering events, detecting conflicts,
scheduling, before/after reasoning.
"""
from __future__ import annotations
import logging
import re

log = logging.getLogger("skill.timeline_reason")

DESCRIPTION = (
    "Temporal and timeline reasoning: ordering events, detecting conflicts, "
    "scheduling dependencies, before/after reasoning, historical sequencing. "
    "Use for 'when did X happen', 'what came first', 'is this timeline consistent?'"
)


def _general_timeline(problem: str, events: list) -> str:
    ev_block = ""
    if events:
        ev_block = "\n**Provided events:**\n" + "\n".join(f"  - {e}" for e in events)
    return f"""**Timeline Reasoning**
Problem: {problem}{ev_block}

**Step 1 — Extract All Temporal Claims**
  Identify every statement of the form:
    - "X happened at time T"
    - "X happened before/after Y"
    - "X and Y overlapped"
    - "X took duration D"
  Label each with a source (who claims it, how certain).

**Step 2 — Build the Constraint Graph**
  Nodes = events.
  Edges = temporal relations (BEFORE, AFTER, OVERLAPS, MEETS, DURING, etc.)
  Use Allen's interval algebra for precision:
    before    | meets     | overlaps  | starts   | during
    finishes  | equals    | and their converses

**Step 3 — Check Consistency**
  Look for cycles in the BEFORE relation: if A < B < C < A, the timeline
  is inconsistent.
  Technique: topological sort of the constraint graph.
  If a consistent total order exists → timeline is coherent.
  If no consistent order exists → identify the conflicting constraints.

**Step 4 — Fill Gaps**
  What events are implied but not stated?
  What is the earliest/latest possible time for each event given constraints?

**Step 5 — Produce the Timeline**
  List events in chronological order (or partial order if not fully determined).
  Flag: [CERTAIN], [INFERRED], [UNCERTAIN] for each event.

**Step 6 — Answer the Question**
  Given the ordered timeline, answer what was asked directly.
"""


def _sequencing(problem: str, events: list) -> str:
    ev_block = ""
    if events:
        ev_block = "\nEvents to sequence:\n" + "\n".join(f"  {i+1}. {e}" for i, e in enumerate(events))
    return f"""**Timeline Reasoning: Sequencing**
Problem: {problem}{ev_block}

**Method: Topological Ordering from Constraints**

1. **List all BEFORE constraints explicitly:**
   For each pair (A, B) where A must precede B, write: A → B

2. **Build DAG (Directed Acyclic Graph):**
   Nodes = events, directed edge = "must come before".

3. **Topological Sort (Kahn's algorithm):**
   a. Find all nodes with no incoming edges (no prerequisites).
   b. Emit one of them, remove it and its outgoing edges.
   c. Repeat until empty (success) or stuck (cycle = contradiction).

4. **Verify Against Given Order (if provided):**
   For each constraint A → B: does A appear before B in the given sequence?
   List any violations.

5. **Report:**
   One valid ordering (there may be many if some events are unordered relative
   to each other).
   Note which pairs are unconstrained (could swap without contradiction).
"""


def _conflict_detection(problem: str, events: list) -> str:
    return f"""**Timeline Reasoning: Conflict / Inconsistency Detection**
Problem: {problem}

**Step 1 — Extract All Temporal Statements**
  Catalogue every "X before Y", "X at time T", "X during [T1, T2]".
  Include implicit claims (if A causes B, then A before B).

**Step 2 — Normalise to Intervals**
  Convert each event to [start, end] if possible.
  Point events: [T, T].

**Step 3 — Check Each Pair for Conflicts**
  Conflict types:
  a) Direct contradiction: A before B AND B before A.
  b) Overlap when mutually exclusive: A and B claimed simultaneous but can't be.
  c) Duration impossibility: event claimed to take time T but start and end
     are less than T apart.
  d) Alibi conflict: person at location X when also claimed to be at Y.

**Step 4 — Report Conflicts**
  For each conflict: state the two contradicting claims and their sources.
  Determine which claim is more reliable (primary source? corroborating evidence?).

**Step 5 — Minimal Correction**
  What is the smallest change to the timeline that resolves all conflicts?
  (Remove one claim? Adjust one timestamp?)
"""


def _scheduling(problem: str, events: list) -> str:
    return f"""**Timeline Reasoning: Scheduling and Dependencies**
Problem: {problem}

**Step 1 — Task Inventory**
  List all tasks with:
    - Duration (best / expected / worst case)
    - Dependencies (what must finish before this starts)
    - Resources required
    - Hard deadlines (if any)

**Step 2 — Dependency Graph**
  Draw directed graph: edge A → B means "A must finish before B starts".
  This is the project's Activity-on-Node (AON) network.

**Step 3 — Critical Path Method (CPM)**
  Forward pass (earliest start/finish):
    ES(start) = 0
    ES(task) = max EF of all predecessors
    EF(task) = ES + duration

  Backward pass (latest start/finish, working from deadline):
    LF(end) = deadline
    LF(task) = min LS of all successors
    LS(task) = LF − duration

  Float = LS − ES = LF − EF
  Critical path = all tasks with Float = 0 (any delay here delays the whole project).

**Step 4 — Identify Bottlenecks**
  Tasks on the critical path. Resource conflicts (same person needed simultaneously).

**Step 5 — Optimise**
  Can any critical path tasks be parallelised? Can resources be reallocated?
  What is the minimum possible project duration?

**Step 6 — Answer**
  Earliest completion date given dependencies.
  Which tasks have slack? Where to focus risk management?
"""

from skills.base_skill import BaseSkill


class TimelineReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "timeline_reason"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth":   {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "events": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        events = kwargs.get("events", [])
        low    = problem.lower()
        if any(k in low for k in ("schedule", "deadline", "dependency", "gantt", "project")):
            return _scheduling(problem, events)
        if any(k in low for k in ("conflict", "contradict", "inconsistent", "overlap")):
            return _conflict_detection(problem, events)
        if any(k in low for k in ("before", "after", "when", "order", "sequence", "first")):
            return _sequencing(problem, events)
        return _general_timeline(problem, events)
