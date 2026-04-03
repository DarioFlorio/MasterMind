"""
agent/ultraplan.py — UltraPlan: deep planning module (Claude Code leak feature).

From the leak:
  "An advanced planning module that allows the AI to spend 10-30 minutes
   creating detailed blueprints for complex tasks before execution."

UltraPlan implements a multi-phase planning pipeline that runs BEFORE
task execution. Unlike the normal CoT or deep_reason skill (which reason
inline), UltraPlan produces a persistent structured Blueprint that:
  - Decomposes the task into phases, steps, sub-steps
  - Identifies required tools, skills, and dependencies
  - Estimates complexity and risks per phase
  - Produces a checkpoint plan (EVE can verify each phase completed)
  - Saves the blueprint to disk so it survives context resets
  - Can be resumed mid-way if interrupted

Trigger: any task prefixed with PLAN: or containing keywords like
"build me", "create a full", "architect", "design a system", "implement"
when accompanied by high-complexity signals.

Usage in query_engine.py / main.py:
    from agent.ultraplan import UltraPlan, should_ultraplan
    if should_ultraplan(user_input):
        planner = UltraPlan(tools=tool_registry, working_dir=WORKING_DIR)
        blueprint = planner.plan(user_input)
        print(blueprint.render())
        # then execute phase by phase
        for phase in blueprint.phases:
            result = planner.execute_phase(phase, query_engine)
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("agent.ultraplan")

# ── Trigger detection ─────────────────────────────────────────────────────────
_PLAN_SIGNALS = re.compile(
    r"\b(ultraplan|ultra.plan|build me a|build a full|create a (full|complete|entire)|"
    r"architect|design a system|implement (a|the) (full|complete|entire)|"
    r"from scratch|end.to.end|full stack|comprehensive plan|detailed plan|"
    r"step by step plan|PLAN:)\b",
    re.IGNORECASE,
)
_COMPLEXITY_SIGNALS = re.compile(
    r"\b(database|api|authentication|deployment|multiple (files|modules|services)|"
    r"production|scalab|integrat|microservice|pipeline|workflow)\b",
    re.IGNORECASE,
)


def should_ultraplan(text: str) -> bool:
    """Return True if this request warrants a full UltraPlan blueprint."""
    if text.strip().upper().startswith("PLAN:"):
        return True
    has_plan   = bool(_PLAN_SIGNALS.search(text))
    has_complex = bool(_COMPLEXITY_SIGNALS.search(text))
    word_count  = len(text.split())
    return has_plan or (has_complex and word_count >= 15)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Step:
    id:          str
    description: str
    tool_hint:   str = ""
    skill_hint:  str = ""
    estimated_m: float = 1.0     # estimated minutes
    status:      str = "pending" # pending | done | failed | skipped
    output:      str = ""


@dataclass
class Phase:
    id:          str
    title:       str
    objective:   str
    steps:       list[Step] = field(default_factory=list)
    risk:        str = "low"     # low | medium | high
    depends_on:  list[str] = field(default_factory=list)
    status:      str = "pending"

    def is_complete(self) -> bool:
        return all(s.status in ("done", "skipped") for s in self.steps)

    def progress(self) -> str:
        done = sum(1 for s in self.steps if s.status == "done")
        return f"{done}/{len(self.steps)}"


@dataclass
class Blueprint:
    task:          str
    created_at:    str
    phases:        list[Phase] = field(default_factory=list)
    total_est_min: float = 0.0
    complexity:    str = "medium"    # low | medium | high | extreme
    risks:         list[str] = field(default_factory=list)
    assumptions:   list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    blueprint_file:str = ""

    def render(self) -> str:
        lines = [
            "╔══ UltraPlan Blueprint ══════════════════════════════════════╗",
            f"║ Task       : {self.task[:58]}",
            f"║ Complexity : {self.complexity.upper()}",
            f"║ Est. time  : {self.total_est_min:.0f} min  ({self.total_est_min/60:.1f}h)",
            f"║ Phases     : {len(self.phases)}",
            f"║ Created    : {self.created_at}",
            "╚════════════════════════════════════════════════════════════╝",
            "",
        ]
        for ph in self.phases:
            risk_icon = {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(ph.risk, "⚪")
            dep_str = f" [after: {', '.join(ph.depends_on)}]" if ph.depends_on else ""
            lines.append(f"Phase {ph.id}: {ph.title} {risk_icon}{dep_str}")
            lines.append(f"  Objective: {ph.objective}")
            for st in ph.steps:
                icon = {"done": "✅", "failed": "❌", "skipped": "⏭", "pending": "⬜"}[st.status]
                tool_str = f" [{st.tool_hint or st.skill_hint}]" if (st.tool_hint or st.skill_hint) else ""
                lines.append(f"    {icon} {st.id}. {st.description}{tool_str} (~{st.estimated_m:.0f}m)")
            lines.append("")

        if self.risks:
            lines.append("⚠ Risks:")
            for r in self.risks:
                lines.append(f"  • {r}")
            lines.append("")

        if self.success_criteria:
            lines.append("✓ Success criteria:")
            for c in self.success_criteria:
                lines.append(f"  • {c}")
            lines.append("")

        done_phases = sum(1 for p in self.phases if p.is_complete())
        lines.append(f"Progress: {done_phases}/{len(self.phases)} phases complete")
        if self.blueprint_file:
            lines.append(f"Blueprint saved: {self.blueprint_file}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "created_at": self.created_at,
            "complexity": self.complexity,
            "total_est_min": self.total_est_min,
            "risks": self.risks,
            "assumptions": self.assumptions,
            "success_criteria": self.success_criteria,
            "phases": [
                {
                    "id": ph.id, "title": ph.title, "objective": ph.objective,
                    "risk": ph.risk, "depends_on": ph.depends_on, "status": ph.status,
                    "steps": [
                        {"id": s.id, "description": s.description,
                         "tool_hint": s.tool_hint, "skill_hint": s.skill_hint,
                         "estimated_m": s.estimated_m, "status": s.status,
                         "output": s.output}
                        for s in ph.steps
                    ]
                }
                for ph in self.phases
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Blueprint":
        bp = cls(task=d["task"], created_at=d["created_at"])
        bp.complexity    = d.get("complexity", "medium")
        bp.total_est_min = d.get("total_est_min", 0)
        bp.risks         = d.get("risks", [])
        bp.assumptions   = d.get("assumptions", [])
        bp.success_criteria = d.get("success_criteria", [])
        for ph_d in d.get("phases", []):
            ph = Phase(
                id=ph_d["id"], title=ph_d["title"],
                objective=ph_d["objective"], risk=ph_d.get("risk","low"),
                depends_on=ph_d.get("depends_on",[]),
                status=ph_d.get("status","pending"),
            )
            for s_d in ph_d.get("steps", []):
                ph.steps.append(Step(
                    id=s_d["id"], description=s_d["description"],
                    tool_hint=s_d.get("tool_hint",""), skill_hint=s_d.get("skill_hint",""),
                    estimated_m=s_d.get("estimated_m",1), status=s_d.get("status","pending"),
                    output=s_d.get("output",""),
                ))
            bp.phases.append(ph)
        return bp


# ── Planner ───────────────────────────────────────────────────────────────────

class UltraPlan:
    """
    Deep planning engine. Call plan(task) to produce a Blueprint,
    then execute phase by phase.
    """

    def __init__(self, tools: list | None = None, working_dir: str = "."):
        self._tools       = {t.name: t for t in (tools or [])}
        self._working_dir = Path(working_dir)
        self._plan_dir    = self._working_dir / ".ultraplan"
        self._plan_dir.mkdir(exist_ok=True)

    def plan(self, task: str) -> Blueprint:
        """Generate a Blueprint for the given task using heuristic decomposition."""
        log.info("UltraPlan: planning — %s", task[:80])
        t0 = time.perf_counter()

        bp = Blueprint(
            task=task,
            created_at=time.strftime("%Y-%m-%d %H:%M"),
        )

        # Detect complexity
        bp.complexity = self._assess_complexity(task)

        # Decompose into phases
        bp.phases = self._decompose(task, bp.complexity)

        # Compute total estimate
        bp.total_est_min = sum(
            s.estimated_m for ph in bp.phases for s in ph.steps
        )

        # Identify risks
        bp.risks = self._identify_risks(task, bp.phases)

        # Assumptions
        bp.assumptions = self._identify_assumptions(task)

        # Success criteria
        bp.success_criteria = self._success_criteria(task)

        # Save blueprint
        slug = re.sub(r"[^a-z0-9]", "_", task.lower())[:40]
        fname = f"plan_{int(time.time())}_{slug}.json"
        fpath = self._plan_dir / fname
        fpath.write_text(json.dumps(bp.to_dict(), indent=2), encoding="utf-8")
        bp.blueprint_file = str(fpath)

        elapsed = time.perf_counter() - t0
        log.info("UltraPlan: blueprint ready in %.2fs — %d phases, %.0f min est",
                 elapsed, len(bp.phases), bp.total_est_min)
        return bp

    def load_blueprint(self, path: str | Path) -> Blueprint:
        """Load a saved blueprint from disk (for resumption)."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        bp = Blueprint.from_dict(d)
        bp.blueprint_file = str(path)
        return bp

    def mark_step(self, blueprint: Blueprint, phase_id: str,
                  step_id: str, status: str, output: str = "") -> None:
        """Update a step's status and save the blueprint."""
        for ph in blueprint.phases:
            if ph.id == phase_id:
                for s in ph.steps:
                    if s.id == step_id:
                        s.status = status
                        s.output = output
                if ph.is_complete():
                    ph.status = "done"
        if blueprint.blueprint_file:
            Path(blueprint.blueprint_file).write_text(
                json.dumps(blueprint.to_dict(), indent=2), encoding="utf-8"
            )

    def next_actionable(self, blueprint: Blueprint) -> Phase | None:
        """Return the next phase whose dependencies are all complete."""
        done_ids = {ph.id for ph in blueprint.phases if ph.is_complete()}
        for ph in blueprint.phases:
            if ph.status != "pending":
                continue
            if all(dep in done_ids for dep in ph.depends_on):
                return ph
        return None

    # ── Decomposition heuristics ───────────────────────────────────────────

    def _assess_complexity(self, task: str) -> str:
        score = 0
        low  = task.lower()
        score += len(task.split()) // 10
        complex_kw = ["database","api","auth","deploy","docker","test","migrate",
                      "refactor","pipeline","microservice","async","concurrent"]
        score += sum(2 for k in complex_kw if k in low)
        if score <= 3: return "low"
        if score <= 7: return "medium"
        if score <= 12: return "high"
        return "extreme"

    def _decompose(self, task: str, complexity: str) -> list[Phase]:
        low = task.lower()
        phases: list[Phase] = []

        # Phase 1: Always — Analysis & Setup
        p1 = Phase("1", "Analysis & Setup",
                   "Understand requirements, scaffold environment, verify dependencies")
        p1.steps = [
            Step("1.1", "Parse task requirements and identify unknowns",
                 skill_hint="deep_reason", estimated_m=2),
            Step("1.2", "List all files, tools, and resources needed",
                 tool_hint="list_dir", estimated_m=1),
            Step("1.3", "Verify environment and install missing deps",
                 tool_hint="bash", estimated_m=3),
        ]
        phases.append(p1)

        # Phase 2: Core implementation (varies by task type)
        if any(k in low for k in ("api", "endpoint", "route", "server", "flask", "fastapi")):
            p2 = Phase("2", "API / Server Layer",
                       "Define routes, request/response schemas, and handlers",
                       risk="medium", depends_on=["1"])
            p2.steps = [
                Step("2.1", "Define API schema / OpenAPI spec", tool_hint="write_file", estimated_m=5),
                Step("2.2", "Implement route handlers", tool_hint="write_file", estimated_m=15),
                Step("2.3", "Add error handling and validation", tool_hint="edit_file", estimated_m=5),
            ]
            phases.append(p2)

        if any(k in low for k in ("database", "db", "sql", "sqlite", "postgres", "mongo")):
            p3 = Phase("3", "Data Layer",
                       "Define schema, migrations, and data access layer",
                       risk="medium", depends_on=["1"])
            p3.steps = [
                Step("3.1", "Design schema / models", skill_hint="deep_reason", estimated_m=5),
                Step("3.2", "Write migration / init scripts", tool_hint="write_file", estimated_m=8),
                Step("3.3", "Implement CRUD / repository layer", tool_hint="write_file", estimated_m=10),
            ]
            phases.append(p3)

        if any(k in low for k in ("test", "unittest", "pytest", "spec", "tdd")):
            p_test = Phase("4", "Testing",
                           "Write and run tests for all implemented components",
                           risk="low", depends_on=["2", "3"] if len(phases) > 1 else ["1"])
            p_test.steps = [
                Step("4.1", "Write unit tests", tool_hint="write_file", estimated_m=10),
                Step("4.2", "Write integration tests", tool_hint="write_file", estimated_m=8),
                Step("4.3", "Run test suite and fix failures", tool_hint="bash", estimated_m=5),
            ]
            phases.append(p_test)

        # If no specific phases matched, add a generic implementation phase
        if len(phases) == 1:
            p2 = Phase("2", "Core Implementation",
                       "Build the primary deliverable",
                       risk="medium", depends_on=["1"])
            n_steps = {"low": 2, "medium": 4, "high": 6, "extreme": 8}.get(complexity, 4)
            for i in range(1, n_steps + 1):
                p2.steps.append(
                    Step(f"2.{i}", f"Implementation step {i}",
                         tool_hint="write_file", estimated_m=5)
                )
            phases.append(p2)

        # Final phase: always — Review & Verify
        last_id = str(len(phases) + 1)
        p_final = Phase(last_id, "Review & Verify",
                        "Run final checks, clean up, confirm success criteria",
                        risk="low", depends_on=[phases[-1].id])
        p_final.steps = [
            Step(f"{last_id}.1", "Run the deliverable end-to-end",
                 tool_hint="bash", estimated_m=2),
            Step(f"{last_id}.2", "Check all success criteria",
                 skill_hint="deep_reason", estimated_m=2),
            Step(f"{last_id}.3", "Write summary and update journal",
                 tool_hint="memory_write", estimated_m=1),
        ]
        phases.append(p_final)
        return phases

    def _identify_risks(self, task: str, phases: list[Phase]) -> list[str]:
        risks = []
        low = task.lower()
        if any(k in low for k in ("database","migrate","schema")):
            risks.append("Data migration may fail — ensure backups before running")
        if any(k in low for k in ("deploy","production","server","cloud")):
            risks.append("Production deployment — test in staging first")
        if any(k in low for k in ("auth","login","password","token","jwt")):
            risks.append("Auth implementation — review for security vulnerabilities")
        if len(phases) > 4:
            risks.append("High phase count — context window may need resets between phases")
        if not risks:
            risks.append("No major risks identified — proceed with normal caution")
        return risks

    def _identify_assumptions(self, task: str) -> list[str]:
        assumptions = [
            "Python environment is configured and accessible",
            "Required packages can be installed via pip",
        ]
        low = task.lower()
        if "api" in low:
            assumptions.append("HTTP server port is available")
        if "database" in low or "db" in low:
            assumptions.append("Database service is running and accessible")
        return assumptions

    def _success_criteria(self, task: str) -> list[str]:
        low = task.lower()
        criteria = ["All phases completed without errors"]
        if "test" in low:
            criteria.append("Test suite passes with 0 failures")
        if "api" in low:
            criteria.append("All API endpoints return expected responses")
        if "database" in low:
            criteria.append("Data persists correctly across restarts")
        criteria.append("Code is readable and documented")
        return criteria
