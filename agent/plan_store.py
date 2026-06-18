"""
agent/plan_store.py — Resumable plan artifact persistence.

GAP IMPLEMENTED: Resumable plan artifacts — plans lost on context reset.
EVE now persists UltraPlan Blueprints to disk and can resume an incomplete
plan after a context reset or restart.

Usage:
    from agent.plan_store import PlanStore
    store = PlanStore(working_dir)

    # Save a blueprint after it's been created
    store.save(blueprint)

    # On startup, check for an incomplete plan
    incomplete = store.get_resumable()
    if incomplete:
        engine.inject_plan(incomplete)

    # Mark a phase complete
    store.mark_phase_done(blueprint_id, phase_index)

    # Mark blueprint fully done
    store.complete(blueprint_id)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PlanPhaseRecord:
    index:       int
    title:       str
    status:      str = "pending"   # pending | done | failed | skipped
    ts_done:     float = 0.0
    error:       str = ""


@dataclass
class PlanArtifact:
    """Serialisable snapshot of an UltraPlan Blueprint."""
    blueprint_id:  str
    goal:          str
    created_at:    float = field(default_factory=time.time)
    last_updated:  float = field(default_factory=time.time)
    status:        str = "active"   # active | complete | abandoned
    phases:        list[PlanPhaseRecord] = field(default_factory=list)
    raw_blueprint: dict = field(default_factory=dict)  # full Blueprint JSON


class PlanStore:
    """
    Persists plan artifacts to memdir/plans/ so they survive context resets.

    Each plan is stored as JSON at memdir/plans/{blueprint_id}.json.
    An index file memdir/plans/index.json tracks all plans.
    """

    def __init__(self, working_dir: str = "") -> None:
        base = Path(working_dir) if working_dir else Path.cwd()
        self._plans_dir  = base / "memdir" / "plans"
        self._index_path = self._plans_dir / "index.json"
        self._plans_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict] = self._load_index()

    # ── Public API ────────────────────────────────────────────────────────────

    def save(self, blueprint) -> str:
        """
        Save a Blueprint (from ultraplan.py) to disk.
        Returns the blueprint_id.
        """
        bid = getattr(blueprint, "id", None) or f"plan_{int(time.time())}"

        phases = []
        for i, phase in enumerate(getattr(blueprint, "phases", [])):
            phase_status = "done" if getattr(phase, "completed", False) else "pending"
            phases.append(PlanPhaseRecord(
                index=i,
                title=getattr(phase, "title", f"Phase {i}"),
                status=phase_status,
            ))

        # Serialise blueprint as dict
        try:
            raw = {
                "id":    bid,
                "goal":  getattr(blueprint, "goal", ""),
                "phases": [
                    {
                        "title":       getattr(p, "title", ""),
                        "description": getattr(p, "description", ""),
                        "steps":       [
                            {"id": getattr(s, "id", ""), "description": getattr(s, "description", ""),
                             "status": getattr(s, "status", "pending")}
                            for s in getattr(p, "steps", [])
                        ],
                        "completed": getattr(p, "completed", False),
                    }
                    for p in getattr(blueprint, "phases", [])
                ],
            }
        except Exception:
            raw = {"id": bid}

        artifact = PlanArtifact(
            blueprint_id=bid,
            goal=getattr(blueprint, "goal", ""),
            phases=phases,
            raw_blueprint=raw,
        )
        self._write(artifact)
        self._index[bid] = {
            "goal":       artifact.goal,
            "status":     artifact.status,
            "created_at": artifact.created_at,
            "last_updated": artifact.last_updated,
        }
        self._save_index()
        return bid

    def get_resumable(self) -> Optional[dict]:
        """
        Return the most recent active (incomplete) plan, or None.
        Returns the raw_blueprint dict so it can be re-injected into context.
        """
        candidates = [
            (meta["last_updated"], bid)
            for bid, meta in self._index.items()
            if meta.get("status") == "active"
        ]
        if not candidates:
            return None
        candidates.sort(reverse=True)
        _, latest_bid = candidates[0]
        artifact = self._read(latest_bid)
        if artifact is None:
            return None
        return artifact.raw_blueprint

    def mark_phase_done(self, blueprint_id: str, phase_index: int) -> None:
        """Mark a specific phase as completed."""
        artifact = self._read(blueprint_id)
        if artifact is None:
            return
        for p in artifact.phases:
            if p.index == phase_index:
                p.status = "done"
                p.ts_done = time.time()
                break
        artifact.last_updated = time.time()
        self._write(artifact)
        self._index[blueprint_id]["last_updated"] = artifact.last_updated
        self._save_index()

    def complete(self, blueprint_id: str) -> None:
        """Mark blueprint as fully completed."""
        artifact = self._read(blueprint_id)
        if artifact:
            artifact.status = "complete"
            artifact.last_updated = time.time()
            self._write(artifact)
        if blueprint_id in self._index:
            self._index[blueprint_id]["status"] = "complete"
            self._index[blueprint_id]["last_updated"] = time.time()
            self._save_index()

    def abandon(self, blueprint_id: str) -> None:
        """Mark blueprint as abandoned (won't show in resumable)."""
        if blueprint_id in self._index:
            self._index[blueprint_id]["status"] = "abandoned"
            self._save_index()

    def list_all(self) -> list[dict]:
        """List all plans with their metadata."""
        return [
            {"id": bid, **meta}
            for bid, meta in sorted(
                self._index.items(),
                key=lambda x: x[1].get("last_updated", 0),
                reverse=True,
            )
        ]

    def render_resumable(self) -> str:
        """
        Return a human-readable summary of the resumable plan for injection
        into the system prompt on startup.
        """
        plan = self.get_resumable()
        if not plan:
            return ""
        goal   = plan.get("goal", "")
        phases = plan.get("phases", [])
        done   = [p for p in phases if p.get("completed")]
        total  = len(phases)
        pct    = int(100 * len(done) / total) if total else 0
        lines  = [
            f"[PlanStore] RESUMABLE PLAN ({pct}% complete): {goal}",
        ]
        for i, p in enumerate(phases):
            mark = "✓" if p.get("completed") else "○"
            lines.append(f"  {mark} Phase {i+1}: {p.get('title', '?')}")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _path_for(self, bid: str) -> Path:
        return self._plans_dir / f"{bid}.json"

    def _write(self, artifact: PlanArtifact) -> None:
        data = {
            "blueprint_id":  artifact.blueprint_id,
            "goal":          artifact.goal,
            "created_at":    artifact.created_at,
            "last_updated":  artifact.last_updated,
            "status":        artifact.status,
            "phases":        [
                {"index": p.index, "title": p.title, "status": p.status,
                 "ts_done": p.ts_done, "error": p.error}
                for p in artifact.phases
            ],
            "raw_blueprint": artifact.raw_blueprint,
        }
        self._path_for(artifact.blueprint_id).write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

    def _read(self, bid: str) -> Optional[PlanArtifact]:
        p = self._path_for(bid)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            phases = [
                PlanPhaseRecord(**ph)
                for ph in data.get("phases", [])
            ]
            return PlanArtifact(
                blueprint_id=data["blueprint_id"],
                goal=data.get("goal", ""),
                created_at=data.get("created_at", 0),
                last_updated=data.get("last_updated", 0),
                status=data.get("status", "active"),
                phases=phases,
                raw_blueprint=data.get("raw_blueprint", {}),
            )
        except Exception:
            return None

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_index(self) -> None:
        self._index_path.write_text(
            json.dumps(self._index, indent=2), encoding="utf-8"
        )
