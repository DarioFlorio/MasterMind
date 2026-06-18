# -*- coding: utf-8 -*-
"""
swarm/team.py — Team management for multi-agent swarms.

MasterMind team orchestration.

Usage:
    from swarm.team import team_manager

    team = team_manager.create("research-team", [
        {"name": "searcher", "task": "Search the web for X"},
        {"name": "analyst",  "task": "Analyse the results"},
    ])
    team.wait_all()
    results = team.collect_outputs()
"""
from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .backends.base import AgentHandle, SwarmBackend

if TYPE_CHECKING:
    pass

log = logging.getLogger("swarm.team")


@dataclass
class AgentSpec:
    name: str
    task: str
    system: str = ""


@dataclass
class Team:
    team_id: str
    name: str
    agents: list[AgentHandle] = field(default_factory=list)
    backend: SwarmBackend | None = None
    created_at: float = field(default_factory=time.time)
    status: str = "running"  # running | done | stopped

    def wait_all(self, timeout: float = 600) -> None:
        """Block until all agents finish."""
        from .backends.inprocess import InProcessBackend
        if isinstance(self.backend, InProcessBackend):
            for handle in self.agents:
                self.backend.wait(handle.agent_id, timeout=timeout)
        else:
            deadline = time.time() + timeout
            while time.time() < deadline:
                statuses = [a.status for a in self.agents]
                if all(s in ("done", "error", "stopped") for s in statuses):
                    break
                time.sleep(1)
        self.status = "done"

    def collect_outputs(self) -> dict[str, str]:
        """Collect outputs from all agents."""
        if not self.backend:
            return {}
        return {
            handle.name: self.backend.get_output(handle.agent_id)
            for handle in self.agents
        }

    def stop_all(self) -> None:
        if self.backend:
            for handle in self.agents:
                self.backend.stop(handle.agent_id)
        self.status = "stopped"

    def summary(self) -> dict:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "status": self.status,
            "agents": [
                {"id": a.agent_id, "name": a.name, "status": a.status}
                for a in self.agents
            ],
            "created_at": self.created_at,
        }


class TeamManager:
    def __init__(self):
        self._teams: dict[str, Team] = {}
        self._backend: SwarmBackend | None = None
        self._engine_factory = None

    def set_backend(self, backend: SwarmBackend) -> None:
        self._backend = backend

    def set_engine_factory(self, factory) -> None:
        self._engine_factory = factory
        # Wire factory into in-process backend if that's what we have
        from .backends.inprocess import InProcessBackend
        if isinstance(self._backend, InProcessBackend):
            self._backend.set_factory(factory)

    def _get_backend(self) -> SwarmBackend:
        if self._backend is None:
            from .backends import detect_best_backend
            self._backend = detect_best_backend()
            if self._engine_factory:
                from .backends.inprocess import InProcessBackend
                if isinstance(self._backend, InProcessBackend):
                    self._backend.set_factory(self._engine_factory)
        return self._backend

    def create(self, name: str, agent_specs: list[dict | AgentSpec],
                wait: bool = False) -> Team:
        """
        Create and launch a team of agents.

        agent_specs: list of dicts with keys: name, task, [system]
        """
        team_id = str(uuid.uuid4())[:8]
        backend = self._get_backend()
        team = Team(team_id=team_id, name=name, backend=backend)

        for spec in agent_specs:
            if isinstance(spec, dict):
                spec = AgentSpec(**spec)
            agent_id = str(uuid.uuid4())[:8]
            try:
                handle = backend.spawn(
                    agent_id=agent_id,
                    name=spec.name,
                    task=spec.task,
                    system=spec.system,
                )
                team.agents.append(handle)
                log.info("Team %s: spawned agent %s (%s)", name, spec.name, agent_id)
            except Exception as e:
                log.error("Failed to spawn agent %s: %s", spec.name, e)

        self._teams[team_id] = team
        log.info("Team %r created with %d agents (id=%s)", name, len(team.agents), team_id)

        from hooks.manager import hook_manager
        hook_manager.fire("swarm_create", team_id=team_id, name=name,
                          agent_count=len(team.agents))

        if wait:
            team.wait_all()

        return team

    def get(self, team_id: str) -> Team | None:
        return self._teams.get(team_id)

    def delete(self, team_id: str) -> bool:
        team = self._teams.get(team_id)
        if not team:
            return False
        team.stop_all()
        del self._teams[team_id]
        from hooks.manager import hook_manager
        hook_manager.fire("swarm_destroy", team_id=team_id)
        log.info("Team %s deleted", team_id)
        return True

    def list_teams(self) -> list[dict]:
        return [t.summary() for t in self._teams.values()]

    def __repr__(self) -> str:
        return f"<TeamManager {len(self._teams)} teams>"


# Singleton
team_manager = TeamManager()
