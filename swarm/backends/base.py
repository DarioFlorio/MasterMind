# -*- coding: utf-8 -*-
"""swarm/backends/base.py — Abstract swarm backend."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class AgentHandle:
    agent_id: str
    name: str
    task: str
    status: str = "running"  # running | done | error | stopped


class SwarmBackend(ABC):
    """Abstract base for swarm execution backends (tmux, in-process, etc.)."""

    @abstractmethod
    def spawn(self, agent_id: str, name: str, task: str, system: str = "") -> AgentHandle:
        """Spawn a new agent. Returns a handle."""
        ...

    @abstractmethod
    def stop(self, agent_id: str) -> bool:
        """Stop a running agent."""
        ...

    @abstractmethod
    def list_agents(self) -> list[AgentHandle]:
        """List all agents managed by this backend."""
        ...

    @abstractmethod
    def get_output(self, agent_id: str) -> str:
        """Get accumulated output from an agent."""
        ...

    @abstractmethod
    def send_input(self, agent_id: str, text: str) -> bool:
        """Send input to an agent."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        return True
