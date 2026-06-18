# -*- coding: utf-8 -*-
"""swarm/backends/inprocess.py — In-process threaded swarm backend."""
from __future__ import annotations
import io
import logging
import threading
import time
import uuid
from .base import SwarmBackend, AgentHandle

log = logging.getLogger("swarm.inprocess")


class InProcessBackend(SwarmBackend):
    """
    Runs each sub-agent in its own thread within the same process.
    No external dependencies required.
    """

    def __init__(self, engine_factory=None):
        self._factory = engine_factory
        self._agents: dict[str, dict] = {}

    def set_factory(self, factory) -> None:
        self._factory = factory

    def spawn(self, agent_id: str, name: str, task: str, system: str = "") -> AgentHandle:
        if not self._factory:
            raise RuntimeError("No engine factory set on InProcessBackend")

        output_buf = io.StringIO()
        handle = AgentHandle(agent_id=agent_id, name=name, task=task, status="running")

        def _run():
            try:
                engine = self._factory(max_turns=20, is_subagent=True)
                result = engine.submit_message(task)
                output_buf.write(result or "")
                handle.status = "done"
            except Exception as e:
                output_buf.write(f"ERROR: {e}")
                handle.status = "error"
                log.error("Swarm agent %s error: %s", agent_id, e)

        t = threading.Thread(target=_run, daemon=True, name=f"swarm-{agent_id}")
        t.start()

        self._agents[agent_id] = {
            "handle": handle,
            "thread": t,
            "output": output_buf,
            "started": time.time(),
        }
        return handle

    def stop(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        agent["handle"].status = "stopped"
        # Threads can't be killed; mark as stopped and let it finish
        return True

    def list_agents(self) -> list[AgentHandle]:
        # Update status based on thread liveness
        for info in self._agents.values():
            if info["handle"].status == "running" and not info["thread"].is_alive():
                info["handle"].status = "done"
        return [info["handle"] for info in self._agents.values()]

    def get_output(self, agent_id: str) -> str:
        agent = self._agents.get(agent_id)
        if not agent:
            return ""
        return agent["output"].getvalue()

    def send_input(self, agent_id: str, text: str) -> bool:
        # In-process agents don't support mid-stream input
        return False

    def wait(self, agent_id: str, timeout: float = 300) -> str:
        agent = self._agents.get(agent_id)
        if not agent:
            return ""
        agent["thread"].join(timeout=timeout)
        return self.get_output(agent_id)

    def wait_all(self, timeout: float = 600) -> dict[str, str]:
        results = {}
        for aid, info in self._agents.items():
            info["thread"].join(timeout=timeout)
            results[aid] = info["output"].getvalue()
        return results

    def is_available(self) -> bool:
        return True
