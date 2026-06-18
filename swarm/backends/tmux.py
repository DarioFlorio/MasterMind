# -*- coding: utf-8 -*-
"""swarm/backends/tmux.py — Tmux-based swarm backend (spawns agents in panes)."""
from __future__ import annotations
import logging
import shutil
import subprocess
import time
from .base import SwarmBackend, AgentHandle

log = logging.getLogger("swarm.tmux")


class TmuxBackend(SwarmBackend):
    """
    Spawns each agent in its own tmux pane.
    Requires tmux to be installed.
    """

    def __init__(self, session_name: str = "mastermind-swarm"):
        self.session_name = session_name
        self._agents: dict[str, dict] = {}
        self._pane_counter = 0

    def is_available(self) -> bool:
        return shutil.which("tmux") is not None

    def _ensure_session(self) -> bool:
        r = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            capture_output=True
        )
        if r.returncode != 0:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self.session_name],
                capture_output=True
            )
        return True

    def spawn(self, agent_id: str, name: str, task: str, system: str = "") -> AgentHandle:
        self._ensure_session()
        self._pane_counter += 1

        # Create new pane
        pane_id = f"{self.session_name}:{self._pane_counter}"
        if self._pane_counter == 1:
            target = f"{self.session_name}:0"
        else:
            subprocess.run(
                ["tmux", "new-window", "-t", self.session_name, "-n", f"agent-{name}"],
                capture_output=True
            )
            target = f"{self.session_name}:{self._pane_counter - 1}"

        # Write task to temp file
        import tempfile, json as _json, sys
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        tmp.write(task)
        tmp.close()

        cmd = f"{sys.executable} -m main --auto -p \"$(cat {tmp.name})\" 2>&1 | tee /tmp/mm-agent-{agent_id}.log"
        subprocess.run(["tmux", "send-keys", "-t", target, cmd, "Enter"], capture_output=True)

        handle = AgentHandle(agent_id=agent_id, name=name, task=task, status="running")
        self._agents[agent_id] = {
            "handle": handle, "pane": target, "log": f"/tmp/mm-agent-{agent_id}.log"
        }
        log.info("Spawned agent %s in tmux pane %s", name, target)
        return handle

    def stop(self, agent_id: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        subprocess.run(["tmux", "send-keys", "-t", agent["pane"], "C-c", ""], capture_output=True)
        agent["handle"].status = "stopped"
        return True

    def list_agents(self) -> list[AgentHandle]:
        for aid, info in self._agents.items():
            if info["handle"].status == "running":
                # Check if log has grown recently (heuristic for completion)
                import os
                log_path = info["log"]
                if os.path.exists(log_path):
                    with open(log_path) as f:
                        content = f.read()
                    if "Goodbye" in content or "Session ended" in content:
                        info["handle"].status = "done"
        return [info["handle"] for info in self._agents.values()]

    def get_output(self, agent_id: str) -> str:
        agent = self._agents.get(agent_id)
        if not agent:
            return ""
        import os
        log_path = agent["log"]
        if os.path.exists(log_path):
            with open(log_path) as f:
                return f.read()
        return ""

    def send_input(self, agent_id: str, text: str) -> bool:
        agent = self._agents.get(agent_id)
        if not agent:
            return False
        subprocess.run(
            ["tmux", "send-keys", "-t", agent["pane"], text, "Enter"],
            capture_output=True
        )
        return True
