# -*- coding: utf-8 -*-
"""tools/team_tool.py — Create and manage swarm agent teams."""
from __future__ import annotations
import json
from tools.base_tool import BaseTool, ToolResult


class TeamCreateTool(BaseTool):
    name = "team_create"
    description = (
        "Create a team of parallel agents. Each agent gets a task and runs concurrently. "
        "Returns team_id for tracking. Use team_status to check progress."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Team name"},
            "agents": {
                "type": "array",
                "description": "List of agents: [{name, task, system?}]",
                "items": {"type": "object"},
            },
            "wait": {
                "type": "boolean",
                "description": "Wait for all agents to finish before returning (default: false)",
            },
        },
        "required": ["name", "agents"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        name = inp.get("name", "team")
        agents = inp.get("agents", [])
        wait = inp.get("wait", False)

        if not agents:
            return ToolResult(output="agents list is required", is_error=True)

        try:
            from swarm.team import team_manager
            team = team_manager.create(name, agents, wait=wait)
            result = {
                "team_id": team.team_id,
                "name": team.name,
                "agents": len(team.agents),
                "status": team.status,
            }
            if wait:
                outputs = team.collect_outputs()
                result["outputs"] = outputs
            return ToolResult(output=json.dumps(result, indent=2))
        except Exception as e:
            return ToolResult(output=f"Team creation failed: {e}", is_error=True)


class TeamDeleteTool(BaseTool):
    name = "team_delete"
    description = "Stop and delete a team of agents."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team ID to delete"},
        },
        "required": ["team_id"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        team_id = inp.get("team_id", "")
        try:
            from swarm.team import team_manager
            if team_manager.delete(team_id):
                return ToolResult(output=f"Team {team_id} stopped and deleted")
            return ToolResult(output=f"Team {team_id} not found", is_error=True)
        except Exception as e:
            return ToolResult(output=f"Error: {e}", is_error=True)


class TeamStatusTool(BaseTool):
    name = "team_status"
    description = "Get status of a team, including per-agent status and outputs."
    input_schema = {
        "type": "object",
        "properties": {
            "team_id": {"type": "string", "description": "Team ID (omit to list all teams)"},
            "collect_outputs": {"type": "boolean", "description": "Include agent outputs (default: false)"},
        },
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        team_id = inp.get("team_id", "")
        collect = inp.get("collect_outputs", False)

        from swarm.team import team_manager

        if not team_id:
            teams = team_manager.list_teams()
            if not teams:
                return ToolResult(output="No active teams")
            return ToolResult(output=json.dumps(teams, indent=2))

        team = team_manager.get(team_id)
        if not team:
            return ToolResult(output=f"Team {team_id} not found", is_error=True)

        summary = team.summary()
        if collect and team.backend:
            summary["outputs"] = team.collect_outputs()
        return ToolResult(output=json.dumps(summary, indent=2))
