# -*- coding: utf-8 -*-
"""tools/mcp_tool.py — Invoke tools from connected MCP servers."""
from __future__ import annotations
import json
from tools.base_tool import BaseTool, ToolResult


class MCPInvokeTool(BaseTool):
    name = "mcp"
    description = (
        "Invoke a tool on a connected MCP (Model Context Protocol) server. "
        "Use mcp_list_servers to see available servers and their tools."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "server": {"type": "string", "description": "MCP server name"},
            "tool": {"type": "string", "description": "Tool name on the server"},
            "arguments": {"type": "object", "description": "Tool arguments"},
        },
        "required": ["server", "tool"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        server = inp.get("server", "")
        tool = inp.get("tool", "")
        arguments = inp.get("arguments", {})

        if not server or not tool:
            return ToolResult(output="server and tool are required", is_error=True)

        from mcp.registry import mcp_registry
        from hooks.manager import hook_manager
        from telemetry.logger import telemetry

        hook_manager.fire("mcp_call:pre", server=server, tool=tool, arguments=arguments)
        telemetry.mcp_call(server=server, tool=tool)

        result = mcp_registry.call(server, tool, arguments)

        hook_manager.fire("mcp_call:post", server=server, tool=tool, result=result)

        if result.get("error"):
            return ToolResult(output=f"MCP error: {result['error']}", is_error=True)
        output = result.get("result", "")
        if not isinstance(output, str):
            output = json.dumps(output, indent=2)
        return ToolResult(output=output or "(no output)")


class MCPListServersTool(BaseTool):
    name = "mcp_list_servers"
    description = "List all connected MCP servers and their available tools."
    input_schema = {"type": "object", "properties": {}}

    def execute(self, inp: dict) -> ToolResult:
        from mcp.registry import mcp_registry
        status = mcp_registry.status()
        if not status:
            return ToolResult(output=(
                "No MCP servers configured.\n"
                "Set MCP_SERVERS=name=url in .env or create .mcp.json"
            ))
        lines = []
        for name, info in status.items():
            conn = "✓" if info["connected"] else "✗"
            lines.append(f"  {conn} {name} ({info['transport']}) — {info['tools']} tools")
            # List tool names
            client = mcp_registry.get(name)
            if client and client._tools:
                for t in client._tools[:10]:
                    lines.append(f"      · {t.name}: {t.description[:50]}")
                if len(client._tools) > 10:
                    lines.append(f"      … and {len(client._tools)-10} more")
        return ToolResult(output="MCP Servers:\n" + "\n".join(lines))
