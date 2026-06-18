# -*- coding: utf-8 -*-
"""
mcp/client.py — Model Context Protocol client.

Connects to MCP servers and exposes their tools as callable Python objects.
Supports HTTP (SSE), WebSocket, and stdio transports.

Config via .env:
    MCP_SERVERS=asana=https://mcp.asana.com/sse,github=https://mcp.github.com/sse

Or register programmatically:
    from mcp.client import MCPClient
    client = MCPClient("asana", "https://mcp.asana.com/sse")
    result = client.call_tool("asana_create_task", {...})
"""
from __future__ import annotations
import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("mcp.client")


@dataclass
class MCPTool:
    """Represents a tool exposed by an MCP server."""
    name: str
    description: str
    input_schema: dict
    server_name: str

    def to_compact(self) -> str:
        props = self.input_schema.get("properties", {})
        req = set(self.input_schema.get("required", []))
        params = ", ".join(p + ("*" if p in req else "") for p in props)
        return f"{self.name}({params}) -- {self.description[:60]}"


class MCPClient:
    """
    Client for a single MCP server.

    transport: "http" | "ws" | "stdio"
    """

    def __init__(self, name: str, url_or_cmd: str, transport: str = "http",
                 api_key: str = ""):
        self.name = name
        self.url_or_cmd = url_or_cmd
        self.transport = transport
        self.api_key = api_key
        self._tools: list[MCPTool] = []
        self._connected = False
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # ── Connection ────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Establish connection and discover tools."""
        try:
            if self.transport == "stdio":
                return self._connect_stdio()
            else:
                return self._connect_http()
        except Exception as e:
            log.error("MCP connect failed for %s: %s", self.name, e)
            return False

    def disconnect(self) -> None:
        self._connected = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc = None

    # ── Tool discovery ────────────────────────────────────────────────────

    def list_tools(self) -> list[MCPTool]:
        if not self._connected:
            self.connect()
        return list(self._tools)

    # ── Tool invocation ───────────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Call a tool on the MCP server.
        Returns {"result": ..., "error": None} or {"result": None, "error": "..."}
        """
        if not self._connected:
            if not self.connect():
                return {"result": None, "error": f"Cannot connect to MCP server {self.name}"}

        try:
            if self.transport == "stdio":
                return self._call_stdio(tool_name, arguments)
            else:
                return self._call_http(tool_name, arguments)
        except Exception as e:
            log.error("MCP call_tool %s.%s failed: %s", self.name, tool_name, e)
            return {"result": None, "error": str(e)}

    # ── HTTP transport ────────────────────────────────────────────────────

    def _connect_http(self) -> bool:
        try:
            import httpx
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            r = httpx.post(
                f"{self.url_or_cmd}/mcp",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
                headers=headers, timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                self._tools = [
                    MCPTool(
                        name=t["name"],
                        description=t.get("description", ""),
                        input_schema=t.get("inputSchema", {}),
                        server_name=self.name,
                    )
                    for t in data.get("result", {}).get("tools", [])
                ]
                self._connected = True
                log.info("MCP %s: connected, %d tools", self.name, len(self._tools))
                return True
        except Exception as e:
            log.warning("MCP HTTP connect %s: %s", self.name, e)
        return False

    def _call_http(self, tool_name: str, arguments: dict) -> dict:
        import httpx
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "id": int(time.time() * 1000),
            "params": {"name": tool_name, "arguments": arguments},
        }
        r = httpx.post(
            f"{self.url_or_cmd}/mcp",
            json=payload, headers=headers, timeout=30
        )
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            return {"result": None, "error": data["error"].get("message", str(data["error"]))}
        result = data.get("result", {})
        # MCP returns content array
        content = result.get("content", [])
        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return {"result": "\n".join(text_parts), "error": None}

    # ── Stdio transport ───────────────────────────────────────────────────

    def _connect_stdio(self) -> bool:
        try:
            cmd = self.url_or_cmd.split()
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True
            )
            # Send initialize
            self._write_jsonrpc({"jsonrpc": "2.0", "method": "initialize", "id": 0,
                                  "params": {"capabilities": {}}})
            resp = self._read_jsonrpc()
            # List tools
            self._write_jsonrpc({"jsonrpc": "2.0", "method": "tools/list", "id": 1})
            resp = self._read_jsonrpc()
            self._tools = [
                MCPTool(
                    name=t["name"],
                    description=t.get("description", ""),
                    input_schema=t.get("inputSchema", {}),
                    server_name=self.name,
                )
                for t in resp.get("result", {}).get("tools", [])
            ]
            self._connected = True
            return True
        except Exception as e:
            log.warning("MCP stdio connect %s: %s", self.name, e)
            return False

    def _call_stdio(self, tool_name: str, arguments: dict) -> dict:
        call_id = int(time.time() * 1000)
        self._write_jsonrpc({
            "jsonrpc": "2.0", "method": "tools/call", "id": call_id,
            "params": {"name": tool_name, "arguments": arguments}
        })
        resp = self._read_jsonrpc()
        if "error" in resp:
            return {"result": None, "error": resp["error"].get("message", str(resp["error"]))}
        content = resp.get("result", {}).get("content", [])
        text_parts = [c.get("text", "") for c in content if c.get("type") == "text"]
        return {"result": "\n".join(text_parts), "error": None}

    def _write_jsonrpc(self, msg: dict) -> None:
        if self._proc and self._proc.stdin:
            line = json.dumps(msg) + "\n"
            self._proc.stdin.write(line)
            self._proc.stdin.flush()

    def _read_jsonrpc(self) -> dict:
        if self._proc and self._proc.stdout:
            line = self._proc.stdout.readline()
            return json.loads(line)
        return {}

    def __repr__(self) -> str:
        status = "connected" if self._connected else "disconnected"
        return f"<MCPClient {self.name!r} {self.transport} {status} tools={len(self._tools)}>"
