# -*- coding: utf-8 -*-
"""
mcp/registry.py — Global MCP server registry.

Auto-loads servers from MCP_SERVERS env var:
    MCP_SERVERS=github=https://mcp.github.com/sse,asana=https://mcp.asana.com/sse

Or from .mcp.json in project root:
    {
      "servers": [
        {"name": "github", "url": "https://mcp.github.com/sse", "transport": "http"}
      ]
    }
"""
from __future__ import annotations
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import MCPClient, MCPTool

log = logging.getLogger("mcp.registry")


class MCPRegistry:
    def __init__(self):
        self._servers: dict[str, "MCPClient"] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._loaded = True
            self._autoload()

    def _autoload(self) -> None:
        from .client import MCPClient

        # From environment variable
        env_servers = os.environ.get("MCP_SERVERS", "")
        for entry in env_servers.split(","):
            entry = entry.strip()
            if "=" in entry:
                name, url = entry.split("=", 1)
                self._servers[name.strip()] = MCPClient(name.strip(), url.strip())

        # From .mcp.json
        for cfg_path in (Path.cwd() / ".mcp.json", Path.home() / ".mcp.json"):
            if cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    for srv in cfg.get("servers", []):
                        name = srv["name"]
                        url = srv.get("url", srv.get("command", ""))
                        transport = srv.get("transport", "http" if url.startswith("http") else "stdio")
                        api_key = srv.get("api_key", "")
                        if name not in self._servers:
                            self._servers[name] = MCPClient(name, url, transport, api_key)
                except Exception as e:
                    log.warning("Failed to load %s: %s", cfg_path, e)
                break

        if self._servers:
            log.info("MCP registry: %d servers auto-loaded", len(self._servers))

    def register(self, client: "MCPClient") -> None:
        self._servers[client.name] = client
        log.info("MCP registered: %s", client.name)

    def add(self, name: str, url: str, transport: str = "http", api_key: str = "") -> "MCPClient":
        from .client import MCPClient
        c = MCPClient(name, url, transport, api_key)
        self._servers[name] = c
        return c

    def get(self, name: str) -> "MCPClient | None":
        self._ensure_loaded()
        return self._servers.get(name)

    def all_tools(self) -> list["MCPTool"]:
        """Return all tools from all connected servers."""
        self._ensure_loaded()
        tools = []
        for client in self._servers.values():
            try:
                tools.extend(client.list_tools())
            except Exception as e:
                log.warning("Failed to list tools from %s: %s", client.name, e)
        return tools

    def call(self, server_name: str, tool_name: str, arguments: dict) -> dict:
        self._ensure_loaded()
        client = self._servers.get(server_name)
        if not client:
            return {"result": None, "error": f"Unknown MCP server: {server_name!r}"}
        return client.call_tool(tool_name, arguments)

    def list_servers(self) -> list[str]:
        self._ensure_loaded()
        return list(self._servers.keys())

    def disconnect_all(self) -> None:
        for c in self._servers.values():
            try:
                c.disconnect()
            except Exception:
                pass

    def status(self) -> dict:
        self._ensure_loaded()
        return {
            name: {"connected": c._connected, "tools": len(c._tools), "transport": c.transport}
            for name, c in self._servers.items()
        }

    def __repr__(self) -> str:
        return f"<MCPRegistry {len(self._servers)} servers>"


# Singleton
mcp_registry = MCPRegistry()
