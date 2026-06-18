# -*- coding: utf-8 -*-
"""tools/lsp_tool.py — LSP (Language Server Protocol) tool for code intelligence."""
from __future__ import annotations
import json
from tools.base_tool import BaseTool, ToolResult
from integrations.lsp import LSPClient

_clients: dict[str, LSPClient] = {}


def _get_client(lang: str, command: str) -> LSPClient:
    key = f"{lang}:{command}"
    if key not in _clients:
        c = LSPClient(lang, command)
        c.start()
        _clients[key] = c
    return _clients[key]


class LSPTool(BaseTool):
    name = "lsp"
    description = (
        "Language Server Protocol tool. Get code intelligence: definitions, "
        "hover info, workspace symbols, diagnostics. Requires a running LSP server."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "description": "Operation: definition | hover | symbols | diagnostics | completions",
            },
            "lang": {
                "type": "string",
                "description": "Language name (python, typescript, rust, etc.)",
            },
            "server": {
                "type": "string",
                "description": "LSP server command (e.g. 'pylsp', 'rust-analyzer')",
            },
            "file": {"type": "string", "description": "File path (for definition/hover/diagnostics)"},
            "line": {"type": "integer", "description": "0-indexed line number"},
            "char": {"type": "integer", "description": "0-indexed character position"},
            "query": {"type": "string", "description": "Symbol query (for symbols op)"},
        },
        "required": ["op", "lang", "server"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        op = inp.get("op", "")
        lang = inp.get("lang", "python")
        server = inp.get("server", "pylsp")

        try:
            client = _get_client(lang, server)
        except Exception as e:
            return ToolResult(output=f"Failed to start LSP server: {e}", is_error=True)

        file = inp.get("file", "")
        line = int(inp.get("line", 0))
        char = int(inp.get("char", 0))
        query = inp.get("query", "")

        try:
            if op == "definition":
                result = client.get_definition(file, line, char)
                if not result:
                    return ToolResult(output="No definition found")
                return ToolResult(output=json.dumps(result, indent=2))

            elif op == "hover":
                result = client.get_hover(file, line, char)
                return ToolResult(output=result or "No hover info")

            elif op == "symbols":
                result = client.workspace_symbols(query)
                if not result:
                    return ToolResult(output="No symbols found")
                lines = [f"{s['name']} ({s['file']}:{s['line']})" for s in result[:30]]
                return ToolResult(output="\n".join(lines))

            elif op == "diagnostics":
                diags = client.get_diagnostics(file or None)
                if not diags:
                    return ToolResult(output="No diagnostics")
                lines = []
                for d in diags:
                    sev = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "HINT"}.get(d.get("severity", 0), "?")
                    rng = d.get("range", {}).get("start", {})
                    lines.append(f"[{sev}] {d.get('message','')} @ L{rng.get('line',0)}")
                return ToolResult(output="\n".join(lines))

            elif op == "completions":
                items = client.get_completions(file, line, char)
                return ToolResult(output="\n".join(items) if items else "No completions")

            else:
                return ToolResult(output=f"Unknown op: {op!r}. Use: definition|hover|symbols|diagnostics|completions",
                                  is_error=True)
        except Exception as e:
            return ToolResult(output=f"LSP error: {e}", is_error=True)
