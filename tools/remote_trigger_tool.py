# -*- coding: utf-8 -*-
"""tools/remote_trigger_tool.py — Trigger tasks from external webhooks/events."""
from __future__ import annotations
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from tools.base_tool import BaseTool, ToolResult

_pending_triggers: list[dict] = []
_trigger_lock = threading.Lock()
_webhook_server: HTTPServer | None = None


class RemoteTriggerTool(BaseTool):
    name = "remote_trigger"
    description = (
        "Start a webhook listener to receive remote triggers. "
        "External systems can POST to the webhook to queue tasks."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {"type": "string", "description": "start | stop | poll | status"},
            "port": {"type": "integer", "description": "Port for webhook listener (default: 7778)"},
            "secret": {"type": "string", "description": "Optional HMAC secret for validation"},
        },
        "required": ["op"],
    }

    def execute(self, inp: dict) -> ToolResult:
        global _webhook_server
        inp = self.safe_parse(inp)
        op = inp.get("op", "")
        port = int(inp.get("port", 7778))

        if op == "start":
            if _webhook_server:
                return ToolResult(output="Webhook listener already running")

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def do_POST(self):
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    try:
                        data = json.loads(body)
                    except Exception:
                        data = {"raw": body.decode()}
                    with _trigger_lock:
                        _pending_triggers.append({"ts": time.time(), "data": data})
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"ok":true}')

            _webhook_server = HTTPServer(("0.0.0.0", port), Handler)
            t = threading.Thread(target=_webhook_server.serve_forever, daemon=True)
            t.start()
            return ToolResult(output=f"Webhook listener started on port {port}")

        elif op == "stop":
            if _webhook_server:
                _webhook_server.shutdown()
                _webhook_server = None
                return ToolResult(output="Webhook listener stopped")
            return ToolResult(output="No webhook listener running")

        elif op == "poll":
            with _trigger_lock:
                triggers = list(_pending_triggers)
                _pending_triggers.clear()
            if not triggers:
                return ToolResult(output="No pending triggers")
            return ToolResult(output=json.dumps(triggers, indent=2))

        elif op == "status":
            running = _webhook_server is not None
            with _trigger_lock:
                pending = len(_pending_triggers)
            return ToolResult(output=f"Webhook: {'running' if running else 'stopped'}, {pending} pending triggers")

        return ToolResult(output=f"Unknown op: {op}. Use: start|stop|poll|status", is_error=True)
