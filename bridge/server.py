# -*- coding: utf-8 -*-
"""
bridge/server.py — HTTP bridge server for remote MasterMind execution.

Exposes the agent engine over a REST+SSE API so IDE extensions,
remote clients, and CI pipelines can drive MasterMind without a terminal.

Endpoints:
  POST /v1/message          — submit a prompt, stream reply as SSE
  POST /v1/tool             — invoke a tool directly
  GET  /v1/session          — get session info
  DELETE /v1/session        — clear session
  GET  /v1/tools            — list available tools
  GET  /v1/skills           — list available skills
  GET  /health              — health check

Start: python -m bridge.server --port 7777
"""
from __future__ import annotations
import json
import logging
import threading
import time
from typing import TYPE_CHECKING

log = logging.getLogger("bridge.server")

if TYPE_CHECKING:
    from agent.query_engine import QueryEngine


class BridgeServer:
    """Lightweight HTTP bridge exposing the QueryEngine over REST/SSE."""

    def __init__(self, engine: "QueryEngine | None" = None,
                 host: str = "127.0.0.1", port: int = 7777,
                 api_key: str = ""):
        self.engine = engine
        self.host = host
        self.port = port
        self.api_key = api_key
        self._server = None
        self._thread: threading.Thread | None = None

    def set_engine(self, engine: "QueryEngine") -> None:
        self.engine = engine

    def start(self, daemon: bool = True) -> None:
        """Start the bridge server in a background thread."""
        self._thread = threading.Thread(
            target=self._serve, daemon=daemon, name="bridge-server"
        )
        self._thread.start()
        log.info("Bridge server started at http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None

    def _serve(self) -> None:
        from http.server import HTTPServer, BaseHTTPRequestHandler

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                log.debug(fmt, *args)

            def _auth(self) -> bool:
                if not server.api_key:
                    return True
                auth = self.headers.get("Authorization", "")
                return auth == f"Bearer {server.api_key}"

            def _json(self, data: dict, status: int = 200) -> None:
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _sse_start(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

            def _sse_write(self, data: str) -> None:
                line = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(line.encode())
                self.wfile.flush()

            def do_GET(self):
                if not self._auth():
                    self._json({"error": "unauthorized"}, 401); return

                if self.path == "/health":
                    self._json({"status": "ok", "engine": server.engine is not None})
                elif self.path == "/v1/session":
                    if server.engine:
                        self._json({
                            "messages": len(server.engine.session),
                            "session_id": server.engine.session.session_id,
                        })
                    else:
                        self._json({"error": "no engine"}, 503)
                elif self.path == "/v1/tools":
                    if server.engine:
                        tools = [
                            {"name": t.name, "description": t.description}
                            for t in server.engine.tools.values()
                        ]
                        self._json({"tools": tools})
                    else:
                        self._json({"tools": []})
                elif self.path == "/v1/skills":
                    try:
                        from tools.skill_tool import SkillTool
                        st = SkillTool()
                        result = st.execute({})
                        self._json({"skills": result.output})
                    except Exception as e:
                        self._json({"error": str(e)}, 500)
                else:
                    self._json({"error": "not found"}, 404)

            def do_DELETE(self):
                if not self._auth():
                    self._json({"error": "unauthorized"}, 401); return
                if self.path == "/v1/session" and server.engine:
                    server.engine.session.clear()
                    self._json({"status": "cleared"})
                else:
                    self._json({"error": "not found"}, 404)

            def do_POST(self):
                if not self._auth():
                    self._json({"error": "unauthorized"}, 401); return

                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}

                if self.path == "/v1/message":
                    prompt = body.get("prompt", "")
                    stream = body.get("stream", True)
                    if not prompt:
                        self._json({"error": "prompt required"}, 400); return
                    if not server.engine:
                        self._json({"error": "engine not ready"}, 503); return

                    if stream:
                        self._sse_start()
                        chunks = []
                        def _chunk(c):
                            chunks.append(c)
                            self._sse_write({"type": "chunk", "text": c})
                        old_cb = getattr(server.engine, "_on_chunk", None)
                        server.engine._on_chunk = _chunk
                        try:
                            result = server.engine.submit_message(prompt)
                            self._sse_write({"type": "done", "text": result or "".join(chunks)})
                        except Exception as e:
                            self._sse_write({"type": "error", "error": str(e)})
                        finally:
                            server.engine._on_chunk = old_cb
                    else:
                        try:
                            result = server.engine.submit_message(prompt)
                            self._json({"result": result})
                        except Exception as e:
                            self._json({"error": str(e)}, 500)

                elif self.path == "/v1/tool":
                    tool_name = body.get("tool", "")
                    inp = body.get("input", {})
                    if not server.engine or tool_name not in server.engine.tools:
                        self._json({"error": f"tool {tool_name!r} not found"}, 404); return
                    try:
                        result = server.engine.tools[tool_name].execute(inp)
                        self._json({"output": result.output, "is_error": result.is_error})
                    except Exception as e:
                        self._json({"error": str(e)}, 500)
                else:
                    self._json({"error": "not found"}, 404)

        httpd = HTTPServer((self.host, self.port), Handler)
        self._server = httpd
        httpd.serve_forever()


# CLI entry point
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7777)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--api-key", default="")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    srv = BridgeServer(host=args.host, port=args.port, api_key=args.api_key)
    print(f"Bridge server on http://{args.host}:{args.port}")
    srv._serve()
