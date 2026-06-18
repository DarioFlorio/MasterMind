# -*- coding: utf-8 -*-
"""
integrations/lsp.py — Language Server Protocol client.

MasterMind built-in's LSPTool. Connects to running language servers
(pylsp, clangd, tsserver, rust-analyzer, etc.) to provide:
  - Go-to-definition
  - Symbol lookup / workspace symbols
  - Diagnostics (errors/warnings)
  - Hover information
  - Code completion hints

Servers auto-detected or configured via LSP_SERVERS env:
    LSP_SERVERS=python=pylsp,typescript=typescript-language-server --stdio

Usage:
    from integrations.lsp import LSPClient
    lsp = LSPClient("python", "pylsp")
    lsp.open_file("main.py")
    defs = lsp.get_definition("main.py", line=10, char=5)
    syms = lsp.workspace_symbols("MyClass")
    diags = lsp.get_diagnostics("main.py")
"""
from __future__ import annotations
import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("integrations.lsp")


class LSPClient:
    """
    LSP client using stdio transport (most common for language servers).
    """

    def __init__(self, lang: str, command: str | list[str],
                 workspace: str | None = None):
        self.lang = lang
        self.command = command if isinstance(command, list) else command.split()
        self.workspace = workspace or str(Path.cwd())
        self._proc: subprocess.Popen | None = None
        self._msg_id = 0
        self._pending: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._open_files: set[str] = set()
        self._diagnostics: dict[str, list[dict]] = {}
        self._ready = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._reader = threading.Thread(
                target=self._read_loop, daemon=True, name=f"lsp-{self.lang}"
            )
            self._reader.start()
            self._initialize()
            return True
        except Exception as e:
            log.error("LSP start failed for %s: %s", self.lang, e)
            return False

    def stop(self) -> None:
        if self._proc:
            try:
                self._send_notification("exit")
                self._proc.terminate()
            except Exception:
                pass
            self._proc = None
        self._ready = False

    def _initialize(self) -> None:
        result = self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": f"file://{self.workspace}",
            "capabilities": {
                "textDocument": {
                    "definition": {"dynamicRegistration": False},
                    "hover": {"dynamicRegistration": False},
                    "completion": {"dynamicRegistration": False},
                    "publishDiagnostics": {"dynamicRegistration": False},
                },
                "workspace": {
                    "symbol": {"dynamicRegistration": False},
                },
            },
            "initializationOptions": {},
        })
        self._send_notification("initialized", {})
        self._ready = True
        log.info("LSP %s initialized", self.lang)

    # ── Public API ─────────────────────────────────────────────────────────

    def open_file(self, path: str) -> None:
        abs_path = str(Path(path).resolve())
        if abs_path in self._open_files:
            return
        try:
            content = Path(abs_path).read_text(encoding="utf-8")
        except Exception as e:
            log.warning("LSP: cannot read %s: %s", path, e)
            return
        lang_id = self._lang_id(abs_path)
        self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": f"file://{abs_path}",
                "languageId": lang_id,
                "version": 1,
                "text": content,
            }
        })
        self._open_files.add(abs_path)

    def get_definition(self, path: str, line: int, char: int) -> list[dict]:
        """Get definition locations for symbol at (line, char). Lines are 0-indexed."""
        self.open_file(path)
        abs_path = str(Path(path).resolve())
        result = self._request("textDocument/definition", {
            "textDocument": {"uri": f"file://{abs_path}"},
            "position": {"line": line, "character": char},
        })
        return self._normalise_locations(result)

    def get_hover(self, path: str, line: int, char: int) -> str:
        """Get hover information for symbol at (line, char)."""
        self.open_file(path)
        abs_path = str(Path(path).resolve())
        result = self._request("textDocument/hover", {
            "textDocument": {"uri": f"file://{abs_path}"},
            "position": {"line": line, "character": char},
        })
        if not result:
            return ""
        contents = result.get("contents", "")
        if isinstance(contents, dict):
            return contents.get("value", "")
        if isinstance(contents, list):
            return " ".join(
                c.get("value", c) if isinstance(c, dict) else c
                for c in contents
            )
        return str(contents)

    def workspace_symbols(self, query: str) -> list[dict]:
        """Search workspace for symbols matching query."""
        result = self._request("workspace/symbol", {"query": query})
        if not result:
            return []
        return [
            {
                "name": s.get("name", ""),
                "kind": s.get("kind", 0),
                "file": s.get("location", {}).get("uri", "").replace("file://", ""),
                "line": s.get("location", {}).get("range", {}).get("start", {}).get("line", 0),
            }
            for s in (result if isinstance(result, list) else [])
        ]

    def get_diagnostics(self, path: str | None = None) -> list[dict]:
        """Return cached diagnostics (populated by server push notifications)."""
        if path:
            abs_path = str(Path(path).resolve())
            return self._diagnostics.get(f"file://{abs_path}", [])
        all_diags = []
        for diags in self._diagnostics.values():
            all_diags.extend(diags)
        return all_diags

    def get_completions(self, path: str, line: int, char: int) -> list[str]:
        self.open_file(path)
        abs_path = str(Path(path).resolve())
        result = self._request("textDocument/completion", {
            "textDocument": {"uri": f"file://{abs_path}"},
            "position": {"line": line, "character": char},
        })
        if not result:
            return []
        items = result.get("items", result) if isinstance(result, dict) else result
        return [item.get("label", "") for item in (items or [])][:20]

    # ── Transport ──────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def _send(self, msg: dict) -> None:
        if not self._proc or not self._proc.stdin:
            return
        body = json.dumps(msg)
        header = f"Content-Length: {len(body)}\r\n\r\n"
        try:
            self._proc.stdin.write((header + body).encode())
            self._proc.stdin.flush()
        except Exception as e:
            log.error("LSP send error: %s", e)

    def _request(self, method: str, params: dict, timeout: float = 10.0) -> Any:
        req_id = self._next_id()
        event = threading.Event()
        self._pending[req_id] = {"event": event, "result": None}
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        event.wait(timeout=timeout)
        return self._pending.pop(req_id, {}).get("result")

    def _send_notification(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _read_loop(self) -> None:
        buf = b""
        while self._proc and self._proc.poll() is None:
            try:
                chunk = self._proc.stdout.read(1)
                if not chunk:
                    break
                buf += chunk
                while b"\r\n\r\n" in buf:
                    header, rest = buf.split(b"\r\n\r\n", 1)
                    length = 0
                    for line in header.decode().split("\r\n"):
                        if line.lower().startswith("content-length:"):
                            length = int(line.split(":", 1)[1].strip())
                    if len(rest) < length:
                        buf = header + b"\r\n\r\n" + rest
                        break
                    body = rest[:length]
                    buf = rest[length:]
                    try:
                        self._handle_message(json.loads(body))
                    except Exception as e:
                        log.warning("LSP parse error: %s", e)
            except Exception:
                break

    def _handle_message(self, msg: dict) -> None:
        if "id" in msg and msg["id"] in self._pending:
            entry = self._pending[msg["id"]]
            entry["result"] = msg.get("result")
            entry["event"].set()
        elif msg.get("method") == "textDocument/publishDiagnostics":
            params = msg.get("params", {})
            uri = params.get("uri", "")
            self._diagnostics[uri] = params.get("diagnostics", [])

    def _normalise_locations(self, result: Any) -> list[dict]:
        if not result:
            return []
        locs = result if isinstance(result, list) else [result]
        out = []
        for loc in locs:
            uri = loc.get("uri", "").replace("file://", "")
            start = loc.get("range", {}).get("start", {})
            out.append({
                "file": uri,
                "line": start.get("line", 0),
                "character": start.get("character", 0),
            })
        return out

    @staticmethod
    def _lang_id(path: str) -> str:
        ext = Path(path).suffix.lower()
        return {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascriptreact", ".tsx": "typescriptreact",
            ".rs": "rust", ".go": "go", ".java": "java",
            ".c": "c", ".cpp": "cpp", ".cs": "csharp",
            ".rb": "ruby", ".php": "php", ".swift": "swift",
            ".kt": "kotlin", ".md": "markdown", ".json": "json",
            ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
            ".html": "html", ".css": "css", ".sh": "shellscript",
        }.get(ext, "plaintext")

    @classmethod
    def from_env(cls) -> dict[str, "LSPClient"]:
        """Create LSP clients from LSP_SERVERS environment variable."""
        env = os.environ.get("LSP_SERVERS", "")
        clients = {}
        for entry in env.split(","):
            entry = entry.strip()
            if "=" in entry:
                lang, cmd = entry.split("=", 1)
                clients[lang.strip()] = cls(lang.strip(), cmd.strip())
        return clients
