# -*- coding: utf-8 -*-
"""bridge/client.py — Client for connecting to a remote MasterMind bridge server."""
from __future__ import annotations
import json
import logging
from typing import Iterator

log = logging.getLogger("bridge.client")


class BridgeClient:
    def __init__(self, base_url: str = "http://127.0.0.1:7777", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def health(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self.base_url}/health", headers=self._headers(), timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def send_message(self, prompt: str, stream: bool = False) -> str | Iterator[str]:
        import httpx
        payload = {"prompt": prompt, "stream": stream}
        if stream:
            return self._stream(payload)
        r = httpx.post(f"{self.base_url}/v1/message", json=payload,
                       headers=self._headers(), timeout=300)
        r.raise_for_status()
        return r.json().get("result", "")

    def _stream(self, payload: dict) -> Iterator[str]:
        import httpx
        with httpx.stream("POST", f"{self.base_url}/v1/message",
                          json=payload, headers=self._headers(), timeout=300) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    if data.get("type") == "chunk":
                        yield data.get("text", "")
                    elif data.get("type") in ("done", "error"):
                        break

    def call_tool(self, tool: str, inp: dict) -> dict:
        import httpx
        r = httpx.post(f"{self.base_url}/v1/tool",
                       json={"tool": tool, "input": inp},
                       headers=self._headers(), timeout=60)
        r.raise_for_status()
        return r.json()

    def list_tools(self) -> list[dict]:
        import httpx
        r = httpx.get(f"{self.base_url}/v1/tools", headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json().get("tools", [])

    def clear_session(self) -> bool:
        import httpx
        r = httpx.delete(f"{self.base_url}/v1/session",
                         headers=self._headers(), timeout=10)
        return r.status_code == 200
