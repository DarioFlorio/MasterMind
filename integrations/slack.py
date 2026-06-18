# -*- coding: utf-8 -*-
"""integrations/slack.py — Slack Web API integration."""
from __future__ import annotations
import logging
import os

log = logging.getLogger("integrations.slack")


class SlackIntegration:
    def __init__(self, token: str = ""):
        self.token = token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._base = "https://slack.com/api"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"}

    def _post(self, method: str, data: dict) -> dict:
        import httpx
        r = httpx.post(f"{self._base}/{method}", json=data,
                       headers=self._headers(), timeout=15)
        r.raise_for_status()
        result = r.json()
        if not result.get("ok"):
            raise RuntimeError(f"Slack API error: {result.get('error', 'unknown')}")
        return result

    def post_message(self, channel: str, text: str, blocks: list | None = None) -> dict:
        payload: dict = {"channel": channel, "text": text}
        if blocks:
            payload["blocks"] = blocks
        return self._post("chat.postMessage", payload)

    def list_channels(self) -> list[dict]:
        import httpx
        r = httpx.get(f"{self._base}/conversations.list",
                      headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json().get("channels", [])

    def get_channel_history(self, channel: str, limit: int = 20) -> list[dict]:
        import httpx
        r = httpx.get(f"{self._base}/conversations.history",
                      params={"channel": channel, "limit": limit},
                      headers=self._headers(), timeout=15)
        r.raise_for_status()
        return r.json().get("messages", [])

    def upload_file(self, channels: list[str], filename: str,
                    content: str, title: str = "") -> dict:
        return self._post("files.upload", {
            "channels": ",".join(channels),
            "filename": filename,
            "content": content,
            "title": title or filename,
        })

    def set_status(self, text: str, emoji: str = "") -> dict:
        return self._post("users.profile.set", {
            "profile": {"status_text": text, "status_emoji": emoji}
        })
