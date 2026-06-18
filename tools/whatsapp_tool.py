# -*- coding: utf-8 -*-
"""tools/whatsapp_tool.py — Send a WhatsApp message via the local Baileys bridge."""
from __future__ import annotations
import json
import re
import urllib.request
from tools.base_tool import BaseTool, ToolResult

_WA_PORT = 5005
_DIGITS_RE = re.compile(r'\d{7,}')   # at least 7 digits = a phone number


def _extract_number(s: str) -> str:
    """Pull digits from a string only if it looks like an actual phone number."""
    digits = "".join(c for c in s if c.isdigit())
    if len(digits) >= 7:
        return digits
    return ""   # "Dario", "user", etc. → ignored


class WhatsAppSendTool(BaseTool):
    name = "whatsapp_send"
    description = (
        "Send a WhatsApp message to the user. "
        "REQUIRED parameter: 'message' (the text to send). "
        "OPTIONAL parameter: 'to' (phone number — the bridge already knows it, omit it). "
        "ALWAYS use the key 'message'. NEVER use 'body', 'text', 'content', or 'msg'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The WhatsApp message text to send.",
            },
            "to": {
                "type": "string",
                "description": "Leave empty. The bridge knows the recipient automatically.",
            },
        },
        "required": ["message"],
    }

    _owner: str = ""

    @classmethod
    def set_owner(cls, jid: str) -> None:
        cls._owner = jid.strip()

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)

        # Pull message text from any plausible key
        text = ""
        for key in ("message", "text", "content", "body", "msg"):
            v = inp.get(key, "")
            if v and isinstance(v, str) and v.strip():
                text = v.strip()
                break

        if not text:
            return ToolResult(output="[whatsapp_send] ERROR: 'message' is missing.", is_error=True)

        # Resolve recipient — only accept values that look like phone numbers
        to_clean = ""
        for key in ("to", "phone_number", "recipient", "number", "phone", "target"):
            v = inp.get(key, "")
            if v and isinstance(v, str):
                candidate = _extract_number(v)
                if candidate:
                    to_clean = candidate
                    break
        # Fall back to stored owner (hardcoded or learned from bridge)
        if not to_clean:
            to_clean = _extract_number(self._owner)
        # Last-ditch live fetch
        if not to_clean:
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{_WA_PORT}/status", timeout=3
                ) as r:
                    data = json.loads(r.read())
                    to_clean = _extract_number(data.get("owner", ""))
                    if to_clean:
                        self.__class__.set_owner(to_clean)
            except Exception:
                pass

        if not to_clean:
            return ToolResult(
                output="[whatsapp_send] ERROR: recipient unknown. Bridge not ready.",
                is_error=True,
            )

        try:
            body = json.dumps({"to": to_clean, "text": text}).encode()
            req = urllib.request.Request(
                f"http://127.0.0.1:{_WA_PORT}/send",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
            if result.get("ok"):
                preview = text[:80].replace("\n", " ")
                return ToolResult(output=f"✓ WhatsApp sent to +{to_clean}: {preview}" + ("…" if len(text) > 80 else ""))
            else:
                return ToolResult(output=f"[whatsapp_send] Bridge error: {result}", is_error=True)
        except Exception as e:
            return ToolResult(output=f"[whatsapp_send] Send failed: {e}", is_error=True)