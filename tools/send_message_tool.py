# -*- coding: utf-8 -*-
"""tools/send_message_tool.py — Send messages between agents in a swarm."""
from __future__ import annotations
import json
import time
from collections import defaultdict
from threading import Lock
from tools.base_tool import BaseTool, ToolResult

# Simple in-process mailbox
_mailbox: dict[str, list[dict]] = defaultdict(list)
_lock = Lock()


class SendMessageTool(BaseTool):
    name = "send_message"
    description = "Send a message to another agent in the swarm, or to a named channel."
    input_schema = {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient agent name or channel (default: 'user')"},
            "message": {"type": "string", "description": "Message content"},
            "metadata": {"type": "object", "description": "Optional metadata dict"},
        },
        "required": ["message"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        to = inp.get("to", "user") or "user"
        message = inp.get("message", "")
        if not message:
            return ToolResult(output="No message provided.", is_error=True)
        envelope = {"from": "agent", "message": message,
                    "ts": time.time(), "meta": inp.get("metadata", {})}
        with _lock:
            _mailbox[to].append(envelope)
        return ToolResult(output=f"Message sent to {to!r}")


class ReceiveMessageTool(BaseTool):
    name = "receive_messages"
    description = "Receive messages sent to a named channel or agent."
    input_schema = {
        "type": "object",
        "properties": {
            "channel": {"type": "string", "description": "Channel/agent name to read from"},
            "clear": {"type": "boolean", "description": "Clear messages after reading (default: true)"},
        },
        "required": ["channel"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        channel = inp.get("channel", "")
        clear = inp.get("clear", True)
        with _lock:
            msgs = list(_mailbox.get(channel, []))
            if clear:
                _mailbox[channel] = []
        if not msgs:
            return ToolResult(output=f"No messages for {channel!r}")
        lines = [f"[{i+1}] from={m['from']}: {m['message']}" for i, m in enumerate(msgs)]
        return ToolResult(output="\n".join(lines))
