from __future__ import annotations
from enum import Enum

_SAFE = {"read_file", "list_dir", "glob", "grep", "todo_read", "memory_read", "web_search", "web_fetch"}
_DESTRUCTIVE = {"bash", "write_file", "edit_file", "agent", "skill", "todo_write", "memory_write"}


class PermMode(str, Enum):
    AUTO = "auto"
    ASK  = "ask"
    DENY = "deny"


class PermissionManager:
    def __init__(self, mode: str = "ask"):
        self.mode = PermMode(mode.lower())
        self._always_allow: set[str] = set()
        self._always_deny:  set[str] = set()

    def set_mode(self, mode: str) -> None:
        self.mode = PermMode(mode.lower())

    def allow_tool(self, name: str) -> None:
        self._always_allow.add(name)

    def deny_tool(self, name: str) -> None:
        self._always_deny.add(name)

    def check(self, tool_name: str, inp: dict) -> bool:
        if tool_name in self._always_deny:
            return False
        if tool_name in self._always_allow:
            return True
        if self.mode == PermMode.AUTO:
            return True
        if self.mode == PermMode.DENY:
            return tool_name in _SAFE
        # ASK mode
        if tool_name in _SAFE:
            return True
        # Prompt user
        return self._ask(tool_name, inp)

    def _ask(self, tool_name: str, inp: dict) -> bool:
        import json, sys
        preview = json.dumps(inp, ensure_ascii=False)[:120]
        try:
            ans = input(f"\n  Allow tool '{tool_name}' with {preview}? [y/N/always/never] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        if ans in ("always", "a"):
            self._always_allow.add(tool_name)
            return True
        if ans in ("never", "n"):
            self._always_deny.add(tool_name)
            return False
        return ans in ("y", "yes")
