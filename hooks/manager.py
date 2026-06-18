# -*- coding: utf-8 -*-
"""
hooks/manager.py — Event hook system MasterMind built-in.

Supports pre/post hooks for: tool_use, session_start, session_end,
message, file_read, file_write, bash_exec, agent_spawn.

Usage:
    from hooks.manager import hook_manager

    @hook_manager.on("tool_use:pre")
    def my_hook(event):
        print(f"About to use tool: {event.name}")

    # Or register programmatically:
    hook_manager.register("bash_exec:post", my_fn)

    # Fire:
    hook_manager.fire("tool_use:pre", name="bash", inp={"command": "ls"})
"""
from __future__ import annotations
import logging
import traceback
from dataclasses import dataclass, field
from typing import Callable, Any

log = logging.getLogger("hooks.manager")


@dataclass
class HookEvent:
    """Payload passed to every hook function."""
    event_type: str
    data: dict = field(default_factory=dict)

    # Convenience accessors
    def __getattr__(self, item):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(item)


class HookManager:
    """
    Lightweight event hook registry.

    Event names follow the pattern  "<category>:<timing>"
    e.g. "tool_use:pre", "tool_use:post", "session_start", "bash_exec:pre"
    """

    KNOWN_EVENTS = {
        "session_start",
        "session_end",
        "message:pre",
        "message:post",
        "tool_use:pre",
        "tool_use:post",
        "tool_error",
        "bash_exec:pre",
        "bash_exec:post",
        "file_read:pre",
        "file_read:post",
        "file_write:pre",
        "file_write:post",
        "agent_spawn",
        "agent_done",
        "plan_enter",
        "plan_exit",
        "worktree_enter",
        "worktree_exit",
        "mcp_call:pre",
        "mcp_call:post",
        "swarm_create",
        "swarm_destroy",
        "compact",
        "memory_write",
        "skill_invoke",
        "cron_fire",
    }

    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {}

    # ── Registration ───────────────────────────────────────────────────────

    def register(self, event: str, fn: Callable) -> None:
        """Register a callable for an event."""
        self._hooks.setdefault(event, []).append(fn)
        log.debug("hook registered: %s → %s", event, fn.__name__ if hasattr(fn, "__name__") else fn)

    def on(self, event: str) -> Callable:
        """Decorator: @hook_manager.on('tool_use:pre')"""
        def decorator(fn: Callable) -> Callable:
            self.register(event, fn)
            return fn
        return decorator

    def unregister(self, event: str, fn: Callable) -> bool:
        """Remove a specific hook. Returns True if found."""
        lst = self._hooks.get(event, [])
        try:
            lst.remove(fn)
            return True
        except ValueError:
            return False

    def clear(self, event: str | None = None) -> None:
        """Clear all hooks for an event, or all hooks if event is None."""
        if event is None:
            self._hooks.clear()
        else:
            self._hooks.pop(event, None)

    # ── Firing ────────────────────────────────────────────────────────────

    def fire(self, event: str, **kwargs) -> list[Any]:
        """
        Fire all hooks registered for `event`.
        kwargs are packed into HookEvent.data.
        Returns list of non-None return values.
        """
        listeners = self._hooks.get(event, [])
        if not listeners:
            return []

        ev = HookEvent(event_type=event, data=kwargs)
        results = []
        for fn in listeners:
            try:
                r = fn(ev)
                if r is not None:
                    results.append(r)
            except Exception:
                log.warning("hook %s raised during event %s:\n%s",
                            getattr(fn, "__name__", fn), event,
                            traceback.format_exc())
        return results

    def fire_blocking(self, event: str, **kwargs) -> bool:
        """
        Like fire() but returns False if any hook returns False
        (allows hooks to veto an action).
        """
        listeners = self._hooks.get(event, [])
        ev = HookEvent(event_type=event, data=kwargs)
        for fn in listeners:
            try:
                if fn(ev) is False:
                    return False
            except Exception:
                log.warning("hook error in %s:\n%s",
                            getattr(fn, "__name__", fn), traceback.format_exc())
        return True

    # ── Introspection ─────────────────────────────────────────────────────

    def list_hooks(self) -> dict[str, list[str]]:
        return {
            ev: [getattr(fn, "__name__", repr(fn)) for fn in fns]
            for ev, fns in self._hooks.items()
        }

    def __repr__(self) -> str:
        total = sum(len(v) for v in self._hooks.values())
        return f"<HookManager {total} hooks across {len(self._hooks)} events>"


# Singleton
hook_manager = HookManager()
