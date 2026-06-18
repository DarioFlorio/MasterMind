"""
Canvas / A2UI — structured UI rendering protocol.
MasterMind canvas widget rendering system.

A2UI (Agent-to-UI) is a JSONL-based protocol that lets agents render
rich UI widgets inside the chat interface. Each line is a JSON action.

Supported widget types (standard catalog):
  text, button, card, column, row, image, audio, video,
  text-field, checkbox, slider, multiple-choice, tabs,
  list, modal, divider, icon

The renderer sends A2UI actions as JSON lines to stdout (or a WebSocket).
A companion UI (web/mobile) renders them.
"""
from __future__ import annotations
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ── Action types ───────────────────────────────────────────────────────────────

@dataclass
class A2UIAction:
    action: str
    id: Optional[str] = None

    def to_jsonl(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d)


@dataclass
class RenderAction(A2UIAction):
    """Render or update a component."""
    action: str = "render"
    component: Optional[str] = None
    props: dict = field(default_factory=dict)

    def to_jsonl(self) -> str:
        d = {"action": self.action}
        if self.id:
            d["id"] = self.id
        if self.component:
            d["component"] = self.component
        d.update(self.props)
        return json.dumps(d)


@dataclass
class ClearAction(A2UIAction):
    """Clear the canvas or a specific component."""
    action: str = "clear"


@dataclass
class EventAction(A2UIAction):
    """Client-to-agent event (button click, input change, etc.)."""
    action: str = "event"
    event_type: str = ""
    payload: dict = field(default_factory=dict)


# ── Canvas renderer ────────────────────────────────────────────────────────────

class Canvas:
    """
    Render rich UI widgets via the A2UI protocol.
    
    By default, renders to stdout as JSONL (one JSON object per line).
    Can also buffer output for testing or send via WebSocket.
    
    Usage:
        canvas = Canvas()
        canvas.text("Hello, World!", style="heading")
        canvas.button("Click me", on_click="handle_click")
        canvas.divider()
        
        with canvas.card(title="Results"):
            canvas.text("Item 1")
            canvas.text("Item 2")
    """

    def __init__(
        self,
        output=None,        # file-like or None (stdout)
        buffer: bool = False,
    ):
        self._out = output or sys.stdout
        self._buffered = buffer
        self._buffer: list[str] = []
        self._id_counter = 0

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"c{self._id_counter:04d}"

    def _emit(self, action: dict) -> str:
        line = json.dumps(action)
        if self._buffered:
            self._buffer.append(line)
        else:
            print(line, file=self._out, flush=True)
        return line

    def _component(self, comp_type: str, props: dict, widget_id: str | None = None) -> str:
        cid = widget_id or self._next_id()
        return self._emit({
            "action": "render",
            "id": cid,
            "component": comp_type,
            **props,
        })

    def flush(self) -> list[str]:
        """Return buffered lines and clear buffer."""
        lines = list(self._buffer)
        self._buffer.clear()
        return lines

    def clear(self, widget_id: str | None = None) -> None:
        action = {"action": "clear"}
        if widget_id:
            action["id"] = widget_id
        self._emit(action)

    # ── Text ──────────────────────────────────────────────────────────────────

    def text(
        self,
        content: str,
        style: str = "body",   # "heading", "subheading", "body", "caption", "code"
        color: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"content": content, "style": style}
        if color:
            props["color"] = color
        return self._component("text", props, widget_id)

    def heading(self, content: str, **kwargs) -> str:
        return self.text(content, style="heading", **kwargs)

    def code(self, content: str, language: str = "", **kwargs) -> str:
        return self.text(f"```{language}\n{content}\n```", style="code", **kwargs)

    # ── Interactive ───────────────────────────────────────────────────────────

    def button(
        self,
        label: str,
        on_click: str | None = None,
        variant: str = "primary",  # "primary", "secondary", "danger"
        disabled: bool = False,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"label": label, "variant": variant}
        if on_click:
            props["onClick"] = on_click
        if disabled:
            props["disabled"] = True
        return self._component("button", props, widget_id)

    def text_field(
        self,
        label: str,
        placeholder: str = "",
        value: str = "",
        on_change: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"label": label, "placeholder": placeholder, "value": value}
        if on_change:
            props["onChange"] = on_change
        return self._component("text-field", props, widget_id)

    def checkbox(
        self,
        label: str,
        checked: bool = False,
        on_change: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"label": label, "checked": checked}
        if on_change:
            props["onChange"] = on_change
        return self._component("checkbox", props, widget_id)

    def slider(
        self,
        label: str,
        value: float = 0.0,
        min_val: float = 0.0,
        max_val: float = 100.0,
        step: float = 1.0,
        on_change: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {
            "label": label, "value": value,
            "min": min_val, "max": max_val, "step": step,
        }
        if on_change:
            props["onChange"] = on_change
        return self._component("slider", props, widget_id)

    def multiple_choice(
        self,
        label: str,
        options: list[str],
        selected: list[str] | None = None,
        multiple: bool = False,
        on_change: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {
            "label": label,
            "options": [{"label": o, "value": o} for o in options],
            "selected": selected or [],
            "multiple": multiple,
        }
        if on_change:
            props["onChange"] = on_change
        return self._component("multiple-choice", props, widget_id)

    # ── Layout ────────────────────────────────────────────────────────────────

    def divider(self, widget_id: str | None = None) -> str:
        return self._component("divider", {}, widget_id)

    def row(self, children: list[dict] | None = None, gap: int = 8,
            widget_id: str | None = None) -> str:
        props: dict = {"gap": gap}
        if children:
            props["children"] = children
        return self._component("row", props, widget_id)

    def column(self, children: list[dict] | None = None, gap: int = 8,
               widget_id: str | None = None) -> str:
        props: dict = {"gap": gap}
        if children:
            props["children"] = children
        return self._component("column", props, widget_id)

    def card(
        self,
        title: str | None = None,
        children: list[dict] | None = None,
        widget_id: str | None = None,
    ) -> "_CardContext":
        return _CardContext(self, title, widget_id)

    def tabs(
        self,
        tabs: list[str],
        active: str | None = None,
        on_change: str | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {
            "tabs": [{"label": t, "value": t} for t in tabs],
            "active": active or (tabs[0] if tabs else ""),
        }
        if on_change:
            props["onChange"] = on_change
        return self._component("tabs", props, widget_id)

    # ── Media ─────────────────────────────────────────────────────────────────

    def image(
        self,
        src: str,
        alt: str = "",
        width: int | None = None,
        height: int | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"src": src, "alt": alt}
        if width:
            props["width"] = width
        if height:
            props["height"] = height
        return self._component("image", props, widget_id)

    def audio(self, src: str, autoplay: bool = False,
              widget_id: str | None = None) -> str:
        return self._component("audio", {"src": src, "autoplay": autoplay}, widget_id)

    def video(self, src: str, autoplay: bool = False,
              controls: bool = True, widget_id: str | None = None) -> str:
        return self._component("video", {
            "src": src, "autoplay": autoplay, "controls": controls
        }, widget_id)

    # ── Complex widgets ───────────────────────────────────────────────────────

    def list_widget(
        self,
        items: list[str | dict],
        ordered: bool = False,
        widget_id: str | None = None,
    ) -> str:
        if items and isinstance(items[0], str):
            items = [{"text": i} for i in items]
        return self._component("list", {"items": items, "ordered": ordered}, widget_id)

    def modal(
        self,
        title: str,
        content: str,
        actions: list[dict] | None = None,
        widget_id: str | None = None,
    ) -> str:
        props: dict = {"title": title, "content": content}
        if actions:
            props["actions"] = actions
        return self._component("modal", props, widget_id)

    def icon(self, name: str, size: int = 24,
             color: str | None = None, widget_id: str | None = None) -> str:
        props: dict = {"name": name, "size": size}
        if color:
            props["color"] = color
        return self._component("icon", props, widget_id)

    # ── Utility ───────────────────────────────────────────────────────────────

    def table(self, headers: list[str], rows: list[list[Any]],
              widget_id: str | None = None) -> str:
        """Convenience: render a table as a card with list items."""
        header_row = " | ".join(f"**{h}**" for h in headers)
        sep = " | ".join("---" for _ in headers)
        data_rows = [" | ".join(str(c) for c in row) for row in rows]
        md = "\n".join([header_row, sep] + data_rows)
        return self.text(md, style="code", widget_id=widget_id)

    def progress(self, value: float, label: str = "", widget_id: str | None = None) -> str:
        return self._component("slider", {
            "label": label, "value": value, "min": 0, "max": 100,
            "disabled": True,
        }, widget_id)

    def update(self, widget_id: str, **props) -> str:
        """Update an existing widget's props."""
        return self._emit({"action": "update", "id": widget_id, **props})


class _CardContext:
    """Context manager for card widgets."""
    def __init__(self, canvas: Canvas, title: str | None, widget_id: str | None):
        self._canvas = canvas
        self._title = title
        self._widget_id = widget_id or canvas._next_id()

    def __enter__(self) -> Canvas:
        props: dict = {}
        if self._title:
            props["title"] = self._title
        props["id"] = self._widget_id
        self._canvas._emit({"action": "render", "component": "card", **props})
        return self._canvas

    def __exit__(self, *args) -> None:
        self._canvas._emit({"action": "end-card", "id": self._widget_id})


# ── A2UI JSONL parser (for incoming client events) ────────────────────────────

def parse_a2ui_line(line: str) -> dict | None:
    """Parse a single A2UI JSONL line."""
    try:
        return json.loads(line.strip())
    except json.JSONDecodeError:
        return None

def parse_a2ui_stream(text: str) -> list[dict]:
    """Parse a multi-line A2UI JSONL stream."""
    results = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            parsed = parse_a2ui_line(line)
            if parsed:
                results.append(parsed)
    return results
