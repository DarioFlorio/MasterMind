"""Canvas / A2UI — structured UI widget rendering protocol."""
from .renderer import Canvas, parse_a2ui_line, parse_a2ui_stream
__all__ = ["Canvas", "parse_a2ui_line", "parse_a2ui_stream"]
