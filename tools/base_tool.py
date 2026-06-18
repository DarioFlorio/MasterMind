# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    output: str
    is_error: bool = False

    def to_xml(self, tool_name: str) -> str:
        tag = "error" if self.is_error else "result"
        return (
            f"<tool_response tool=\"{tool_name}\">\n"
            f"  <{tag}>{self.output}</{tag}>\n"
            f"</tool_response>"
        )


class BaseTool(ABC):
    name: str
    description: str
    input_schema: dict

    @abstractmethod
    def execute(self, inp: dict) -> ToolResult: ...

    def to_xml_schema(self) -> str:
        props = self.input_schema.get("properties", {})
        req   = self.input_schema.get("required", [])
        lines = []
        for p, s in props.items():
            r = " (required)" if p in req else ""
            lines.append(f"    <param name='{p}' type='{s.get('type','string')}'{r}>{s.get('description','')}</param>")
        return (f"<tool>\n  <n>{self.name}</n>\n"
                f"  <description>{self.description}</description>\n"
                f"  <params>\n" + "\n".join(lines) + "\n  </params>\n</tool>")

    def to_compact_schema(self) -> str:
        """Compact one-liner schema ~13x smaller than to_xml_schema.
        Format: name(required*, optional) -- short description
        Used in system prompt to cut prefill cost.
        """
        props = self.input_schema.get("properties", {})
        req   = set(self.input_schema.get("required", []))
        params = sorted(props.keys(), key=lambda p: (p not in req, p))
        param_str = ", ".join(p + ("*" if p in req else "") for p in params)
        desc = self.description[:80].rstrip()
        if len(self.description) > 80:
            desc = desc + "..."
        return f"{self.name}({param_str}) -- {desc}"

    def safe_parse(self, raw) -> dict:
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}