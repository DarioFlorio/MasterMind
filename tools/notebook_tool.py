# -*- coding: utf-8 -*-
"""tools/notebook_tool.py — Jupyter notebook (.ipynb) editor tool."""
from __future__ import annotations
from tools.base_tool import BaseTool, ToolResult
from integrations.notebook import NotebookEditor


class NotebookTool(BaseTool):
    name = "notebook"
    description = (
        "Read and edit Jupyter notebooks (.ipynb). Operations: read, insert, "
        "edit, delete, append, execute. Clears outputs on cell edits."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "op": {
                "type": "string",
                "description": "read | insert | edit | delete | append | execute | summary",
            },
            "path": {"type": "string", "description": "Path to .ipynb file"},
            "index": {"type": "integer", "description": "Cell index (0-based)"},
            "cell_type": {
                "type": "string",
                "description": "Cell type: code | markdown (for insert/append)",
            },
            "source": {"type": "string", "description": "Cell source code/markdown"},
        },
        "required": ["op", "path"],
    }

    def execute(self, inp: dict) -> ToolResult:
        inp = self.safe_parse(inp)
        op = inp.get("op", "")
        path = inp.get("path", "")
        index = inp.get("index", 0)
        cell_type = inp.get("cell_type", "code")
        source = inp.get("source", "")

        if not path:
            return ToolResult(output="path is required", is_error=True)

        try:
            nb = NotebookEditor(path)
        except Exception as e:
            return ToolResult(output=f"Cannot open notebook: {e}", is_error=True)

        try:
            if op == "summary":
                return ToolResult(output=nb.summary())

            elif op == "read":
                cells = nb.cells()
                if not cells:
                    return ToolResult(output="(empty notebook)")
                parts = []
                for c in cells:
                    out_text = f"\n--- Output ---\n{c.output_text()}" if c.outputs else ""
                    parts.append(f"[{c.index}] {c.cell_type}\n{c.source}{out_text}")
                return ToolResult(output="\n\n".join(parts))

            elif op == "insert":
                if not source:
                    return ToolResult(output="source is required for insert", is_error=True)
                actual = nb.insert_cell(index, cell_type, source)
                nb.save()
                return ToolResult(output=f"Cell inserted at index {actual}")

            elif op == "append":
                if not source:
                    return ToolResult(output="source is required for append", is_error=True)
                actual = nb.append_cell(cell_type, source)
                nb.save()
                return ToolResult(output=f"Cell appended at index {actual}")

            elif op == "edit":
                if not source:
                    return ToolResult(output="source is required for edit", is_error=True)
                if not nb.edit_cell(index, source):
                    return ToolResult(output=f"Cell index {index} not found", is_error=True)
                nb.save()
                return ToolResult(output=f"Cell {index} updated")

            elif op == "delete":
                if not nb.delete_cell(index):
                    return ToolResult(output=f"Cell index {index} not found", is_error=True)
                nb.save()
                return ToolResult(output=f"Cell {index} deleted")

            elif op == "execute":
                result = nb.execute_cell(index)
                return ToolResult(output=result)

            else:
                return ToolResult(
                    output=f"Unknown op: {op!r}. Use: summary|read|insert|append|edit|delete|execute",
                    is_error=True
                )
        except Exception as e:
            return ToolResult(output=f"Notebook error: {e}", is_error=True)
