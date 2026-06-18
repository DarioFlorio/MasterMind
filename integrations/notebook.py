# -*- coding: utf-8 -*-
"""
integrations/notebook.py — Jupyter notebook (.ipynb) editor.

MasterMind built-in's NotebookEditTool. Supports:
  - Reading cells
  - Inserting cells (code/markdown)
  - Editing cell source
  - Deleting cells
  - Reading cell outputs
  - Executing cells via jupyter kernel (if available)
"""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("integrations.notebook")


@dataclass
class NotebookCell:
    cell_type: str  # "code" | "markdown" | "raw"
    source: str
    index: int
    outputs: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    execution_count: int | None = None

    def output_text(self) -> str:
        """Extract plain text from all outputs."""
        parts = []
        for out in self.outputs:
            if out.get("output_type") in ("stream", "display_data", "execute_result"):
                text = out.get("text", out.get("data", {}).get("text/plain", ""))
                if isinstance(text, list):
                    parts.append("".join(text))
                elif text:
                    parts.append(str(text))
        return "\n".join(parts)


class NotebookEditor:
    """Read and edit Jupyter notebooks (.ipynb files)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._nb: dict = {}
        self._dirty = False
        self._load()

    # ── Load / Save ────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            self._nb = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            # New notebook
            self._nb = {
                "nbformat": 4,
                "nbformat_minor": 5,
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"},
                },
                "cells": [],
            }

    def save(self, path: str | Path | None = None) -> None:
        dest = Path(path) if path else self.path
        dest.write_text(json.dumps(self._nb, indent=1), encoding="utf-8")
        self._dirty = False
        log.info("Notebook saved: %s", dest)

    # ── Read ──────────────────────────────────────────────────────────────

    def cells(self) -> list[NotebookCell]:
        raw = self._nb.get("cells", [])
        return [
            NotebookCell(
                cell_type=c.get("cell_type", "code"),
                source="".join(c.get("source", [])) if isinstance(c.get("source"), list)
                       else c.get("source", ""),
                index=i,
                outputs=c.get("outputs", []),
                metadata=c.get("metadata", {}),
                execution_count=c.get("execution_count"),
            )
            for i, c in enumerate(raw)
        ]

    def get_cell(self, index: int) -> NotebookCell | None:
        cells = self.cells()
        if 0 <= index < len(cells):
            return cells[index]
        return None

    def summary(self) -> str:
        cells = self.cells()
        lines = [f"Notebook: {self.path.name}  ({len(cells)} cells)"]
        for c in cells:
            src_preview = c.source[:60].replace("\n", " ")
            out_flag = " [has output]" if c.outputs else ""
            lines.append(f"  [{c.index}] {c.cell_type:8} | {src_preview}{out_flag}")
        return "\n".join(lines)

    # ── Write ─────────────────────────────────────────────────────────────

    def insert_cell(self, index: int, cell_type: str, source: str) -> int:
        """Insert a cell at index. Returns the actual index."""
        raw_cells = self._nb.setdefault("cells", [])
        new_cell = {
            "cell_type": cell_type,
            "source": source,
            "metadata": {},
        }
        if cell_type == "code":
            new_cell["outputs"] = []
            new_cell["execution_count"] = None
        index = max(0, min(index, len(raw_cells)))
        raw_cells.insert(index, new_cell)
        self._dirty = True
        return index

    def edit_cell(self, index: int, source: str) -> bool:
        raw_cells = self._nb.get("cells", [])
        if not (0 <= index < len(raw_cells)):
            return False
        raw_cells[index]["source"] = source
        if raw_cells[index]["cell_type"] == "code":
            raw_cells[index]["outputs"] = []
            raw_cells[index]["execution_count"] = None
        self._dirty = True
        return True

    def delete_cell(self, index: int) -> bool:
        raw_cells = self._nb.get("cells", [])
        if not (0 <= index < len(raw_cells)):
            return False
        raw_cells.pop(index)
        self._dirty = True
        return True

    def append_cell(self, cell_type: str, source: str) -> int:
        raw_cells = self._nb.setdefault("cells", [])
        return self.insert_cell(len(raw_cells), cell_type, source)

    # ── Execution (optional, requires jupyter) ─────────────────────────────

    def execute_cell(self, index: int) -> str:
        """Execute a single cell via jupyter nbconvert (if available)."""
        import shutil, subprocess, tempfile
        if not shutil.which("jupyter"):
            return "jupyter not found — install with: pip install jupyter"
        cell = self.get_cell(index)
        if not cell:
            return f"Cell {index} not found"
        if cell.cell_type != "code":
            return "Only code cells can be executed"

        # Save to temp, execute, read back outputs
        with tempfile.NamedTemporaryFile(suffix=".ipynb", delete=False) as f:
            tmp = Path(f.name)
        self.save(tmp)
        try:
            result = subprocess.run(
                ["jupyter", "nbconvert", "--to", "notebook",
                 "--execute", "--inplace", str(tmp)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return f"Execution error:\n{result.stderr[:500]}"
            nb = NotebookEditor(tmp)
            executed_cell = nb.get_cell(index)
            return executed_cell.output_text() if executed_cell else "(no output)"
        except subprocess.TimeoutExpired:
            return "Execution timed out"
        except Exception as e:
            return f"Error: {e}"
        finally:
            tmp.unlink(missing_ok=True)
