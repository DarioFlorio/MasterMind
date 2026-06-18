"""
tools/memory_search_tool.py — User-specific vector store search.

GAP IMPLEMENTED: User-specific vector store — searchable past interactions.
Provides EVE with a rich semantic + keyword search over all three memory tiers.

Usage (tool call):
    <tool_use><n>memory_search</n><input>{"query": "python file writing", "k": 5}</input></tool_use>

    <tool_use><n>memory_search</n><input>{
        "query": "authentication errors",
        "tier": "semantic",
        "k": 3
    }</input></tool_use>
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from tools.base_tool import BaseTool, ToolResult

log = logging.getLogger("tools.memory_search")


class MemorySearchTool(BaseTool):
    """
    Semantic + keyword search over the three-tier vector memory store.
    Returns ranked results with tier labels and relevance scores.
    """

    name = "memory_search"
    description = (
        "Search EVE's persistent memory (working, episodic, semantic tiers) "
        "for relevant past interactions, facts, and insights. "
        "Use this to recall context from previous sessions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in memory",
            },
            "k": {
                "type": "integer",
                "description": "Max results to return (default: 5)",
            },
            "tier": {
                "type": "string",
                "enum": ["all", "working", "episodic", "semantic"],
                "description": "Which memory tier to search (default: all)",
            },
        },
        "required": ["query"],
    }

    def __init__(self, working_dir: str = "") -> None:
        self._base = Path(working_dir) if working_dir else Path.cwd()
        self._mem_dir = str(self._base / "memdir" / "three_tier")
        self._mem: object = None
        self._init_memory()

    def _init_memory(self) -> None:
        try:
            from memory.three_tier import ThreeTierMemory
            self._mem = ThreeTierMemory(db_path=self._mem_dir)
        except Exception as exc:
            log.warning("MemorySearchTool: ThreeTierMemory init failed: %s", exc)

    def execute(self, inp: dict) -> ToolResult:
        inp   = self.safe_parse(inp)
        query = inp.get("query", "").strip()
        k     = int(inp.get("k", 5))
        tier  = inp.get("tier", "all")

        if not query:
            return ToolResult(output="query is required", is_error=True)

        # Try three-tier vector search
        results: list[str] = []
        if self._mem and getattr(self._mem, "_ready", False):
            try:
                results = self._mem.retrieve(query, k=k)
            except Exception as exc:
                log.debug("three_tier retrieve: %s", exc)

        # Fallback: keyword search over journal + facts
        if not results:
            results = self._keyword_fallback(query, k)

        # Also search semantic memory in manager (plain JSON facts)
        plain_facts = self._search_plain_facts(query, k=3)
        if plain_facts:
            results = plain_facts + results

        # Deduplicate
        seen: set[str] = set()
        deduped: list[str] = []
        for r in results:
            key = r[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(r)

        if not deduped:
            return ToolResult(
                output=f"No memory results found for: {query}\n"
                       "(Memory may be empty or not yet indexed.)"
            )

        lines = [f"Memory search results for '{query}' ({len(deduped)} hits):"]
        for i, r in enumerate(deduped[:k], 1):
            lines.append(f"\n[{i}] {r}")

        return ToolResult(output="\n".join(lines))

    # ── Fallback search ───────────────────────────────────────────────────────

    def _keyword_fallback(self, query: str, k: int) -> list[str]:
        """Keyword overlap search over the in-memory buffer."""
        results: list[str] = []
        try:
            from memory.manager import _load_journal, _load_facts
            journal = _load_journal()
            facts   = _load_facts()

            qwords = set(query.lower().split())

            scored: list[tuple[float, str]] = []
            for entry in journal:
                text = entry.get("note", "")
                ewords = set(text.lower().split())
                score  = len(qwords & ewords) / max(len(qwords), 1)
                if score > 0:
                    scored.append((score, f"[journal|{entry.get('ts','')}] {text[:300]}"))

            for key, val in facts.items():
                text = f"{key}: {val.get('content', '')}"
                ewords = set(text.lower().split())
                score  = len(qwords & ewords) / max(len(qwords), 1)
                if score > 0:
                    scored.append((score, f"[fact] {text[:300]}"))

            scored.sort(reverse=True)
            results = [r for _, r in scored[:k]]
        except Exception as exc:
            log.debug("Keyword fallback: %s", exc)
        return results

    def _search_plain_facts(self, query: str, k: int) -> list[str]:
        """Search plain-text facts from memory.manager."""
        try:
            from memory.manager import _load_facts
            facts = _load_facts()
            qwords = set(query.lower().split())
            scored: list[tuple[float, str]] = []
            for key, val in facts.items():
                text = f"{key}: {val.get('content', '')}"
                ewords = set(text.lower().split())
                score = len(qwords & ewords) / max(len(qwords), 1)
                if score > 0.2:
                    scored.append((score, f"[semantic|fact] {text[:200]}"))
            scored.sort(reverse=True)
            return [r for _, r in scored[:k]]
        except Exception:
            return []
