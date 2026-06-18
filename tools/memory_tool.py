from __future__ import annotations
from tools.base_tool import BaseTool, ToolResult


class MemoryWriteTool(BaseTool):
    name = "memory_write"
    description = (
        "Save a fact or note to persistent memory. "
        "Use for information that should persist across sessions."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "key":     {"type": "string", "description": "Short label/key for this memory"},
            "content": {"type": "string", "description": "Content to remember"},
            "value":   {"type": ["string", "object", "array"], "description": "Alias for content — use either 'content' or 'value'"},
        },
        "required": ["key"],
    }

    def execute(self, inp: dict) -> ToolResult:
        key     = inp.get("key", "note").strip()
        # Accept both 'content' and 'value' (EVE sometimes uses 'value')
        content = inp.get("content", inp.get("value", ""))
        if isinstance(content, (dict, list)):
            import json as _json
            content = _json.dumps(content, indent=2, ensure_ascii=False)
        content = str(content).strip()
        if not content:
            return ToolResult("No content provided.", is_error=True)
        try:
            # Write to JSON store (appears in agent's next system prompt)
            from memory.manager import save_fact
            save_fact(key, content)

            # Also index in memory_core so it's searchable via FTS+vector
            try:
                from memory_core.manager import get_memory_manager
                get_memory_manager().ingest_text(f"[{key}] {content}", label=f"fact:{key}")
            except Exception:
                pass  # vector indexing is best-effort

            return ToolResult(f"Saved memory: [{key}]")
        except Exception as e:
            return ToolResult(str(e), is_error=True)


class MemoryReadTool(BaseTool):
    name = "memory_read"
    description = (
        "Search persistent memory. Pass a query to find relevant stored facts "
        "and session notes using hybrid keyword+vector search."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query (or key to look up directly)"},
        },
    }

    def execute(self, inp: dict) -> ToolResult:
        query = inp.get("query", inp.get("key", "")).strip()
        try:
            # Try memory_core hybrid search first (richer results)
            from memory_core.manager import get_memory_manager
            mgr = get_memory_manager()
            st = mgr.status()
            if st["total_chunks"] > 0 and query:
                results = mgr.search_hybrid(query, limit=8)
                if results:
                    lines = []
                    for r in results:
                        src = r.chunk.path
                        lines.append(f"[{r.match_type}|{r.score:.2f}] {src}\n{r.snippet[:400]}")
                    return ToolResult("\n\n---\n".join(lines))

            # Fallback: JSON key/value store
            from memory.manager import load_context, load_fact
            if query:
                val = load_fact(query)
                if val:
                    return ToolResult(val)
            return ToolResult(load_context() or "No memories stored yet.")
        except Exception as e:
            return ToolResult(str(e), is_error=True)

