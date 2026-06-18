"""
tools/wiki_tool.py — EVE Second Brain (Karpathy three-folder structure)
=======================================================================

Implements a structured personal knowledge base inspired by Andrew Karpathy's
"LLM wiki / second brain" concept with three clearly separated folders:

  📥 INBOX      Raw captures, quick notes, things to process later.
                Fast, zero-friction. Anything goes in here.

  📝 NOTES      Working notes, active research, structured thinking.
                Processed and organised but still evolving.

  📚 REFERENCE  Permanent knowledge base, organised by topic/domain.
                Final, searchable, dense. Your LLM wiki.

Each folder contains Markdown files. Files are date-stamped and topic-tagged.
Cross-linking between notes is supported via [[WikiLink]] syntax.
Full-text search runs across all folders.

Usage examples:
  wiki_write  - Write to inbox/notes/reference
  wiki_read   - Read a specific note
  wiki_search - Search across all notes
  wiki_list   - List notes (with filters)
  wiki_link   - Find all notes linking to a given note
  wiki_promote - Move a note from inbox → notes → reference

System prompt teaching the format:
  INBOX: quick capture, raw ideas, unprocessed.
  NOTES: structured research, active thinking, evolving.
  REFERENCE: finished knowledge, permanent wiki entries.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool, ToolResult
from config.settings import WORKING_DIR

# ── Folder layout ──────────────────────────────────────────────────────────────
_FOLDERS = {
    "inbox":     "📥 Inbox (raw captures)",
    "notes":     "📝 Notes (working/active)",
    "reference": "📚 Reference (permanent wiki)",
}
_FOLDER_ALIASES = {
    "i": "inbox", "in": "inbox",
    "n": "notes", "note": "notes", "working": "notes",
    "r": "reference", "ref": "reference", "wiki": "reference",
    "kb": "reference", "knowledge": "reference",
}

_MAX_SEARCH_CHARS = 600    # chars shown per result in search
_MAX_READ_CHARS   = 40_000 # chars shown when reading a full note


def _canon_folder(raw: str) -> str:
    s = (raw or "inbox").lower().strip()
    return _FOLDER_ALIASES.get(s, s if s in _FOLDERS else "inbox")


def _slug(title: str) -> str:
    """Convert title → safe filename slug."""
    s = re.sub(r"[^\w\s-]", "", title.lower())
    s = re.sub(r"[\s_]+", "-", s.strip())
    return s[:80] or "untitled"


def _now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _extract_tags(content: str) -> list[str]:
    return re.findall(r"#([A-Za-z][A-Za-z0-9_/-]*)", content)


def _extract_links(content: str) -> list[str]:
    return re.findall(r"\[\[([^\]]+)\]\]", content)


class _WikiBase:
    """Shared base for all wiki tool variants."""

    def __init__(self, working_dir: str = WORKING_DIR):
        self._root = Path(working_dir) / "wiki"
        for folder in _FOLDERS:
            (self._root / folder).mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / ".index.json"

    # ── Index helpers ──────────────────────────────────────────────────────────

    def _load_index(self) -> dict:
        if self._index_path.exists():
            try:
                return json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_index(self, idx: dict) -> None:
        try:
            self._index_path.write_text(
                json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    def _update_index(self, folder: str, slug: str, meta: dict) -> None:
        idx = self._load_index()
        key = f"{folder}/{slug}"
        idx[key] = meta
        self._save_index(idx)

    def _remove_from_index(self, folder: str, slug: str) -> None:
        idx = self._load_index()
        idx.pop(f"{folder}/{slug}", None)
        self._save_index(idx)

    # ── File helpers ───────────────────────────────────────────────────────────

    def _find_file(self, folder: str, slug_or_title: str) -> Optional[Path]:
        """Find a note by slug or partial title match."""
        folder_path = self._root / folder
        # Direct hit
        p = folder_path / f"{slug_or_title}.md"
        if p.exists():
            return p
        # Slug match
        slug = _slug(slug_or_title)
        p = folder_path / f"{slug}.md"
        if p.exists():
            return p
        # Partial filename match
        for f in folder_path.glob("*.md"):
            if slug_or_title.lower() in f.stem.lower():
                return f
        return None

    def _all_notes(self) -> list[tuple[str, Path]]:
        """Return (folder, path) for every note across all folders."""
        results = []
        for folder in _FOLDERS:
            for p in sorted((self._root / folder).glob("*.md")):
                results.append((folder, p))
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  WRITE
# ══════════════════════════════════════════════════════════════════════════════

class WikiWriteTool(BaseTool, _WikiBase):
    name = "wiki_write"
    description = (
        "Write to your personal knowledge base (second brain). "
        "Three folders: 'inbox' (quick captures), 'notes' (working/active), "
        "'reference' (permanent wiki). Content is Markdown. "
        "Use #tags and [[WikiLinks]] for cross-referencing. "
        "If a note with the same title exists it is updated (appended or overwritten)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title":   {"type": "string",  "description": "Note title (becomes filename)"},
            "content": {"type": "string",  "description": "Markdown content"},
            "folder":  {"type": "string",  "description": "'inbox', 'notes', or 'reference' (default: inbox)"},
            "append":  {"type": "boolean", "description": "If true, append to existing note instead of overwriting"},
            "tags":    {"type": "array",   "items": {"type": "string"}, "description": "Extra tags to add"},
        },
        "required": ["title", "content"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        BaseTool.__init__(self)
        _WikiBase.__init__(self, working_dir)

    def execute(self, inp: dict) -> ToolResult:
        title   = str(inp.get("title", "")).strip()
        content = str(inp.get("content", "")).strip()
        folder  = _canon_folder(inp.get("folder", "inbox"))
        append  = bool(inp.get("append", False))
        extra_tags = [str(t).strip().lstrip("#") for t in (inp.get("tags") or [])]

        if not title:
            return ToolResult("No title provided.", is_error=True)
        if not content:
            return ToolResult("No content provided.", is_error=True)

        slug   = _slug(title)
        path   = self._root / folder / f"{slug}.md"
        today  = _now_stamp()

        # Build header block (front-matter style, human readable)
        tag_str = " ".join(f"#{t}" for t in extra_tags) if extra_tags else ""
        header = f"# {title}\n\n> 📅 {today}  |  📂 {_FOLDERS[folder]}{('  |  ' + tag_str) if tag_str else ''}\n\n"

        if append and path.exists():
            existing = path.read_text(encoding="utf-8")
            new_content = existing + f"\n\n---\n\n_Updated {today}_\n\n{content}"
            path.write_text(new_content, encoding="utf-8")
            action = "appended"
        else:
            path.write_text(header + content, encoding="utf-8")
            action = "written"

        tags  = list(set(_extract_tags(path.read_text(encoding="utf-8")) + extra_tags))
        links = _extract_links(content)
        meta  = {
            "title": title, "folder": folder, "slug": slug,
            "tags": tags, "links": links,
            "created": today, "updated": today,
            "size": path.stat().st_size,
        }
        self._update_index(folder, slug, meta)

        return ToolResult(
            f"✅ Note '{title}' {action} → wiki/{folder}/{slug}.md\n"
            f"   Tags: {', '.join('#'+t for t in tags) or 'none'}  "
            f"Links: {', '.join('[['+l+']]' for l in links) or 'none'}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  READ
# ══════════════════════════════════════════════════════════════════════════════

class WikiReadTool(BaseTool, _WikiBase):
    name = "wiki_read"
    description = (
        "Read a note from your knowledge base. "
        "Specify folder ('inbox'/'notes'/'reference') and title or slug. "
        "If folder is omitted, searches all three."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title":  {"type": "string", "description": "Note title or slug"},
            "folder": {"type": "string", "description": "Folder to look in (omit to search all)"},
        },
        "required": ["title"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        BaseTool.__init__(self)
        _WikiBase.__init__(self, working_dir)

    def execute(self, inp: dict) -> ToolResult:
        title  = str(inp.get("title", "")).strip()
        folder = inp.get("folder", "")

        if not title:
            return ToolResult("No title provided.", is_error=True)

        search_in = [_canon_folder(folder)] if folder else list(_FOLDERS.keys())
        for f in search_in:
            p = self._find_file(f, title)
            if p:
                content = p.read_text(encoding="utf-8")
                if len(content) > _MAX_READ_CHARS:
                    content = content[:_MAX_READ_CHARS] + f"\n\n… [{len(content)-_MAX_READ_CHARS:,} chars truncated]"
                tags  = _extract_tags(content)
                links = _extract_links(content)
                return ToolResult(
                    f"📂 wiki/{f}/{p.stem}.md\n"
                    f"Tags: {', '.join('#'+t for t in tags) or 'none'}  "
                    f"Links: {', '.join('[['+l+']]' for l in links) or 'none'}\n\n"
                    + content
                )

        folders_searched = ", ".join(search_in)
        return ToolResult(
            f"Note '{title}' not found in [{folders_searched}].\n"
            f"Use wiki_list to see available notes.",
            is_error=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH
# ══════════════════════════════════════════════════════════════════════════════

class WikiSearchTool(BaseTool, _WikiBase):
    name = "wiki_search"
    description = (
        "Full-text search across your entire knowledge base (all three folders). "
        "Searches titles, content, and tags. Returns ranked results with excerpts."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query":   {"type": "string",  "description": "Search terms"},
            "folder":  {"type": "string",  "description": "Limit to 'inbox'/'notes'/'reference' (optional)"},
            "tag":     {"type": "string",  "description": "Filter by tag (optional)"},
            "max":     {"type": "integer", "description": "Max results (default 10)"},
        },
        "required": ["query"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        BaseTool.__init__(self)
        _WikiBase.__init__(self, working_dir)

    def execute(self, inp: dict) -> ToolResult:
        query  = str(inp.get("query", "")).strip().lower()
        folder = inp.get("folder", "")
        tag    = str(inp.get("tag", "")).strip().lstrip("#").lower()
        max_r  = max(1, min(int(inp.get("max", 10)), 30))

        if not query:
            return ToolResult("No query provided.", is_error=True)

        terms = query.split()
        search_in = [_canon_folder(folder)] if folder else list(_FOLDERS.keys())

        results: list[dict] = []
        for f in search_in:
            for p in sorted((self._root / f).glob("*.md")):
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                content_lower = content.lower()
                # Tag filter
                if tag and f"#{tag}" not in content_lower:
                    continue
                # Score: count term occurrences (title match = 3x)
                score = 0
                for t in terms:
                    score += content_lower.count(t)
                    if t in p.stem.lower():
                        score += 3
                if score == 0:
                    continue
                # Find best excerpt
                best_pos = max(
                    (content_lower.find(t) for t in terms if t in content_lower),
                    default=0,
                )
                start = max(0, best_pos - 60)
                excerpt = content[start:start + _MAX_SEARCH_CHARS].strip()
                results.append({"folder": f, "stem": p.stem, "score": score, "excerpt": excerpt})

        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:max_r]

        if not results:
            return ToolResult(f"No notes found matching '{query}'.")

        lines = [f"🔍 Search: '{query}' — {len(results)} result(s)\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**{i}. wiki/{r['folder']}/{r['stem']}.md** (score={r['score']})")
            lines.append(f"   {r['excerpt'][:300].replace(chr(10), ' ')}")
            lines.append("")
        return ToolResult("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  LIST
# ══════════════════════════════════════════════════════════════════════════════

class WikiListTool(BaseTool, _WikiBase):
    name = "wiki_list"
    description = (
        "List notes in your knowledge base. "
        "Filter by folder, tag, or title prefix. "
        "Shows the three-folder overview if no filter is given."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "folder": {"type": "string", "description": "'inbox', 'notes', or 'reference' (omit for all)"},
            "tag":    {"type": "string", "description": "Filter by tag (optional)"},
            "prefix": {"type": "string", "description": "Filter by title prefix (optional)"},
        },
        "required": [],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        BaseTool.__init__(self)
        _WikiBase.__init__(self, working_dir)

    def execute(self, inp: dict) -> ToolResult:
        folder = inp.get("folder", "")
        tag    = str(inp.get("tag", "")).strip().lstrip("#").lower()
        prefix = str(inp.get("prefix", "")).lower()

        search_in = [_canon_folder(folder)] if folder else list(_FOLDERS.keys())

        lines = ["📚 **EVE Knowledge Base**\n"]
        total = 0
        for f in search_in:
            folder_notes = []
            for p in sorted((self._root / f).glob("*.md")):
                try:
                    content = p.read_text(encoding="utf-8")
                except Exception:
                    continue
                if tag and f"#{tag}" not in content.lower():
                    continue
                if prefix and not p.stem.lower().startswith(prefix):
                    continue
                tags  = _extract_tags(content)
                links = _extract_links(content)
                first_line = content.splitlines()[0].lstrip("# ").strip() if content else p.stem
                folder_notes.append(
                    f"  • `{p.stem}.md` — {first_line[:60]}"
                    + (f" | Tags: {', '.join('#'+t for t in tags[:4])}" if tags else "")
                    + (f" | Links: {len(links)}" if links else "")
                )
            if folder_notes:
                lines.append(f"\n{_FOLDERS[f]} ({len(folder_notes)} notes)")
                lines.extend(folder_notes)
                total += len(folder_notes)

        if total == 0:
            filter_desc = " | ".join(x for x in [
                f"folder={folder}" if folder else "",
                f"tag=#{tag}" if tag else "",
                f"prefix={prefix}" if prefix else "",
            ] if x)
            return ToolResult(f"No notes found{' matching ' + filter_desc if filter_desc else ''}.")

        lines.append(f"\nTotal: {total} note(s)")
        return ToolResult("\n".join(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  PROMOTE (inbox → notes → reference)
# ══════════════════════════════════════════════════════════════════════════════

class WikiPromoteTool(BaseTool, _WikiBase):
    name = "wiki_promote"
    description = (
        "Promote a note from inbox → notes → reference (or any folder → folder). "
        "Used to process and organise captured knowledge. "
        "Optionally add a summary or restructure the content."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "title":       {"type": "string", "description": "Note title or slug"},
            "from_folder": {"type": "string", "description": "Source folder (default: inbox)"},
            "to_folder":   {"type": "string", "description": "Destination folder (default: notes)"},
            "new_content": {"type": "string", "description": "Optional: replace content during promotion"},
        },
        "required": ["title"],
    }

    def __init__(self, working_dir: str = WORKING_DIR):
        BaseTool.__init__(self)
        _WikiBase.__init__(self, working_dir)

    def execute(self, inp: dict) -> ToolResult:
        title       = str(inp.get("title", "")).strip()
        from_folder = _canon_folder(inp.get("from_folder", "inbox"))
        to_folder   = _canon_folder(inp.get("to_folder",   "notes"))
        new_content = inp.get("new_content", "")

        if not title:
            return ToolResult("No title provided.", is_error=True)
        if from_folder == to_folder:
            return ToolResult(f"Source and destination are the same folder: {from_folder}.", is_error=True)

        src = self._find_file(from_folder, title)
        if not src:
            return ToolResult(f"Note '{title}' not found in {from_folder}.", is_error=True)

        content = new_content.strip() if new_content.strip() else src.read_text(encoding="utf-8")
        dest    = self._root / to_folder / src.name
        dest.write_text(content, encoding="utf-8")
        src.unlink()

        self._remove_from_index(from_folder, src.stem)
        tags  = _extract_tags(content)
        links = _extract_links(content)
        self._update_index(to_folder, dest.stem, {
            "title": title, "folder": to_folder, "slug": dest.stem,
            "tags": tags, "links": links,
            "updated": _now_stamp(),
        })

        return ToolResult(
            f"✅ Promoted '{title}'\n"
            f"   {_FOLDERS[from_folder]}  →  {_FOLDERS[to_folder]}\n"
            f"   Path: wiki/{to_folder}/{dest.name}"
        )
