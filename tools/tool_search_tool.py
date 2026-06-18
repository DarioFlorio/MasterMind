# -*- coding: utf-8 -*-
"""
tools/tool_search_tool.py — IDS: Intelligent Dispatch Search

Aggressively indexes all harness nodes — tools, skills, wiki docs — and
returns ranked results via BM25-style scoring.  The model should call
tool_search at the START of every task and before each major step so it
always picks up the best available capability.

Usage:
  tool_search(query="edit python file surgically")
  tool_search(query="reason about cause and effect", kind="skill")
  tool_search(list_all=True, kind="skill")
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tools.base_tool import BaseTool, ToolResult

# ── Node ────────────────────────────────────────────────────────────────────

@dataclass
class IDSNode:
    id:          str
    kind:        str          # "tool" | "skill" | "doc"
    name:        str
    description: str
    keywords:    list[str] = field(default_factory=list)
    tags:        list[str] = field(default_factory=list)


# ── BM25-ish index ────────────────────────────────────────────────────────────

_STOP = frozenset({
    "a","an","the","is","are","for","to","of","in","and","or","with","by",
    "at","on","as","it","its","this","that","be","from","do","use","using",
    "you","your","can","will","have","has","all","any","not","more","get",
})

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in _STOP and len(t) > 1]


class IDSIndex:
    """Lightweight BM25-ish in-memory index over harness nodes."""

    K1 = 1.5   # term saturation
    B  = 0.75  # length normalisation

    def __init__(self):
        self._nodes:    dict[str, IDSNode]   = {}
        self._tf:       dict[str, Counter]   = {}   # node_id → term_freq
        self._df:       Counter              = Counter()
        self._dl:       dict[str, int]       = {}   # node_id → doc_len
        self._avgdl:    float                = 1.0
        self._N:        int                  = 0
        self._dirty:    bool                 = True

    def add(self, node: IDSNode):
        # Weight name tokens 3× and keyword tokens 2×
        tokens = (
            _tokenize(node.name)          * 3 +
            _tokenize(node.description)       +
            _tokenize(" ".join(node.keywords)) * 2 +
            _tokenize(" ".join(node.tags))
        )
        self._nodes[node.id] = node
        self._tf[node.id]    = Counter(tokens)
        self._dl[node.id]    = len(tokens)
        for t in set(tokens):
            self._df[t] += 1
        self._dirty = True

    def _build(self):
        self._N     = len(self._nodes)
        self._avgdl = (sum(self._dl.values()) / self._N) if self._N else 1.0
        self._dirty = False

    def search(
        self,
        query:       str,
        top_k:       int           = 8,
        kind_filter: Optional[str] = None,
    ) -> list[tuple[IDSNode, float]]:
        if self._dirty:
            self._build()

        qtokens = _tokenize(query)
        if not qtokens:
            nodes = [n for n in self._nodes.values()
                     if not kind_filter or n.kind == kind_filter]
            return [(n, 1.0) for n in nodes[:top_k]]

        N      = max(self._N, 1)
        avgdl  = self._avgdl
        scores: dict[str, float] = defaultdict(float)

        for nid, tf in self._tf.items():
            node = self._nodes[nid]
            if kind_filter and node.kind != kind_filter:
                continue
            dl   = self._dl[nid]
            norm = 1 - self.B + self.B * (dl / avgdl)
            score = 0.0
            for t in qtokens:
                if not tf[t]:
                    continue
                idf = math.log(1 + (N - self._df.get(t, 0) + 0.5) /
                               (self._df.get(t, 0) + 0.5))
                tf_norm = tf[t] * (self.K1 + 1) / (tf[t] + self.K1 * norm)
                score  += idf * tf_norm

            if score <= 0:
                continue

            # Bonus: exact or prefix name match
            qlow = query.lower().replace("_", " ").replace("-", " ")
            nlow = node.name.lower().replace("_", " ").replace("-", " ")
            if qlow == nlow:
                score *= 3.0
            elif qlow in nlow or nlow in qlow:
                score *= 2.0
            elif nlow.startswith(qlow.split()[0]) if qlow.split() else False:
                score *= 1.5

            scores[nid] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(self._nodes[nid], s) for nid, s in ranked[:top_k]]


# ── Global state ──────────────────────────────────────────────────────────────

_index:    IDSIndex            = IDSIndex()
_registry: dict[str, BaseTool] = {}
_ready:    bool                = False
_ROOT      = Path(__file__).parent.parent


# ── Index builders ────────────────────────────────────────────────────────────

def _index_tools():
    for name, tool in _registry.items():
        desc = getattr(tool, "description", "") or ""
        schema = getattr(tool, "input_schema", {}) or {}
        # Harvest param names as extra keywords
        props = schema.get("properties", {})
        kw = list(props.keys()) + _tokenize(desc)[:20]
        _index.add(IDSNode(
            id=f"tool:{name}", kind="tool",
            name=name, description=desc[:300],
            keywords=kw[:30], tags=["tool"],
        ))


def _index_skills():
    skills_dir = _ROOT / "skills"
    if not skills_dir.exists():
        return
    for py in sorted(skills_dir.glob("*.py")):
        if py.name.startswith("_") or py.name in (
            "base_skill.py", "skill_router.py", "skill_tool.py",
            "thinking_controller.py", "__init__.py",
        ):
            continue
        skill_name = py.stem
        try:
            src = py.read_text(encoding="utf-8", errors="replace")
            # Extract first docstring or first comment block
            m = re.search(r'"""(.+?)"""', src, re.DOTALL)
            if m:
                raw = m.group(1).strip().replace("\n", " ")[:250]
            else:
                lines = [l.lstrip("#").strip() for l in src.splitlines()
                         if l.strip().startswith("#")]
                raw = " ".join(lines[:3])[:250] or skill_name
        except Exception:
            raw = skill_name
        _index.add(IDSNode(
            id=f"skill:{skill_name}", kind="skill",
            name=skill_name, description=raw,
            keywords=_tokenize(raw)[:20] + [skill_name.replace("_", " ")],
            tags=["skill", "reasoning"],
        ))

    # Skill markdown reference docs
    for md in sorted(skills_dir.glob("SKILL_*.md")):
        doc_name = md.stem[6:].lower()   # strip "SKILL_"
        try:
            content = md.read_text(encoding="utf-8", errors="replace")
            first_para = re.sub(r"#[^\n]*\n", "", content)[:400].replace("\n", " ")
        except Exception:
            first_para = doc_name
        _index.add(IDSNode(
            id=f"doc:{doc_name}", kind="doc",
            name=doc_name, description=first_para,
            keywords=_tokenize(first_para)[:20],
            tags=["doc", "skill", "reference"],
        ))


def register_all(tools: dict[str, BaseTool]) -> None:
    _registry.update(tools)
    _rebuild_index()


def _rebuild_index():
    global _index, _ready
    _index = IDSIndex()
    _index_tools()
    _index_skills()
    _index._build()
    _ready = True


# ── Tool ─────────────────────────────────────────────────────────────────────

class ToolSearchTool(BaseTool):
    name = "tool_search"
    description = (
        "IDS — Intelligent Dispatch Search: aggressively find the best harness node "
        "(tool, skill, or reference doc) for the current task. "
        "Call this at the START of any task and before each major step to latch "
        "onto the highest-scoring capability. "
        "Returns ranked results with scores. "
        "Examples: query='edit file surgically', kind='tool'; "
        "query='bayesian reasoning', kind='skill'."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query":    {"type": "string",  "description": "Free-text description of what you need to do"},
            "kind":     {"type": "string",  "description": "Optional filter: 'tool', 'skill', or 'doc'"},
            "top_k":    {"type": "integer", "description": "Max results to return (default 8, max 30)"},
            "list_all": {"type": "boolean", "description": "List all indexed nodes of the given kind"},
        },
        "required": [],
    }

    def __init__(self, tools: "dict[str, BaseTool] | None" = None):
        if tools:
            _registry.update(tools)

    def execute(self, inp: dict) -> ToolResult:
        inp      = self.safe_parse(inp)
        query    = inp.get("query", "").strip()
        kind     = (inp.get("kind", "") or "").strip().lower() or None
        top_k    = min(max(int(inp.get("top_k", 8)), 1), 30)
        list_all = bool(inp.get("list_all", False))

        if not _ready:
            _rebuild_index()

        if list_all or not query:
            nodes = list(_index._nodes.values())
            if kind:
                nodes = [n for n in nodes if n.kind == kind]
            results: list[tuple[IDSNode, float]] = [
                (n, 1.0) for n in sorted(nodes, key=lambda n: (n.kind, n.name))[:top_k]
            ]
        else:
            results = _index.search(query, top_k=top_k, kind_filter=kind)

        if not results:
            hint = f" of kind '{kind}'" if kind else ""
            return ToolResult(
                output=f"IDS: no nodes matched {query!r}{hint}.\n"
                       f"Try: tool_search(list_all=True, kind='tool') to browse all tools."
            )

        ICON = {"tool": "🔧", "skill": "🧠", "doc": "📄"}
        header = f"IDS — {len(results)} node(s) for {query!r}" if query else \
                 f"IDS — {len(results)} node(s)" + (f" [{kind}]" if kind else "")
        lines  = [header, ""]
        for node, score in results:
            icon   = ICON.get(node.kind, "○")
            desc   = (node.description[:85] + "…") if len(node.description) > 85 else node.description
            lines.append(f"  {icon} {node.name:<30}  score={score:5.2f}")
            lines.append(f"     [{node.kind}] {desc}")
            if node.kind == "tool":
                lines.append(f"     → use: tool_name=\"{node.name}\"")
            elif node.kind == "skill":
                lines.append(f"     → use: skill(\"{node.name}\", problem=<your question>)")
            lines.append("")
        return ToolResult(output="\n".join(lines).rstrip())
