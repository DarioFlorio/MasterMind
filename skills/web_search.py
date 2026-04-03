"""
skills/web_search.py — Deep web search skill with BFS + IDS traversal.

Strategy
--------
  BFS  (Breadth-First Search)  — used for broad queries, news, current events.
        Fires N parallel top-level queries, collects all results, deduplicates.
        Good when you want wide coverage fast.

  IDS  (Iterative-Deepening Search) — used for deep research / single topics.
        Starts with a root query, extracts sub-queries from result snippets,
        then fans out depth-first to a configurable depth limit.
        Good when you need to drill down into a specific subject.

  AUTO — picks BFS for multi-word news/event queries; IDS for narrow topics.
         Can be overridden via the `strategy` parameter.

Installation
------------
  pip install ddgs          # renamed from duckduckgo-search
  pip install httpx         # for optional page fetch
"""
from __future__ import annotations

import re
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import httpx as _httpx
    _HTTPX = True
except ImportError:
    _HTTPX = False

# ── ddgs import (supports both old and new package names) ────────────────────
_DDGS = None
_AVAILABLE = False

for _pkg in ("ddgs", "duckduckgo_search"):
    try:
        if _pkg == "ddgs":
            from ddgs import DDGS as _DDGS
        else:
            from duckduckgo_search import DDGS as _DDGS
        _AVAILABLE = True
        break
    except ImportError:
        continue

try:
    from tools.base_tool import BaseTool, ToolResult
except ImportError:
    class BaseTool:  # type: ignore
        pass
    class ToolResult:  # type: ignore
        def __init__(self, output: str, is_error: bool = False):
            self.output   = output
            self.is_error = is_error


# ── Thread-local DDGS session (avoids per-call handshake overhead) ───────────
_local = threading.local()

def _get_ddgs():
    """Return a cached DDGS session for the current thread."""
    if not hasattr(_local, "ddgs") or _local.ddgs is None:
        _local.ddgs = _DDGS()
    return _local.ddgs


def _safe_search(query: str, max_results: int = 8) -> list:
    """Run one DuckDuckGo search, return raw result dicts. Never raises."""
    try:
        ddgs = _get_ddgs()
        results = list(ddgs.text(query.strip(), max_results=max_results))
        return results
    except Exception:
        _local.ddgs = None
        try:
            results = list(_get_ddgs().text(query.strip(), max_results=max_results))
            return results
        except Exception:
            return []


# ── Sub-query extractor ───────────────────────────────────────────────────────
_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can need dare used to "
    "and or but nor so yet for of in on at by with from as into "
    "through about against between into during before after above "
    "below to from up down out off over under again further then "
    "once that this these those it its they them their there here "
    "when where which who whom what how why all each every more most "
    "other some such no only same than too very just because while "
    "although though".split()
)


def _extract_subqueries(base_query: str, snippets: list, n: int = 3) -> list:
    base_words = set(base_query.lower().split())
    freq: dict = {}

    for snippet in snippets:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-']{2,}\b", snippet.lower())
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if w1 in _STOP or w2 in _STOP:
                continue
            if w1 in base_words and w2 in base_words:
                continue
            phrase = f"{w1} {w2}"
            freq[phrase] = freq.get(phrase, 0) + 1

    ranked = sorted(freq, key=lambda p: -freq[p])
    candidates = []
    for phrase in ranked:
        sub = f"{base_query} {phrase}"
        candidates.append(sub)
        if len(candidates) >= n:
            break

    return candidates


# ── BFS search ────────────────────────────────────────────────────────────────

def _bfs_search(queries: list, max_results_per_query: int = 8, max_workers: int = 6) -> list:
    """Parallel BFS: fire all queries at once, collect & deduplicate results."""
    seen_urls: set = set()
    all_results: list = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_safe_search, q, max_results_per_query): q for q in queries}
        for fut in as_completed(futures):
            for r in fut.result():
                url = r.get("href", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

    return all_results


# ── IDS search ────────────────────────────────────────────────────────────────

def _ids_search(root_query: str, depth_limit: int = 2, branch_factor: int = 3,
                results_per_node: int = 6) -> list:
    """
    Iterative-Deepening Search: start from root_query, extract sub-queries
    from snippets, recurse to depth_limit.
    """
    seen_urls: set = set()
    visited_queries: set = set()
    all_results: list = []

    queue: deque = deque()
    queue.append((root_query, 0))

    while queue:
        query, depth = queue.popleft()
        if query in visited_queries:
            continue
        visited_queries.add(query)

        results = _safe_search(query, results_per_node)
        new_snippets: list = []

        for r in results:
            url = r.get("href", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)
                snippet = r.get("body", "")
                if snippet:
                    new_snippets.append(snippet)

        if depth < depth_limit and new_snippets:
            sub_queries = _extract_subqueries(root_query, new_snippets, branch_factor)
            for sq in sub_queries:
                if sq not in visited_queries:
                    queue.append((sq, depth + 1))

    return all_results


# ── Keyword expansion ─────────────────────────────────────────────────────────

def _expand_keywords(query: str, n: int = 4) -> list:
    """Generate n related keyword variants for parallel IDS sweep."""
    words = [w for w in query.lower().split() if w not in _STOP and len(w) > 2]
    variants: list = []
    # focus-shifted: drop first word
    if len(words) >= 3:
        variants.append(" ".join(words[1:]))
    # reversed-focus
    if len(words) >= 2:
        variants.append(" ".join(list(reversed(words))[:3]))
    # recency anchor
    year = time.strftime("%Y")
    if year not in query:
        variants.append(f"{query} {year}")
    # explanation angle
    variants.append(f"{query} explained")
    # deduplicate and strip matches identical to original
    seen = {query.lower()}
    out = []
    for v in variants:
        v = v.strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
        if len(out) >= n:
            break
    return out


# ── URL fetch + summarise ─────────────────────────────────────────────────────

def _fetch_url(url: str, timeout: float = 6.0, max_chars: int = 1200) -> str:
    """Fetch a URL via httpx and return stripped plain-text summary."""
    if not _HTTPX or not url:
        return ""
    try:
        r = _httpx.get(
            url, timeout=timeout, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; EVE-search/1.0)"},
        )
        text = r.text
        text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.I)
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.I)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


# ── Parallel IDS sweep across keyword variants ────────────────────────────────

def _parallel_ids_sweep(queries: list, depth_limit: int = 2,
                        branch_factor: int = 3, results_per_node: int = 6,
                        max_workers: int = 4) -> list:
    """Run IDS on each keyword variant in parallel, merge results."""
    seen_urls: set = set()
    all_results: list = []
    lock = threading.Lock()

    def _ids_worker(q):
        return _ids_search(q, depth_limit, branch_factor, results_per_node)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_ids_worker, q): q for q in queries}
        for fut in as_completed(futures):
            for r in fut.result():
                url = r.get("href", "")
                with lock:
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(r)
    return all_results


_NEWS_SIGNALS = re.compile(
    r"\b(latest|recent|today|yesterday|last night|breaking|news|update|"
    r"announce|release|leak|report|2024|2025|2026|this week|this month)\b",
    re.I,
)

def _pick_strategy(query: str, hint: Optional[str]) -> str:
    if hint and hint.lower() in ("bfs", "ids", "auto"):
        if hint.lower() != "auto":
            return hint.lower()
    if _NEWS_SIGNALS.search(query):
        return "bfs"
    return "bfs" if len(query.split()) >= 4 else "ids"


# ── Result formatter ──────────────────────────────────────────────────────────

def _fmt_results(query: str, results: list, strategy: str,
                 fetched: dict | None = None) -> str:
    if not results:
        return f"No results found for: {query}"

    lines = [
        f"Search results for: {query}",
        f"Strategy: {strategy.upper()}  |  {len(results)} unique results",
        "",
    ]
    for i, r in enumerate(results, 1):
        title   = r.get("title", "No title")
        url     = r.get("href", "")
        snippet = r.get("body", "")
        lines.append(f"{i}. {title}")
        if url:
            lines.append(f"   URL: {url}")
        if snippet:
            lines.append(f"   {snippet}")
        # Append fetched page summary if available
        if fetched and url and url in fetched and fetched[url]:
            lines.append(f"   [Page content]: {fetched[url]}")
        lines.append("")

    return "\n".join(lines)


# ── Tool class ────────────────────────────────────────────────────────────────

class WebSearchSkill(BaseTool):
    name        = "web_search"
    description = (
        "Search the web for current information, news, leaks, documentation, "
        "or anything you don't know.\n\n"
        "Uses two traversal strategies:\n"
        "  BFS — parallel broad search across multiple queries (fast, wide coverage).\n"
        "  IDS — iterative deepening from a root query into sub-topics (thorough deep dive).\n"
        "The strategy is chosen automatically based on the query, or you can set it explicitly.\n\n"
        "Returns FULL untruncated titles, URLs, and snippets — no token limits."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Primary search query.",
            },
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional list of additional parallel queries (BFS mode). "
                    "If omitted, sub-queries are auto-generated."
                ),
            },
            "strategy": {
                "type": "string",
                "enum": ["auto", "bfs", "ids"],
                "description": (
                    "'bfs' = parallel broad search (best for news/current events); "
                    "'ids' = iterative deepening (best for deep research); "
                    "'auto' = chosen automatically (default)."
                ),
            },
            "depth": {
                "type": "integer",
                "description": "IDS depth limit (default 2). Ignored in BFS mode.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results per query node (default 8). No upper cap.",
            },
            "fetch_pages": {
                "type": "boolean",
                "description": "Fetch and summarise actual page content for top results (default true).",
            },
            "fetch_top_n": {
                "type": "integer",
                "description": "Number of top result URLs to fetch and summarise (default 5).",
            },
        },
        "required": ["query"],
    }

    def execute(self, input_data: dict) -> ToolResult:
        if not _AVAILABLE:
            return ToolResult(
                output=(
                    "web_search requires the 'ddgs' package.\n"
                    "Run:  pip install ddgs\n"
                    "(The old 'duckduckgo-search' package has been renamed.)"
                ),
                is_error=True,
            )

        root_query = (input_data.get("query") or "").strip()
        if not root_query:
            return ToolResult(output="'query' is required.", is_error=True)

        extra_queries: list = input_data.get("queries") or []
        strategy_hint: str  = input_data.get("strategy", "auto")
        depth:         int  = int(input_data.get("depth", 2))
        max_results:   int  = int(input_data.get("max_results", 8))
        fetch_pages:   bool = bool(input_data.get("fetch_pages", True))
        fetch_top_n:   int  = int(input_data.get("fetch_top_n", 5))

        strategy = _pick_strategy(root_query, strategy_hint)

        t0 = time.perf_counter()

        try:
            # ── Phase 1: BFS for broad top-level results ──────────────────
            bfs_queries = [root_query] + extra_queries
            bfs_results = _bfs_search(
                queries               = bfs_queries,
                max_results_per_query = max_results,
                max_workers           = max(4, len(bfs_queries) * 2),
            )

            # ── Phase 2: IDS sweep across keyword variants (parallel) ─────
            kw_variants = _expand_keywords(root_query, n=4)
            ids_results: list = []
            if kw_variants:
                ids_results = _parallel_ids_sweep(
                    queries        = kw_variants,
                    depth_limit    = depth,
                    branch_factor  = 3,
                    results_per_node = max(4, max_results // 2),
                    max_workers    = min(len(kw_variants), 4),
                )

            # ── Merge BFS + IDS, deduplicate ──────────────────────────────
            seen_urls: set = set()
            all_results: list = []
            for r in bfs_results + ids_results:
                url = r.get("href", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)

            # ── Phase 3: Fetch & summarise top-N URLs (IDS crawl style) ──
            fetched: dict = {}
            if fetch_pages and _HTTPX and all_results:
                top_urls = [r.get("href", "") for r in all_results[:fetch_top_n] if r.get("href")]
                with ThreadPoolExecutor(max_workers=min(len(top_urls), 6)) as pool:
                    fut_map = {pool.submit(_fetch_url, u): u for u in top_urls}
                    for fut in as_completed(fut_map):
                        url = fut_map[fut]
                        content = fut.result()
                        if content:
                            fetched[url] = content

        except Exception as exc:
            return ToolResult(output=f"Search failed: {exc}", is_error=True)

        elapsed = time.perf_counter() - t0
        output  = _fmt_results(root_query, all_results, strategy, fetched)
        output += f"\n[BFS: {len(bfs_results)} results | IDS variants: {len(kw_variants)} | " \
                  f"Pages fetched: {len(fetched)} | {elapsed:.2f}s]"

        return ToolResult(output=output)


# ── Standalone smoke test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    class _TR:
        def __init__(self, o, e=False):
            self.output   = o
            self.is_error = e

    ToolResult = _TR  # type: ignore

    query = " ".join(sys.argv[1:]) or "anthropic claude leak today"
    skill = WebSearchSkill()
    result = skill.execute({"query": query, "strategy": "auto"})
    print(result.output)

# ─────────────────────────────────────────────────────────────────────────────
# BaseSkill wrapper so web_search registers in SKILL_REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
from skills.base_skill import BaseSkill


class WebSearchBaseSkill(BaseSkill):
    """Adapter that wraps WebSearchSkill under the BaseSkill interface."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information, news, or anything you don't know. "
            "Uses BFS (broad) or IDS (deep) traversal strategies."
        )

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem":     {"type": "string", "description": "Primary search query"},
                "depth":       {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
                "queries":     {"type": "array",   "items": {"type": "string"}},
                "strategy":    {"type": "string",  "enum": ["auto", "bfs", "ids"]},
                "max_results": {"type": "integer"},
                "fetch_pages": {"type": "boolean"},
                "fetch_top_n": {"type": "integer"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        input_data = {"query": problem}
        for key in ("queries", "strategy", "depth", "max_results", "fetch_pages", "fetch_top_n"):
            if key in kwargs:
                input_data[key] = kwargs[key]
        skill  = WebSearchSkill()
        result = skill.execute(input_data)
        return result.output
