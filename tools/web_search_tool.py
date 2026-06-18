"""
tools/web_search_tool.py
========================
Full-stack web research pipeline:
  1. DDG search  → top N results (title, url, snippet)
  2. BFS Layer 1 → parallel-fetch every result URL
                   extract: main text, title, outbound links
  3. IDS Layers  → score outbound links by query-term overlap
                   iteratively fetch top-K at each depth
  4. Synthesis   → deduplicate, rank, format with per-source excerpts

Mirrors how a research assistant browses: search → read → follow best leads.
"""
from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FutureTimeout
from urllib.parse import urljoin, urlparse, urlunparse
from typing import Optional

from tools.base_tool import BaseTool, ToolResult

# ── DDG package compat (renamed ddgs / duckduckgo_search) ────────────────────
_DDGS = None
_DDG_OK = False
for _pkg in ("ddgs", "duckduckgo_search"):
    try:
        if _pkg == "ddgs":
            from ddgs import DDGS as _DDGS
        else:
            from duckduckgo_search import DDGS as _DDGS
        _DDG_OK = True
        break
    except ImportError:
        continue

# ── Tuning knobs ──────────────────────────────────────────────────────────────
_MAX_PAGE_CHARS  = 12_000   # chars kept per fetched page
_MAX_EXCERPT     = 500      # chars shown per source in output
_FETCH_TIMEOUT   = 10       # seconds per HTTP GET
_BFS_WORKERS     = 10       # parallel fetchers
_SKIP_EXT = re.compile(
    r"\.(pdf|docx?|xlsx?|pptx?|zip|gz|tar|exe|dmg|mp[34]|mkv|avi"
    r"|jpg|jpeg|png|gif|webp|svg|ico|woff2?|ttf|eot|otf)$",
    re.I,
)
_SKIP_HOST = re.compile(
    r"(facebook\.com|twitter\.com|x\.com|instagram\.com|tiktok\.com"
    r"|youtube\.com|linkedin\.com/feed|reddit\.com/login"
    r"|accounts\.google\.|login\.|signup\.)",
    re.I,
)

# ── Thread-local DDG session ──────────────────────────────────────────────────
_local = threading.local()

def _ddg_inst():
    if not _DDG_OK:
        return None
    if not getattr(_local, "inst", None):
        _local.inst = _DDGS()
    return _local.inst

def _ddg_search(query: str, n: int = 8) -> list[dict]:
    try:
        return list(_ddg_inst().text(query.strip(), max_results=n))
    except Exception:
        _local.inst = None
        try:
            return list(_ddg_inst().text(query.strip(), max_results=n))
        except Exception:
            return []

# ── HTTP fetch + HTML extraction ──────────────────────────────────────────────

def _fetch_page(url: str) -> Optional[dict]:
    """
    Fetch URL → {url, title, text, links} or None.
    Strips boilerplate, extracts main content + all outbound links.
    """
    if _SKIP_EXT.search(url) or _SKIP_HOST.search(url):
        return None
    try:
        import httpx
        r = httpx.get(
            url, timeout=_FETCH_TIMEOUT, follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/124.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if r.status_code >= 400:
            return None
        ct = r.headers.get("content-type", "")
        if "text/html" not in ct and "text/plain" not in ct:
            return None
        raw = r.text
    except Exception:
        return None

    # Parse HTML
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # No BS4 — return raw text truncated
        return {"url": url, "title": url, "text": raw[:_MAX_PAGE_CHARS], "links": []}

    soup = BeautifulSoup(raw, "html.parser")

    # Title
    t = soup.find("title")
    title = t.get_text(strip=True) if t else url

    # Strip boilerplate elements
    for tag in soup(["script", "style", "noscript", "iframe",
                     "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    # Strip ad/cookie/social divs by class/id heuristic
    junk_re = re.compile(
        r"(cookie|banner|popup|modal|overlay|ad[s_-]|sidebar"
        r"|comment|share|social|newsletter|subscribe|promo)",
        re.I,
    )
    for tag in soup.find_all(True):
        cls = " ".join(tag.get("class", []))
        eid = tag.get("id", "")
        if junk_re.search(cls) or junk_re.search(eid):
            tag.decompose()

    # Prefer semantic content containers
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find(id=re.compile(r"(content|article|post|entry|main)", re.I))
        or soup.find(class_=re.compile(r"(content|article|post|entry|main)", re.I))
        or soup.find("body")
        or soup
    )

    raw_text = main.get_text(separator="\n", strip=True)
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    # Drop very short lines that are likely nav remnants
    lines = [ln for ln in lines if len(ln) > 20 or ln.endswith((".", ":", "?", "!"))]
    text = "\n".join(lines)[:_MAX_PAGE_CHARS]

    # Extract outbound links with anchor text
    base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
    links: list[dict] = []
    seen_hrefs: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        abs_url = urljoin(base, href)
        p = urlparse(abs_url)
        if p.scheme not in ("http", "https"):
            continue
        clean = urlunparse(p._replace(fragment=""))
        if clean in seen_hrefs:
            continue
        seen_hrefs.add(clean)
        anchor = a.get_text(strip=True)[:120]
        links.append({"url": clean, "anchor": anchor})

    return {"url": url, "title": title, "text": text, "links": links}

# ── Relevance scoring ─────────────────────────────────────────────────────────

def _tok(text: str) -> list[str]:
    return re.findall(r"\b[a-z]{3,}\b", text.lower())

def _score_text(text: str, qtok: list[str]) -> float:
    """TF-based relevance: query-term frequency in page text."""
    if not text or not qtok:
        return 0.0
    toks = _tok(text[:5000])
    if not toks:
        return 0.0
    return sum(toks.count(t) for t in qtok) / len(toks)

def _score_link(link: dict, qtok: list[str]) -> float:
    """Score a candidate link: anchor text + URL path keywords."""
    a = _tok(link.get("anchor", ""))
    u = _tok(link.get("url", ""))
    return sum(1 for t in qtok if t in a) * 2.0 + sum(1 for t in qtok if t in u)

# ── BFS parallel fetch ────────────────────────────────────────────────────────

def _bfs_fetch(urls: list[str], qtok: list[str]) -> list[dict]:
    """Fetch URLs in parallel, score and sort by relevance."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=_BFS_WORKERS) as pool:
        futs = {pool.submit(_fetch_page, u): u for u in urls}
        try:
            for fut in as_completed(futs, timeout=_FETCH_TIMEOUT + 8):
                try:
                    page = fut.result(timeout=1)
                    if page and page.get("text"):
                        page["score"] = _score_text(page["text"], qtok)
                        results.append(page)
                except Exception:
                    pass
        except _FutureTimeout:
            # Some fetches didn't finish in time — collect what's already done
            for fut in futs:
                if fut.done() and not fut.cancelled():
                    try:
                        page = fut.result(timeout=0)
                        if page and page.get("text"):
                            page["score"] = _score_text(page["text"], qtok)
                            if page not in results:
                                results.append(page)
                    except Exception:
                        pass
    results.sort(key=lambda p: p.get("score", 0), reverse=True)
    return results

# ── IDS crawler ───────────────────────────────────────────────────────────────

def _ids_crawl(
    seed_pages: list[dict],
    qtok: list[str],
    max_depth: int,
    max_per_depth: int,
    visited: set[str],
) -> list[dict]:
    """
    Iterative deepening crawl.
    Each depth: score all outbound links from current frontier → fetch top K.
    Stops at max_depth or when no new candidates remain.
    """
    all_new: list[dict] = []
    frontier = seed_pages

    for depth in range(1, max_depth + 1):
        # Collect candidate links from frontier pages
        candidates: list[dict] = []
        seen_cand: set[str] = set()
        for page in frontier:
            for link in page.get("links", []):
                u = link["url"]
                if u in visited or u in seen_cand:
                    continue
                if _SKIP_EXT.search(u) or _SKIP_HOST.search(u):
                    continue
                seen_cand.add(u)
                candidates.append({**link, "score": _score_link(link, qtok)})

        if not candidates:
            break

        candidates.sort(key=lambda c: c["score"], reverse=True)
        to_fetch = [c["url"] for c in candidates[:max_per_depth]]
        new_pages = _bfs_fetch(to_fetch, qtok)

        for p in new_pages:
            visited.add(p["url"])
            p["depth"] = depth

        all_new.extend(new_pages)
        frontier = new_pages or []

        if not frontier:
            break

    return all_new

# ── Output formatter ──────────────────────────────────────────────────────────

def _format(query: str, ddg: list[dict], fetched: list[dict], elapsed: float) -> str:
    # Deduplicate fetched pages
    seen: set[str] = set()
    unique: list[dict] = []
    for p in sorted(fetched, key=lambda x: x.get("score", 0), reverse=True):
        if p["url"] not in seen:
            seen.add(p["url"])
            unique.append(p)

    lines = [
        f"## Research: {query}",
        f"DDG: {len(ddg)} hits · fetched: {len(unique)} pages · {elapsed:.1f}s\n",
        "### Sources\n",
    ]

    for i, p in enumerate(unique[:10], 1):
        title  = (p.get("title") or p["url"])[:100]
        url    = p["url"]
        depth  = p.get("depth", 0)
        text   = p.get("text", "")
        # Pull best excerpt: first paragraph with >80 chars
        paras  = [ln for ln in text.split("\n") if len(ln) > 80]
        excerpt = paras[0][:_MAX_EXCERPT] if paras else text[:_MAX_EXCERPT]
        dlabel = f" [crawled depth {depth}]" if depth else " [direct result]"
        lines += [
            f"**{i}. {title}**{dlabel}",
            f"   {url}",
            f"   {excerpt}",
            "",
        ]

    # DDG-only results (not fetched) as lightweight footnotes
    fetched_urls = {p["url"] for p in unique}
    extras = [r for r in ddg if r.get("href") not in fetched_urls]
    if extras:
        lines.append("### Additional snippets (not crawled)\n")
        for r in extras[:5]:
            body = (r.get("body") or "")[:200]
            lines += [f"- **{r.get('title','')}** — {r.get('href','')}", f"  {body}", ""]

    return "\n".join(lines)

# ── Tool ──────────────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Web research: DDG search → BFS parallel fetch of result pages → "
        "IDS iterative crawl of top outbound links → per-source summaries. "
        "Returns full extracted content per source, not just snippets. "
        "Use for any question needing current, detailed, or multi-source information."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "query":   {"type": "string",  "description": "Primary search query"},
            "queries": {"type": "array",   "items": {"type": "string"},
                        "description": "Extra parallel queries for wider coverage (optional)"},
            "max":     {"type": "integer", "description": "Max DDG results 1-12 (default 6)"},
            "depth":   {"type": "integer", "description": "IDS crawl depth 0-3 (default 1). "
                                           "0=fetch results only, 1=+1 link layer, 2=+2 layers"},
            "region":  {"type": "string",  "description": "DDG region e.g. 'uk-en', 'us-en' (default 'wt-wt')"},
        },
        "required": ["query"],
    }

    def execute(self, inp: dict) -> ToolResult:
        if not _DDG_OK:
            return ToolResult(
                "Web search unavailable. Install: pip install ddgs\n"
                "(Package was renamed from 'duckduckgo-search' to 'ddgs'.)",
                is_error=True,
            )

        query  = str(inp.get("query", "")).strip()
        max_r  = max(1, min(int(inp.get("max",  6)), 12))
        depth  = max(0, min(int(inp.get("depth", 1)),  3))
        region = str(inp.get("region", "wt-wt"))
        extras = [str(q).strip() for q in (inp.get("queries") or []) if str(q).strip()]

        if not query:
            return ToolResult("No query provided.", is_error=True)

        t0 = time.perf_counter()

        # 1. DDG — fire all queries in parallel
        all_queries = [query] + extras
        ddg_results: list[dict] = []
        seen_urls: set[str] = set()
        with ThreadPoolExecutor(max_workers=len(all_queries)) as pool:
            futs = [pool.submit(_ddg_search, q, max_r) for q in all_queries]
            try:
                for fut in as_completed(futs, timeout=20):
                    try:
                        for r in fut.result():
                            u = r.get("href", "")
                            if u and u not in seen_urls:
                                seen_urls.add(u)
                                ddg_results.append(r)
                    except Exception:
                        pass
            except _FutureTimeout:
                # Collect whatever DDG results finished in time
                for fut in futs:
                    if fut.done() and not fut.cancelled():
                        try:
                            for r in fut.result(timeout=0):
                                u = r.get("href", "")
                                if u and u not in seen_urls:
                                    seen_urls.add(u)
                                    ddg_results.append(r)
                        except Exception:
                            pass

        if not ddg_results:
            return ToolResult(f"No DDG results for '{query}'.")

        qtok = _tok(query)

        # 2. BFS: fetch all result URLs in parallel
        seed_urls = [r["href"] for r in ddg_results if r.get("href")]
        seed_pages = _bfs_fetch(seed_urls, qtok)
        for p in seed_pages:
            p.setdefault("depth", 0)
        visited = {p["url"] for p in seed_pages} | set(seed_urls)

        # 3. IDS: crawl outbound links from seed pages
        deep_pages: list[dict] = []
        if depth > 0 and seed_pages:
            # max_per_depth shrinks as depth grows to keep total fetches bounded
            mpd = max(2, 8 // depth)
            deep_pages = _ids_crawl(seed_pages, qtok, depth, mpd, visited)

        all_fetched = seed_pages + deep_pages
        elapsed = time.perf_counter() - t0

        return ToolResult(_format(query, ddg_results, all_fetched, elapsed))