"""
tools/web_fetch_tool.py
=======================
Fetch a single URL → clean text + extracted links.
Links are returned so the agent can hand them back to web_search (IDS crawl)
or fetch them individually for deep research.
"""
from __future__ import annotations
import re
from urllib.parse import urljoin, urlparse, urlunparse

from tools.base_tool import BaseTool, ToolResult

_MAX_CHARS   = 20_000
_SKIP_EXT = re.compile(
    r"\.(pdf|docx?|xlsx?|zip|gz|tar|exe|mp[34]|mkv|jpg|jpeg|png|gif|webp|svg|ico)$",
    re.I,
)


def _extract(url: str, html: str, prompt: str = "") -> str:
    """
    Parse HTML → clean text + top outbound links.
    When prompt is given, ranks paragraphs by term-overlap and shows best first.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        lines = [ln.strip() for ln in html.splitlines() if ln.strip()]
        return "\n".join(lines)[:_MAX_CHARS]

    soup = BeautifulSoup(html, "html.parser")

    try:
        for tag in soup(["script", "style", "noscript", "iframe",
                         "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        junk = re.compile(
            r"(cookie|banner|popup|ad[s_-]|sidebar|comment|social|newsletter|subscribe)",
            re.I,
        )
        for tag in soup.find_all(True):
            try:
                cls = " ".join(tag.get("class", []) or [])
                eid = tag.get("id", "") or ""
            except AttributeError:
                continue
            if junk.search(cls) or junk.search(eid):
                tag.decompose()

        # Title
        t = soup.find("title")
        title = t.get_text(strip=True) if t else url

        # Main content
        main = (
            soup.find("article") or soup.find("main")
            or soup.find(id=re.compile(r"(content|article|post|main)", re.I))
            or soup.find(class_=re.compile(r"(content|article|post|main)", re.I))
            or soup.find("body") or soup
        )
        raw = main.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in raw.splitlines()
                 if ln.strip() and len(ln.strip()) > 15]
        text = "\n".join(lines)

        # If prompt given, surface most relevant paragraphs first
        if prompt:
            qtok = set(re.findall(r"\b[a-z]{3,}\b", prompt.lower()))
            paras = [p for p in text.split("\n") if len(p) > 60]
            scored = sorted(
                paras,
                key=lambda p: sum(1 for t in qtok if t in p.lower()),
                reverse=True,
            )
            # Put top-scoring paragraphs at the top, rest follows
            top = scored[:6]
            rest = [p for p in paras if p not in set(top)]
            text = "\n".join(top) + "\n---\n" + "\n".join(rest)

        # Extract outbound links
        base_parsed = urlparse(url)
        base = f"{base_parsed.scheme}://{base_parsed.netloc}"
        links: list[str] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue
            abs_url = urljoin(base, href)
            p = urlparse(abs_url)
            if p.scheme not in ("http", "https"):
                continue
            clean = urlunparse(p._replace(fragment=""))
            if clean in seen or clean == url:
                continue
            seen.add(clean)
            anchor = a.get_text(strip=True)[:80]
            links.append(f"  {clean}" + (f"  [{anchor}]" if anchor else ""))

        header = f"[Fetched: {url}]\n[Title: {title}]"
        if prompt:
            header += f"\n[Query: {prompt}]"
        header += "\n"

        body = text[:_MAX_CHARS]

        link_section = ""
        if links:
            link_section = "\n\n[Outbound links — top 30]\n" + "\n".join(links[:30])

        return header + "\n" + body + link_section

    except Exception as parse_err:
        # Fallback: return raw text stripped of HTML tags
        raw = re.sub(r"<[^>]+>", " ", html)
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and len(ln.strip()) > 20]
        text = "\n".join(lines)[:_MAX_CHARS]
        return f"[Fetched: {url}]\n[Parse warning: {parse_err}]\n\n{text}"


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "Fetch a URL and return clean extracted text plus outbound links. "
        "Pass a 'prompt' to surface the most relevant paragraphs first. "
        "The returned links can be fed back to web_search or web_fetch for deeper research."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "url":    {"type": "string", "description": "URL to fetch"},
            "prompt": {"type": "string", "description": "Optional: surface content relevant to this"},
        },
        "required": ["url"],
    }

    def execute(self, inp: dict) -> ToolResult:
        url    = str(inp.get("url", "")).strip()
        prompt = str(inp.get("prompt", "")).strip()

        if not url:
            return ToolResult("No URL provided.", is_error=True)
        if not url.startswith(("http://", "https://")):
            return ToolResult("URL must start with http:// or https://", is_error=True)
        if _SKIP_EXT.search(url):
            return ToolResult(f"Skipping binary file: {url}", is_error=True)

        try:
            import httpx
            r = httpx.get(
                url, timeout=15, follow_redirects=True,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,*/*;q=0.9",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
        except Exception as e:
            return ToolResult(f"Fetch error: {e}", is_error=True)

        if "text/html" in ct:
            text = _extract(url, r.text, prompt)
        elif "application/json" in ct:
            text = f"[Fetched: {url}]\n\n{r.text[:_MAX_CHARS]}"
        else:
            # Plain text or unknown
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            text = f"[Fetched: {url}]\n\n" + "\n".join(lines)[:_MAX_CHARS]

        return ToolResult(text)