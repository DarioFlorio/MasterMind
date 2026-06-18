"""
agent/cfe_compressor.py — Context Feature Engineer (CFE) for EVE.

Ports the CFE engine from CFE/CLAW v12.1 into EVE's architecture.

WHAT IT DOES
────────────
Raw conversation text → typed, salience-scored feature atoms:

    FACT|0.85|EVE is running on Gemma-4 at Q8 quantisation
    GOAL|0.90|User wants a full Tetris game with 10 levels
    CODE|0.75|def fibonacci(n): return n if n<=1 else fibonacci(n-1)+fibonacci(n-2)
    ENTITY|0.70|stress_test.py in agent_workspace
    CONSTRAINT|0.60|Must use pygame; no external APIs
    OPEN|0.50|Whether level speed progression is linear or exponential

These atoms are:
  • Deduplicated — superseded GOALs/ENTITYs are evicted when overwritten
  • Salience-ranked — lower salience evicted first when context budget fills
  • Rendered into a compact prompt section that replaces raw history excerpts

WHY THIS MATTERS
────────────────
EVE already has ContextBudget (tool schema trimming) and Session (message
window sliding).  But neither compresses the *content* of what was said —
they just drop old messages or trim descriptions.

CFE compresses semantics: a 2000-token conversation might yield 6 feature
atoms totalling 80 tokens, injected into the system prompt section where they
carry more weight than buried history.

INTEGRATION (query_engine.py)
──────────────────────────────
    from agent.cfe_compressor import ContextFeatureEngineer

    # In QueryEngine.__init__:
    self._cfe = ContextFeatureEngineer(token_budget=700)

    # In _run_loop, after session.add_user(user_text):
    self._cfe.ingest("user", user_text)

    # In _run_loop, after getting tool result:
    self._cfe.ingest("tool", result_text)

    # In _get_system_prompt, append to prompt string:
    cfe_block = self._cfe.render(query=current_task)
    if cfe_block:
        prompt += f"\\n\\n{cfe_block}"
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict

import logging
log = logging.getLogger("agent.cfe_compressor")


# ── Feature types ─────────────────────────────────────────────────────────────

class FType(str, Enum):
    FACT       = "FACT"
    GOAL       = "GOAL"
    CONSTRAINT = "CONSTRAINT"
    ENTITY     = "ENTITY"
    OPEN       = "OPEN"
    CODE       = "CODE"


@dataclass
class Feature:
    ftype:      FType
    content:    str
    salience:   float = 1.0
    turn:       int   = 0
    superseded: bool  = False
    ts:         float = field(default_factory=time.time)


# ── LLM extraction prompt ─────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a context feature extractor for an AI agent.
Output AT MOST 6 features, one per line, in EXACTLY this format:
  TYPE|SALIENCE|content

Rules (violating any rule → discard the line):
  TYPE     : one of FACT, GOAL, CONSTRAINT, ENTITY, OPEN, CODE
  SALIENCE : float 0.1–1.0 (how important is this for the active task?)
  content  : SHORT, specific, unique — never the word "content" literally
             never repeat a prior line
             never describe the format itself
  Stop after 6 features.
  If nothing meaningful to extract, output only: NONE
""".strip()

# Signals that the LLM hallucinated a code snippet rather than a feature atom
_HALLUCINATED_CODE_SIGNALS = [
    "import requests", "response.status_code", "BeautifulSoup",
    "requests.get(", "response.text", "html.parser",
    "urllib.request", "http.client",
    "# In practice", "# This is a simplified", "# For demonstration",
]

MAX_FEATURES_PER_CALL = 6


# ── Similarity dedup ──────────────────────────────────────────────────────────

def _word_overlap(a: str, b: str, threshold: float = 0.60) -> bool:
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) > threshold


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())[:60]


def _token_est(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


# ── Feature state ─────────────────────────────────────────────────────────────

class FeatureState:
    def __init__(self) -> None:
        self._features: List[Feature] = []
        self.turn:      int           = 0
        self.raw_tokens: int          = 0
        self.fe_tokens:  int          = 0

    def add(self, f: Feature) -> None:
        # Supersede old GOAL / ENTITY with the same key
        if f.ftype in (FType.GOAL, FType.ENTITY):
            k = _key(f.content)
            for existing in self._features:
                if existing.ftype == f.ftype and _key(existing.content) == k:
                    existing.superseded = True
        self._features.append(f)

    def active(self) -> List[Feature]:
        return [f for f in self._features if not f.superseded]

    def evict(self, budget: int) -> int:
        """Evict lowest-salience non-CODE features until under budget. Returns tokens freed."""
        evictable = sorted(
            [f for f in self._features
             if not f.superseded and f.ftype != FType.CODE],
            key=lambda x: x.salience,
        )
        freed = 0
        for f in evictable:
            if self.fe_tokens <= budget:
                break
            tokens = _token_est(f.content)
            f.superseded = True
            self.fe_tokens -= tokens
            freed += tokens
        return freed

    def render(self, query: str = "", top_k: int = 25) -> str:
        active = self.active()

        if query:
            qwords = set(query.lower().split())
            def _rank(f: Feature) -> float:
                fwords = set(f.content.lower().split())
                overlap = len(qwords & fwords) / max(len(qwords), 1)
                return overlap + f.salience * 0.3
            active = sorted(active, key=_rank, reverse=True)

        active = active[:top_k]

        sections: Dict[FType, List[str]] = {t: [] for t in FType}
        for f in active:
            sections[f.ftype].append(f.content)

        _LABELS = {
            FType.GOAL:       "## Active Goals",
            FType.CONSTRAINT: "## Constraints",
            FType.ENTITY:     "## Key Entities",
            FType.FACT:       "## Established Facts",
            FType.OPEN:       "## Open Questions",
            FType.CODE:       "## Code / Exact Values",
        }
        lines = []
        for ftype, label in _LABELS.items():
            if sections[ftype]:
                lines.append(label)
                for item in sections[ftype]:
                    lines.append(f"- {item}")
        return "\n".join(lines)


# ── Line parser ───────────────────────────────────────────────────────────────

def _parse_features(raw: str, turn: int) -> List[Feature]:
    results: List[Feature] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        if line.startswith(("#", "-", "*", "```")):
            continue
        parts = line.split("|", 2)
        if len(parts) < 3:
            continue
        ftype_raw, sal_raw, content = parts[0], parts[1], parts[2].strip()
        if not content or content.lower() in ("content", "...", "<content>"):
            continue
        if sal_raw.strip().lower() in ("salience", "0.x", "x"):
            continue
        if ftype_raw.strip().upper() == "CODE":
            if any(sig in content for sig in _HALLUCINATED_CODE_SIGNALS):
                continue
            if content.count("\n") > 5:
                continue
        try:
            ftype    = FType(ftype_raw.strip().upper())
            salience = max(0.0, min(1.0, float(sal_raw.strip())))
            if any(_word_overlap(content, r.content) for r in results):
                continue
            feat = Feature(ftype=ftype, content=content, salience=salience, turn=turn)
            results.append(feat)
            if len(results) >= MAX_FEATURES_PER_CALL:
                break
        except Exception:
            continue
    return results


# ── Main class ────────────────────────────────────────────────────────────────

class ContextFeatureEngineer:
    """
    Drop-in context compressor for EVE.

    Usage::
        cfe = ContextFeatureEngineer(token_budget=700)
        cfe.ingest("user", user_message_text)
        cfe.ingest("tool", tool_result_text)
        prompt += "\\n\\n" + cfe.render(query=current_task)
    """

    # System prompt section header
    SECTION_HEADER = "## Compressed Context (CFE)"

    def __init__(
        self,
        token_budget:  int = 700,
        llm_call_fn   = None,        # callable(prompt, system, temperature, max_tokens) → str
        top_k:         int = 20,
    ) -> None:
        self._budget  = token_budget
        self._llm     = llm_call_fn   # injected after construction if needed
        self._top_k   = top_k
        self._state   = FeatureState()
        self._cot_cache: List[str] = []
        self._lock    = threading.RLock()

    def set_llm(self, fn) -> None:
        """Inject the LLM callable after construction (avoids circular imports)."""
        self._llm = fn

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest(self, role: str, text: str) -> List[Feature]:
        """
        Extract features from `text` and add them to the feature state.
        If no LLM callable is set, falls back to a lightweight regex extractor.
        Returns the list of features added this call.
        """
        if not text or len(text.split()) < 4:
            return []
        with self._lock:
            self._state.turn += 1
            turn = self._state.turn
            self._state.raw_tokens += _token_est(text)

        features = self._extract(text, turn)

        with self._lock:
            for f in features:
                self._state.add(f)
                self._state.fe_tokens += _token_est(f.content)
            # Evict if over budget
            if self._state.fe_tokens > self._budget:
                freed = self._state.evict(self._budget)
                if freed:
                    log.debug("CFE evicted %d tokens (budget=%d)", freed, self._budget)

        log.debug(
            "CFE turn=%d  raw=%d tok  new_features=%d  total_active=%d",
            turn, _token_est(text), len(features), len(self._state.active())
        )
        return features

    def add_thought(self, thought: str) -> None:
        """Record a chain-of-thought string as a CODE feature."""
        with self._lock:
            self._cot_cache.append(thought)
            self._cot_cache = self._cot_cache[-5:]
            f = Feature(ftype=FType.CODE, content=f"CoT: {thought}",
                        salience=0.9, turn=self._state.turn)
            self._state.add(f)
            self._state.fe_tokens += _token_est(f.content)

    def render(self, query: str = "") -> str:
        """
        Render the active feature state into a compact prompt section.
        Returns an empty string if there are no active features.
        """
        with self._lock:
            body = self._state.render(query=query, top_k=self._top_k)
            cot  = self._cot_cache[-3:]

        if not body and not cot:
            return ""

        parts = [self.SECTION_HEADER]
        if cot:
            parts.append("## Recent Reasoning Chain")
            for c in cot:
                parts.append(f"- {c}")
        if body:
            parts.append(body)
        return "\n".join(parts)

    def metrics(self) -> dict:
        with self._lock:
            raw  = self._state.raw_tokens
            comp = self._state.fe_tokens
        return {
            "raw_tokens":        raw,
            "compressed_tokens": comp,
            "compression_ratio": round(raw / comp, 2) if comp else 0,
            "active_features":   len(self._state.active()),
            "turn":              self._state.turn,
        }

    def dump(self) -> List[dict]:
        """Return all active features as dicts (for debugging / export)."""
        with self._lock:
            return [
                {"type": f.ftype.value, "salience": f.salience,
                 "content": f.content, "turn": f.turn}
                for f in self._state.active()
            ]

    # ── Internal ───────────────────────────────────────────────────────────────

    def _extract(self, text: str, turn: int) -> List[Feature]:
        if self._llm is not None:
            return self._llm_extract(text, turn)
        return self._regex_extract(text, turn)

    def _llm_extract(self, text: str, turn: int) -> List[Feature]:
        try:
            raw = self._llm(
                f"Extract features:\n\n{text}",
                system=_EXTRACT_SYSTEM,
                temperature=0.05,
                max_tokens=256,
            )
            return _parse_features(raw, turn)
        except Exception as exc:
            log.warning("CFE LLM extraction failed: %s", exc)
            return self._regex_extract(text, turn)

    def _regex_extract(self, text: str, turn: int) -> List[Feature]:
        """
        Fallback extractor that works without an LLM.
        Pulls out goals, code blocks, URLs, and key nouns heuristically.
        Not as good as LLM extraction but never fails.
        """
        features: List[Feature] = []

        # Goal: sentences with imperative verbs
        for m in re.finditer(
            r"(?:I want|please|could you|can you|I need|make|create|write|build|fix|help)\s+(.{10,120})",
            text, re.IGNORECASE
        ):
            content = m.group(1).strip().rstrip(".,!?")
            f = Feature(FType.GOAL, content, salience=0.8, turn=turn)
            if not any(_word_overlap(content, x.content) for x in features):
                features.append(f)

        # Code: inline code blocks
        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL):
            snippet = m.group(1).strip()[:200]
            if snippet:
                f = Feature(FType.CODE, snippet, salience=0.7, turn=turn)
                if not any(_word_overlap(snippet, x.content) for x in features):
                    features.append(f)

        # Entity: file paths, URLs, quoted names
        for pattern in [r"[\w./\\]+\.py\b", r"https?://\S+", r'"([^"]{4,60})"']:
            for m in re.finditer(pattern, text):
                content = m.group(0)
                f = Feature(FType.ENTITY, content, salience=0.6, turn=turn)
                if not any(_word_overlap(content, x.content) for x in features):
                    features.append(f)

        return features[:MAX_FEATURES_PER_CALL]
