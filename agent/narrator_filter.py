# ════════════════════════════════════════════════════════════════════════════
# Narrator-bleed suppression — strips internal monologue from model output.
# Used both post-generation (_clean_output) and during streaming.
# ════════════════════════════════════════════════════════════════════════════
import re as _re

_NARRATOR_BLOCK_RE = _re.compile(
    r"""
    (?:^|\n)            # start of text or newline
    [ \t]*              # optional indent / leading space
    (?:
        Okay[,. ]|Alright[,. ]|
        Let me |
        I need to |I'll |I will |I'm going to |I should |I have to |I can see |
        So I |So the |
        Wait[,. ]|
        But (?:the |I |we |according)|
        Since |
        First[,. ]|
        Now[,. ]|
        The user |
        Looking at |
        Note that |
        Keep it |
        Actually[,. ]|
        Hmm[,. ]
    )
    [^\n]*              # rest of first sentence
    (?:\n(?!\n)[^\n]*)* # continuation lines until blank line
    """,
    _re.VERBOSE | _re.IGNORECASE,
)

# Patterns specific to Gemma 4 E2B tool-echoing behaviour
_TOOL_ECHO_RE = _re.compile(
    r"""
    (?:^|\n)
    [ \t]*
    (?:
        # Model narrating its own tool usage
        (?:To |In order to )?(?:execute|perform|run|conduct|do) (?:the |a )?(?:web |file |this )?(?:search|task|request|query|analysis)\b|
        (?:Based on the (?:given |provided |above )?(?:information|context|instructions?|tools?|results?))|
        (?:For (?:the )?(?:initial|this) (?:response|query|request|task))|
        (?:The (?:primary |main )?(?:skills?|tools?) (?:involved|needed|required|applicable)\b)|
        (?:Here(?:'s| is) (?:a |the )?step-by-step)|
        (?:To (?:solve|address|answer|handle|analyze|determine) this\b)|
        (?:\*\*(?:Answer|Step|Solution|Approach|Result)\*\*\s*:)|
        (?:Answer\s*:)|
        (?:\[THINKING mode:)|
        (?:remember to use actual results)|
        # Tool signature echoes
        (?:web_fetch\(url\*)|
        (?:web_search\(query\*)|
        (?:bash\(command\*)|
        (?:read_file\(path\*)
    )
    [^\n]*
    (?:\n(?!\n)[^\n]*)*
    """,
    _re.VERBOSE | _re.IGNORECASE,
)

# Hard truncation: if output starts with any of these, the whole thing is leaked narration
_HARD_TRUNCATE_RE = _re.compile(
    r"""^
    (?:
        Based\ on\ the\ (?:given|provided)|
        For\ the\ initial\ response|
        To\ execute\ the\ web\ search|
        To\ analyze\ the\ dataset|
        The\ (?:primary|main)\ skills?\ involved|
        \[THINKING\ mode:|
        remember\ to\ use\ actual\ results
    )
    """,
    _re.VERBOSE | _re.IGNORECASE,
)


def _strip_narrator(text: str) -> str:
    """Remove internal monologue paragraphs from a completed text block."""
    stripped = text.strip()

    # Hard truncation — entire output is leaked narration
    if _HARD_TRUNCATE_RE.match(stripped):
        return ""

    cleaned = _NARRATOR_BLOCK_RE.sub("", text)
    cleaned = _TOOL_ECHO_RE.sub("", cleaned)
    cleaned = _re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


class _StreamingNarratorFilter:
    """
    Buffer-based narrator filter for token-by-token streaming.
    Feed chunks in; get back only the clean, safe-to-display portion.
    Call flush() at end of turn to drain any remaining buffer.
    """
    _LOOKAHEAD = 120  # reduced from 300 — was causing short replies to be swallowed

    def __init__(self):
        self._buf = ""
        self._suppressing = False
        self._hard_killed = False  # set if we detected full-output narration

    def feed(self, chunk: str) -> str:
        if self._hard_killed:
            return ""
        self._buf += chunk

        # Hard kill: if first 120 chars match a full-narration opener, suppress everything
        if len(self._buf) >= 60 and not self._suppressing:
            if _HARD_TRUNCATE_RE.match(self._buf.lstrip()):
                self._hard_killed = True
                self._buf = ""
                return ""

        return self._flush_safe()

    def flush(self) -> str:
        if self._hard_killed:
            self._buf = ""
            self._hard_killed = False
            self._suppressing = False
            return ""
        result = _strip_narrator(self._buf)
        self._buf = ""
        self._suppressing = False
        return result

    def reset(self):
        self._buf = ""
        self._suppressing = False
        self._hard_killed = False

    def _flush_safe(self) -> str:
        if not self._buf:
            return ""

        stripped_start = self._buf.lstrip()
        if _NARRATOR_BLOCK_RE.match(stripped_start) or _TOOL_ECHO_RE.match(stripped_start) or self._suppressing:
            self._suppressing = True
            blank_idx = self._buf.find("\n\n")
            if blank_idx != -1:
                self._buf = self._buf[blank_idx + 2:]
                self._suppressing = False
                return self._flush_safe()
            self._buf = ""
            return ""

        if len(self._buf) < self._LOOKAHEAD:
            return ""

        safe_up_to = len(self._buf) - self._LOOKAHEAD
        cut = self._buf.rfind("\n\n", 0, safe_up_to)
        if cut == -1:
            for punct in (". ", "! ", "? "):
                c = self._buf.rfind(punct, 0, safe_up_to)
                if c != -1:
                    cut = c + len(punct)
                    break

        if cut <= 0:
            return ""

        out = self._buf[:cut]
        self._buf = self._buf[cut:]
        return out
