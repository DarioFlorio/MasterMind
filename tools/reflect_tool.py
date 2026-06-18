"""
tools/reflect_tool.py — MasterMind-style self-critique and reflection.

Lets the model step back and critique its own draft output before presenting
it to the user. Zero model calls — returns a structured critique template
that the model fills in, then uses to revise.

The model calls this tool with its draft answer, gets back a structured
critique scaffold, then produces a revised answer addressing each point.
"""
from __future__ import annotations

from tools.base_tool import BaseTool, ToolResult


_CRITIQUE_TEMPLATE = """## Self-critique of draft answer

**Draft:** {draft_preview}

Please evaluate your draft on each dimension:

### 1. Correctness (are the facts right?)
→ 

### 2. Completeness (did I answer the full question?)
→ 

### 3. Clarity (is this easy to understand?)
→ 

### 4. Conciseness (any unnecessary padding?)
→ 

### 5. Actionability (can the user actually use this?)
→ 

### 6. What I would change
→ 

Now write a REVISED answer that addresses the weaknesses you identified above.
"""

_DEEP_TEMPLATE = """## Deep reflection — {aspect}

**Context:** {context}

Reflect carefully on the following questions:
1. What assumptions am I making that might be wrong?
2. What am I missing or underweighting?
3. What would a critic say about this analysis?
4. What would strengthen the argument or solution?
5. What is the single most important thing to change?

Provide your reflection, then a revised answer.
"""

_CODE_TEMPLATE = """## Code review of draft

**Code:**
{draft_preview}

Review on:
1. **Correctness** — does it do what was asked?
2. **Edge cases** — what inputs could break it?
3. **Efficiency** — obvious performance issues?
4. **Readability** — is naming and structure clear?
5. **Security** — any obvious vulnerabilities?

List issues found, then provide the corrected version.
"""


class ReflectTool(BaseTool):
    name = "reflect"
    description = (
        "Trigger a structured self-critique of a draft answer or plan. "
        "Call with your draft before presenting it to the user to catch "
        "errors, gaps, and unclear reasoning. The tool returns a critique "
        "scaffold — you fill it in and write a revised answer. "
        "modes: general (default), code (for code review), deep (for analysis)."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "draft": {
                "type": "string",
                "description": "Your draft answer, plan, or code to critique.",
            },
            "mode": {
                "type": "string",
                "enum": ["general", "code", "deep"],
                "description": "Critique style: general (default), code, deep.",
            },
            "aspect": {
                "type": "string",
                "description": "For mode=deep: the specific aspect to reflect on.",
            },
            "context": {
                "type": "string",
                "description": "For mode=deep: additional context about the problem.",
            },
        },
        "required": ["draft"],
    }

    def execute(self, inp: dict) -> ToolResult:
        draft   = (inp.get("draft") or "").strip()
        mode    = (inp.get("mode") or "general").lower()
        aspect  = inp.get("aspect") or "the approach"
        context = inp.get("context") or ""

        if not draft:
            return ToolResult("'draft' is required — provide the text to critique.", is_error=True)

        preview = draft[:500] + ("…" if len(draft) > 500 else "")

        if mode == "code":
            text = _CODE_TEMPLATE.format(draft_preview=preview)
        elif mode == "deep":
            text = _DEEP_TEMPLATE.format(
                aspect=aspect,
                context=context or draft[:200],
            )
        else:
            text = _CRITIQUE_TEMPLATE.format(draft_preview=preview)

        return ToolResult(text)
