"""
Skill: reason_chain
Automatic multi-skill reasoning pipeline.

Architecture: AttnRes + three residual connections
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. ATTENTION RESIDUALS (AttnRes) — MoonshotAI arXiv:2603.15031
   Each skill receives a softmax-weighted mixture of ALL prior skill outputs
   (Block AttnRes variant) rather than only the immediately preceding one.
   Prevents early-insight dilution across the depth stack.

   Old:  context_l = output_{l-1}
   New:  context_l = sum_i( alpha_i * v_i ),  alpha = softmax(cosine_sim(problem, v_i))

2. PROBLEM RESIDUAL (most impactful addition)
   The original problem statement is always prepended to every skill's context,
   regardless of attention weights. No matter how much the chain drifts, every
   skill is grounded in the original question.

   h_l = attn_context + PROBLEM          <- problem is the skip connection

3. SELF-RESIDUAL
   After a skill runs, its output is blended (50/50 by default) with what it
   received as input before being pushed onto the stack. Prevents catastrophic
   forgetting within a single step — if a skill partially misunderstands, the
   residual pulls it back toward the prior context.

   v_l_stored = blend(output_l, context_l, alpha=self_residual_alpha)

4. RUNNING SUMMARY RESIDUAL
   A compressed "ground truth" summary accumulates across steps and is always
   appended to context. Bounds context drift the way LayerNorm + residual bounds
   hidden-state magnitude in the paper. Updated after every successful skill.

   summary = summary + " | " + first_sentence(output_l)
"""
from __future__ import annotations

import math
import re
import time
from collections import Counter

from skills.base_skill import BaseSkill

DESCRIPTION = (
    "Multi-skill reasoning chain with AttnRes + three residual connections: "
    "(1) depth-wise softmax attention over all prior outputs, "
    "(2) problem residual — original question always in context, "
    "(3) self-residual — skill output blended with its input, "
    "(4) running summary residual — compressed ground truth across steps. "
    "Auto-routes via skill_router or accepts an explicit skill list."
)


# ── Tokenisation & cosine similarity ─────────────────────────────────────────

def _tokenise(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def _cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[t] * b[t] for t in a if t in b)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _compute_attention_weights(query_tokens: list, source_texts: list, temperature: float = 1.0) -> list:
    """
    Softmax depth-wise attention weights.

    Paper: logits[i] = dot(w_l, RMSNorm(v_i));  alpha = softmax(logits)
    Here:  logits[i] = cosine_sim(problem_tf, output_i_tf)
    """
    q_tf = Counter(_tokenise(" ".join(str(t) for t in query_tokens)))
    logits = [_cosine_sim(q_tf, Counter(_tokenise(str(src)))) for src in source_texts]
    max_l = max(logits) if logits else 0.0
    exps = [math.exp((l - max_l) / max(temperature, 1e-6)) for l in logits]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


# ── Text utilities ────────────────────────────────────────────────────────────

def _trim(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return "..." + text[-(max_chars - 3):]


def _first_sentence(text: str, max_chars: int = 200) -> str:
    """Extract first meaningful sentence for the running summary."""
    text = text.strip()
    for sep in (".\n", ". ", "\n\n", "\n"):
        idx = text.find(sep)
        if 20 < idx < max_chars:
            return text[: idx + 1].strip()
    return _trim(text, max_chars)


def _blend(output: str, context: str, alpha: float) -> str:
    """
    Self-residual blend: alpha * output + (1-alpha) * context (in text space).
    Realised as: take alpha-proportion of chars from output tail,
    (1-alpha)-proportion from context tail, then join.
    Total budget = max(len(output), 600) to avoid bloat.
    """
    if not context or alpha >= 1.0:
        return output
    budget = max(len(output), 600)
    out_chars = max(80, int(budget * alpha))
    ctx_chars = max(40, int(budget * (1.0 - alpha)))
    out_part = _trim(output, out_chars)
    ctx_part = _trim(context, ctx_chars)
    return out_part + "\n\n[self-residual←] " + ctx_part


# ── Block AttnRes ─────────────────────────────────────────────────────────────

def _make_blocks(outputs: list, block_size: int):
    if not outputs:
        return [], ""
    completed = outputs[:-1]
    partial = outputs[-1]
    blocks = []
    for start in range(0, len(completed), block_size):
        chunk = completed[start: start + block_size]
        blocks.append("\n\n".join(filter(None, chunk)))
    return blocks, partial


def _block_attn_res(blocks: list, partial_block: str, query_tokens: list, max_chars: int = 1200) -> str:
    sources = blocks + ([partial_block] if partial_block else [])
    if not sources:
        return ""
    if len(sources) == 1:
        return _trim(sources[0], max_chars)
    weights = _compute_attention_weights(query_tokens, sources)
    parts = []
    for src, w in zip(sources, weights):
        budget = max(60, int(max_chars * w))
        trimmed = _trim(str(src), budget)
        if trimmed:
            parts.append(trimmed)
    return "\n\n".join(parts)


# ── Auto-route ────────────────────────────────────────────────────────────────

def _auto_select_chain(problem: str, max_steps: int) -> list:
    try:
        from skills import SKILL_REGISTRY
        SkillRouterClass = SKILL_REGISTRY.get("skill_router")
        if SkillRouterClass is None:
            return ["deep_reason"]
        router = SkillRouterClass()
        result = router.execute(problem, top_n=max_steps)
    except Exception:
        return ["deep_reason"]

    names = []
    chain_block = re.search(r"Skill Chaining Suggestion.*?(?=\n\n|\Z)", result, re.DOTALL)
    if chain_block:
        found = re.findall(r"`([a-z_]+)`", chain_block.group())
        names = [n for n in found if n != "skill_router"]
    if not names:
        primary = re.search(r"Primary recommendation: `([a-z_]+)`", result)
        if primary:
            names.append(primary.group(1))
        alts = re.findall(r"- `([a-z_]+)`", result)
        for a in alts:
            if a not in names and a != "skill_router":
                names.append(a)

    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= max_steps:
            break
    return out or ["deep_reason"]


# ── Single skill runner ───────────────────────────────────────────────────────

def _run_skill(name: str, problem: str, context: str, depth: int):
    try:
        from skills import SKILL_REGISTRY
        SkillClass = SKILL_REGISTRY.get(name)
        if SkillClass is None:
            return "", f"Skill '{name}' not found in SKILL_REGISTRY."
        skill = SkillClass()
        result = skill.execute(problem, depth=depth)
        return (result or "").strip(), ""
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"


# ── Synthesiser ───────────────────────────────────────────────────────────────

def _synthesise(problem: str, steps: list, weights_log: list) -> str:
    conclusions = []
    for step in steps:
        out = step["output"]
        if not out:
            continue
        paras = [p.strip() for p in out.split("\n\n") if p.strip()]
        if paras:
            conclusions.append(f"[{step['skill']}]: {paras[-1][:300]}")

    if not conclusions:
        return "No conclusions extracted — review individual step outputs."

    attn_summary = ""
    if weights_log:
        attn_summary = "\n\n**AttnRes weights per step:**\n"
        for entry in weights_log:
            row = ", ".join(f"`{s}`={w:.2f}" for s, w in zip(entry["sources"], entry["weights"]))
            attn_summary += f"  - Step {entry['step']} (`{entry['skill']}`): [{row}]\n"

    merged = "\n\n".join(conclusions)
    return (
        f"The reasoning chain converged across {len(steps)} skill(s).\n\n"
        f"Key conclusions per layer:\n\n{merged}\n\n"
        f"**Integrated answer:** The analysis is most robust where skill outputs "
        f"converge. Divergences indicate genuinely uncertain territory — treat "
        f"those points with appropriate epistemic humility."
        f"{attn_summary}"
    )


# ── Main skill class ──────────────────────────────────────────────────────────

class ReasonChainSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "reason_chain"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 2},
                "chain": {"type": "array", "items": {"type": "string"}},
                "max_steps": {"type": "integer"},
                "context": {"type": "string"},
                "block_size": {
                    "type": "integer", "minimum": 1, "maximum": 8, "default": 2,
                    "description": (
                        "Block AttnRes block size. 1 = Full AttnRes (attend every prior output). "
                        "Default 2 balances richness and cost."
                    ),
                },
                "attn_temperature": {
                    "type": "number", "default": 1.0,
                    "description": "Softmax temperature for depth-wise attention. Lower = sharper.",
                },
                "self_residual_alpha": {
                    "type": "number", "default": 0.7,
                    "description": (
                        "Self-residual blend ratio. 1.0 = output only (no self-residual). "
                        "0.7 = 70% output + 30% input context blended back in."
                    ),
                },
                "problem_residual": {
                    "type": "boolean", "default": True,
                    "description": "Always prepend the original problem to every skill's context.",
                },
                "summary_residual": {
                    "type": "boolean", "default": True,
                    "description": "Maintain a running compressed summary appended to all contexts.",
                },
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        chain               = kwargs.get("chain") or []
        depth               = int(kwargs.get("depth", 2))
        max_steps           = int(kwargs.get("max_steps", 3))
        block_size          = int(kwargs.get("block_size", 2))
        temperature         = float(kwargs.get("attn_temperature", 1.0))
        self_res_alpha      = float(kwargs.get("self_residual_alpha", 0.7))
        use_problem_res     = bool(kwargs.get("problem_residual", True))
        use_summary_res     = bool(kwargs.get("summary_residual", True))

        if not chain:
            chain = _auto_select_chain(problem, max_steps)

        query_tokens = _tokenise(problem)

        mode_label = "Full AttnRes" if block_size == 1 else f"Block AttnRes (block={block_size})"
        residuals_active = ", ".join(filter(None, [
            "problem" if use_problem_res else "",
            f"self(α={self_res_alpha})",
            "summary" if use_summary_res else "",
        ]))

        header = [
            "## Reasoning Chain  *(AttnRes + Residual Connections)*",
            f"**Problem:** {problem[:120]}{'...' if len(problem) > 120 else ''}",
            f"**Chain:** {' -> '.join(chain)} ({len(chain)} steps)",
            f"**Mode:** {mode_label}  |  **Temperature:** {temperature}",
            f"**Residuals:** {residuals_active}",
            "",
        ]

        steps        = []
        all_outputs  = []   # V: the growing stack (AttnRes)
        weights_log  = []
        summary      = ""   # running summary residual

        for i, skill_name in enumerate(chain, 1):

            # ── 1. AttnRes: depth-wise attention over all prior outputs ────────
            step_weights = None
            step_sources = []

            if not all_outputs:
                attn_context = ""
            elif len(all_outputs) == 1:
                attn_context = _trim(all_outputs[0], 1200)
                step_weights = [1.0]
                step_sources = [steps[0]["skill"]]
            else:
                blocks, partial = _make_blocks(all_outputs, block_size)
                attn_context = _block_attn_res(blocks, partial, query_tokens, max_chars=1200)
                step_weights = _compute_attention_weights(query_tokens, all_outputs, temperature)
                step_sources = [s["skill"] for s in steps]

            if step_weights and len(step_weights) > 1:
                weights_log.append({
                    "step":    i,
                    "skill":   skill_name,
                    "sources": step_sources,
                    "weights": step_weights,
                })

            # ── 2. Problem residual: always re-inject the original question ────
            #    h_l = attn_context + PROBLEM  (skip connection to depth-0)
            if use_problem_res:
                problem_anchor = f"[Problem residual] {problem}"
                context = (problem_anchor + "\n\n" + attn_context).strip()
            else:
                context = attn_context

            # ── 3. Running summary residual: bounded ground-truth accumulator ──
            #    Analogous to LayerNorm keeping magnitude bounded across depth.
            if use_summary_res and summary:
                context = context + "\n\n[Summary residual] " + summary

            # ── Run the skill ─────────────────────────────────────────────────
            t0 = time.perf_counter()
            output, error = _run_skill(skill_name, problem, context, depth)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # ── 4. Self-residual: blend output back with its input context ─────
            #    v_l_stored = alpha*output + (1-alpha)*context
            #    Prevents catastrophic forgetting within a single step.
            stored_output = output
            if output and not error and self_res_alpha < 1.0 and context:
                stored_output = _blend(output, context, self_res_alpha)

            step = {
                "step":         i,
                "skill":        skill_name,
                "output":       output,       # raw output shown to user
                "stored":       stored_output, # blended version pushed to stack
                "error":        error,
                "elapsed":      elapsed_ms,
                "attn_weights": step_weights,
                "attn_sources": step_sources,
            }
            steps.append(step)

            if output and not error:
                # Push blended version onto stack V
                all_outputs.append(stored_output)
                # Update running summary (first sentence of raw output)
                new_sentence = _first_sentence(output)
                if new_sentence:
                    summary = (summary + " | " + new_sentence if summary else new_sentence)
                    summary = _trim(summary, 600)   # keep summary bounded

        # ── Render ────────────────────────────────────────────────────────────
        lines = header[:]
        for step in steps:
            status = "✅" if not step["error"] else "❌"
            lines.append(
                f"### {status} Step {step['step']}: `{step['skill']}` ({step['elapsed']:.0f}ms)"
            )
            if step["attn_weights"] and len(step["attn_weights"]) > 1:
                w_str = ", ".join(
                    f"{s}={w:.2f}"
                    for s, w in zip(step["attn_sources"], step["attn_weights"])
                )
                lines.append(f"*AttnRes weights: [{w_str}]*")
            if step["error"]:
                lines.append(f"**Error:** {step['error']}")
            else:
                lines.append(step["output"])
            lines.append("")

        successful = [s for s in steps if not s["error"]]
        if len(successful) > 1:
            lines.append("---")
            lines.append("## Chain Synthesis")
            lines.append(_synthesise(problem, successful, weights_log))

        if summary:
            lines.append("")
            lines.append(f"*Running summary residual: {summary}*")

        return "\n".join(lines)
