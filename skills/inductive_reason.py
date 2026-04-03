"""
Skill: inductive_reason — Find rules, laws, and patterns from examples.
Handles: number sequences, symbolic patterns, analogical rule extraction,
         scientific generalisation, and classification rules.

Imported from EVE (Embedded Virtual Entity) — pure-Python, no LLM required.
"""
from __future__ import annotations
import math
import re

DESCRIPTION = (
    "Inductive reasoning: observe examples, extract the underlying rule, "
    "validate it, and predict the next item. Best for number sequences, "
    "pattern series, classification from examples, and scientific laws."
)


def _analyse_sequence(nums: list, question: str) -> str:
    lines = [f"**Sequence:** {', '.join(str(n) for n in nums)}\n"]

    # Try rules in order of simplicity
    result = None
    rule_name = ""

    r = _try_arithmetic(nums)
    if r:
        result, rule_name = r, "Arithmetic (constant difference)"
    if not result:
        r = _try_geometric(nums)
        if r:
            result, rule_name = r, "Geometric (constant ratio)"
    if not result:
        r = _try_quadratic(nums)
        if r:
            result, rule_name = r, "Quadratic (2nd differences constant)"
    if not result:
        r = _try_fibonacci_like(nums)
        if r:
            result, rule_name = r, "Fibonacci-like (each term = sum of prior two)"
    if not result:
        r = _try_alternating(nums)
        if r:
            result, rule_name = r, "Alternating (two interleaved sequences)"
    if not result:
        r = _try_power(nums)
        if r:
            result, rule_name = r, "Power sequence (n^k or similar)"
    if not result:
        r = _try_factorial(nums)
        if r:
            result, rule_name = r, "Factorial-related"
    if not result:
        r = _try_prime(nums)
        if r:
            result, rule_name = r, "Prime numbers"
    if not result:
        # Fallback: differences of differences
        result = _describe_differences(nums)
        rule_name = "Higher-order difference analysis"

    lines.append(f"**Rule detected:** {rule_name}")
    lines.append(f"**Explanation:** {result['explanation']}")
    lines.append(f"**Next term(s):** {', '.join(str(n) for n in result['next'])}")
    lines.append(f"**Confidence:** {result['confidence']}")

    if question:
        lines.append(f"\n**Question:** {question}")
        lines.append(f"**Answer:** {_answer_question(question, nums, result)}")

    lines.append("\n**Inductive Method:**")
    lines.append("  1. Observed examples → detected recurring regularity")
    lines.append(f"  2. Proposed rule: {rule_name}")
    lines.append("  3. Validated rule holds for all provided terms")
    lines.append("  4. Applied rule forward to predict next term(s)")
    lines.append("\n*Note: inductive conclusions are probable, not logically certain.*")

    return "\n".join(lines)


def _try_arithmetic(nums: list) -> dict | None:
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    if len(set(round(d, 9) for d in diffs)) == 1:
        d = diffs[0]
        next_terms = [nums[-1] + d * (i+1) for i in range(3)]
        return {
            "explanation": f"Each term increases by {d}. Formula: a(n) = {nums[0]} + {d}*(n-1)",
            "next": next_terms,
            "confidence": "HIGH"
        }
    return None


def _try_geometric(nums: list) -> dict | None:
    if any(n == 0 for n in nums):
        return None
    ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1)]
    if len(set(round(r, 9) for r in ratios)) == 1:
        r = ratios[0]
        next_terms = [nums[-1] * r**(i+1) for i in range(3)]
        next_terms = [int(n) if n == int(n) else round(n, 4) for n in next_terms]
        return {
            "explanation": f"Each term is multiplied by {round(r, 6)}. Formula: a(n) = {nums[0]} * {round(r,6)}^(n-1)",
            "next": next_terms,
            "confidence": "HIGH"
        }
    return None


def _try_quadratic(nums: list) -> dict | None:
    if len(nums) < 4:
        return None
    diffs1 = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    diffs2 = [diffs1[i+1] - diffs1[i] for i in range(len(diffs1)-1)]
    if len(set(round(d, 9) for d in diffs2)) == 1:
        d2 = diffs2[0]
        next_d1 = diffs1[-1] + d2
        next_terms = [nums[-1] + next_d1 + d2*i for i in range(3)]
        # Recompute properly
        nxt = [nums[-1]]
        nd = diffs1[-1]
        for _ in range(3):
            nd += d2
            nxt.append(nxt[-1] + nd)
        return {
            "explanation": f"2nd differences are constant ({d2}). Quadratic growth.",
            "next": nxt[1:4],
            "confidence": "HIGH"
        }
    return None


def _try_fibonacci_like(nums: list) -> dict | None:
    if len(nums) < 4:
        return None
    for i in range(2, len(nums)):
        if round(nums[i], 9) != round(nums[i-1] + nums[i-2], 9):
            return None
    next_terms = [
        nums[-1] + nums[-2],
        nums[-1] + nums[-2] + nums[-1],
    ]
    n3 = nums[-1] + nums[-2]
    n4 = n3 + nums[-1]
    n5 = n4 + n3
    return {
        "explanation": f"Each term = sum of previous two. Starts: {nums[0]}, {nums[1]}",
        "next": [n3, n4, n5],
        "confidence": "HIGH"
    }


def _try_alternating(nums: list) -> dict | None:
    if len(nums) < 5:
        return None
    evens = nums[0::2]
    odds  = nums[1::2]
    r_e = _try_arithmetic(evens) or _try_geometric(evens)
    r_o = _try_arithmetic(odds)  or _try_geometric(odds)
    if r_e and r_o:
        # Rebuild interleaved
        next_e = r_e["next"][0]
        next_o = r_o["next"][0]
        next_seq = [next_o, next_e] if len(nums) % 2 == 0 else [next_e, next_o]
        return {
            "explanation": f"Two interleaved sequences: evens follow {r_e['explanation'][:40]}; odds follow {r_o['explanation'][:40]}",
            "next": next_seq[:3],
            "confidence": "MEDIUM"
        }
    return None


def _try_power(nums: list) -> dict | None:
    if any(n <= 0 for n in nums):
        return None
    # Check if nums[i] = (i+1)^k for some k
    for k in (2, 3, 4, 0.5):
        predicted = [(i+1)**k for i in range(len(nums))]
        if all(round(predicted[i], 6) == round(nums[i], 6) for i in range(len(nums))):
            next_terms = [(len(nums)+i+1)**k for i in range(3)]
            return {
                "explanation": f"a(n) = n^{k}",
                "next": [int(n) if n == int(n) else round(n,4) for n in next_terms],
                "confidence": "HIGH"
            }
    return None


def _try_factorial(nums: list) -> dict | None:
    facts = [math.factorial(i) for i in range(len(nums)+4)]
    if all(nums[i] in facts for i in range(len(nums))):
        # Find offset
        for offset in range(8):
            if all(math.factorial(i + offset) == nums[i] for i in range(len(nums))):
                next_terms = [math.factorial(len(nums) + offset + i) for i in range(3)]
                return {
                    "explanation": f"a(n) = (n+{offset-1})! — factorial sequence",
                    "next": next_terms,
                    "confidence": "HIGH"
                }
    return None


def _try_prime(nums: list) -> dict | None:
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5)+1):
            if n % i == 0: return False
        return True

    if not all(is_prime(int(n)) for n in nums):
        return None

    primes = []
    candidate = 2
    while len(primes) < len(nums) + 4:
        if is_prime(candidate):
            primes.append(candidate)
        candidate += 1

    if primes[:len(nums)] == [int(n) for n in nums]:
        return {
            "explanation": "Sequence of prime numbers.",
            "next": primes[len(nums):len(nums)+3],
            "confidence": "HIGH"
        }
    return None


def _describe_differences(nums: list) -> dict:
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    diffs2 = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)] if len(diffs) > 1 else []
    explanation = f"1st differences: {diffs}"
    if diffs2:
        explanation += f"; 2nd differences: {diffs2}"
    # Extrapolate using last 1st difference
    d = diffs[-1] if diffs else 0
    next_terms = [nums[-1] + d*(i+1) for i in range(3)]
    return {
        "explanation": explanation + " — no simple closed form found; extrapolating from last difference.",
        "next": next_terms,
        "confidence": "LOW"
    }


# ── Rule from (input, output) pairs ───────────────────────────────────────────

def _rule_from_pairs(pairs: list, question: str) -> str:
    lines = ["**Input/Output pairs:**"]
    for inp, out in pairs:
        lines.append(f"  {inp} → {out}")
    lines.append("")

    # Try linear: out = a*inp + b
    xs = [float(p[0]) for p in pairs]
    ys = [float(p[1]) for p in pairs]
    rule = _fit_linear(xs, ys)

    lines.append(f"**Induced rule:** {rule['formula']}")
    lines.append(f"**Confidence:** {rule['confidence']}")

    if question:
        lines.append(f"\n**Question:** {question}")
        # If question contains a number, apply rule
        q_nums = _parse_numbers(question)
        if q_nums:
            pred = rule["fn"](q_nums[-1])
            lines.append(f"**Predicted output for {q_nums[-1]}:** {round(pred, 4)}")

    return "\n".join(lines)


def _fit_linear(xs, ys) -> dict:
    n = len(xs)
    if n < 2:
        return {"formula": "insufficient data", "confidence": "NONE", "fn": lambda x: x}
    sx = sum(xs); sy = sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x,y in zip(xs,ys))
    denom = n*sxx - sx*sx
    if abs(denom) < 1e-12:
        # Constant
        c = sy / n
        return {"formula": f"output = {round(c,4)} (constant)", "confidence": "HIGH", "fn": lambda x: c}
    a = (n*sxy - sx*sy) / denom
    b = (sy - a*sx) / n
    # Validate
    residuals = [abs(ys[i] - (a*xs[i]+b)) for i in range(n)]
    max_err = max(residuals)
    conf = "HIGH" if max_err < 0.01 else "MEDIUM" if max_err < 1 else "LOW"
    a_s = round(a, 4); b_s = round(b, 4)
    formula = f"output = {a_s} × input + {b_s}" if b_s != 0 else f"output = {a_s} × input"
    return {"formula": formula, "confidence": conf, "fn": lambda x: a*x + b}


# ── Textual induction ──────────────────────────────────────────────────────────

def _textual_induction(text: str, question: str) -> str:
    lines = [f"**Examples provided:**\n{text}\n"]
    lines.append("**Inductive Analysis:**")
    lines.append("")
    lines.append("**Step 1 — Identify commonalities:**")
    lines.append("  Look for shared attributes, structure, or category across all examples.")
    lines.append("")
    lines.append("**Step 2 — Find the differentiators:**")
    lines.append("  What varies between examples? This reveals the variable of interest.")
    lines.append("")
    lines.append("**Step 3 — Propose a general rule:**")
    lines.append("  Generalise beyond the examples to a minimal covering rule.")
    lines.append("")
    lines.append("**Step 4 — Test for counter-examples:**")
    lines.append("  Could any case violate the rule? If yes, narrow it.")
    lines.append("")
    if question:
        lines.append(f"**Question:** {question}")
        lines.append("**Approach:** Apply the induced rule to the question. "
                     "If uncertain, flag which assumption might fail.")
    lines.append("")
    lines.append("*Inductive reasoning produces strong generalisations, not certainties.*")
    return "\n".join(lines)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_numbers(text: str) -> list:
    tokens = re.findall(r"-?\d+(?:\.\d+)?", str(text))
    return [float(t) if '.' in t else int(t) for t in tokens]


def _answer_question(question: str, nums: list, result: dict) -> str:
    low = question.lower()
    if any(k in low for k in ("next", "what comes after", "predict", "following")):
        return f"{result['next'][0]}"
    if "5th" in low or "fifth" in low:
        needed = 5 - len(nums)
        if needed <= 0: return str(nums[4])
        nxt = list(result["next"])
        all_nums = nums + nxt
        return str(all_nums[4]) if len(all_nums) > 4 else str(nxt[0])
    if "rule" in low or "pattern" in low or "formula" in low:
        return result["explanation"]
    return result["explanation"]

from skills.base_skill import BaseSkill


class InductiveReasonSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "inductive_reason"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth":   {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "examples": {"type": ["string", "array"]},
                "sequence": {"type": ["string", "array"]},
                "question": {"type": "string"},
                "query":    {"type": "string"},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        # problem holds the main sequence / examples input
        examples = kwargs.get("examples") or kwargs.get("sequence") or problem
        question = kwargs.get("question") or kwargs.get("query") or ""

        if isinstance(examples, str):
            nums = _parse_numbers(examples)
            if len(nums) >= 3:
                return _analyse_sequence(nums, question)
            return _textual_induction(examples, question)

        if isinstance(examples, list):
            if examples and isinstance(examples[0], (int, float)):
                return _analyse_sequence(examples, question)
            if examples and isinstance(examples[0], (list, tuple)) and len(examples[0]) == 2:
                return _rule_from_pairs(examples, question)
            return _textual_induction("\n".join(str(e) for e in examples), question)

        return _textual_induction(str(examples), question)
