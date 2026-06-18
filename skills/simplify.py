"""
skills/simplify.py — Simplify: Code Review and Cleanup

Ports the src simplify skill. Reviews all changed files for reuse,
quality, and efficiency using three parallel review agents, then fixes
all found issues.
"""
from __future__ import annotations
from skills.base_skill import BaseSkill


SIMPLIFY_PROMPT = """# Simplify: Code Review and Cleanup

Review all changed files for reuse, quality, and efficiency. Fix any issues found.

## Phase 1: Identify Changes

Run `git diff` (or `git diff HEAD` if there are staged changes) to see what changed. If there are no git changes, review the most recently modified files that the user mentioned or that you edited earlier in this conversation.

## Phase 2: Launch Three Review Agents in Parallel

Use the agent tool to launch all three agents concurrently in a single message. Pass each agent the full diff so it has complete context.

### Agent 1: Code Reuse Review

For each change:
1. **Search for existing utilities and helpers** that could replace newly written code. Look for similar patterns elsewhere in the codebase — common locations are utility directories, shared modules, and files adjacent to the changed ones.
2. **Flag any new function that duplicates existing functionality.** Suggest the existing function to use instead.
3. **Flag any inline logic that could use an existing utility** — hand-rolled string manipulation, manual path handling, custom environment checks, ad-hoc type guards, and similar patterns are common candidates.

### Agent 2: Code Quality Review

Review the same changes for hacky patterns:
1. **Redundant state**: state that duplicates existing state, cached values that could be derived
2. **Parameter sprawl**: adding new parameters instead of generalising existing ones
3. **Copy-paste with slight variation**: near-duplicate code blocks that should be unified
4. **Leaky abstractions**: exposing internal details, breaking existing abstraction boundaries
5. **Stringly-typed code**: using raw strings where constants or enums already exist
6. **Unnecessary nesting**: wrapper elements that add no layout value
7. **Unnecessary comments**: comments explaining WHAT the code does — delete; keep only non-obvious WHY

### Agent 3: Efficiency Review

Review the same changes for efficiency:
1. **Unnecessary work**: redundant computations, repeated file reads, duplicate API calls, N+1 patterns
2. **Missed concurrency**: independent operations run sequentially when they could run in parallel
3. **Hot-path bloat**: new blocking work added to startup or per-request hot paths
4. **Recurring no-op updates**: state updates that fire unconditionally — add a change-detection guard
5. **Unnecessary existence checks**: pre-checking file/resource existence before operating (TOCTOU anti-pattern)
6. **Memory**: unbounded data structures, missing cleanup, event listener leaks
7. **Overly broad operations**: reading entire files when only a portion is needed

## Phase 3: Fix Issues

Wait for all three agents to complete. Aggregate their findings and fix each issue directly. If a finding is a false positive or not worth addressing, note it and move on.

When done, briefly summarise what was fixed (or confirm the code was already clean).
"""


class SimplifySkill(BaseSkill):
    """Review changed code for reuse, quality, and efficiency, then fix issues found."""

    @property
    def name(self) -> str:
        return "simplify"

    @property
    def description(self) -> str:
        return (
            "Review all changed files for reuse, quality, and efficiency using three "
            "parallel review agents (reuse, quality, efficiency), then fix any issues "
            "found. Run after implementing a feature or fix."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        prompt = SIMPLIFY_PROMPT
        additional_focus = problem.strip()
        if additional_focus and additional_focus.lower() not in ("run", "simplify", "review"):
            prompt += f"\n\n## Additional Focus\n\n{additional_focus}"
        return prompt
