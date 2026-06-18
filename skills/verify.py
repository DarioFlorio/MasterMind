"""
skills/verify.py — Verify: End-to-end change verification

Ports the src verify skill. Verifies a code change does what it should
by running the app, executing tests, and checking the result end-to-end.
"""
from __future__ import annotations
from skills.base_skill import BaseSkill


VERIFY_PROMPT = """# Verify: End-to-End Change Verification

Verify that the recent code change does what it should by running the app and checking the result.

## Phase 1: Understand the Change

Review recent git changes with `git diff HEAD~1` (or as specified by the user). Identify:
- What was changed and why
- What behaviour should result
- What the acceptance criteria are

## Phase 2: Run the Test Suite

1. Find and run the project's test suite:
   - Check for `package.json` scripts: `npm test`, `bun test`
   - Check for `pytest`, `go test ./...`, `cargo test`, `make test`
   - Run all tests and note any failures

2. If tests fail, triage:
   - Are they pre-existing failures unrelated to this change?
   - Are they new failures caused by this change?
   - Fix any new failures introduced by this change.

**Success criteria**: All tests pass (or failures are documented as pre-existing).

## Phase 3: End-to-End Verification

Based on the type of change, choose the appropriate e2e approach:

**For CLI/backend changes:**
- Start the application if needed
- Exercise the changed behaviour directly via command line or API
- Verify the output matches expected results

**For library/module changes:**
- Write a small inline test script that exercises the changed interface
- Run it and verify correctness

**For data/config changes:**
- Load the changed config and verify it parses and applies correctly

**For refactors:**
- Verify existing behaviour is preserved by running the full test suite
- Spot-check a few key entry points manually

## Phase 4: Report

Produce a concise verification report:
- ✅ What passed
- ❌ What failed (with details)
- ⚠️ What was not verifiable and why
- Final verdict: PASS / FAIL / PARTIAL

If FAIL or PARTIAL, describe what needs to be fixed before this change can be considered complete.
"""


class VerifySkill(BaseSkill):
    """Verify a code change does what it should by running tests and exercising the app end-to-end."""

    @property
    def name(self) -> str:
        return "verify"

    @property
    def description(self) -> str:
        return (
            "Verify a code change does what it should by running the app and checking "
            "the result end-to-end. Runs tests, exercises changed behaviour, and "
            "produces a PASS/FAIL/PARTIAL verification report."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        parts = [VERIFY_PROMPT.strip()]
        if problem.strip():
            parts.append(f"\n## User Request\n\n{problem.strip()}")
        return "\n\n".join(parts)
