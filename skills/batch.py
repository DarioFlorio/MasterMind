"""
skills/batch.py — Batch: Parallel Work Orchestration

Ports the src batch skill logic into EVE's BaseSkill architecture.
Plans a large-scale change, then executes it in parallel across 5–30
isolated worktree agents, each opening a PR.
"""
from __future__ import annotations
import subprocess
from skills.base_skill import BaseSkill


MIN_AGENTS = 5
MAX_AGENTS = 30

WORKER_INSTRUCTIONS = """After you finish implementing the change:
1. **Simplify** — invoke the skill tool with skill: "simplify" to review and clean up your changes.
2. **Run unit tests** — run the project's test suite (check for package.json scripts, Makefile targets, or commands like `npm test`, `bun test`, `pytest`, `go test`). Fix failures.
3. **Test end-to-end** — follow the e2e test recipe from the coordinator's prompt. Skip if instructed.
4. **Commit and push** — commit all changes with a clear message, push the branch, and create a PR with `gh pr create`. Use a descriptive title. If `gh` is unavailable, note it.
5. **Report** — end with a single line: `PR: <url>` so the coordinator can track it. If no PR was created: `PR: none — <reason>`."""


def _build_prompt(instruction: str) -> str:
    return f"""# Batch: Parallel Work Orchestration

You are orchestrating a large, parallelisable change across this codebase.

## User Instruction

{instruction}

## Phase 1: Research and Plan (Plan Mode)

Enter plan mode now, then:

1. **Understand the scope.** Launch one or more subagents to deeply research what this instruction touches. Find all files, patterns, and call sites that need to change. Understand existing conventions so the migration is consistent.

2. **Decompose into independent units.** Break the work into {MIN_AGENTS}–{MAX_AGENTS} self-contained units. Each unit must:
   - Be independently implementable in an isolated git worktree (no shared state with sibling units)
   - Be mergeable on its own without depending on another unit's PR landing first
   - Be roughly uniform in size (split large units, merge trivial ones)

   Scale the count to the actual work: few files → closer to {MIN_AGENTS}; hundreds of files → closer to {MAX_AGENTS}. Prefer per-directory or per-module slicing over arbitrary file lists.

3. **Determine the e2e test recipe.** Figure out how a worker can verify its change actually works end-to-end. Look for: CLI verifier, dev-server + curl pattern, or an existing e2e/integration test suite. If you cannot find a concrete e2e path, ask the user.

4. **Write the plan** including: research summary, numbered list of work units (title, files, one-line description), e2e recipe, and worker instructions template.

5. Exit plan mode and present the plan for approval.

## Phase 2: Spawn Workers (After Plan Approval)

Once the plan is approved, spawn one background agent per work unit using the agent tool with `isolation: "worktree"` and `run_in_background: true`. Launch all in a single message block so they run in parallel.

For each agent, the prompt must be fully self-contained. Include: overall goal, this unit's specific task, codebase conventions, e2e recipe, and these worker instructions verbatim:

```
{WORKER_INSTRUCTIONS}
```

## Phase 3: Track Progress

After launching all workers, render an initial status table:

| # | Unit | Status | PR |
|---|------|--------|-----|
| 1 | <title> | running | — |

As completion notifications arrive, parse the `PR: <url>` line from each agent's result and re-render the table with updated status (done/failed) and PR links. Keep a brief failure note for any agent that did not produce a PR.

When all agents have reported, render the final table and a one-line summary (e.g., "22/24 units landed as PRs").
"""


def _is_git_repo() -> bool:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


class BatchSkill(BaseSkill):
    """Parallel work orchestration across git worktrees — plans then spawns 5–30 agents."""

    @property
    def name(self) -> str:
        return "batch"

    @property
    def description(self) -> str:
        return (
            "Research and plan a large-scale change, then execute it in parallel across "
            "5–30 isolated worktree agents that each open a PR. Use for sweeping, "
            "mechanical changes (migrations, refactors, bulk renames) that can be "
            "decomposed into independent parallel units."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        instruction = problem.strip()
        if not instruction:
            return (
                "Provide an instruction describing the batch change you want to make.\n\n"
                "Examples:\n"
                "  batch: migrate from react to vue\n"
                "  batch: replace all uses of lodash with native equivalents\n"
                "  batch: add type annotations to all untyped function parameters"
            )

        if not _is_git_repo():
            return (
                "This is not a git repository. The batch skill requires a git repo because "
                "it spawns agents in isolated git worktrees and creates PRs from each. "
                "Initialise a repo first, or run this from inside an existing one."
            )

        return _build_prompt(instruction)
