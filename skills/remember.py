"""
skills/remember.py — Remember: Multi-layer memory curation and review

Ports the src remember skill. Reviews the user's memory landscape across
all layers (auto-memory, facts, journal, CLAUDE.md equivalents) and
proposes promotions, cleanups, and resolutions — without applying any
changes until the user approves.
"""
from __future__ import annotations
from skills.base_skill import BaseSkill


SKILL_PROMPT = """# Memory Review

## Goal
Review the user's memory landscape and produce a clear report of proposed changes, grouped by action type. Do NOT apply changes — present proposals for user approval.

## Steps

### 1. Gather all memory layers
Read from all available memory stores:
- `memdir/facts.json` — persistent facts store
- `memdir/journal.json` — session journal entries
- Auto-memory files in `memdir/` (any .md or .json files)
- Any CLAUDE.md or project-level instruction files in the working directory
- Memory loaded in your current context (system prompt memories)

Note which memory sections exist and what they contain.

**Success criteria**: You have the contents of all memory layers and can compare them.

### 2. Classify each memory entry
For each substantive entry in memory, determine the best destination:

| Destination | What belongs there | Examples |
|---|---|---|
| **facts.json** | Stable, project-agnostic facts about the user/environment | "User prefers Python", "API key is at ~/.config/key", "test command is pytest" |
| **journal.json** | Session-specific events and working notes | "Implemented auth flow on 2025-01-15", "Discovered race condition in queue" |
| **Auto-memory (.md)** | Durable knowledge about this project's architecture, conventions, or gotchas | "Auth routes use JWT bearer tokens", "Database migrations run on startup" |
| **Project CLAUDE.md** | Instructions for EVE that all contributors should follow | "use bun not npm", "API routes use kebab-case" |
| **Stay where it is** | Working notes, temporary context, or entries that don't clearly fit elsewhere | Session-specific observations, uncertain patterns |

**Important distinctions:**
- Project conventions (naming, tooling, test commands) → auto-memory or project instructions
- Personal workflow preferences → facts.json
- When unsure, ask rather than guess

**Success criteria**: Each entry has a proposed destination or is flagged as ambiguous.

### 3. Identify cleanup opportunities
Scan across all layers for:
- **Duplicates**: entries that appear in multiple stores → propose keeping only the canonical one
- **Outdated**: entries contradicted by newer information → propose updating or removing
- **Conflicts**: contradictions between any two layers → propose resolution, noting which is more recent
- **Stale journal entries**: entries older than 30 days that have no ongoing relevance → propose pruning

**Success criteria**: All cross-layer issues identified.

### 4. Present the report
Output a structured report grouped by action type:
1. **Promotions** — entries to move, with destination and rationale
2. **Cleanup** — duplicates, outdated entries, conflicts to resolve
3. **Pruning** — stale journal entries proposed for removal
4. **Ambiguous** — entries where you need the user's input on destination
5. **No action needed** — brief note on entries that should stay put

If memory is empty, say so and offer to review any project-level instruction files for cleanup.

**Success criteria**: User can review and approve/reject each proposal individually.

## Rules
- Present ALL proposals before making any changes
- Do NOT modify files without explicit user approval
- Do NOT create new files unless the target doesn't exist yet
- Ask about ambiguous entries — don't guess
- After user approves specific changes, apply them using the appropriate memory tools
"""


class RememberSkill(BaseSkill):
    """Review memory entries and propose promotions, cleanups, and conflict resolutions."""

    @property
    def name(self) -> str:
        return "remember"

    @property
    def description(self) -> str:
        return (
            "Review all memory layers (facts, journal, auto-memory, project instructions) "
            "and propose promotions, deduplication, pruning, and conflict resolution. "
            "Use when the user wants to review, organise, or clean up their memory entries. "
            "Never applies changes without explicit user approval."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        prompt = SKILL_PROMPT
        if problem.strip():
            prompt += f"\n## Additional context from user\n\n{problem.strip()}"
        return prompt
