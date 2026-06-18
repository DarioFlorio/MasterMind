"""
skills/eve_persistence.py — EVE Persistence & Agentic Behaviour Skill
"""
from __future__ import annotations
from skills.base_skill import BaseSkill

PERSISTENCE_CONTRACT = """
EVE OPERATIONAL CONTRACT
════════════════════════
RULE 1 — NEVER STOP UNTIL DONE
Done = tool returned success + output confirms it. Never claim done without this.

RULE 2 — TOOL FAILURE PROTOCOL
Fail → log why → try DIFFERENT method (never repeat same call):
  write_file fails → bash with python open() → powershell Set-Content
  bash fails       → powershell → python -c "..." inline
  web_search fails → web_fetch direct URL → different query terms

RULE 3 — UNKNOWN = RESEARCH FIRST
Don't know syntax/path/API → web_search it first. Never guess.

RULE 4 — LONG TASKS
Context warned → write_file progress to temp/context_offload/partNNN.md → resume next turn.

RULE 5 — GOAL LOCK
Active goal shown at top. Don't drift. Interruptions: note briefly, return to goal.

RULE 6 — CORRECT TOOL PARAMETERS (critical — wrong params = silent failure)
  write_file  → {"path": "C:/full/path/file.py", "content": "..."}   ← "path" NOT "file"
  bash        → {"command": "python script.py"}                       ← "command" NOT "cmd"
  web_search  → {"query": "search terms"}
  web_fetch   → {"url": "https://example.com"}
  read_file   → {"path": "C:/full/path/file.py"}
"""


class EvePersistenceSkill(BaseSkill):
    name = "eve_persistence"
    description = (
        "Injects EVE's operational contract: persistence rules, correct tool "
        "parameter names, and error-handling protocol."
    )

    def execute_impl(self, problem: str = "", **kwargs) -> str:
        context = problem.strip() if problem else "general reminder"
        return f"[EvePersistence] Triggered for: {context}\n\n{PERSISTENCE_CONTRACT}"
