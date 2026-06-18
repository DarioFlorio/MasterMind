"""
skills/stuck.py — Stuck: Diagnose frozen/slow EVE sessions

Ports the src stuck skill. Investigates other EVE/Python processes on
this machine for signs of being frozen, stuck, or very slow.
"""
from __future__ import annotations
from skills.base_skill import BaseSkill


STUCK_PROMPT = """# /stuck — diagnose frozen/slow EVE sessions

The user thinks another EVE session on this machine is frozen, stuck, or very slow. Investigate and report findings.

## What to look for

Scan for other EVE/Python processes (excluding the current one). Process names are typically `python`, `python3`, or `eve`.

Signs of a stuck session:
- **High CPU (≥90%) sustained** — likely an infinite loop. Sample twice, 1–2s apart, to confirm it's not a transient spike.
- **Process state `D` (uninterruptible sleep)** — often an I/O hang. The `state` column in `ps` output; first character matters.
- **Process state `T` (stopped)** — user probably hit Ctrl+Z by accident.
- **Process state `Z` (zombie)** — parent isn't reaping.
- **Very high RSS (≥4GB)** — possible memory leak making the session sluggish.
- **Stuck child process** — a hung `git`, `python`, or shell subprocess can freeze the parent. Check `pgrep -lP <pid>` for each session.

## Investigation steps

1. **List all EVE/Python processes** (macOS/Linux):
   ```
   ps -axo pid=,pcpu=,rss=,etime=,state=,comm=,command= | grep -E '(python|python3|eve)' | grep -v grep
   ```
   Filter to rows where the command path contains "eve" or "mind_eve".

2. **For anything suspicious**, gather more context:
   - Child processes: `pgrep -lP <pid>`
   - If high CPU: sample again after 1–2s to confirm it's sustained
   - If a child looks hung, note its full command line: `ps -p <child_pid> -o command=`
   - Check session logs if available: `~/.eve/logs/` or `./logs/`

3. **Consider a stack dump** for a truly frozen process (advanced, optional):
   - macOS: `sample <pid> 3` gives a 3-second native stack sample
   - Linux: `py-spy dump --pid <pid>` if py-spy is installed

## Report

If every session looks healthy, tell the user that directly.

If you found a stuck/slow session, prepare a report with:
- PID, CPU%, RSS, state, uptime, command line, child processes
- Your diagnosis of what's likely wrong
- Relevant log tail or stack sample if you captured it

## Notes
- Don't kill or signal any processes — this is diagnostic only.
- If the user gave a specific PID or symptom, focus there first.
"""


class StuckSkill(BaseSkill):
    """Investigate frozen/stuck/slow EVE sessions on this machine and produce a diagnostic report."""

    @property
    def name(self) -> str:
        return "stuck"

    @property
    def description(self) -> str:
        return (
            "Investigate frozen/stuck/slow EVE sessions on this machine. "
            "Scans running processes for signs of hangs, high CPU, zombie processes, "
            "or memory leaks and produces a diagnostic report."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        prompt = STUCK_PROMPT
        if problem.strip():
            prompt += f"\n## User-provided context\n\n{problem.strip()}\n"
        return prompt
