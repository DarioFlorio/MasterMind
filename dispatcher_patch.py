"""
dispatcher_patch.py — Patch instructions for agent/dispatcher.py
Apply these changes to wire inductive_reason into the routing engine.

HOW TO APPLY
============
1. In dispatcher.py, find the _SKILL_CORE pattern and ADD these lines
   inside the alternation group (before the closing r")\\b"):

     # inductive_reason
     r"find the rule|what is the rule|what's the pattern|"
     r"number sequence|next number|predict the sequence|"
     r"what comes next in|series:|sequence:|"
     r"inductive|generalise|generalize from examples|"
     r"what rule generates|"

2. In _pick_core_skill(), ADD before the final `return "deep_reason"` line:

    if any(k in low for k in (
        "find the rule", "number sequence", "next number",
        "what comes next in", "series:", "sequence:",
        "inductive", "generalise", "generalize from",
        "what rule", "predict the sequence",
    )):
        return "inductive_reason"

3. In skill_router.py's _SKILL_REGISTRY list, ADD:

    {
        "name": "inductive_reason",
        "keywords": ["sequence", "pattern", "rule", "next number", "series",
                     "inductive", "generalise", "generalize", "number series",
                     "what comes next", "find the rule"],
        "description": "Inductive reasoning: find rules from sequences or examples.",
        "best_for": "Number sequences, pattern series, rule extraction, scientific laws.",
    },

A ready-to-paste code block for each change is below.
"""

# ─── PASTE BLOCK 1: into _SKILL_CORE regex ────────────────────────────────────
SKILL_CORE_ADDITION = r"""
    # inductive_reason (NEW — EVE import)
    r"find the rule|what is the rule|what's the pattern|"
    r"next number|predict the sequence|"
    r"what comes next in|inductive reasoning|"
    r"generalise from examples|generalize from examples|"
    r"what rule generates|number sequence|"
"""

# ─── PASTE BLOCK 2: into _pick_core_skill() ───────────────────────────────────
PICK_CORE_ADDITION = '''
    if any(k in low for k in (
        "find the rule", "number sequence", "next number",
        "what comes next in", "inductive", "generalise", "generalize",
        "what rule", "predict the sequence", "what is the rule",
    )):
        return "inductive_reason"
'''

# ─── PASTE BLOCK 3: into _SKILL_REGISTRY in skill_router.py ──────────────────
SKILL_REGISTRY_ENTRY = '''    {
        "name": "inductive_reason",
        "keywords": ["sequence", "pattern", "rule", "next number", "series",
                     "inductive", "generalise", "generalize", "number series",
                     "what comes next", "find the rule"],
        "description": "Inductive reasoning: find rules from sequences or examples.",
        "best_for": "Number sequences, pattern series, rule extraction, scientific laws.",
    },'''

# ─── PASTE BLOCK 4: into main.py imports section ──────────────────────────────
MAIN_IMPORTS = '''from tools.journal_tool import JournalTool'''

# ─── PASTE BLOCK 5: into main.py tool list (where tools are registered) ───────
MAIN_TOOL_REGISTRATION = '''    JournalTool(),'''

if __name__ == "__main__":
    print("dispatcher_patch.py — copy the PASTE BLOCKs into the corresponding files.")
    print("Files to edit:")
    print("  agent/dispatcher.py    (BLOCKS 1 + 2)")
    print("  skills/skill_router.py (BLOCK 3)")
    print("  main.py                (BLOCKS 4 + 5)")
    print("")
    print("New files to copy:")
    print("  skills/inductive_reason.py   → project/skills/")
    print("  skills/lateral_thinking.py   → project/skills/  (replaces existing)")
    print("  tools/web_search_tool.py     → project/tools/   (replaces existing)")
    print("  tools/journal_tool.py        → project/tools/   (new)")
