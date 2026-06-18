"""
skills/skillify.py — Skillify: Capture session as a reusable skill

Ports the src skillify skill. Interviews the user to understand the
repeatable process performed in this session, then writes a SKILL.md
or Python skill file that can be re-invoked later.
"""
from __future__ import annotations
from skills.base_skill import BaseSkill


SKILLIFY_PROMPT = """# Skillify {user_description_block}

You are capturing this session's repeatable process as a reusable skill.

## Your Session Context

Review the conversation history above — focus on:
- What repeatable process was performed
- What the inputs/parameters were
- The distinct steps (in order)
- Where the user corrected or steered you
- What tools were needed

## Your Task

### Step 1: Analyse the Session

Before asking any questions, identify:
- What repeatable process was performed
- What the inputs/parameters were
- The distinct steps (in order)
- The success criteria for each step
- Where the user corrected or steered you
- What tools and permissions were needed

### Step 2: Interview the User

Use the ask_user tool for ALL questions. Iterate until the user is happy.

**Round 1: High level confirmation**
- Suggest a name and description based on your analysis
- Suggest high-level goal(s) and success criteria

**Round 2: More details**
- Present the high-level steps as a numbered list
- If the skill needs arguments, suggest them based on what you observed
- Ask where to save: this project (skills/skill_name.py) or personal (~/.eve/skills/)

**Round 3: Breaking down each step**
For each major step:
- What does this step produce that later steps need?
- What proves that this step succeeded?
- Should the user confirm before proceeding? (especially for irreversible actions)
- Are any steps independent and could run in parallel?
- What are the hard constraints or preferences?

**Round 4: Final questions**
- Confirm when this skill should be invoked, with trigger phrases
- Any gotchas or things to watch out for?

Stop interviewing once you have enough information. Don't over-ask for simple processes!

### Step 3: Write the Skill

Create a Python skill file using the BaseSkill pattern:

```python
\"\"\"
skills/{skill_name}.py — {description}
\"\"\"
from __future__ import annotations
from skills.base_skill import BaseSkill


SKILL_PROMPT = \"\"\"# {Skill Title}
{description}

## Inputs
- `$arg_name`: Description of this input

## Goal
{clearly stated goal}

## Steps

### 1. Step Name
What to do in this step. Be specific and actionable.

**Success criteria**: What proves this step is done.

...
\"\"\"


class {ClassName}Skill(BaseSkill):
    \"\"\"{description}\"\"\"

    @property
    def name(self) -> str:
        return "{skill_name}"

    @property
    def description(self) -> str:
        return "{description}"

    def execute_impl(self, problem: str, **kwargs) -> str:
        prompt = SKILL_PROMPT
        if problem.strip():
            prompt += f"\\n\\n## User Request\\n\\n{{problem}}"
        return prompt
```

**Frontmatter as docstring rules:**
- `when_to_use`: Start with "Use when..." and include trigger phrases
- `allowed_tools`: Minimum permissions needed
- Keep simple skills simple — a 2-step skill doesn't need elaborate documentation

### Step 4: Confirm and Save

Before writing the file, output the complete skill content so the user can review it. Then ask for confirmation using ask_user. After writing, tell the user where it was saved and how to invoke it.
"""


class SkillifySkill(BaseSkill):
    """Capture this session's repeatable process into a reusable skill."""

    @property
    def name(self) -> str:
        return "skillify"

    @property
    def description(self) -> str:
        return (
            "Capture this session's repeatable process into a reusable skill. "
            "Call at end of the process you want to capture with an optional description. "
            "Interviews the user, then writes a skill file that can be re-invoked later."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        user_description_block = ""
        if problem.strip():
            user_description_block = f'The user described this process as: "{problem.strip()}"'

        return SKILLIFY_PROMPT.replace("{user_description_block}", user_description_block)
