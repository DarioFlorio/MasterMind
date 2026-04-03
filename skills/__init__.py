"""
skills/__init__.py — Auto‑discover and register all reasoning skills.

Each skill module should export a class that inherits from BaseSkill.
The class name is arbitrary, but the skill's `.name` attribute is used
as the identifier in `skill` tool calls.
"""

from pathlib import Path
from typing import Dict, Type

from skills.base_skill import BaseSkill  # you need to have this base class

# Registry: skill_name -> SkillClass
SKILL_REGISTRY: Dict[str, Type[BaseSkill]] = {}

# Auto‑discover all .py files in this directory (except this one and base_skill)
_this_dir = Path(__file__).parent
_skip = {"__init__.py", "base_skill.py", "skill_tool.py"}

for py_file in _this_dir.glob("*.py"):
    if py_file.name in _skip:
        continue

    module_name = f"skills.{py_file.stem}"
    try:
        # Import the module dynamically
        module = __import__(module_name, fromlist=["*"])
    except Exception as e:
        print(f"  \033[91m✖\033[0m  {module_name}: {e}", flush=True)
        continue

    # Find any class that is a subclass of BaseSkill
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseSkill)
            and obj is not BaseSkill
        ):
            # Instantiate temporarily to get its .name
            try:
                instance = obj()
                skill_name = instance.name
                SKILL_REGISTRY[skill_name] = obj
                print(f"  \033[38;5;67m◈\033[0m  {skill_name}", flush=True)
            except Exception as e:
                print(f"  \033[91m✖\033[0m  {obj.__name__}: {e}", flush=True)

# Also register any skills that are defined directly in this __init__.py
# (none by default, but you could add manual overrides here)


def get_skill(skill_name: str) -> Type[BaseSkill] | None:
    """Return the skill class for a given name, or None."""
    return SKILL_REGISTRY.get(skill_name)


def list_skills() -> list[str]:
    """Return sorted list of all registered skill names."""
    return sorted(SKILL_REGISTRY.keys())