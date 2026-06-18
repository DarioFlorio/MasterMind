"""
skills/__init__.py — Auto-discover and register all reasoning skills.
Prints go to stderr only (never stdout) to avoid interleaving with the UI.
"""
import sys
from pathlib import Path
from typing import Dict, Type
from skills.base_skill import BaseSkill

SKILL_REGISTRY: Dict[str, Type[BaseSkill]] = {}

_this_dir = Path(__file__).parent
_skip = {"__init__.py", "base_skill.py", "skill_tool.py"}

for _py in _this_dir.glob("*.py"):
    if _py.name in _skip:
        continue
    _mod = f"skills.{_py.stem}"
    try:
        _m = __import__(_mod, fromlist=["*"])
    except Exception as _e:
        print(f"  [skill load error] {_mod}: {_e}", file=sys.stderr)
        continue
    for _attr in dir(_m):
        _obj = getattr(_m, _attr)
        if isinstance(_obj, type) and issubclass(_obj, BaseSkill) and _obj is not BaseSkill:
            try:
                _name = _obj().name
                SKILL_REGISTRY[_name] = _obj
            except Exception as _e:
                print(f"  [skill init error] {_attr}: {_e}", file=sys.stderr)


def get_skill(name: str) -> Type[BaseSkill] | None:
    return SKILL_REGISTRY.get(name)


def list_skills() -> list[str]:
    return sorted(SKILL_REGISTRY.keys())