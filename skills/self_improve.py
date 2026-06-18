"""
skills/self_improve.py — Recursive self-improvement loop for EVE.

Core logic:
1. Monitor errors via error_learner.py
2. Extract patterns (e.g., "write_file requires old_str")
3. Generate micro-skills to fix common errors
4. Test fixes in sandbox before deployment
5. Deploy successful fixes to skills/ directory
"""
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from agent.error_learner import ErrorLearner
from memory.three_tier import ThreeTierMemory

log = logging.getLogger("skills.self_improve")


class SelfImprover:
    def __init__(self, error_learner: ErrorLearner, memory: ThreeTierMemory):
        self._learner = error_learner
        self._memory = memory
        self._sandbox_dir = Path("temp/sandbox")
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

    def run_improvement_cycle(self) -> List[str]:
        """Run one full improvement cycle. Returns list of deployed skill names."""
        errors = self._collect_errors()
        patterns = self._extract_patterns(errors)
        new_skills = []
        
        for pattern in patterns:
            skill_code = self._generate_skill(pattern)
            if skill_code and self._test_skill(skill_code):
                skill_name = self._deploy_skill(skill_code)
                new_skills.append(skill_name)
                self._memory.store(f"Deployed new skill: {skill_name}", role="system")
        
        return new_skills

    def _collect_errors(self) -> List[dict]:
        """Collect recent errors from error_learner and memory."""
        errors = []
        for tool, failures in self._learner._session_failures.items():
            errors.extend(failures)
        
        # Also check memory for historical errors
        error_snippets = self._memory.retrieve("error", k=10)
        for snippet in error_snippets:
            if "error" in snippet.lower():
                errors.append({"error": snippet})
        
        return errors

    def _extract_patterns(self, errors: List[dict]) -> List[str]:
        """Extract common error patterns using regex and frequency analysis."""
        patterns = defaultdict(int)
        
        for error in errors:
            error_text = error.get("error", "")
            
            # Common patterns
            if "old_str is required" in error_text:
                patterns["write_file_missing_old_str"] += 1
            elif "to and message are required" in error_text:
                patterns["send_message_missing_to"] += 1
            elif "No such file or directory" in error_text:
                patterns["file_not_found"] += 1
            elif "UnicodeEncodeError" in error_text:
                patterns["unicode_error"] += 1
        
        # Return top 3 patterns
        return [p for p, cnt in sorted(patterns.items(), key=lambda x: -x[1])[:3]]

    def _generate_skill(self, pattern: str) -> Optional[str]:
        """Generate a micro-skill to fix a specific error pattern."""
        if pattern == "write_file_missing_old_str":
            return self._generate_write_file_fix()
        elif pattern == "send_message_missing_to":
            return self._generate_send_message_fix()
        elif pattern == "file_not_found":
            return self._generate_file_check_fix()
        elif pattern == "unicode_error":
            return self._generate_unicode_fix()
        return None

    def _generate_write_file_fix(self) -> str:
        """Generate skill to handle write_file missing old_str."""
        return '''
"""
skills/write_file_safe.py — Safe write_file wrapper that checks for old_str.
"""
from tools.write_file_tool import write_file
from tools.read_file_tool import read_file


def write_file_safe(path: str, content: str, old_str: str = None) -> bool:
    """Safe write_file that checks for old_str if file exists."""
    if old_str is None and path.exists():
        old_content = read_file(path)
        if "old_str is required" in old_content:
            old_str = ""  # Default empty old_str for existing files
    
    try:
        write_file(path, content, old_str=old_str)
        return True
    except Exception as e:
        print(f"write_file_safe failed: {e}")
        return False
'''

    def _generate_send_message_fix(self) -> str:
        """Generate skill to handle send_message missing 'to'."""
        return '''
"""
skills/send_message_safe.py — Safe send_message wrapper.
"""
from tools.send_message_tool import send_message


def send_message_safe(message: str, to: str = "user") -> bool:
    """Safe send_message with default 'to' parameter."""
    try:
        send_message(message=message, to=to, is_user_facing=True)
        return True
    except Exception as e:
        print(f"send_message_safe failed: {e}")
        return False
'''

    def _test_skill(self, skill_code: str) -> bool:
        """Test generated skill in sandbox."""
        skill_path = self._sandbox_dir / "test_skill.py"
        skill_path.write_text(skill_code)
        
        try:
            # Execute skill in subprocess
            import subprocess
            result = subprocess.run(
                ["python", str(skill_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception as e:
            log.error(f"Skill test failed: {e}")
            return False

    def _deploy_skill(self, skill_code: str) -> str:
        """Deploy tested skill to skills/ directory."""
        skill_name = f"auto_fix_{int(time.time())}.py"
        skill_path = Path("skills") / skill_name
        skill_path.write_text(skill_code)
        return skill_name


# Helper for defaultdict
from collections import defaultdict