from skills.base_skill import BaseSkill

class PMSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "pm"

    @property
    def description(self) -> str:
        return "Project management: tasks, milestones, planning"

    def execute_impl(self, problem: str, **kwargs) -> str:
        # Your PM logic here (safe – exceptions are caught by base)
        return f"Project management result for: {problem}"