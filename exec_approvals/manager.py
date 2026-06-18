"""
Exec Approvals — user must approve shell commands before they run.
MasterMind shell command approval system.

Features:
  - Allowlist: patterns that auto-approve (e.g. "git *", "ls *")
  - Blocklist: patterns always denied
  - Interactive: prompt user in terminal for unlisted commands
  - Socket mode: external UI can handle approvals (e.g. a GUI companion app)
  - Audit log: all decisions recorded to JSONL
"""
from __future__ import annotations
import fnmatch
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


_APPROVALS_PATH = Path.home() / ".mastermind" / "exec-approvals.json"
_AUDIT_LOG_PATH = Path.home() / ".mastermind" / "exec-audit.jsonl"


class ApprovalDecision(str, Enum):
    ALLOW    = "allow"
    DENY     = "deny"
    PENDING  = "pending"


class AskMode(str, Enum):
    ALWAYS   = "always"    # Always ask, even if allowlisted
    ON_MISS  = "on-miss"   # Ask only if not on allowlist
    NEVER    = "never"     # Never ask; deny on miss


@dataclass
class ApprovalPolicy:
    ask: AskMode = AskMode.ON_MISS
    ask_fallback: ApprovalDecision = ApprovalDecision.DENY
    auto_allow_skills: bool = True
    allowlist: list[str] = field(default_factory=list)   # glob patterns
    blocklist: list[str] = field(default_factory=list)   # glob patterns


@dataclass
class ApprovalRequest:
    command: str
    argv: list[str]
    cwd: str
    agent_id: str = "main"
    request_id: str = field(default_factory=lambda: os.urandom(8).hex())
    timestamp: float = field(default_factory=time.time)


@dataclass
class ApprovalResult:
    decision: ApprovalDecision
    command: str
    request_id: str
    reason: str = ""
    timestamp: float = field(default_factory=time.time)


class ExecApprovalManager:
    """
    Manages shell command approval workflow.

    Usage:
        mgr = ExecApprovalManager()
        result = mgr.request_approval("git commit -am 'fix'", ["git", "commit", "-am", "fix"], "/repo")
        if result.decision == ApprovalDecision.ALLOW:
            # run the command
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        prompt_fn: Optional[Callable[[str], bool]] = None,
        audit_log: bool = True,
    ):
        self._config_path = Path(config_path or _APPROVALS_PATH)
        self._audit_log = audit_log
        self._lock = threading.Lock()
        self._policy: dict[str, ApprovalPolicy] = {}
        self._default_policy = ApprovalPolicy()
        self._prompt_fn = prompt_fn or self._terminal_prompt
        self._pending: dict[str, ApprovalResult] = {}

        self._load_config()

    # ── Config ─────────────────────────────────────────────────────────────────

    def _load_config(self) -> None:
        if not self._config_path.exists():
            self._save_default_config()
            return
        try:
            data = json.loads(self._config_path.read_text())
            defaults = data.get("defaults", {})
            self._default_policy = self._parse_policy(defaults)
            for agent_id, agent_cfg in data.get("agents", {}).items():
                self._policy[agent_id] = self._parse_policy(agent_cfg)
        except Exception:
            pass

    def _parse_policy(self, cfg: dict) -> ApprovalPolicy:
        return ApprovalPolicy(
            ask=AskMode(cfg.get("ask", "on-miss")),
            ask_fallback=ApprovalDecision(cfg.get("askFallback", "deny")),
            auto_allow_skills=cfg.get("autoAllowSkills", True),
            allowlist=cfg.get("allowlist", []),
            blocklist=cfg.get("blocklist", []),
        )

    def _save_default_config(self) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        default = {
            "version": 1,
            "defaults": {
                "ask": "on-miss",
                "askFallback": "deny",
                "autoAllowSkills": True,
                "allowlist": [
                    "git status", "git log*", "git diff*",
                    "ls*", "cat *", "echo *", "pwd", "whoami",
                    "python --version", "python3 --version",
                    "pip list*", "pip show*",
                ],
                "blocklist": [
                    "rm -rf /*", "sudo rm*", ":(){:|:&};:",
                    "dd if=* of=/dev/*", "mkfs*",
                ],
            },
            "agents": {},
        }
        self._config_path.write_text(json.dumps(default, indent=2))

    def get_policy(self, agent_id: str) -> ApprovalPolicy:
        return self._policy.get(agent_id, self._default_policy)

    def add_to_allowlist(self, pattern: str, agent_id: str = "default") -> None:
        policy = self._policy.setdefault(agent_id, ApprovalPolicy())
        if pattern not in policy.allowlist:
            policy.allowlist.append(pattern)
        self._save_config()

    def _save_config(self) -> None:
        data = {
            "version": 1,
            "defaults": {
                "ask": self._default_policy.ask.value,
                "askFallback": self._default_policy.ask_fallback.value,
                "autoAllowSkills": self._default_policy.auto_allow_skills,
                "allowlist": self._default_policy.allowlist,
                "blocklist": self._default_policy.blocklist,
            },
            "agents": {
                aid: {
                    "ask": p.ask.value,
                    "askFallback": p.ask_fallback.value,
                    "autoAllowSkills": p.auto_allow_skills,
                    "allowlist": p.allowlist,
                    "blocklist": p.blocklist,
                }
                for aid, p in self._policy.items()
            },
        }
        self._config_path.write_text(json.dumps(data, indent=2))

    # ── Decision logic ─────────────────────────────────────────────────────────

    def _matches_any(self, command: str, patterns: list[str]) -> bool:
        cmd_lower = command.lower().strip()
        for pattern in patterns:
            if fnmatch.fnmatch(cmd_lower, pattern.lower()):
                return True
            # Also try prefix match
            if cmd_lower == pattern.lower().rstrip('*').strip():
                return True
        return False

    def _sanitize_display(self, command: str) -> str:
        """Sanitize command for display (truncate long commands)."""
        if len(command) > 200:
            return command[:197] + "..."
        return command

    def request_approval(
        self,
        command: str,
        argv: list[str],
        cwd: str = ".",
        agent_id: str = "main",
    ) -> ApprovalResult:
        """
        Request approval for a shell command. Blocks until a decision is made.
        Returns an ApprovalResult with the decision.
        """
        req = ApprovalRequest(command=command, argv=argv, cwd=cwd, agent_id=agent_id)
        policy = self.get_policy(agent_id)

        # Check blocklist first
        if self._matches_any(command, policy.blocklist + self._default_policy.blocklist):
            result = ApprovalResult(
                decision=ApprovalDecision.DENY,
                command=command,
                request_id=req.request_id,
                reason="blocklist",
            )
            self._audit(req, result)
            return result

        # Check allowlist
        all_allowlist = list(set(policy.allowlist + self._default_policy.allowlist))
        is_allowed = self._matches_any(command, all_allowlist)

        if is_allowed and policy.ask != AskMode.ALWAYS:
            result = ApprovalResult(
                decision=ApprovalDecision.ALLOW,
                command=command,
                request_id=req.request_id,
                reason="allowlist",
            )
            self._audit(req, result)
            return result

        # Ask user if mode permits
        if policy.ask == AskMode.NEVER:
            decision = policy.ask_fallback
            result = ApprovalResult(
                decision=decision,
                command=command,
                request_id=req.request_id,
                reason=f"ask=never, fallback={decision.value}",
            )
            self._audit(req, result)
            return result

        # Interactive prompt
        safe_cmd = self._sanitize_display(command)
        approved = self._prompt_fn(safe_cmd)
        decision = ApprovalDecision.ALLOW if approved else ApprovalDecision.DENY
        result = ApprovalResult(
            decision=decision,
            command=command,
            request_id=req.request_id,
            reason="user_approved" if approved else "user_denied",
        )

        if approved:
            self.add_to_allowlist(command, agent_id)

        self._audit(req, result)
        return result

    def is_approved(self, command: str, agent_id: str = "main") -> bool:
        """Quick check without prompting — True only if on allowlist."""
        policy = self.get_policy(agent_id)
        all_allowlist = list(set(policy.allowlist + self._default_policy.allowlist))
        return self._matches_any(command, all_allowlist)

    # ── Terminal prompt ────────────────────────────────────────────────────────

    @staticmethod
    def _terminal_prompt(command: str) -> bool:
        """Default interactive terminal prompt."""
        print(f"\n\033[33m⚠  Exec Approval Required\033[0m")
        print(f"   Command: \033[36m{command}\033[0m")
        try:
            answer = input("   Allow? [y/N/always]: ").strip().lower()
            if answer in ('y', 'yes'):
                return True
            elif answer in ('always', 'a'):
                return True  # caller adds to allowlist
            else:
                return False
        except (KeyboardInterrupt, EOFError):
            print("\n   Denied (interrupt)")
            return False

    # ── Audit log ──────────────────────────────────────────────────────────────

    def _audit(self, req: ApprovalRequest, result: ApprovalResult) -> None:
        if not self._audit_log:
            return
        entry = {
            "ts": result.timestamp,
            "agent": req.agent_id,
            "command": req.command,
            "decision": result.decision.value,
            "reason": result.reason,
            "request_id": req.request_id,
        }
        try:
            _AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_AUDIT_LOG_PATH, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        try:
            lines = _AUDIT_LOG_PATH.read_text().splitlines()[-limit:]
            return [json.loads(l) for l in lines if l.strip()]
        except Exception:
            return []


# ── Module-level singleton ─────────────────────────────────────────────────────

_GLOBAL_APPROVALS: Optional[ExecApprovalManager] = None
_GLOBAL_APPROVALS_LOCK = threading.Lock()

def get_approval_manager(
    config_path: str | Path | None = None,
    prompt_fn: Optional[Callable] = None,
) -> ExecApprovalManager:
    global _GLOBAL_APPROVALS
    with _GLOBAL_APPROVALS_LOCK:
        if _GLOBAL_APPROVALS is None:
            _GLOBAL_APPROVALS = ExecApprovalManager(config_path, prompt_fn)
        return _GLOBAL_APPROVALS


def wrap_bash_tool(
    original_run_fn: Callable,
    agent_id: str = "main",
    manager: Optional[ExecApprovalManager] = None,
) -> Callable:
    """
    Wrap a bash execution function with exec approval checking.

    Usage:
        run_bash = wrap_bash_tool(original_run_bash, agent_id="main")
    """
    mgr = manager or get_approval_manager()

    def wrapped(command: str, cwd: str = ".", **kwargs):
        argv = command.split()
        result = mgr.request_approval(command, argv, cwd=cwd, agent_id=agent_id)
        if result.decision != ApprovalDecision.ALLOW:
            return f"[exec denied] Command not approved: {command}"
        return original_run_fn(command, cwd=cwd, **kwargs)

    return wrapped
