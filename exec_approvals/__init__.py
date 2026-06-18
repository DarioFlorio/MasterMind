"""Exec Approvals — user confirmation gate for shell commands."""
from .manager import ExecApprovalManager, ApprovalDecision, ApprovalPolicy, get_approval_manager, wrap_bash_tool
__all__ = ["ExecApprovalManager", "ApprovalDecision", "ApprovalPolicy", "get_approval_manager", "wrap_bash_tool"]
