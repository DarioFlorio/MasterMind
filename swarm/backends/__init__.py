from .base import SwarmBackend, AgentHandle
from .inprocess import InProcessBackend
from .tmux import TmuxBackend

__all__ = ["SwarmBackend", "AgentHandle", "InProcessBackend", "TmuxBackend"]


def detect_best_backend():
    """Return the best available backend for the current system."""
    tmux = TmuxBackend()
    if tmux.is_available():
        return tmux
    return InProcessBackend()
