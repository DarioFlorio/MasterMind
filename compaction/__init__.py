"""Session Compaction — LLM-based context compression."""
from .compactor import SessionCompactor, CompactionConfig, CompactionResult, compact_messages
__all__ = ["SessionCompactor", "CompactionConfig", "CompactionResult", "compact_messages"]
