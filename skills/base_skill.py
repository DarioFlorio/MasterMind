"""
skills/base_skill.py — Robust base class for all reasoning skills.

Provides:
  - Input validation with schema support
  - Automatic retries with exponential backoff
  - Execution timeout protection
  - Structured result (ToolResult‑compatible)
  - Detailed error logging and fallback responses
  - Optional caching for repeat invocations
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

# Configure logger
logger = logging.getLogger("skills.base")

# Optional: try to import ToolResult for seamless integration
try:
    from tools.base_tool import ToolResult
    HAS_TOOL_RESULT = True
except ImportError:
    HAS_TOOL_RESULT = False
    # Fallback: define a minimal ToolResult
    class ToolResult:
        def __init__(self, output: str, is_error: bool = False):
            self.output = output
            self.is_error = is_error

# Type for skill execution result
T = TypeVar('T')


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator: retry skill execution on failure."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    logger.warning(
                        f"Skill execution failed (attempt {attempt}/{max_attempts}): {e}"
                    )
                    if attempt < max_attempts:
                        time.sleep(delay)
                        delay *= backoff
            raise last_exc
        return wrapper
    return decorator


class BaseSkill(ABC):
    """
    Abstract base class for all reasoning skills.

    Subclasses must implement:
        name: str property (unique identifier)
        execute(self, problem: str, **kwargs) -> str

    Optional overrides:
        description, input_schema, supports_streaming, cache_results
    """

    # ------------------------------------------------------------------
    # Required properties (must be overridden)
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique skill name (e.g., 'bayes_reason')."""
        pass

    # ------------------------------------------------------------------
    # Optional properties (can be overridden)
    # ------------------------------------------------------------------
    @property
    def description(self) -> str:
        """Short human-readable description."""
        return f"Skill '{self.name}'"

    @property
    def input_schema(self) -> Dict[str, Any]:
        """
        JSON Schema for input validation.
        Default: accepts any string 'problem'.
        Override to enforce structured inputs.
        """
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["problem"],
            "additionalProperties": False,
        }

    @property
    def cache_results(self) -> bool:
        """Whether to cache results for identical problem + kwargs."""
        return False

    @property
    def supports_streaming(self) -> bool:
        """If True, skill can yield chunks via yield (not yet used)."""
        return False

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------
    def __init__(self):
        self._cache: Dict[str, str] = {}
        self._stats = {
            "calls": 0,
            "cache_hits": 0,
            "errors": 0,
            "total_time": 0.0,
        }

    # ------------------------------------------------------------------
    # Public API (with built-in error handling, timeout, retry, validation)
    # ------------------------------------------------------------------
    @with_retry(max_attempts=2, base_delay=0.5, exceptions=(ValueError, RuntimeError))
    def execute(self, problem: str, timeout_seconds: float = 30.0, **kwargs) -> str:
        """
        Execute the skill with full error protection.

        Args:
            problem: The main input text.
            timeout_seconds: Maximum execution time (raises TimeoutError).
            **kwargs: Additional parameters (validated against schema).

        Returns:
            String output, or an error message prefixed with "[Skill Error]".
        """
        self._stats["calls"] += 1
        start_time = time.time()

        # 1. Validate inputs
        try:
            self._validate_input(problem, kwargs)
        except ValueError as ve:
            self._stats["errors"] += 1
            return self._error_result(f"Input validation failed: {ve}")

        # 2. Check cache
        cache_key = self._compute_cache_key(problem, kwargs)
        if self.cache_results and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            logger.debug(f"Skill '{self.name}' cache hit for key {cache_key[:16]}...")
            return self._cache[cache_key]

        # 3. Execute with timeout
        try:
            result = self._run_with_timeout(problem, timeout_seconds, **kwargs)
        except TimeoutError:
            self._stats["errors"] += 1
            msg = f"Skill '{self.name}' exceeded {timeout_seconds}s timeout"
            logger.error(msg)
            return self._error_result(msg)
        except Exception as e:
            self._stats["errors"] += 1
            error_detail = traceback.format_exc()
            logger.error(f"Skill '{self.name}' failed: {e}\n{error_detail}")
            return self._error_result(f"Skill execution error: {e}")

        # 4. Post-process result
        if not isinstance(result, str):
            result = str(result)

        # 5. Store in cache
        if self.cache_results:
            self._cache[cache_key] = result

        elapsed = time.time() - start_time
        self._stats["total_time"] += elapsed
        logger.debug(f"Skill '{self.name}' completed in {elapsed:.2f}s")

        return result

    # ------------------------------------------------------------------
    # Subclass override point (actual logic)
    # ------------------------------------------------------------------
    @abstractmethod
    def execute_impl(self, problem: str, **kwargs) -> str:
        """
        The actual skill logic. Override this, not execute() directly.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper methods (can be overridden for custom behaviour)
    # ------------------------------------------------------------------
    def _validate_input(self, problem: str, kwargs: Dict[str, Any]) -> None:
        """Validate problem and kwargs against input_schema."""
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("'problem' must be a non‑empty string")

        # Basic schema validation (expand with jsonschema if needed)
        schema = self.input_schema
        if "properties" in schema:
            # Check required fields
            required = schema.get("required", [])
            for req in required:
                if req == "problem":
                    continue  # already checked
                if req not in kwargs and req not in schema.get("default", {}):
                    raise ValueError(f"Missing required parameter: {req}")

            # Check extra keys
            allowed = set(schema["properties"].keys())
            provided = set(kwargs.keys())
            extra = provided - allowed - {"problem"}
            if extra:
                raise ValueError(f"Unexpected parameters: {extra}")

        # Depth validation if present
        depth = kwargs.get("depth", 3)
        if not isinstance(depth, int) or depth < 1 or depth > 10:
            raise ValueError("'depth' must be an integer between 1 and 10")

    def _compute_cache_key(self, problem: str, kwargs: Dict[str, Any]) -> str:
        """Generate a deterministic cache key."""
        data = {"problem": problem.strip(), **kwargs}
        # Sort keys for consistency
        sorted_data = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(sorted_data.encode()).hexdigest()

    def _run_with_timeout(self, problem: str, timeout_sec: float, **kwargs) -> str:
        """Run execute_impl with a timeout using threading."""
        import threading

        result_container = []
        error_container = []

        def target():
            try:
                res = self.execute_impl(problem, **kwargs)
                result_container.append(res)
            except Exception as e:
                error_container.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout_sec)

        if thread.is_alive():
            raise TimeoutError()
        if error_container:
            raise error_container[0]
        return result_container[0]

    def _error_result(self, message: str) -> str:
        """Return a consistent error string."""
        return f"[Skill Error: {self.name}] {message}"

    # ------------------------------------------------------------------
    # Integration with ToolResult (if available)
    # ------------------------------------------------------------------
    def to_tool_result(self, output: str, is_error: bool = False) -> Any:
        """Convert skill output to ToolResult object."""
        if HAS_TOOL_RESULT:
            return ToolResult(output=output, is_error=is_error)
        # Fallback: return plain string (error marker included)
        return output

    # ------------------------------------------------------------------
    # Stats & introspection
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return execution statistics."""
        return {**self._stats, "name": self.name, "cached_items": len(self._cache)}

    def clear_cache(self) -> None:
        """Clear internal result cache."""
        self._cache.clear()
        logger.info(f"Cache cleared for skill '{self.name}'")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"


# ----------------------------------------------------------------------
# Example of a minimal concrete skill (for testing)
# ----------------------------------------------------------------------
class EchoSkill(BaseSkill):
    """Example: just returns the problem unchanged."""

    @property
    def name(self) -> str:
        return "echo"

    def execute_impl(self, problem: str, **kwargs) -> str:
        return f"Echo: {problem}"


# ----------------------------------------------------------------------
# Utility function to instantiate a skill safely
# ----------------------------------------------------------------------
def create_skill(skill_class: type, **init_kwargs) -> BaseSkill:
    """Factory to create a skill with error handling."""
    try:
        return skill_class(**init_kwargs)
    except Exception as e:
        logger.error(f"Failed to instantiate {skill_class.__name__}: {e}")
        # Return a dummy error skill
        class _ErrorSkill(BaseSkill):
            @property
            def name(self) -> str:
                return "error"

            def execute_impl(self, problem: str, **kwargs) -> str:
                return f"Skill instantiation failed: {e}"
        return _ErrorSkill()