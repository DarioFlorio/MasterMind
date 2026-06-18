"""
tools/structured_output_tool.py — StructuredOutputTool

Forces schema-validated JSON output at the end of a pipeline or TaskFlow run.
When called, the tool validates the provided JSON payload against a supplied
JSON Schema and stores it as the pipeline's structured result.

Ported from src SyntheticOutputTool pattern.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tools.base_tool import BaseTool, ToolResult

log = logging.getLogger("tools.structured_output")


def _validate_against_schema(data: Any, schema: dict) -> list[str]:
    """
    Basic JSON Schema validation without external dependencies.
    Returns a list of error messages (empty = valid).
    """
    errors = []
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(data, dict):
            errors.append(f"Expected object, got {type(data).__name__}")
            return errors
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field!r}")
        props = schema.get("properties", {})
        for key, sub_schema in props.items():
            if key in data:
                sub_errors = _validate_against_schema(data[key], sub_schema)
                errors.extend(f"{key}: {e}" for e in sub_errors)

    elif schema_type == "array":
        if not isinstance(data, list):
            errors.append(f"Expected array, got {type(data).__name__}")
        else:
            items_schema = schema.get("items")
            if items_schema:
                for i, item in enumerate(data):
                    sub_errors = _validate_against_schema(item, items_schema)
                    errors.extend(f"[{i}]: {e}" for e in sub_errors)

    elif schema_type == "string":
        if not isinstance(data, str):
            errors.append(f"Expected string, got {type(data).__name__}")
        min_len = schema.get("minLength")
        if min_len is not None and isinstance(data, str) and len(data) < min_len:
            errors.append(f"String too short (min {min_len})")

    elif schema_type == "integer":
        if not isinstance(data, int) or isinstance(data, bool):
            errors.append(f"Expected integer, got {type(data).__name__}")

    elif schema_type == "number":
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            errors.append(f"Expected number, got {type(data).__name__}")

    elif schema_type == "boolean":
        if not isinstance(data, bool):
            errors.append(f"Expected boolean, got {type(data).__name__}")

    enum = schema.get("enum")
    if enum is not None and data not in enum:
        errors.append(f"Value {data!r} not in enum {enum}")

    return errors


class StructuredOutputTool(BaseTool):
    """
    Emit schema-validated structured JSON as the pipeline's final output.

    Use this at the end of a TaskFlow pipeline or agentic chain to
    produce a machine-readable result that downstream consumers can parse.
    The tool validates the payload against the provided schema before
    accepting it.
    """

    name = "structured_output"
    description = (
        "Emit validated structured JSON as the final output of a pipeline or agentic chain. "
        "Provide the JSON payload and an optional JSON Schema to validate against. "
        "Use this as the LAST tool call in a pipeline to produce a machine-readable result. "
        "If no schema is provided, the payload is accepted as-is. "
        "Input: {\"payload\": {...}, \"schema\": {...}, \"label\": \"optional label\"}"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "description": "The structured JSON data to emit as pipeline output.",
            },
            "schema": {
                "type": "object",
                "description": (
                    "Optional JSON Schema to validate the payload against. "
                    "If provided and validation fails, the tool returns an error."
                ),
            },
            "label": {
                "type": "string",
                "description": "Optional human-readable label for this output (e.g. 'analysis_result').",
            },
        },
        "required": ["payload"],
    }

    def __init__(self):
        super().__init__()
        self._last_output: dict | None = None

    def execute(self, inp: dict) -> ToolResult:
        payload = inp.get("payload")
        schema = inp.get("schema")
        label = inp.get("label", "output")

        if payload is None:
            return ToolResult("Error: 'payload' is required.", is_error=True)

        # Validate against schema if provided
        if schema:
            try:
                errors = _validate_against_schema(payload, schema)
            except Exception as exc:
                return ToolResult(f"Schema validation error: {exc}", is_error=True)

            if errors:
                error_list = "\n".join(f"  - {e}" for e in errors)
                return ToolResult(
                    f"Schema validation failed for {label!r}:\n{error_list}",
                    is_error=True,
                )

        # Store and emit
        self._last_output = {"label": label, "payload": payload}
        try:
            pretty = json.dumps(payload, indent=2, default=str)
        except Exception:
            pretty = str(payload)

        log.info("StructuredOutputTool: emitted output %r (%d bytes)", label, len(pretty))
        return ToolResult(
            f"Structured output accepted ({label!r}):\n```json\n{pretty}\n```"
        )

    def get_last_output(self) -> dict | None:
        """Return the last validated output (for pipeline consumers)."""
        return self._last_output