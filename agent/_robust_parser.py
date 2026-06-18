"""
agent/_robust_parser.py — Forgiving tool-call parser.

Recovers from malformed model output:
  1. Close-tag corruption / missing </input>
  2. Garbage tail between } and </tool_use>
  3. Fenced JSON blocks
  4. Function-call syntax  (allowlist only)
  5. Python-style quotes / trailing commas
  6. OpenAI/Mistral bare JSON: {"name":"x","arguments":{...}}
  7. Tool calls wrapped in <final_response>...</final_response>
  8. Broken close-tags: </arg_key> instead of </n>
  9. [NEW] <tooluse>NAME<input>...</input></tooluse> variant (DeepSeek / small models)
 10. [NEW] <tool_use>NAME<input>...</input></tool_use> missing <n> wrapper
"""
from __future__ import annotations
import json
import re
from typing import List, Tuple

# Primary: tolerant <tool_use> block. Accepts </n>, </name>, </arg_key> as close tag.
_TOOL_USE_RE = re.compile(
    r"<tool_use>\s*"
    r"<(?:n|name)>\s*(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)\s*</(?:n|name|arg_key)>\s*"
    r"<input>\s*(?P<input>.*?)"
    r"(?:</input>\s*)?"
    r"</tool_use>",
    re.DOTALL,
)

# [NEW] Variant used by DeepSeek and other small models:
# <tooluse>tool_name<input>{"key": "val"}</input></tooluse>  OR  </tool_use>
# Also catches the missing-<n> variant: <tool_use>tool_name<input>...</input></tool_use>
_TOOL_USE_COMPACT_RE = re.compile(
    r"<tool(?:_use|use|call)>\s*"          # opening: <tool_use>, <tooluse>, <toolcall>
    r"(?:<(?:n|name)>)?\s*"                # optional <n> or <name> wrapper
    r"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"   # the tool name (no underscore variant too)
    r"\s*(?:</(?:n|name)>)?\s*"            # optional </n> or </name>
    r"<input>\s*(?P<input>.*?)"            # <input> then JSON
    r"(?:</input>)?\s*"                    # optional </input>
    r"</tool(?:_use|use|call)>",           # closing tag (any variant)
    re.DOTALL,
)

# Fenced JSON: ```json {...} ```
_JSON_FENCE_RE = re.compile(
    r"```(?:json|tool_use|tool)?\s*(\{.*?\})\s*```",
    re.DOTALL,
)

# Function-call syntax: tool_name(key="val")
_FUNC_CALL_RE = re.compile(
    r"\b(?P<name>[a-z][a-z0-9_]{2,40})\s*\(\s*(?P<args>[^)]*)\s*\)",
)

# OpenAI/Mistral bare JSON tool call (not fenced):
# {"name": "tool_name", "arguments": {...}}
_OPENAI_BARE_RE = re.compile(
    r'\{\s*"name"\s*:\s*"(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"\s*,'
    r'\s*"(?:arguments?|parameters?|input|args)"\s*:\s*(?P<args>\{.*?\}|".*?")',
    re.DOTALL,
)

# Strip <final_response> wrapper — tool calls inside it are still real
_FINAL_RESPONSE_RE = re.compile(
    r"<final_response>\s*(.*?)\s*</final_response>",
    re.DOTALL | re.IGNORECASE,
)

# Populated by query_engine.py so function-call fallback only fires for real tools.
KNOWN_TOOL_NAMES: set[str] = set()


def _extract_first_json_object(s: str) -> str:
    s = s.lstrip()
    if not s.startswith("{"):
        idx = s.find("{")
        if idx < 0:
            return ""
        s = s[idx:]
    depth = 0
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[: i + 1]
    return ""


def _safe_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return {}
    extracted = _extract_first_json_object(raw)
    if extracted:
        raw = extracted
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"[^}\]]+$", "", repaired).rstrip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _parse_func_args(s: str) -> dict:
    out = {}
    if not s.strip():
        return out
    parts = re.findall(
        r'([a-zA-Z_]\w*)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,]+)',
        s,
    )
    for k, v in parts:
        v = v.strip()
        if v.startswith(('"', "'")):
            v = v[1:-1]
        else:
            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
        out[k] = v
    return out


def _unwrap_final_response(text: str) -> str:
    """Extract inner content from <final_response> if it contains a tool call."""
    m = _FINAL_RESPONSE_RE.search(text)
    if m:
        inner = m.group(1)
        if "<tool_use>" in inner or "<tooluse>" in inner or '"name"' in inner:
            return inner
    return text


# Map of compact (no-underscore) names → canonical underscore names
_COMPACT_NAME_MAP = {
    "memorysearch":   "memory_search",
    "memoryread":     "memory_read",
    "memorywrite":    "memory_write",
    "websearch":      "web_search",
    "webfetch":       "web_fetch",
    "readfile":       "read_file",
    "writefile":      "write_file",
    "editfile":       "edit_file",
    "listdir":        "list_dir",
    "testrunner":     "test_runner",
    "todowrite":      "todo_write",
    "todoread":       "todo_read",
    "journaltool":    "journal",
    "skiltool":       "skill",
    "agentool":       "agent",
    "taskoutput":     "task_output",
    "taskget":        "task_get",
    "tasklist":       "task_list",
    "taskcreate":     "task_create",
    "taskupdate":     "task_update",
    "taskstop":       "task_stop",
    "teamcreate":     "team_create",
    "teamdelete":     "team_delete",
    "teamstatus":     "team_status",
    "croncreate":     "cron_create",
    "cronlist":       "cron_list",
    "crondelete":     "cron_delete",
    "sendmessage":    "send_message",
    "receivemessage": "receive_message",
    "powershell":     "powershell",
    "whatsappsend":   "whatsapp_send",
    "structuredoutput": "structured_output",
    "toolsearch":     "tool_search",
    "remotetrigger":  "remote_trigger",
    "askuser":        "ask_user",
}


def _normalise_tool_name(name: str) -> str:
    """Map compact no-underscore tool names to their canonical form."""
    if KNOWN_TOOL_NAMES and name in KNOWN_TOOL_NAMES:
        return name  # already canonical
    lower = name.lower().replace("-", "_")
    return _COMPACT_NAME_MAP.get(lower, name)


def parse_tool_calls(text: str) -> List[Tuple[str, dict]]:
    """Extract (tool_name, args_dict) pairs. Never raises. Returns [] if nothing found."""
    if not text:
        return []

    text = _unwrap_final_response(text)
    calls: List[Tuple[str, dict]] = []

    # 1. Primary: forgiving <tool_use> XML (proper format with <n> wrapper)
    for m in _TOOL_USE_RE.finditer(text):
        name = (m.group("name") or "").strip()
        raw_input = m.group("input") or ""
        if not name:
            continue
        args = _safe_json(raw_input)
        if args is None:
            args = {"raw": raw_input.strip()}
        calls.append((name, args))
    if calls:
        return calls

    # 1b. [NEW] Compact variant: <tooluse>name<input>...</input></tooluse>
    #     Used by DeepSeek-R1, Qwen, and other small models that skip the <n> wrapper
    for m in _TOOL_USE_COMPACT_RE.finditer(text):
        name = (m.group("name") or "").strip()
        raw_input = m.group("input") or ""
        if not name:
            continue
        # Normalise tool names: models sometimes drop underscores
        # e.g. "memorysearch" → "memory_search", "websearch" → "web_search"
        name = _normalise_tool_name(name)
        args = _safe_json(raw_input)
        if args is None:
            args = {"raw": raw_input.strip()}
        calls.append((name, args))
    if calls:
        return calls

    # 2. Fenced JSON blocks
    for raw in _JSON_FENCE_RE.findall(text):
        obj = _safe_json(raw)
        if not isinstance(obj, dict):
            continue
        name = (obj.get("tool") or obj.get("name") or obj.get("action") or "").strip()
        if not name:
            continue
        args = (obj.get("input") or obj.get("args")
                or obj.get("parameters") or obj.get("arguments") or {})
        if not isinstance(args, dict):
            args = {"raw": str(args)}
        calls.append((name, args))
    if calls:
        return calls

    # 3. OpenAI/Mistral bare JSON: {"name":"x","arguments":{...}}
    for m in _OPENAI_BARE_RE.finditer(text):
        name = m.group("name").strip()
        args_raw = m.group("args").strip()
        if args_raw.startswith('"'):
            try:
                args_raw = json.loads(args_raw)
            except Exception:
                pass
        args = _safe_json(args_raw) if isinstance(args_raw, str) else args_raw
        if not isinstance(args, dict):
            args = {"raw": str(args_raw)}
        if name:
            calls.append((name, args))
    if calls:
        return calls

    # 4. Function-call syntax (allowlist only)
    if KNOWN_TOOL_NAMES:
        for m in _FUNC_CALL_RE.finditer(text):
            name = m.group("name")
            if name in KNOWN_TOOL_NAMES:
                args = _parse_func_args(m.group("args"))
                calls.append((name, args))
    if calls:
        return calls

    # 5. Raw <toolname>...</toolname> XML (model forgot to wrap in <tool_use>)
    #    Only fires for known tool names to avoid false positives.
    if KNOWN_TOOL_NAMES:
        for name in KNOWN_TOOL_NAMES:
            pat = re.compile(
                rf"<{re.escape(name)}\b[^>]*>(.*?)</{re.escape(name)}>",
                re.DOTALL | re.IGNORECASE,
            )
            for m in pat.finditer(text):
                inner = m.group(1).strip()
                # Try to parse inner as JSON
                args = _safe_json(inner)
                if args is None:
                    # Parse <key>value</key> children
                    args = {}
                    for cm in re.finditer(r"<(\w+)>(.*?)</\1>", inner, re.DOTALL):
                        args[cm.group(1)] = cm.group(2).strip()
                if not args:
                    args = {"raw": inner}
                calls.append((name, args))
        if calls:
            return calls

    return calls


if __name__ == "__main__":
    KNOWN_TOOL_NAMES.update({"web_search", "bash", "read_file"})
    tests = [
        '<tool_use><n>web_search</n><input>{"query":"x"}></tool_use>',
        '<tool_use><n>bash</n><input>{"command":"ls"}}}></tool_use>',
        '```json\n{"tool": "web_search", "input": {"query": "x"}}\n```',
        'web_search(query="latest news")',
        '{"name": "web_search", "arguments": {"query": "python docx"}}',
        '<final_response><tool_use><n>read_file</n><input>{"path":"x"}</input></tool_use></final_response>',
        '<tool_use><n>read_file</arg_key>path</n><input>{"path":"x"}</input></tool_use>',
    ]
    for t in tests:
        print(parse_tool_calls(t), " <= ", t[:80])
