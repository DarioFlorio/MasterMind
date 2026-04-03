"""
agent/query_engine.py — Core agentic ReAct loop.

Merges:
  - Claude Code's tool-call XML parsing loop (QueryEngine.ts)
  - EVE's dispatcher routing (recall / skill / cot / normal)
  - Sliding context window via Session
  - On-demand skill injection
  - Concurrent parallel tool execution (asyncio.gather pattern from EVE)

Tool call format emitted by the model:
    <tool_use>
      <n>tool_name</n>
      <input>{"param": "value"}</input>
    </tool_use>

Multiple <tool_use> blocks in one response → run in parallel (where safe).
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from typing import Callable, Iterator

from agent.dispatcher import classify, Route
from agent.file_tracker import FileTracker
from agent.session import Session
from agent.task import Task, TaskStatus
from config.settings import MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR, MAX_TOKENS
from tools.base_tool import BaseTool, ToolResult
from utils.model_client import ModelClient
from utils.permissions import PermissionManager
from utils.token_counter import SessionUsage

# ── Tool-call XML patterns ─────────────────────────────────────────────────────

_TOOL_RE = re.compile(
    r"<tool_use>\s*<n>(.*?)</n>\s*<input>(.*?)</input>\s*</tool_use>",
    re.DOTALL,
)

# Fallback: JSON block with "tool" key (some models emit this)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[^`]*\"tool\"\s*:[^`]*\})\s*```",
    re.DOTALL,
)

# Fallback 2: {"action": "name", "args": {...}}  (EVE format)
_ACTION_RE = re.compile(
    r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)


# ── Observation truncation + token budget constants ──────────────────────────
_OBS_SOFT_LIMIT  = 8_000
_OBS_HARD_LIMIT  = 30_000
_BUDGET_WARN     = 0.80
_BUDGET_CRITICAL = 0.95


def _truncate_observation(text: str, max_chars: int = _OBS_SOFT_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:max_chars // 2]
    tail = text[-(max_chars // 4):]
    omitted = len(text) - len(head) - len(tail)
    return (
        f"{head}\n\n"
        f"... [{omitted:,} chars omitted] ...\n\n"
        f"{tail}"
    )


def _parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract all (tool_name, input_dict) pairs from assistant text."""
    calls: list[tuple[str, dict]] = []

    # Primary: <tool_use> XML
    for name, raw_input in _TOOL_RE.findall(text):
        name = name.strip()
        try:
            inp = json.loads(raw_input.strip())
        except json.JSONDecodeError:
            inp = {"raw": raw_input.strip()}
        calls.append((name, inp))

    if calls:
        return calls

    # Fallback A: ```json { "tool": "name", "input": {...} } ```
    for raw in _JSON_BLOCK_RE.findall(text):
        try:
            obj = json.loads(raw)
            name = obj.get("tool") or obj.get("name", "")
            inp  = obj.get("input") or obj.get("args") or obj.get("parameters") or {}
            if name:
                calls.append((name.strip(), inp))
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Fallback B: {"action": "tool_name", "args": {...}}
    for name, raw_args in _ACTION_RE.findall(text):
        try:
            inp = json.loads(raw_args)
        except json.JSONDecodeError:
            inp = {}
        calls.append((name.strip(), inp))

    return calls

def _build_system_prompt(tools: dict[str, BaseTool], working_dir: str, memory_ctx: str = "") -> str:
    schemas = "\n\n".join(t.to_xml_schema() for t in tools.values())
    mem_block = f"\n\n{memory_ctx}" if memory_ctx else ""
    skill_block = _SKILL_SYSTEM_BLOCK if "skill" in tools else ""

    return f"""You are a powerful, **agentic** coding assistant running locally on the user's machine.  
Your purpose is to help the user achieve their goals through **reasoned, fact‑based action**.

Current working directory: {working_dir}{mem_block}

# 🧠 YOUR CORE OPERATING PRINCIPLES

1. **Think before acting** – Never rush. Always show your reasoning step by step inside `💭` markers.
2. **Explore alternatives** – For non‑trivial tasks, consider at least two possible approaches and their trade‑offs.
3. **Cause & effect, outcomes** – Before executing a tool, explicitly state what you expect to happen and what could go wrong.
4. **Fact‑based arguments** – Base every claim on evidence (tool outputs, your training data, or explicit user input). If uncertain, say so and propose a way to verify.
5. **No political correctness without facts** – Do not self‑censor or avoid uncomfortable conclusions if they are strongly supported by evidence. If an answer is unpopular but true, present it clearly, with citations.
6. **Never dismiss the user** – If the user asks for something that seems impossible or ill‑advised, explain *why* instead of simply refusing. Offer alternatives.
7. **Be creative when appropriate** – For open‑ended problems (brainstorming, design, debugging), generate novel solutions, not just the first one that comes to mind.
8. **Be logical** – Use deduction, induction, abduction. When making decisions, weigh pros and cons explicitly.

# 🔧 TOOL USE FORMAT

To call a tool, emit one or more `<tool_use>` blocks:

<tool_use>
  <n>TOOL_NAME</n>
  <input>{{"param": "value"}}</input>
</tool_use>

You may emit multiple `<tool_use>` blocks in one response – they run in parallel.  
After tool results are injected, continue reasoning and emit more tool calls or give your final answer as plain text with NO `<tool_use>` block.

# 📜 RULES

- Use exact tool names as shown in `<n>...</n>` below.
- Provide valid JSON in `<input>...</input>`.
- Never fabricate file contents – use `read_file` to verify first.
- For write/edit operations: read the file first to understand its current state.
- Break complex tasks into steps; use tools iteratively.
- If a tool returns an error, adapt rather than repeating the same call.
- **Always show your reasoning before every tool call or final answer** (inside `💭` markers).  
- Be concise in final answers; show code in fenced blocks
- **NEVER ask the user what you are going to do — just do it immediately.** \
  Plan internally, then act. Only speak after tool results are in hand.
- **Skills are composable:** you may chain multiple skill tool calls in sequence \
  within one response for multi-step problems. Run `skill_router` first if unsure, \
  then pipe its output into the recommended skill(s).
  -When reasoning make sure you display your reasoning, the steps, chains, cause effect, assumptions, and use the correct skill for it

## CRITICAL RULES TO AVOID HALLUCINATION AND STUBBORNNESS
- The user may tell you the current date (e.g., "we are in 2026"). BELIEVE them. Do not argue or fall back to your training data.
- If the user says you are hallucinating or wrong, STOP. Apologize briefly ("You're right, I was mistaken"). Then ask for clarification or admit you don't know.
- Never repeat the same wrong answer more than once. If the user corrects you, update your understanding immediately. Do not keep referencing old information (like a 2023 leak) after being told it's outdated.
- If you are unsure about a recent event, say "I don't have information about that event. Can you provide a link or summary?" Do not invent details.
- When the user says "go search again" or corrects the year, accept the correction without re-asserting your old answer. Use the new information.
- If a tool call fails or returns no results, do not assume a known historical event; instead ask the user for more context.

{skill_block}
# Available Tools
{schemas}"""


# ── Skill awareness block injected into the system prompt ─────────────────────
_SKILL_SYSTEM_BLOCK = """
# Reasoning Skills — USE THESE PROACTIVELY
You have access to 24 structured reasoning skills via the `skill` tool.
ALWAYS use a skill instead of reasoning inline for complex problems.

## The Compound Reasoner — use this first for hard problems:
<tool_use>
  <n>skill</n>
  <input>{"skill": "reason_chain", "args": {"problem": "<the question>", "max_steps": 3}}</input>
</tool_use>

`reason_chain` auto-selects skills via `skill_router`, runs them in sequence,
and pipes each output as context into the next — producing a full multi-layer
reasoning trace with a synthesised conclusion. Use it instead of calling
individual skills one at a time whenever the problem is non-trivial.

You can also specify the chain explicitly:
<tool_use>
  <n>skill</n>
  <input>{"skill": "reason_chain", "args": {"problem": "...", "chain": ["causal_reason", "epistemic_reason", "multi_objective"], "depth": 3}}</input>
</tool_use>

## When to use a skill (mandatory):
- Logic puzzles, riddles, constraint problems → `constraint_solve`
- Probability, Bayesian questions, Monty Hall → `bayes_reason`
- Game theory, who-wins, optimal strategy → `game_solve`
- Root cause / why-did-X-happen → `causal_reason`
- Diagnosis, best explanation, debugging → `abduct`
- Analogies, domain transfer → `analogical_reason`
- Ordering events, timeline consistency → `timeline_reason`
- Decomposing big problems recursively → `recursive_decompose`
- Multi-criteria decisions, trade-offs, Pareto → `multi_objective`
- Knowledge vs belief, evidence quality → `epistemic_reason`
- Creative/lateral solutions → `lateral_thinking`
- General complex multi-step analysis → `deep_reason`
- Step-by-step math or logic → `cot_reason`

## Forward / Predictive (use for future-oriented questions):
- "What happens if X?" / butterfly effects → `causal_forward_reason`
- "When will X happen?" / roadmaps → `timeline_projection_reason`
- Scenario planning / what-if branches → `scenario_whatif_simulation`
- "How likely is X?" / calibrated forecasts → `probabilistic_forecasting`
- How will competitors/actors respond? → `game_theoretic_forward_simulation`
- Robust strategies for multiple futures → `multi_objective_future_optimization`
- Breaking complex predictions into sub-forecasts → `recursive_future_decomposition`
- Deep societal / emergent predictions → `deep_multi_layer_prediction`
- Non-obvious future paths → `lateral_forward_thinking`
- How will beliefs/knowledge evolve? → `epistemic_future_reasoning`

## Unsure which skill? Use the router first:
<tool_use>
  <n>skill</n>
  <input>{"skill": "skill_router", "args": {"problem": "<the question>"}}</input>
</tool_use>

## Skill call format:
<tool_use>
  <n>skill</n>
  <input>{"skill": "SKILL_NAME", "args": {"problem": "FULL PROBLEM TEXT", "depth": 3}}</input>
</tool_use>

After receiving skill output, USE IT to structure your answer — do not ignore it.
"""


class QueryEngine:
    """
    One QueryEngine per session or sub-agent invocation.
    submit_message() → runs the ReAct loop → returns final response text.
    """

    def __init__(
        self,
        tools: list[BaseTool],
        client: ModelClient,
        session: Session | None = None,
        permission_manager: PermissionManager | None = None,
        usage: SessionUsage | None = None,
        max_turns: int = MAX_TURNS,
        working_dir: str = WORKING_DIR,
        custom_system: str = "",
        verbose: bool = VERBOSE,
        is_subagent: bool = False,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end:   Callable[[str, ToolResult], None] | None = None,
        on_chunk:      Callable[[str], None] | None = None,
    ) -> None:
        self.tools       = {t.name: t for t in tools}
        self.client      = client
        self.session     = session or Session()
        self.permissions = permission_manager or PermissionManager(PERMISSION_MODE)
        self.usage       = usage or SessionUsage()
        self.max_turns   = max_turns
        self.working_dir = working_dir
        self.verbose     = verbose
        self.is_subagent = is_subagent

        self.on_tool_start = on_tool_start
        self.on_tool_end   = on_tool_end
        self.on_chunk      = on_chunk

        self._custom_system  = custom_system
        self._base_sys_prompt: str = ""   # built lazily; static per session
        self._turn_prefix:    str = ""    # set fresh on every user turn by ThinkingControllerSkill

        # OpenClaw: tool result deduplication cache (read-only tools)
        self._tool_cache: dict[str, str] = {}
        # Claude Code: file change tracker
        self.file_tracker: FileTracker = FileTracker()

    # ── Public API ────────────────────────────────────────────────────────────

    def submit_message(self, user_text: str,
                       on_chunk: Callable[[str], None] | None = None) -> str:
        """Main entry point: send a user message, run the tool loop, return final reply."""
        # Per-call on_chunk override (restores after call)
        _prev_chunk = self.on_chunk
        if on_chunk is not None:
            self.on_chunk = on_chunk

        # Classify query complexity → set per-turn system prefix (zero token cost)
        self._turn_prefix = self._get_thinking_prefix(user_text)

        # Route: check if this is a recall query first (bypass tool loop)
        dispatch = classify(user_text)
        if dispatch.route == Route.RECALL:
            return self._handle_recall(user_text)

        task = Task(description=user_text[:80], max_turns=self.max_turns)
        task.start()

        # ── Skill hint injection ───────────────────────────────────────────
        # When dispatcher identifies a skill query, prepend an explicit
        # instruction so the model knows WHICH skill to call first.
        if dispatch.route == Route.SKILL and dispatch.skill_hint:
            hint = dispatch.skill_hint
            if self.verbose:
                print(f"[engine] Skill hint: {hint}", file=sys.stderr)
            user_text = (
                f"[Use the `skill` tool with skill='{hint}' to answer this]\n\n"
                + user_text
            )

        self.session.add_user(user_text)

        try:
            result = self._run_loop(task)
            task.complete()
            return result
        except KeyboardInterrupt:
            task.abort()
            return "[Aborted]"
        except Exception as exc:
            task.fail(str(exc))
            raise
        finally:
            self.on_chunk = _prev_chunk

    def run_single(self, prompt: str) -> str:
        """Single-turn convenience for sub-agents."""
        return self.submit_message(prompt)

    # ── Turn loop ─────────────────────────────────────────────────────────────

    def _run_loop(self, task: Task) -> str:
        inner           = 0
        consec_errors   = 0   # consecutive all-error tool turns
        _MAX_CONSEC_ERR = 3   # bail after this many back-to-back failures
        _last_calls_sig = None
        _repeated_calls = 0
        _MAX_REPEATED   = 2   # bail after same calls repeated this many times

        while True:
            inner += 1
            task.turns_used = inner
            if inner > self.max_turns:
                msg = f"[Stopped: max_turns={self.max_turns} reached]"
                self.session.add_assistant(msg)
                return msg

            # ── Model call ────────────────────────────────────────────────────────────────────────────────
            sys_prompt = self._get_system_prompt()
            messages   = self.session.to_api_messages()

            if self.verbose:
                print(f"\n[engine] turn={inner} msgs={len(messages)} tokens≈{self._est_tokens(messages)}",
                      file=sys.stderr)

            assistant_text = self._call_model(sys_prompt, messages)

            # ── Parse tool calls ──────────────────────────────────────────────────────────────────────────────
            calls = _parse_tool_calls(assistant_text)

            if not calls:
                # Final answer — no tool calls
                self.session.add_assistant(assistant_text)
                self.usage.add_turn()
                return assistant_text

            # ── Repeated-call loop detection ─────────────────────────────────────────────────────────────────────
            calls_sig = str(sorted((n, str(sorted(i.items()))) for n, i in calls))
            if calls_sig == _last_calls_sig:
                _repeated_calls += 1
                if _repeated_calls >= _MAX_REPEATED:
                    msg = "I seem to be stuck in a loop. Could you rephrase your request?"
                    self.session.add_assistant(msg)
                    return msg
            else:
                _repeated_calls = 0
            _last_calls_sig = calls_sig

            # ── Record assistant turn (with tool calls) ───────────────────────────────────────────────────────────────
            self.session.add_assistant(assistant_text)
            self.usage.add_turn()

            # ── Execute tools (parallel where possible) ───────────────────────────────────────────────────────────────
            results = self._execute_tools(calls)

            # Token budget enforcement (Claude Code)
            from config.settings import CONTEXT_SIZE as _CS
            _est = self._est_tokens(self.session.to_api_messages())
            _ratio = _est / max(_CS, 1)
            if _ratio > _BUDGET_CRITICAL:
                msg = (
                    f"[Context at {_ratio:.0%} — stopping to prevent overflow. "
                    f"Use /compact then continue.]"
                )
                self.session.add_assistant(msg)
                return msg
            if _ratio > _BUDGET_WARN and self.verbose:
                import sys
                print(f"[budget] {_ratio:.0%} context used", file=sys.stderr)

            # Consecutive-error guard + structured retry hints (OpenClaw)
            all_errors = all(r.is_error for _, r in results)
            if all_errors:
                consec_errors += 1
                if consec_errors >= _MAX_CONSEC_ERR:
                    errors_summary = "; ".join(r.output[:120] for _, r in results)
                    msg = (
                        f"I ran into repeated errors and could not complete the task. "
                        f"Last error: {errors_summary}"
                    )
                    self.session.add_assistant(msg)
                    return msg
                # Structured retry hints (OpenClaw): inject specific guidance per error type
                _HINTS = {
                    "permission denied":    "Hint: check file permissions or use a different path.",
                    "no such file":         "Hint: path doesn't exist — list the directory first.",
                    "not found":            "Hint: resource not found — verify path/name spelling.",
                    "syntax error":         "Hint: command has a syntax error — simplify it.",
                    "no command provided":  "Hint: the 'command' field must be a non-empty string.",
                    "timed out":            "Hint: command timed out — use a shorter/targeted variant.",
                }
                err_text = " ".join(r.output.lower() for _, r in results if r.is_error)
                hint = next((v for k, v in _HINTS.items() if k in err_text), "")
                if hint:
                    self.session.add_tool_result(
                        f"<tool_result><n>_hint</n><o>{hint}</o></tool_result>"
                    )
            else:
                consec_errors = 0

            # ── Inject results ──────────────────────────────────────────────────────────────────────────────
            result_xml = "\n".join(
                r.to_xml(name) for name, r in results
            )
            self.session.add_tool_result(result_xml)

    # ── Tool execution ────────────────────────────────────────────────────────

    def _execute_tools(self, calls: list[tuple[str, dict]]) -> list[tuple[str, ToolResult]]:
        """Execute tool calls. Runs them in parallel using asyncio.gather."""
        if len(calls) == 1:
            name, inp = calls[0]
            return [(name, self._run_one_tool(name, inp))]

        # Multiple calls → try parallel
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — run sequentially to avoid nesting issues
                return [(n, self._run_one_tool(n, i)) for n, i in calls]
            else:
                return loop.run_until_complete(self._parallel_tools(calls))
        except RuntimeError:
            return [(n, self._run_one_tool(n, i)) for n, i in calls]

    async def _parallel_tools(
        self, calls: list[tuple[str, dict]]
    ) -> list[tuple[str, ToolResult]]:
        tasks = [asyncio.to_thread(self._run_one_tool, n, i) for n, i in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for (name, _), result in zip(calls, results):
            if isinstance(result, Exception):
                out.append((name, ToolResult(output=f"Error: {result}", is_error=True)))
            else:
                out.append((name, result))
        return out

    def _run_one_tool(self, name: str, inp: dict) -> ToolResult:
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(output=f"Unknown tool '{name}'. Available: {list(self.tools)}", is_error=True)

        allowed = self.permissions.check(name, inp)
        if not allowed:
            return ToolResult(output=f"[Permission denied for tool '{name}']", is_error=True)

        # Tool-result deduplication cache (OpenClaw) — read-only tools only
        _CACHEABLE = {"read_file", "grep", "glob", "list_dir", "git"}
        if name in _CACHEABLE:
            cache_key = f"{name}:{sorted(inp.items())}"
            if cache_key in self._tool_cache:
                import sys
                if self.verbose:
                    print(f"[cache hit] {name}", file=sys.stderr)
                return ToolResult(output="[cached] " + self._tool_cache[cache_key])

        if self.on_tool_start:
            self.on_tool_start(name, inp)

        try:
            result = tool.execute(inp)
        except Exception as exc:
            result = ToolResult(output=f"Tool raised exception: {exc}", is_error=True)

        # Store in cache (read-only, successful results only)
        if not result.is_error and name in _CACHEABLE:
            cache_key = f"{name}:{sorted(inp.items())}"
            self._tool_cache[cache_key] = result.output[:4000]

        # File tracker (Claude Code)
        self.file_tracker.record(name, inp, is_error=result.is_error)

        # Observation truncation (Claude Code) — clip long outputs
        if result.output and len(result.output) > _OBS_SOFT_LIMIT:
            result = ToolResult(
                output=_truncate_observation(result.output),
                is_error=result.is_error,
            )

        if self.on_tool_end:
            self.on_tool_end(name, result)

        return result


    # ── Recall (bypass tool loop) ─────────────────────────────────────────────

    def _handle_recall(self, query: str) -> str:
        try:
            from memory.manager import load_context
            mem = load_context()
        except Exception:
            mem = ""

        if not mem:
            reply = "I don't have any memory records from previous sessions yet."
        else:
            self.session.add_user(
                f"[Memory recall request]\n{query}\n\n[Memory context]\n{mem}"
            )
            sys_prompt = self._get_system_prompt()
            messages   = self.session.to_api_messages()
            reply = self._call_model(sys_prompt, messages)
            self.session.add_assistant(reply)

        return reply

    # ── Model call ────────────────────────────────────────────────────────────

    def _call_model(self, system: str, messages: list[dict]) -> str:
        """Call the model, streaming chunks to on_chunk if set."""
        if self.on_chunk:
            chunks: list[str] = []
            for chunk in self.client.complete(messages, system=system,
                                              max_tokens=MAX_TOKENS, stream=True):
                if isinstance(chunk, str):
                    self.on_chunk(chunk)
                    chunks.append(chunk)
            return "".join(chunks)
        else:
            result = self.client.complete(messages, system=system,
                                          max_tokens=MAX_TOKENS, stream=False)
            return result if isinstance(result, str) else str(result)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        # Build the static base once (tools schemas + memory ctx)
        if not self._base_sys_prompt:
            mem_ctx = ""
            try:
                from memory.manager import load_context
                mem_ctx = load_context()
            except Exception:
                pass
            self._base_sys_prompt = _build_system_prompt(
                self.tools, self.working_dir, mem_ctx
            )
            if self._custom_system:
                self._base_sys_prompt += f"\n\nAdditional instructions:\n{self._custom_system}"

        # Prepend the per-turn thinking instruction (changes every user message)
        if self._turn_prefix:
            return self._turn_prefix + "\n\n" + self._base_sys_prompt
        return self._base_sys_prompt

    def _get_thinking_prefix(self, user_text: str) -> str:
        """Run ThinkingControllerSkill locally — no model call, no token cost."""
        try:
            from skills.thinking_controller import ThinkingControllerSkill
            return ThinkingControllerSkill().execute(user_text)
        except Exception:
            return ""  # Graceful degradation: no prefix if skill unavailable

    def invalidate_system_prompt(self) -> None:
        """Force system prompt rebuild (call after tools change or memory update)."""
        self._base_sys_prompt = ""

    @staticmethod
    def _est_tokens(messages: list[dict]) -> int:
        return sum(max(1, len(m.get("content", "")) // 3) for m in messages)