"""
agent/query_engine.py — Core agentic ReAct loop for MasterMind.

FIX LOG (openMind_fixed3):
  [F1] KV-CACHE BUG: turn_prefix is now appended to the static system prompt
       instead of being inserted mid-message-list. Keeps token positions stable
       so llama.cpp fully reuses its KV cache. Dead lag before first token gone.
  [F2] GEMMA 4 THINK STRIP: add_assistant() stores think-stripped version in
       session history per Google spec ("historical output must not include
       thought content from previous turns"). Raw text still returned to caller.
  [F3] STREAMING BUFFER: on_chunk fires every _CHUNK_BUFFER_SIZE chars (default 8)
       instead of every single token → word-sized bursts in the UI.
  [F4] thinking_prefix baked into system prompt at first build, not per-turn.
  [F5] Gemma 4 <|channel>thought...<channel|> blocks stripped from output + history.
  [F6] SYSTEM PROMPT LEAK FIX: aggressive regex strips model regurgitating its own
       system prompt / instructions back into visible output (Gemma 4 regression).
  [F7] VISION GUARD: if image payload sent but mmproj not loaded, return clear notice
       instead of a silent error or hallucinated description.
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
from agent._robust_parser import (
    parse_tool_calls as _parse_tool_calls_ROBUST,
    KNOWN_TOOL_NAMES as _ROBUST_KNOWN_TOOL_NAMES,
)
from agent.task import Task, TaskStatus
from config.settings import MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR, MAX_TOKENS, SYSTEM_PROMPT_EXTRA
from tools.base_tool import BaseTool, ToolResult
from utils.model_client import ModelClient
from utils.permissions import PermissionManager
from utils.token_counter import SessionUsage
from agent.error_learner import ErrorLearner
from agent.goal_tracker import GoalTracker
from agent.context_offloader import ContextOffloader
from agent.plan_store import PlanStore
from agent.candidate_scorer import CandidateScorer

_TOOL_RE = re.compile(
    r"<tool_use>\s*<n>(.*?)</n>\s*<input>(.*?)</input>\s*</tool_use>",
    re.DOTALL,
)
_JSON_BLOCK_RE = re.compile(
    r"```(?:json)?\s*(\{[^`]*\"tool\"\s*:[^`]*\})\s*```",
    re.DOTALL,
)
_ACTION_RE = re.compile(
    r'\{\s*"action"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)

_THINK_TAG_RE = re.compile(
    r"<think>.*?</think>|<think>.*$|^\s*</think>",
    re.DOTALL | re.MULTILINE,
)
# [F5] Gemma 4 native thinking block format
_GEMMA4_THINK_RE = re.compile(
    r"<\|channel>thought\n.*?<channel\|>",
    re.DOTALL,
)
_TOOL_USE_LEAK_RE = re.compile(r"<tool_use>.*?</tool_use>", re.DOTALL)
_TOOL_RESULT_LEAK_RE = re.compile(r"<tool_result>.*?</tool_result>", re.DOTALL)
_DANGLING_TAG_RE = re.compile(r"</?(?:tool_use|tool_result|think|n|input)>")
_SUBAGENT_BLEED_RE = re.compile(r"\[Sub-agent result\]\s*<think>.*", re.DOTALL)

# [F6] Catch model regurgitating its own system prompt / instruction boilerplate.
# Gemma 4 (and other instruction-tuned models) sometimes echo their system prompt
# verbatim, especially on the first turn or after a tool result.
_SYSTEM_LEAK_PHRASES = [
    r"Never output plain text for tool instructions",
    r"Always use the <tool_use> format",
    r"Never reveal tool names or XML tags",
    r"When you have a result.*speak naturally",
    r"Never output the file content to the chat",
    r"If you.?re writing a file.? just do it",
    r"Do not reveal tool names or XML tags",
    r"TOOLS vs SKILLS",
    r"CRITICAL: write_file, bash, read_file",
    r"CRITICAL: If a skill name is NOT",
    r"CRITICAL: Never fabricate tool results",
    r"CRITICAL: NEVER print a <tool_use>",
    r"CRITICAL: NEVER narrate your reasoning",
    r"DIRECT TOOLS — exact names only",
    r"SKILLS — call ONLY via:",
    r"OUTPUT RULES — NEVER BREAK THESE",
    r"You are (?:MasterMind|EVE).*agentic AI",
    # ── Leak pattern: model explaining its own tools as prose ──────────────
    r"I need to (?:instruct|determine|explain) (?:the user|how) (?:to use|to run) (?:web_|the )",
    r"Based on the given information.*web (?:search|tools)",
    r"web_fetch\(url\*",
    r"web_search\(query\*",
    r"use `?web_fetch`? to search for",
    r"use `?web_search`? to verify",
    r"After retrieving the results.*use web_",
    r"This process (?:ensures|allows) (?:that )?you (?:first )?fetch",
    r"To execute the web search and verify",
    r"perform the following steps",
]
_SYSTEM_LEAK_RE = re.compile(
    "|".join(f"(?:{p})" for p in _SYSTEM_LEAK_PHRASES),
    re.IGNORECASE | re.DOTALL,
)

# [F6] If the output contains any leak phrase, strip everything from that phrase onward
# (the model typically dumps the entire prompt once it starts leaking).
_SYSTEM_LEAK_BLOCK_RE = re.compile(
    r"(?:" + "|".join(f"(?:{p})" for p in _SYSTEM_LEAK_PHRASES) + r")[\s\S]*",
    re.IGNORECASE | re.DOTALL,
)

# chars buffered before firing on_chunk — lower = faster visible streaming
_CHUNK_BUFFER_SIZE = 4

from agent.narrator_filter import _strip_narrator, _StreamingNarratorFilter


def _clean_output(text: str) -> str:
    text = _THINK_TAG_RE.sub("", text)
    text = _GEMMA4_THINK_RE.sub("", text)
    text = _TOOL_USE_LEAK_RE.sub("", text)
    text = _TOOL_RESULT_LEAK_RE.sub("", text)
    text = _SUBAGENT_BLEED_RE.sub("[Sub-agent result]", text)
    text = _DANGLING_TAG_RE.sub("", text)
    # [F6] Strip system prompt leaks — truncate at first leaked phrase
    text = _SYSTEM_LEAK_BLOCK_RE.sub("", text)
    # [F6b] Line-level pass: drop any line that contains a leak phrase
    # (catches mid-output injections that the block regex misses)
    clean_lines = []
    for line in text.splitlines():
        if _SYSTEM_LEAK_RE.search(line):
            break   # once leak starts, everything after is noise too
        clean_lines.append(line)
    text = "\n".join(clean_lines)
    # Strip model-generated "Answer:" / "Answer: " label prefix (Gemma 4 regression)
    text = re.sub(r"^\s*Answer\s*:\s*", "", text, flags=re.IGNORECASE)
    text = _strip_narrator(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_think_for_history(text: str) -> str:
    """[F2] Strip think blocks before storing in session history (Gemma 4 spec)."""
    text = _THINK_TAG_RE.sub("", text)
    text = _GEMMA4_THINK_RE.sub("", text)
    # [F6] Also strip leaks from history
    text = _SYSTEM_LEAK_BLOCK_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_OBS_SOFT_LIMIT  = 8_000
_OBS_HARD_LIMIT  = 30_000
_BUDGET_WARN     = 0.80
_BUDGET_CRITICAL = 0.95
_GOAL_ANCHOR_INTERVAL = 3


def _truncate_observation(text: str, max_chars: int = _OBS_SOFT_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:max_chars // 2]
    tail = text[-(max_chars // 4):]
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n\n... [{omitted:,} chars omitted] ...\n\n{tail}"


def _parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    calls: list[tuple[str, dict]] = []
    for name, raw_input in _TOOL_RE.findall(text):
        name = name.strip()
        try:
            inp = json.loads(raw_input.strip())
        except json.JSONDecodeError:
            inp = {"raw": raw_input.strip()}
        calls.append((name, inp))
    if calls:
        return calls
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
    for name, raw_args in _ACTION_RE.findall(text):
        try:
            inp = json.loads(raw_args)
        except json.JSONDecodeError:
            inp = {}
        calls.append((name.strip(), inp))
    return calls

# --- PATCHED: robust parser wins ---
_parse_tool_calls_ORIGINAL = _parse_tool_calls
_parse_tool_calls = _parse_tool_calls_ROBUST



SKILL_DESCRIPTIONS = {
    "skill_router":      "not sure which skill? route here first",
    "mode_switch":       "auto-detect reasoning mode, switch mid-execution",
    "reason_chain":      "chain multiple skills with residual context",
    "surgical_debug":    "bugs/crashes/unexpected behaviour - read before touching",
    "compound_fix":      "full fix methodology: measure, test, verify, ship",
    "deep_reason":       "deep multi-step analysis for complex questions",
    "cot_reason":        "step-by-step chain-of-thought for math and logic",
    "causal_reason":     "root cause analysis, 5-why, counterfactuals",
    "causal_forward_reason": "trace cascading consequences forward",
    "abduct":            "best explanation / diagnosis by inference",
    "lateral_thinking":  "creative, unexpected, non-obvious solutions",
    "lateral_forward_thinking": "non-obvious future paths and wild cards",
    "multi_objective":   "trade-offs, Pareto analysis, conflicting criteria",
    "multi_objective_future_optimization": "robust strategies across multiple futures",
    "epistemic_reason":  "evaluate evidence quality, knowledge vs belief",
    "epistemic_future_reasoning": "predict how knowledge and beliefs will evolve",
    "bayes_reason":      "Bayesian inference, base rates, conditional probability",
    "probabilistic_forecasting": "calibrated probability estimates for future events",
    "constraint_solve":  "logic puzzles, CSP, zebra riddles, knight/knave",
    "game_solve":        "minimax, Nash equilibrium, optimal game strategy",
    "game_theoretic_forward_simulation": "predict moves and counter-moves",
    "analogical_reason": "structural mapping between domains, analogies",
    "timeline_reason":   "order events, detect conflicts, schedule dependencies",
    "timeline_projection_reason": "project milestones and future sequences",
    "recursive_decompose": "break big problems into sub-problems recursively",
    "recursive_future_decomposition": "break complex forecasts into sub-forecasts",
    "inductive_reason":  "find patterns and rules from sequences or examples",
    "scenario_whatif_simulation": "what-if branches, best/worst case, stress test",
    "deep_multi_layer_prediction": "long-arc societal and emergent predictions",
    "adaptive_reason":   "adapt reasoning strategy mid-problem",
    "web_search":        "deep BFS/IDS web research (vs web_search TOOL for quick lookup)",
    "debug":             "structured debugging with hypothesis testing",
    "pm":                "project management reasoning and planning",
}

def _build_system_prompt(tools: dict[str, "BaseTool"], working_dir: str,
                         memory_ctx: str = "", thinking_prefix: str = "") -> str:
    # Build tool name list
    tool_names = sorted(tools.keys())
    tool_list  = "  " + ", ".join(tool_names)

    skill_names = []
    try:
        import skills as _s
        # temporal_cognition is orchestrator-level — hidden from the model's
        # direct skill list (it fires automatically, not via skill tool calls)
        skip = {"thinking_controller", "wakefulness", "goal_anchor", "temporal_cognition"}
        skill_names = sorted(k for k in _s.SKILL_REGISTRY.keys() if k not in skip)
    except Exception:
        skill_names = ["deep_reason", "cot_reason", "causal_reason", "web_search"]
    skill_list = "  " + ", ".join(skill_names)

    mem_block   = f"\n\nMemory:\n{memory_ctx}" if memory_ctx else ""
    extra_block = f"\n\n{SYSTEM_PROMPT_EXTRA}" if SYSTEM_PROMPT_EXTRA else ""
    think_block = f"\n\n{thinking_prefix}" if thinking_prefix else ""

    # GAP: PowerShell native support — prefer PowerShell on Windows
    import sys as _sys
    ps_block = (
        "\n\nWINDOWS TOOL PREFERENCE: Use `powershell` tool for ALL Windows-native "
        "operations (files, registry, services, system info, scheduled tasks). "
        "Use `bash` only for Python scripts or cross-platform commands. "
        "The `powershell` tool now supports UTF-8, scripts, and stdin."
        if _sys.platform == "win32" else ""
    )

    # GAP: Memory search reminder
    if "memory_search" in tool_names:
        mem_search_block = (
            "\n\nMEMORY: Use `memory_search` to recall past conversations, "
            "decisions, and context from previous sessions before starting any task."
        )
    else:
        mem_search_block = ""

    return f"""You are EVE, a persistent autonomous AI agent running locally on Windows for Dario.
Working directory: {working_dir}{mem_block}{extra_block}{think_block}{ps_block}{mem_search_block}

You have tools and skills. Use them — do not describe them.

CALL A TOOL:
<tool_use><n>TOOL_NAME</n><input>{{"key": "value"}}</input></tool_use>

CALL A SKILL:
<tool_use><n>skill</n><input>{{"skill": "SKILL_NAME", "args": {{"problem": "..."}}}}</input></tool_use>

Available tools:
{tool_list}

Available skills:
{skill_list}

RULES (never break):
1. NEVER STOP until task is verifiably done (tool returned success).
2. DECOMPOSE FIRST: for any multi-step task, before doing anything else write a numbered
   plan inside <think> tags: "Step 1: ..., Step 2: ..., Step N: ...". Then execute step 1.
   After each tool result, state which step just completed and call the next tool immediately.
   Never wait for user confirmation between steps.
3. TOOL FAILURE: never repeat same call. Research correct approach first, then try differently.
4. UNKNOWN SYNTAX: web_search it before acting. Never guess.
5. LONG TASK: write_file progress to temp/context_offload/ when context warned. Resume next turn.
6. GOAL LOCK: active goal shown above. Don't drift. Interruptions: acknowledge briefly, resume.
7. RESTART RECOVERY: if session history shows an interrupted task, re-read the last tool
   results and continue from the next incomplete step without asking the user what to do.
8. NO PRETENDING: never say "done" without a successful tool result proving it.
9. Output ONLY the XML when calling a tool. Nothing else. Wait for result.
10. Keep reasoning inside <think>...</think> only.

SECOND BRAIN (wiki tools — Karpathy three-folder structure):
  wiki_write   — capture to knowledge base. folder: inbox/notes/reference
  wiki_read    — read a note. {{"title": "...", "folder": "reference"}}
  wiki_search  — full-text search. {{"query": "...", "folder": "reference"}}
  wiki_list    — list notes. {{"folder": "reference", "tag": "ai"}}
  wiki_promote — promote inbox → notes → reference
  USE wiki_reference for permanent knowledge (LLM wikis, research, facts).
  USE wiki_inbox for quick captures. USE wiki_notes for active working notes.

CORRECT TOOL PARAMETER NAMES (wrong name = silent "No path/command provided" error):
  write_file  {{"path": "C:/Users/dario/OneDrive/Documenten/Mind_EVE/out.py", "content": "..."}}
              NEVER use "file" or "filename" - only "path"
  bash        {{"command": "python script.py"}}
              NEVER use "cmd" or "shell" - only "command"
  web_search  {{"query": "search terms"}}
  web_fetch       {{"url": "https://example.com"}}
  read_file       {{"path": "C:/full/path/file.py"}}
  list_dir        {{"path": "C:/Users/dario/OneDrive/Documenten/Mind_EVE"}}
  whatsapp_send   {{"message": "Hello!"}}
                  ALWAYS use "message" — never "body", "text", "content", or "msg"

TOOL CALL FORMAT:
✓ <tool_use><n>write_file</n><input>{{"path": "C:/Users/dario/OneDrive/Documenten/Mind_EVE/out.py", "content": "print(1)"}}</input></tool_use>
✗ {{"file": "..."}} or {{"cmd": "..."}} ← WRONG KEYS, always fail"""


class QueryEngine:
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
        on_plan:       Callable[[str, object], None] | None = None,
    ) -> None:
        self.tools       = {t.name: t for t in tools}
        _ROBUST_KNOWN_TOOL_NAMES.clear()
        _ROBUST_KNOWN_TOOL_NAMES.update(self.tools.keys())
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
        self.on_plan       = on_plan
        self._custom_system  = custom_system
        self._base_sys_prompt: str = ""
        self._thinking_prefix_for_build: str = ""
        self._tool_cache: dict[str, str] = {}
        self.file_tracker: FileTracker = FileTracker()
        self._tool_history: list[str] = []
        self._memory_writes_this_turn = 0
        self._error_learner = ErrorLearner(working_dir=working_dir)
        self._goal_tracker = GoalTracker(working_dir=working_dir)
        # FIX: restore interrupted goal from disk so restart picks up where we left off
        self._goal_text: str = (
            self._goal_tracker.goal
            if self._goal_tracker.goal and not self._goal_tracker.completed
            else ""
        )
        from config.settings import CONTEXT_SIZE as _CS_INIT
        self._ctx_offloader = ContextOffloader(working_dir, _CS_INIT)
        self._tool_attempt_counts: dict[str, int] = {}

        # ── GAP: Resumable plan artifacts ────────────────────────────────────
        self._plan_store = PlanStore(working_dir=working_dir) if working_dir else PlanStore()

        # ── GAP: Candidate scoring on retry + Reasoning gap ──────────────────
        self._candidate_scorer = CandidateScorer(
            error_learner=self._error_learner,
            working_dir=working_dir,
        )

        # ── GAP: Active memory curation (idle consolidation) ─────────────────
        try:
            from services.idle_consolidation import IdleConsolidation
            self._idle_consolidation = IdleConsolidation(working_dir=working_dir)
            self._idle_consolidation.start()
        except Exception:
            self._idle_consolidation = None

        # ── GAP: Daily digest + proactive surfacing ───────────────────────────
        try:
            from services.daily_digest import DailyDigest
            self._daily_digest = DailyDigest(working_dir=working_dir)
        except Exception:
            self._daily_digest = None

        # ── GAP: Context quality (active pruning) ─────────────────────────────
        try:
            from agent.context_budget import ContextBudget
            from config.settings import CONTEXT_SIZE as _CB_CS
            self._context_budget = ContextBudget(context_size=_CB_CS)
        except Exception:
            self._context_budget = None

        # ── Resumable plan: inject any incomplete plan on startup ─────────────
        self._inject_resumable_plan()

    def submit_message(self, user_text: str,
                       on_chunk: Callable[[str], None] | None = None) -> str:
        _prev_chunk = self.on_chunk
        if on_chunk is not None:
            self.on_chunk = on_chunk

        # GAP: Active memory curation — reset idle timer on every user message
        if self._idle_consolidation:
            try:
                self._idle_consolidation.ping()
            except Exception:
                pass

        # GAP: Proactive surfacing — surface pending insights before processing
        if self._idle_consolidation:
            try:
                insight = self._idle_consolidation.pop_insight()
                if insight:
                    user_text = "[Proactive insight from memory]: " + insight + "\n\n" + user_text
            except Exception:
                pass

        # GAP: Daily digest + Scheduled reminders — check on each message
        if self._daily_digest:
            try:
                due_reminders = self._daily_digest.check_reminders()
                if due_reminders:
                    reminder_block = "\n".join(due_reminders)
                    user_text = reminder_block + "\n\n" + user_text
            except Exception:
                pass

        # [F7] Vision guard — detect image payload but no mmproj loaded
        if "[IMG:" in user_text:
            vision_ok = getattr(self.client, "_vision_enabled", False)
            if not vision_ok:
                reply = (
                    "[Vision unavailable — no mmproj file loaded.\n\n"
                    "To enable image support: open Settings (gear icon) → paste the path "
                    "to your mmproj .gguf file → Apply & Save → reload.\n\n"
                    "You can download the Gemma 4 mmproj from the same HuggingFace repo "
                    "as your model (look for mmproj-gemma-4-E2B-it-f16.gguf).]"
                )
                return reply

        # [F1][F4] compute thinking prefix once; baked into system prompt on first build
        if not self._base_sys_prompt:
            self._thinking_prefix_for_build = self._get_thinking_prefix(user_text)

        # ── Reflector brain (pre-conscious deliberation, ≤1.2s) ──────────────
        # Runs BEFORE the intent gate so injected context is always present.
        try:
            from reflector_agent import _REFLECTOR  # lazily initialised in main.py
            _aug, _reflex = _REFLECTOR.process(user_text)
            if _reflex.blocked:
                return "[REFLECTOR] input blocked by reflex rule."
            if _aug != user_text:
                user_text = _aug
            # Ingest user message so it becomes part of working memory
            _REFLECTOR.ingest_turn(user_text[:500], role="user",
                                   turn=len(self.session._messages))
        except (ImportError, AttributeError):
            pass  # reflector not initialised — safe no-op

        # ── Intent gate: classify BEFORE setting any goal ────────────────────
        # STOP/CHAT/QUESTION/COMMAND must never enter the full agentic loop.
        try:
            from skills.conversational_intent import classify_intent, IntentType
            _intent = classify_intent(user_text)

            if _intent == IntentType.STOP:
                # ── Hard stop: clear ALL goal state + trim session ──────────
                self._goal_text = ""
                self._goal_tracker.clear()
                self._tool_history.clear()
                self._tool_attempt_counts.clear()
                # Trim session: keep only last 4 messages so ghost context
                # from old tasks cannot bleed into the next task
                try:
                    self.session._messages = self.session._messages[-4:]
                except Exception:
                    pass
                self.session.add_user(user_text)
                stop_reply = "Understood — stopping everything. Ready for your next request."
                self.session.add_assistant(stop_reply)
                return stop_reply

            if _intent == IntentType.CHAT:
                # Conversational reply — skip the goal/tool loop entirely.
                # Do NOT update goal — just chat.
                self.session.add_user(user_text)
                raw  = self._call_model(self._get_system_prompt(), self.session.to_api_messages())
                text = _clean_output(raw)
                self.session.add_assistant(_strip_think_for_history(raw))
                return text

            if _intent == IntentType.QUESTION:
                # Single-pass answer — no persistent goal.
                self.session.add_user(user_text)
                raw  = self._call_model(self._get_system_prompt(), self.session.to_api_messages())
                # One web_search allowed; handle inline if present
                calls = _parse_tool_calls(raw)
                if calls and calls[0][0] == "web_search":
                    result = self._run_one_tool(*calls[0])
                    self.session.add_tool_result(
                        f"<tool_result><n>web_search</n><o>{result.output}</o></tool_result>"
                    )
                    raw  = self._call_model(self._get_system_prompt(), self.session.to_api_messages())
                text = _clean_output(raw)
                self.session.add_assistant(_strip_think_for_history(raw))
                return text

            if _intent == IntentType.COMMAND:
                # Single tool call — no persistent goal, stops after one action.
                self.session.add_user(user_text)
                raw   = self._call_model(self._get_system_prompt(), self.session.to_api_messages())
                calls = _parse_tool_calls(raw)
                if calls:
                    name, inp = calls[0]
                    res  = self._run_one_tool(name, inp)
                    self.session.add_tool_result(
                        f"<tool_result><n>{name}</n><o>{res.output}</o></tool_result>"
                    )
                    raw  = self._call_model(self._get_system_prompt(), self.session.to_api_messages())
                text = _clean_output(raw)
                self.session.add_assistant(_strip_think_for_history(raw))
                return text

            # ── TASK: full agentic loop ──────────────────────────────────────
            # ALWAYS update the active goal to the current message.
            # This prevents "ghost tasks" where an old goal keeps injecting.
            self._goal_text = user_text.strip()
            self._goal_tracker.set_goal(user_text.strip())
            # Inject active goal status now that intent is confirmed as TASK
            if not self.is_subagent:
                user_text = self._goal_tracker.inject(user_text)

        except Exception as _ie:
            if self.verbose:
                print(f"[intent gate] classify_intent failed: {_ie}", file=sys.stderr)
            # Fallback: set goal if not already set
            if not self._goal_text and user_text.strip():
                self._goal_text = user_text.strip()
                self._goal_tracker.set_goal(user_text.strip())
            if self._goal_text and not self.is_subagent:
                user_text = self._goal_tracker.inject(user_text)
        # ── end intent gate ──────────────────────────────────────────────────

        dispatch = classify(user_text)
        if dispatch.route == Route.RECALL:
            return self._handle_recall(user_text)

        task = Task(description=user_text[:80], max_turns=self.max_turns)
        task.start()

        if dispatch.route == Route.SKILL and dispatch.skill_hint:
            hint = dispatch.skill_hint
            if self.verbose:
                print(f"[engine] Skill hint: {hint}", file=sys.stderr)
            user_text = (
                f"[Use the `skill` tool with skill='{hint}' to answer this]\n\n"
                + user_text
            )

        self.session.add_user(user_text)
        self._memory_writes_this_turn = 0

        # ── UltraPlan: deep planning pass for complex tasks ───────────────────
        # Runs before the main ReAct loop; streams plan phases via on_plan.
        if self.on_plan and not self.is_subagent:
            try:
                from agent.ultraplan import UltraPlan, should_ultraplan
                if should_ultraplan(user_text):
                    planner = UltraPlan(
                        tools=list(self.tools.values()),
                        working_dir=self.working_dir,
                    )
                    blueprint = planner.plan(user_text, on_step=self.on_plan)
                    # Inject the rendered blueprint into context so the agent
                    # knows the plan it's about to execute.
                    from agent.ultraplan import Blueprint as _BP
                    self.session.add_tool_result(
                        "<tool_result><n>ultraplan</n>"
                        f"<o>{blueprint.render()}</o></tool_result>"
                    )
                    # GAP: Resumable plan artifacts — persist to disk
                    try:
                        bid = self._plan_store.save(blueprint)
                        if self.verbose:
                            print(f"[PlanStore] Saved blueprint {bid}", file=sys.stderr)
                    except Exception as _ps_err:
                        if self.verbose:
                            print(f"[PlanStore] Save failed: {_ps_err}", file=sys.stderr)
            except Exception as _up_err:
                if self.verbose:
                    print(f"[engine] UltraPlan skipped: {_up_err}", file=sys.stderr)
        # ─────────────────────────────────────────────────────────────────────

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
        return self.submit_message(prompt)

    async def complete_simple(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        result = self.client.complete(msgs, max_tokens=512, stream=False)
        return result if isinstance(result, str) else str(result)

    def _run_loop(self, task: Task) -> str:
        inner           = 0
        consec_errors   = 0
        _MAX_CONSEC_ERR = 3
        _last_calls_sig = None
        _repeated_calls = 0
        _MAX_REPEATED   = 2
        _consec_nudges  = 0       # FIX: track consecutive bare-text turns
        _MAX_NUDGES     = 3       # give up and surface text after 3 nudges

        while True:
            inner += 1
            task.turns_used = inner
            # Only enforce max_turns if explicitly set below the auto-unlimited threshold.
            # For long tasks (novels, research), we auto-manage via context offloader.
            from config.settings import UNLIMITED_CONTEXT as _UNLIMITED
            _effective_max = self.max_turns if not _UNLIMITED else max(self.max_turns, 500)
            if inner > _effective_max:
                msg = f"[Stopped: max_turns={_effective_max} reached — task may need splitting]"
                self.session.add_assistant(msg)
                return msg

            # Goal anchor fires every turn for non-subagents
            if self._goal_text and not self.is_subagent and inner > 1:
                self._run_goal_anchor(inner)
            # Temporal cognition fires every turn — lightweight pattern check
            if not self.is_subagent and inner > 1:
                self._run_temporal_cognition(inner)
            self._goal_tracker.tick()

            sys_prompt = self._get_system_prompt()
            # [F1] plain message list — no mid-list insertion
            messages   = self.session.to_api_messages()

            # GAP: Context quality — active pruning of low-value messages
            if self._context_budget:
                try:
                    messages = self._context_budget.prune_messages(messages)
                except Exception:
                    pass

            if self.verbose:
                print(f"\n[engine] turn={inner} msgs={len(messages)} tokens≈{self._est_tokens(messages)}",
                      file=sys.stderr)

            raw_text = self._call_model(sys_prompt, messages)
            calls = _parse_tool_calls(raw_text)

            if not calls:
                display_text = _clean_output(raw_text)
                if not display_text and raw_text.strip():
                    display_text = "[No visible response — model produced only internal reasoning. Try rephrasing.]"
                # [F2] store think-stripped version in history
                self.session.add_assistant(_strip_think_for_history(raw_text))
                self.usage.add_turn()
                self._run_wakefulness(display_text, inner)
                # ── Reflector
                try:
                    from reflector_agent import _REFLECTOR
                    if _REFLECTOR is not None and display_text:
                        _REFLECTOR.ingest_turn(display_text, role="agent", turn=inner)
                except (ImportError, AttributeError):
                    pass

                # FIX: bare-text during an active TASK should not exit the loop.
                # Only exit when: no active goal, OR model signals done.
                _DONE_RE = re.compile(
                    r"(?:task\s+(?:complete|done|finished)|"
                    r"^done[\s.!]*$|^finished[\s.!]*$|^complete[\s.!]*$|"
                    r"all\s+(?:steps?\s+)?(?:complete|done|finished)|"
                    r"successfully\s+(?:completed?|finished?|done))",
                    re.IGNORECASE | re.MULTILINE,
                )
                _is_done_signal = bool(_DONE_RE.search(display_text))
                _has_active_goal = bool(self._goal_text)

                if not _has_active_goal or _is_done_signal:
                    _consec_nudges = 0
                    return display_text

                _consec_nudges += 1
                if _consec_nudges >= _MAX_NUDGES:
                    _consec_nudges = 0
                    return display_text

                self.session.add_tool_result(
                    f"<tool_result><n>_continuation_nudge</n>"
                    f"<o>[harness] Reasoning without a tool call. "
                    f"Active goal: {self._goal_text[:200]}. "
                    f"Call a tool now — do not narrate.</o></tool_result>"
                )
                continue

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
            _consec_nudges = 0  # model issued a tool call — reset nudge counter

            # [F2] store think-stripped version in history
            self.session.add_assistant(_strip_think_for_history(raw_text))
            self.usage.add_turn()

            for name, _ in calls:
                self._tool_history.append(name)

            filtered_calls = []
            for name, inp in calls:
                if name == "memory_write":
                    self._memory_writes_this_turn += 1
                    if self._memory_writes_this_turn > 1:
                        if self.verbose:
                            print(f"[engine] Suppressed duplicate memory_write (turn dedup)", file=sys.stderr)
                        continue
                filtered_calls.append((name, inp))
            calls = filtered_calls

            results = self._execute_tools(calls)

            from config.settings import CONTEXT_SIZE as _CS
            _est = self._est_tokens(self.session.to_api_messages())
            _ratio = _est / max(_CS, 1)
            if _ratio > _BUDGET_CRITICAL:
                # ── Auto-compact: drop oldest non-system messages to free space
                # and inject a save-and-continue instruction. NEVER hard-stop.
                try:
                    msgs = self.session.to_api_messages()
                    # Drop the oldest 20% of messages (keep recent context + system)
                    _trim_count = max(2, len(msgs) // 5)
                    self.session._messages = self.session._messages[_trim_count:]
                    if self.verbose:
                        print(f"[budget] Auto-compacted: dropped {_trim_count} msgs", file=sys.stderr)
                except Exception:
                    pass
                offload_msg = self._ctx_offloader.check(
                    self.session.to_api_messages(), goal=self._goal_text
                )
                _inject_offload = offload_msg or (
                    f"\n[AutoCompact] Context was trimmed to free space. "
                    f"Goal: {self._goal_text[:120] if self._goal_text else 'continue task'}. "
                    f"IMPORTANT: save completed work to disk with write_file, then continue from where you left off.\n"
                )
                self.session.add_tool_result(
                    f"<tool_result><n>_context_offloader</n><o>{_inject_offload}</o></tool_result>"
                )
            elif _ratio > _BUDGET_WARN:
                # Warn but inject offloader guidance
                offload_warn = self._ctx_offloader.check(
                    self.session.to_api_messages(), goal=self._goal_text
                )
                if offload_warn:
                    self.session.add_tool_result(
                        f"<tool_result><n>_context_offloader</n><o>{offload_warn}</o></tool_result>"
                    )
                if self.verbose:
                    print(f"[budget] {_ratio:.0%} context used", file=sys.stderr)

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
                # ── Smart error injection from ErrorLearner ──────────────────
                for t_name, t_result in results:
                    if t_result.is_error:
                        key = t_name
                        self._tool_attempt_counts[key] = self._tool_attempt_counts.get(key, 0) + 1
                        attempt_n = self._tool_attempt_counts[key]
                        hint = self._error_learner.get_hint(
                            t_name,
                            {},  # inp not available here; learner uses error text
                            t_result.output,
                            attempt_n,
                        )
                        # Also warn if this pattern failed in past sessions
                        cross_warn = self._error_learner.cross_session_warnings(t_name, {})
                        full_hint = hint
                        if cross_warn:
                            full_hint = cross_warn + "\n" + full_hint

                        # GAP: Candidate scoring on retry — ranked alternative approaches
                        try:
                            candidates = self._candidate_scorer.score_retry_candidates(
                                tool_name=t_name,
                                inp={},
                                error=t_result.output,
                                attempt=attempt_n,
                                top_k=3,
                            )
                            if candidates:
                                cand_lines = ["\n[CandidateScorer] Ranked retry approaches:"]
                                for ci, c in enumerate(candidates, 1):
                                    cand_lines.append(
                                        f"  {ci}. {c.approach} (score={c.score:.0%}) — {c.rationale}"
                                    )
                                full_hint += "\n".join(cand_lines)
                        except Exception:
                            pass

                        self.session.add_tool_result(
                            f"<tool_result><n>_error_learner</n><o>{full_hint}</o></tool_result>"
                        )
                # ── Goal tracker reminder after error ────────────────────────
                if self._goal_tracker.goal and not self.is_subagent:
                    reminder = self._goal_tracker.persistence_reminder()
                    if reminder:
                        self.session.add_tool_result(
                            f"<tool_result><n>_goal_reminder</n><o>{reminder}</o></tool_result>"
                        )
            else:
                consec_errors = 0
                # Reset attempt counter on success
                for t_name, t_result in results:
                    if not t_result.is_error and t_name in self._tool_attempt_counts:
                        prev_attempts = self._tool_attempt_counts.get(t_name, 0)
                        self._tool_attempt_counts[t_name] = 0
                        # GAP: Cross-session error DB — record what fixed the error
                        if prev_attempts > 0:
                            try:
                                self._error_learner.record_success(t_name, {}, approach="retry succeeded")
                                self._candidate_scorer.record_success(t_name, t_name)
                            except Exception:
                                pass

            self._memory_writes_this_turn = 0

            result_xml = "\n".join(r.to_xml(name) for name, r in results)
            self.session.add_tool_result(result_xml)

    def _run_goal_anchor(self, inner: int) -> None:
        try:
            from skills.goal_anchor import GoalAnchorSkill
            anchor = GoalAnchorSkill()
            result = anchor.check(
                goal=self._goal_text,
                tool_history=self._tool_history,
                session=self.session,
                turn=inner,
            )
            if result:
                if self.verbose:
                    print(f"[GoalAnchor] drift at turn {inner}: {result[:80]}", file=sys.stderr)
                self.session.add_tool_result(
                    f"<tool_result><n>goal_anchor</n><o>[GoalAnchor] {result}</o></tool_result>"
                )
        except Exception as exc:
            if self.verbose:
                print(f"[GoalAnchor] check failed: {exc}", file=sys.stderr)

    def _run_temporal_cognition(self, inner: int) -> None:
        """
        Auto-fires every turn. Reads aggregate episode counts (no model call,
        no heavy I/O). Injects a temporal alert into the session only when a
        genuine pattern warrants it — silent the rest of the time.
        """
        try:
            from skills.temporal_cognition import TemporalCognitionSkill
            tc     = TemporalCognitionSkill()
            result = tc.check(
                goal=self._goal_text or "",
                tool_history=self._tool_history,
                session=self.session,
                turn=inner,
            )
            if result:
                if self.verbose:
                    print(f"[TemporalCognition] alert at turn {inner}: {result[:80]}", file=sys.stderr)
                self.session.add_tool_result(
                    f"<tool_result><n>temporal_cognition</n><o>{result}</o></tool_result>"
                )
        except Exception as exc:
            if self.verbose:
                print(f"[TemporalCognition] check failed: {exc}", file=sys.stderr)

    def _run_wakefulness(self, text: str, inner: int) -> None:
        try:
            from skills.wakefulness import WakefulnessSkill
            monitor = WakefulnessSkill()
            result  = monitor.check(text=text, session=self.session, turn=inner)
            if result:
                if self.verbose:
                    print(f"[Wakefulness] alert at turn {inner}: {result[:80]}", file=sys.stderr)
                self.session.add_tool_result(
                    f"<tool_result><n>wakefulness</n><o>{result}</o></tool_result>"
                )
        except Exception as exc:
            if self.verbose:
                print(f"[Wakefulness] check failed: {exc}", file=sys.stderr)

    def _inject_resumable_plan(self) -> None:
        """
        GAP: Resumable plan artifacts — on startup, check if there's an
        incomplete plan and inject its status into the first prompt.
        """
        try:
            summary = self._plan_store.render_resumable()
            if summary:
                # Inject as a synthetic tool result so it's part of context
                self.session.add_tool_result(
                    f"<tool_result><n>_plan_store</n><o>{summary}</o></tool_result>"
                )
                if self.verbose:
                    print(f"[PlanStore] Resumable plan injected", file=sys.stderr)
        except Exception as exc:
            if self.verbose:
                print(f"[PlanStore] inject failed: {exc}", file=sys.stderr)

    def get_startup_digest(self) -> str:
        """
        GAP: Daily digest + proactive surfacing.
        Returns a digest string if one is due; "" otherwise.
        Call from UI/main.py to show on startup.
        """
        if not self._daily_digest:
            return ""
        try:
            return self._daily_digest.get_startup_digest()
        except Exception:
            return ""

    def _execute_tools(self, calls: list[tuple[str, dict]]) -> list[tuple[str, ToolResult]]:
        if len(calls) == 1:
            name, inp = calls[0]
            return [(name, self._run_one_tool(name, inp))]
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return [(n, self._run_one_tool(n, i)) for n, i in calls]
            else:
                return loop.run_until_complete(self._parallel_tools(calls))
        except RuntimeError:
            return [(n, self._run_one_tool(n, i)) for n, i in calls]

    async def _parallel_tools(self, calls: list[tuple[str, dict]]) -> list[tuple[str, ToolResult]]:
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
            # Fuzzy-match: suggest the closest real tool names
            import difflib
            available = sorted(self.tools.keys())
            close = difflib.get_close_matches(name, available, n=3, cutoff=0.5)
            suggestion = f" Did you mean: {close}?" if close else ""
            # Special case: catch common hallucinated tool names from fallback models
            _hallucinated = {
                "execute_python": "bash",
                "run_python":     "bash",
                "python_repl":    "bash",
                "execute_code":   "bash",
                "run_code":       "bash",
                "call_tool":      None,
                "use_tool":       None,
                "tool_call":      None,
            }
            if name in _hallucinated:
                alt = _hallucinated[name]
                alt_msg = (
                    f" Use `bash` with command='python -c \"...\"' or write a .py file and run it."
                    if alt == "bash" else ""
                )
                return ToolResult(
                    output=(
                        f"Tool '{name}' does not exist — this is a hallucinated tool name.{alt_msg}\n"
                        f"Available tools: {available}\n"
                        "Use the correct XML format: "
                        "<tool_use><n>bash</n><input>{\"command\": \"python script.py\"}</input></tool_use>"
                    ),
                    is_error=True,
                )
            return ToolResult(
                output=(
                    f"Unknown tool '{name}'.{suggestion}\n"
                    f"Available tools: {available}\n"
                    "Tool call format: <tool_use><n>TOOL_NAME</n><input>{...}</input></tool_use>"
                ),
                is_error=True,
            )
        allowed = self.permissions.check(name, inp)
        if not allowed:
            return ToolResult(output=f"[Permission denied for tool '{name}']", is_error=True)
        _CACHEABLE = {"read_file", "grep", "glob", "list_dir", "git"}
        if name in _CACHEABLE:
            cache_key = f"{name}:{sorted(inp.items())}"
            if cache_key in self._tool_cache:
                if self.verbose:
                    print(f"[cache hit] {name}", file=sys.stderr)
                return ToolResult(output="[cached] " + self._tool_cache[cache_key])
        # Warn if this exact call pattern has failed before
        cross_warn = self._error_learner.cross_session_warnings(name, inp)
        if cross_warn and self.verbose:
            print(f"[ErrorLearner] {cross_warn[:120]}", file=sys.stderr)
        if self.on_tool_start:
            self.on_tool_start(name, inp)
        try:
            result = tool.execute(inp)
        except Exception as exc:
            result = ToolResult(output=f"Tool raised exception: {exc}", is_error=True)
        if not result.is_error and name in _CACHEABLE:
            cache_key = f"{name}:{sorted(inp.items())}"
            self._tool_cache[cache_key] = result.output[:4000]
        self.file_tracker.record(name, inp, is_error=result.is_error)
        if result.output and len(result.output) > _OBS_SOFT_LIMIT:
            result = ToolResult(
                output=_truncate_observation(result.output),
                is_error=result.is_error,
            )
        # Record failure in error learner
        if result.is_error:
            self._error_learner.record_failure(name, inp, result.output)
        if self.on_tool_end:
            self.on_tool_end(name, result)
        return result

    def _handle_recall(self, query: str) -> str:
        mem = ""
        try:
            from memory_core.manager import get_memory_manager
            mgr = get_memory_manager()
            results = mgr.search_hybrid(query, limit=8)
            if results:
                lines = []
                for r in results:
                    src = f"{r.chunk.path}:{r.chunk.start_line}" if r.chunk.start_line else r.chunk.path
                    lines.append(f"[{r.match_type}|score={r.score:.2f}] {src}\n{r.snippet[:400]}")
                mem = "\n\n---\n".join(lines)
        except Exception:
            pass
        if not mem:
            try:
                from memory.manager import load_context
                mem = load_context()
            except Exception:
                pass
        if not mem:
            reply = "I don't have any memory records from previous sessions yet."
        else:
            self.session.add_user(
                f"[Memory recall request]\n{query}\n\n[Retrieved memory context]\n{mem}"
            )
            sys_prompt = self._get_system_prompt()
            messages   = self.session.to_api_messages()
            reply = self._call_model(sys_prompt, messages)
            reply = _clean_output(reply)
            self.session.add_assistant(_strip_think_for_history(reply))
        return reply

    def _call_model(self, system: str, messages: list[dict]) -> str:
        """[F3] Buffer streaming chunks to _CHUNK_BUFFER_SIZE chars before firing on_chunk."""
        if self.on_chunk:
            chunks: list[str] = []
            buf = ""
            for chunk in self.client.complete(messages, system=system,
                                              max_tokens=MAX_TOKENS, stream=True):
                if isinstance(chunk, str):
                    buf += chunk
                    chunks.append(chunk)
                    if len(buf) >= _CHUNK_BUFFER_SIZE or "\n" in buf:
                        self.on_chunk(buf)
                        buf = ""
            if buf:
                self.on_chunk(buf)
            return "".join(chunks)
        else:
            result = self.client.complete(messages, system=system,
                                          max_tokens=MAX_TOKENS, stream=False)
            return result if isinstance(result, str) else str(result)

    def _get_system_prompt(self) -> str:
        """[F1] Static system prompt with thinking prefix baked in at first build."""
        if not self._base_sys_prompt:
            mem_ctx = ""
            try:
                from memory.manager import load_context
                mem_ctx = load_context()
            except Exception:
                pass
            try:
                from memory_core.manager import get_memory_manager
                st = get_memory_manager().status()
                if st["total_chunks"] > 0:
                    mode = "vector+keyword" if st["vector_enabled"] else "keyword-only"
                    idx_note = (
                        f"\n\n[Searchable memory index: {st['total_chunks']} chunks "
                        f"({mode}). Use the memory_read tool or ask me to recall "
                        f"something specific to query it.]"
                    )
                    mem_ctx = (mem_ctx + idx_note).strip()
            except Exception:
                pass

            self._base_sys_prompt = _build_system_prompt(
                self.tools, self.working_dir, mem_ctx,
                self._thinking_prefix_for_build
            )
            if self._custom_system:
                self._base_sys_prompt += f"\n\nAdditional instructions:\n{self._custom_system}"
        return self._base_sys_prompt

    def _get_thinking_prefix(self, user_text: str) -> str:
        try:
            from skills.thinking_controller import ThinkingControllerSkill, _MODE_INSTRUCTIONS
            mode = ThinkingControllerSkill()._classify(user_text)
            return _MODE_INSTRUCTIONS[mode]
        except Exception:
            return ""

    def invalidate_system_prompt(self) -> None:
        self._base_sys_prompt = ""

    @staticmethod
    def _est_tokens(messages: list[dict]) -> int:
        return sum(max(1, len(m.get("content", "")) // 3) for m in messages)