"""
skills/conversational_intent.py — Intent classifier for EVE.

This is the fix for EVE's "compulsive completion loop" problem.

THE PROBLEM
===========
EVE was treating every user message as a task goal, then injecting
"DO NOT STOP UNTIL COMPLETE" into every turn. So when you said
"stop" or "how are you?", EVE turned those into goals and obsessively
tried to prove it had completed them — writing files, sending messages,
looping forever.

THE APPROACH
============
Before anything becomes a goal, classify the intent. There are five
distinct input types that each need radically different handling:

  STOP     — "stop", "cancel", "abort", "pause", "forget it"
             → halt all agentic loops immediately, clear active goal,
               reply with a single acknowledgement sentence. No tools.

  CHAT     — greetings, status checks, opinions, casual questions
             → respond conversationally in plain text. No goal set.
               No tool loop. No persistence pressure.

  QUESTION — factual/knowledge questions, "what is X", "who is Y"
             → answer directly (may use web_search once). No goal set.
               Single-pass, not a multi-turn agentic loop.

  COMMAND  — short imperative directives ("save that", "show me X",
              "list files") that need one tool call and done.
             → execute the single obvious tool, reply, stop.
               Set no persistent goal.

  TASK     — multi-step work that genuinely requires an agentic loop:
              building things, writing code, research projects, etc.
             → set as goal, enable full agentic loop with persistence.

INTEGRATION
===========
This skill is called by GoalTracker and QueryEngine BEFORE the agentic
loop starts. The result determines:
  - Whether a goal is set at all
  - Whether goal_anchor fires on every turn
  - Whether persistence_reminder is injected on errors
  - What system prompt rule 1 says

Callable as:
  classify_intent(text) -> IntentType   (fast, no model call)
  or via skill tool for model introspection
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from skills.base_skill import BaseSkill


class IntentType(str, Enum):
    STOP     = "stop"
    CHAT     = "chat"
    QUESTION = "question"
    COMMAND  = "command"
    TASK     = "task"


# ── Pattern banks ─────────────────────────────────────────────────────────────

# STOP: any of these → immediately halt, no goal, no loop
_STOP_PATTERNS = re.compile(
    r"^\s*("
    r"stop|halt|cancel|abort|pause|quit|exit|enough|"
    r"drop\s+(?:it|that|this|everything)|"
    r"forget\s+(?:it|that|this)|"
    r"never\s+mind|nevermind|"
    r"don['\u2019]?t\s+(?:do|work\s+on|continue|keep)\s+(?:that|this|it|anything)|"
    r"stop\s+(?:working|doing|that|this|everything)|"
    r"leave\s+it|"
    r"i\s+said\s+stop|"
    r"(?:please\s+)?(?:just\s+)?stop"
    r")\s*[!.]*\s*$",
    re.IGNORECASE,
)

# STOP qualifiers: longer phrases that contain stop intent
_STOP_PHRASES = [
    # Original
    "stop working on",
    "don't work on",
    "do not work on",
    "stop doing",
    "cancel that",
    "abort that",
    "leave that alone",
    "put that on hold",
    "hold off on",
    "set that aside",
    "don't touch",
    "do not touch",
    "ready for another task",
    "ready for a new task",
    "need you ready",
    "stand by",
    "stand down",
    # ── Task-switch / task-reset phrases ──────────────────────────────────
    "forget about that task",
    "forget about that",
    "forget the previous",
    "ignore the previous",
    "dismiss and archive",
    "archive any past task",
    "drop that task",
    "move on from",
    "new task now",
    "start fresh",
    "clear the task",
    "clear your task",
    "full attention now",
    "i need your full attention",
    "your full attention",
    "put that aside",
    "set aside",
    "stop churning",
    "too much on this",
    "churning too much",
    "not the user",
    "i am dario",
    "start a new",
    "begin a new",
    "fresh start",
    "disregard",
    "moving on",
    "new one",          # "get ready for a new one"
    "i need you now",
    "speak to you at once",
    "at once",
    "stop thinking about",
    "stop everything",
    "ready for",
    "get ready for",
]

# CHAT: social/status messages — respond but set no goal
_CHAT_PATTERNS = re.compile(
    r"^\s*("
    r"hi|hey|hello|sup|yo|oi|howdy|"
    r"how\s+(?:are\s+you|you\s+doing|is\s+it\s+going|are\s+things)|"
    r"do\s+you\s+feel|are\s+you\s+(?:ok|ready|there|good)|"
    r"what['\u2019]?s\s+up|"
    r"good\s+(?:morning|afternoon|evening|night|job|work)|"
    r"thanks?(?:\s+you)?|thank\s+you|ty|"
    r"nice\s+(?:work|job|one)|"
    r"cool|awesome|great|perfect|got\s+it|ok(?:ay)?|"
    r"you['\u2019]?re\s+(?:welcome|good|ready)|"
    r"who\s+are\s+you|what\s+are\s+you|tell\s+me\s+about\s+yourself"
    r")\s*[!?.]*\s*$",
    re.IGNORECASE,
)

_CHAT_PHRASES = [
    "how are you",
    "how you doing",
    "how do you feel",
    "are you ready",
    "tell me about yourself",
    "who are you",
    "what can you do",
    "what are your capabilities",
    "what do you know",
    "you doing ok",
]

# QUESTION: knowledge/factual lookups — answer and stop
_QUESTION_STARTERS = re.compile(
    r"^\s*("
    r"what\s+is|what\s+are|what\s+was|what\s+were|"
    r"who\s+is|who\s+are|who\s+was|who\s+were|"
    r"where\s+is|where\s+are|where\s+was|"
    r"when\s+(?:is|was|did|will)|"
    r"why\s+(?:is|are|was|does|do|did)|"
    r"how\s+(?:does|do|did|many|much|long|far|old)|"
    r"can\s+you\s+(?:tell|explain|describe)|"
    r"do\s+you\s+know|"
    r"explain\s+(?:to\s+me\s+)?(?:what|how|why|the)|"
    r"define\s+|"
    r"what['\u2019]?s\s+the\s+(?:difference|meaning|definition)"
    r")",
    re.IGNORECASE,
)

# COMMAND: short single-action imperatives
_COMMAND_PATTERNS = re.compile(
    r"^\s*("
    r"show\s+(?:me\s+)?(?:the\s+)?|"
    r"list\s+(?:the\s+)?(?:files?|dirs?|folders?|tasks?)|"
    r"open\s+|"
    r"read\s+(?:the\s+)?(?:file\s+)?|"
    r"save\s+(?:that|this|it)|"
    r"run\s+(?:that|this|it)\s*$|"
    r"check\s+(?:the\s+)?(?:status|logs?|output)\s*$|"
    r"print\s+|display\s+|"
    r"search\s+for\s+|find\s+(?:the\s+)?(?:file\s+)?"
    r")",
    re.IGNORECASE,
)

# Signals that something is definitely a multi-step TASK
_TASK_SIGNALS = [
    "build", "create", "write", "implement", "develop", "make me",
    "design", "set up", "install", "configure", "fix", "refactor",
    "update", "modify", "add", "remove", "delete", "migrate",
    "research", "analyse", "analyze", "investigate", "compile",
    "generate", "produce", "draft", "plan", "organise", "organize",
    "automate", "script", "test", "deploy", "integrate",
    "and then", "then also", "after that", "followed by",
    "step by step", "multiple", "all of the", "everything",
    # ── Continuity / execution signals ───────────────────────────────────
    "run it", "run the", "debug", "continue", "keep going", "keep working",
    "finish", "finish working", "complete the", "get it done", "get the job done",
    "don't stop", "do not stop", "why did you stop", "why are you stopping",
    "without stopping", "stop stopping", "keep on", "carry on",
]


def classify_intent(text: str) -> IntentType:
    """
    Classify the user's intent. Fast, no model call.

    Rules applied in priority order:
      1. STOP — any stop signal wins immediately
      2. CHAT — social/status messages
      3. QUESTION — factual questions (ends in ?)
      4. COMMAND — short single-action imperatives
      5. TASK — everything else (default for substantive requests)
    """
    stripped = text.strip()
    lower = stripped.lower()

    # ── 1. STOP (highest priority) ──────────────────────────────────────────
    if _STOP_PATTERNS.match(stripped):
        return IntentType.STOP
    if any(phrase in lower for phrase in _STOP_PHRASES):
        return IntentType.STOP

    # ── 2. CHAT ─────────────────────────────────────────────────────────────
    if _CHAT_PATTERNS.match(stripped):
        return IntentType.CHAT
    if any(phrase in lower for phrase in _CHAT_PHRASES):
        # Only if it's short — longer messages with these phrases may be tasks
        if len(stripped.split()) <= 12:
            return IntentType.CHAT

    # ── 3. QUESTION ─────────────────────────────────────────────────────────
    if _QUESTION_STARTERS.match(stripped):
        # A question that also has task signals is actually a TASK
        # e.g. "How do I build a web scraper?" → TASK
        if not any(sig in lower for sig in _TASK_SIGNALS):
            return IntentType.QUESTION
    if stripped.endswith("?") and len(stripped.split()) <= 15:
        if not any(sig in lower for sig in _TASK_SIGNALS):
            return IntentType.QUESTION

    # ── 4. COMMAND ──────────────────────────────────────────────────────────
    if _COMMAND_PATTERNS.match(stripped) and len(stripped.split()) <= 10:
        return IntentType.COMMAND

    # ── 5. TASK (default for substantive input) ──────────────────────────────
    return IntentType.TASK


def is_agentic(intent: IntentType) -> bool:
    """Should this intent trigger the full agentic goal loop?"""
    return intent == IntentType.TASK


def needs_goal(intent: IntentType) -> bool:
    """Should this intent set a persistent goal?"""
    return intent in (IntentType.TASK,)


def should_stop_loop(intent: IntentType) -> bool:
    """Should this intent immediately halt any running agentic loop?"""
    return intent == IntentType.STOP


# ── Skill class (for model-side introspection via skill tool) ─────────────────

class ConversationalIntentSkill(BaseSkill):
    """
    Classify the intent of a user message before goal-tracking decisions.
    Use this to determine whether a message is a STOP command, casual CHAT,
    a QUESTION, a single-shot COMMAND, or a multi-step TASK.
    """

    @property
    def name(self) -> str:
        return "conversational_intent"

    @property
    def description(self) -> str:
        return (
            "Classify user intent as STOP / CHAT / QUESTION / COMMAND / TASK. "
            "Use before setting goals or starting agentic loops. "
            "STOP = halt everything. CHAT/QUESTION = reply conversationally, no goal. "
            "COMMAND = one tool call. TASK = full agentic loop."
        )

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "The user message to classify.",
                }
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        intent = classify_intent(problem)
        lines = [
            f"Intent: {intent.value.upper()}",
            "",
        ]
        explanations = {
            IntentType.STOP: (
                "This is a STOP signal. Halt all agentic loops immediately. "
                "Clear the active goal. Reply with one short acknowledgement sentence. "
                "Do NOT write files, create tasks, or take any further actions."
            ),
            IntentType.CHAT: (
                "This is casual conversation. Reply naturally in plain text. "
                "Do NOT set a goal. Do NOT start a tool loop. "
                "Just answer like a person would."
            ),
            IntentType.QUESTION: (
                "This is a factual question. Answer directly. "
                "You may use web_search once if needed. "
                "Do NOT set a persistent goal or start a multi-turn loop."
            ),
            IntentType.COMMAND: (
                "This is a single-action command. Execute the one obvious tool call "
                "and reply with the result. Do NOT set a persistent goal."
            ),
            IntentType.TASK: (
                "This is a multi-step task. Set it as the active goal. "
                "Use the full agentic loop. Persist until complete."
            ),
        }
        lines.append(explanations[intent])
        return "\n".join(lines)