"""
Skill: lateral_thinking — Solve riddles, trick questions, lateral puzzles.
Enhanced with EVE's extended riddle corpus (40+ fast-paths from EVE's DB).
"""
from __future__ import annotations
import re

DESCRIPTION = "Lateral thinking for riddles, trick questions, and non-obvious problems."


def _known_riddle(text: str) -> str:
    """
    EVE-enhanced riddle corpus. Each entry is ([required_keywords], answer).
    Matches when ALL keywords in a group are found in the lowercased problem.
    """
    low = text.lower()

    patterns = [
        # ── Silence / echo / sound ─────────────────────────────────────────
        (["saying its name", "breaks it"],         "Silence"),
        (["saying its name", "breaks"],            "Silence"),
        (["fragile", "breath breaks it"],          "Silence"),
        (["fragile", "name breaks"],               "Silence"),
        (["speak", "breaks"],                      "Silence"),
        (["no voice", "mouth", "hear"],            "An echo"),
        (["speak without mouth", "hear without"],  "An echo"),
        (["speak without", "no ears"],             "An echo"),
        (["heard but not seen"],                   "An echo"),

        # ── Water / shadow / weather ───────────────────────────────────────
        (["doesn't get wet", "touches water"],     "A shadow"),
        (["can't get wet"],                        "A shadow"),
        (["no weight", "seen in mirror", "no reflection"], "A shadow"),
        (["follows you", "no weight"],             "A shadow"),
        (["runs but never walks", "mouth but never talks"], "A river"),
        (["has a mouth", "never eats", "has a bed", "never sleeps"], "A river"),
        (["has banks", "no money"],                "A river"),

        # ── Time ───────────────────────────────────────────────────────────
        (["yesterday", "today", "tomorrow", "never actually"], "Time"),
        (["comes once in a minute", "twice in a moment"], "The letter M"),
        (["yesterday before today"],               "A dictionary"),

        # ── Footsteps / stairs ────────────────────────────────────────────
        (["more you take", "leaves behind"],       "Footsteps"),
        (["more you take", "more you leave"],      "Footsteps"),

        # ── Letters / words ───────────────────────────────────────────────
        (["spelled incorrectly", "every dictionary"], "Incorrectly"),
        (["spelled wrong", "dictionary"],          "Incorrectly"),
        (["what word", "spelled wrong", "everyone agrees"], "Wrong"),
        (["three letters", "remove one", "you have none"], "The word 'few'"),
        (["starts with e", "ends with e", "one letter"], "Envelope"),
        (["has head and tail", "no body", "coin"],  "A coin"),
        (["has head", "no body", "tail", "no legs"], "A coin"),
        (["head and tail", "no legs"],             "A coin"),

        # ── Holes / emptiness ─────────────────────────────────────────────
        (["full of holes", "holds water"],         "A sponge"),
        (["more holes", "stronger"],               "A net"),
        (["has holes", "strong"],                  "A net"),
        (["what has holes", "holds water"],        "A sponge"),

        # ── Age / growth ──────────────────────────────────────────────────
        (["older i get", "smaller"],               "A candle"),
        (["the older it gets", "shorter"],         "A candle"),
        (["shorter as it gets older"],             "A candle"),

        # ── Fire / light ──────────────────────────────────────────────────
        (["what has no mouth", "eats everything"], "Fire"),
        (["eats", "drinks water", "born in fire"], "Fire"),
        (["light as a feather", "strong man", "can't hold"],  "Breath"),
        (["lighter than feather", "strongest man"],           "Breath"),

        # ── Cold / temperature ────────────────────────────────────────────
        (["catch", "can't be seen", "cold"],       "A cold"),
        (["you can catch", "invisible", "sneeze"], "A cold"),

        # ── Clocks / time-pieces ──────────────────────────────────────────
        (["hands but no", "clock"],                "A clock"),
        (["hands but can't clap"],                 "A clock"),
        (["hands but", "can't hold"],              "A clock"),
        (["goes up", "doesn't move", "stairs"],    "A staircase"),
        (["always in front", "never seen"],        "The future"),
        (["always behind you"],                    "The past"),

        # ── Newspapers / books ────────────────────────────────────────────
        (["black and white", "read all over"],     "A newspaper"),
        (["black", "white", "red all over"],       "A newspaper"),

        # ── Mirrors ───────────────────────────────────────────────────────
        (["you look at", "you see me", "i can't see you"], "A mirror"),
        (["see yourself", "can't see"],            "A mirror"),

        # ── Darkness ──────────────────────────────────────────────────────
        (["what am i", "faster than light"],       "Darkness"),
        (["travels faster than light"],            "Darkness"),
        (["can fill a room", "take up no space"],  "Light"),

        # ── Gloves / fingers ──────────────────────────────────────────────
        (["fingers", "no flesh", "no bones", "glove"], "A glove"),
        (["five fingers", "not a hand"],           "A glove"),

        # ── Teapot ────────────────────────────────────────────────────────
        (["teeth", "no mouth", "chews"],           "A comb"),
        (["short", "long in your mouth"],          "A toothbrush"),
        (["you throw away the outside", "eat the inside"],  "An ear of corn"),

        # ── Man in a field ────────────────────────────────────────────────
        (["man in field", "pack on his back", "didn't land"],  "A skydiver whose parachute failed"),
        (["man found dead", "field", "unopened package"],      "A skydiver whose parachute failed"),

        # ── Surgeon riddle ────────────────────────────────────────────────
        (["boy in accident", "can't operate", "my son"],       "The surgeon is the boy's mother"),
        (["surgeon said", "my son"],                           "The surgeon is the boy's mother"),

        # ── Coin in bottle ────────────────────────────────────────────────
        (["coin in bottle", "cork", "without breaking"],       "Push the cork in"),

        # ── Alive / dead ──────────────────────────────────────────────────
        (["alive but does not breathe", "cold", "stone"],      "Ice"),
        (["cold as death", "never thirsty"],                   "Ice"),

        # ── Mountain climber ─────────────────────────────────────────────
        (["same mountain", "same path", "same time", "different days"], "Same time of day, same place on path"),
    ]

    for keywords, answer in patterns:
        if all(k in low for k in keywords):
            return answer

    return ""


def _generate_candidates(text: str) -> list[str]:
    low = text.lower()

    if any(k in low for k in ("dark", "light", "shadow", "follow")):
        return ["A shadow", "Darkness", "Light", "A reflection"]
    if any(k in low for k in ("letter", "word", "spell", "alphabet")):
        return ["A letter of the alphabet", "The word itself", "Silence", "Incorrectly"]
    if any(k in low for k in ("water", "wet", "river", "rain", "bank")):
        return ["A shadow (doesn't get wet)", "A river", "A sponge", "A cloud"]
    if any(k in low for k in ("time", "when", "minute", "hour", "day")):
        return ["Time", "A clock", "The letter M", "Tomorrow"]
    if any(k in low for k in ("hand", "finger", "hold")):
        return ["A clock", "A glove", "Breath", "A promise"]
    if any(k in low for k in ("fire", "burn", "heat", "warm")):
        return ["Fire", "A candle", "The sun", "Lava"]
    if any(k in low for k in ("old", "age", "smaller", "shorter")):
        return ["A candle", "A pencil", "Ice", "A secret"]
    if any(k in low for k in ("dead", "alive", "breathe")):
        return ["Ice", "Fire", "A river", "A flame"]
    if any(k in low for k in ("man", "woman", "person", "surgeon", "father", "mother")):
        return ["Hidden gender assumption", "A family member you didn't consider",
                "The person is the opposite gender to what you assumed"]

    return ["Silence", "Time", "Shadow", "An echo", "A mirror",
            "The letter 'e'", "A promise", "Your breath", "A thought"]

from skills.base_skill import BaseSkill


class LateralThinkingSkill(BaseSkill):
    @property
    def name(self) -> str:
        return "lateral_thinking"

    @property
    def description(self) -> str:
        return DESCRIPTION

    @property
    def input_schema(self):
        return {
            "type": "object",
            "properties": {
                "problem": {"type": "string", "description": "The problem to solve"},
                "depth":   {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
            "required": ["problem"],
        }

    def execute_impl(self, problem: str, **kwargs) -> str:
        result = []
        result.append(f"**Puzzle:** {problem}\n")
        result.append("**Lateral Thinking Approach:**\n")
        answer = _known_riddle(problem)
        if answer:
            result.append(f"**Pattern recognised:** Classic riddle/puzzle")
            result.append(f"**Answer:** {answer}")
            return "\n".join(result)
        result.append("**Step 1 — Challenge assumptions:**")
        result.append("  What words in the problem are doing double duty?")
        result.append("  What have we unconsciously assumed that may be wrong?")
        result.append("")
        result.append("**Step 2 — Reframe:**")
        result.append("  What if the literal reading is NOT the intended reading?")
        result.append("  Look for homophones, double meanings, metaphors.")
        result.append("")
        result.append("**Step 3 — Consider all perspectives:**")
        result.append("  Who or what could 'it' refer to?")
        result.append("  Could the answer be something abstract (e.g. silence, time, shadow)?")
        result.append("")
        result.append("**Step 4 — Test candidate answers:**")
        candidates = _generate_candidates(problem)
        for c in candidates:
            result.append(f"  • {c}")
        result.append("")
        result.append("**Insight:** Lateral thinking puzzles typically exploit a single "
                      "assumption the reader makes automatically. Finding which assumption "
                      "is wrong usually reveals the answer.")
        return "\n".join(result)
