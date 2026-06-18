"""
skills/loop.py — Loop: Schedule a recurring prompt

Ports the src loop skill. Parses an interval + prompt from the user's
input, converts to a cron expression, schedules it with CronCreateTool,
then immediately executes the prompt once.
"""
from __future__ import annotations
import math
import re
from skills.base_skill import BaseSkill


DEFAULT_INTERVAL = "10m"
DEFAULT_MAX_AGE_DAYS = 7

USAGE_MESSAGE = f"""Usage: loop [interval] <prompt>

Run a prompt or skill on a recurring interval.

Intervals: Ns, Nm, Nh, Nd (e.g. 5m, 30m, 2h, 1d). Minimum granularity is 1 minute.
If no interval is specified, defaults to {DEFAULT_INTERVAL}.

Examples:
  loop 5m /simplify
  loop 30m check the deploy
  loop 1h run tests
  loop check the deploy          (defaults to {DEFAULT_INTERVAL})
  loop check the deploy every 20m"""


def _parse_interval_and_prompt(args: str) -> tuple[str, str]:
    """
    Parse args into (interval, prompt).
    Rules (in priority order):
    1. Leading token matching ^\\d+[smhd]$ → interval; rest is prompt.
    2. Trailing 'every <N><unit>' clause → extract interval, strip from prompt.
    3. Default: interval = DEFAULT_INTERVAL, entire input is prompt.
    """
    args = args.strip()

    # Rule 1: leading token
    leading_match = re.match(r'^(\d+[smhd])\s+(.*)', args, re.DOTALL)
    if leading_match:
        return leading_match.group(1), leading_match.group(2).strip()

    # Rule 2: trailing 'every N unit'
    trailing_match = re.search(
        r'\s+every\s+(\d+)\s*(s|sec(?:ond)?s?|m|min(?:ute)?s?|h|hr?s?|hour?s?|d|day?s?)$',
        args, re.IGNORECASE
    )
    if trailing_match:
        n = int(trailing_match.group(1))
        unit_raw = trailing_match.group(2).lower()
        if unit_raw.startswith('s'):
            unit = 's'
        elif unit_raw.startswith('m'):
            unit = 'm'
        elif unit_raw.startswith('h'):
            unit = 'h'
        else:
            unit = 'd'
        interval = f"{n}{unit}"
        prompt = args[:trailing_match.start()].strip()
        return interval, prompt

    # Rule 3: default
    return DEFAULT_INTERVAL, args


def _interval_to_cron(interval: str) -> tuple[str, str]:
    """Convert an interval string to (cron_expr, human_readable). Returns (expr, note)."""
    match = re.match(r'^(\d+)([smhd])$', interval)
    if not match:
        raise ValueError(f"Invalid interval format: {interval!r}. Use Ns, Nm, Nh, or Nd.")

    n = int(match.group(1))
    unit = match.group(2)

    if unit == 's':
        # Round up to nearest minute
        minutes = max(1, math.ceil(n / 60))
        return f"*/{minutes} * * * *", f"every {minutes} minute(s) [rounded from {n}s]"
    elif unit == 'm':
        if n <= 59:
            return f"*/{n} * * * *", f"every {n} minute(s)"
        else:
            hours = n // 60
            return f"0 */{hours} * * *", f"every {hours} hour(s) [rounded from {n}m]"
    elif unit == 'h':
        if n <= 23:
            return f"0 */{n} * * *", f"every {n} hour(s)"
        else:
            days = n // 24
            return f"0 0 */{days} * *", f"every {days} day(s) [rounded from {n}h]"
    elif unit == 'd':
        return f"0 0 */{n} * *", f"every {n} day(s) at midnight"
    else:
        raise ValueError(f"Unknown unit: {unit}")


def _build_prompt(args: str) -> str:
    interval, prompt = _parse_interval_and_prompt(args)

    if not prompt:
        return USAGE_MESSAGE

    try:
        cron_expr, human_readable = _interval_to_cron(interval)
    except ValueError as e:
        return f"Error parsing interval: {e}\n\n{USAGE_MESSAGE}"

    return f"""# /loop — schedule a recurring prompt

Parsed: interval=`{interval}`, prompt=`{prompt}`

## Action

1. Call the CronCreateTool with:
   - `cron`: `{cron_expr}`
   - `prompt`: `{prompt}`
   - `recurring`: `true`

2. Briefly confirm:
   - What's scheduled
   - Cron expression: `{cron_expr}` ({human_readable})
   - That recurring tasks auto-expire after {DEFAULT_MAX_AGE_DAYS} days
   - That the user can cancel sooner with CronDeleteTool (include the job ID)

3. **Then immediately execute the parsed prompt now** — don't wait for the first cron fire. If it's a skill, invoke it via the skill tool; otherwise act on it directly.

## Prompt to execute now

{prompt}
"""


class LoopSkill(BaseSkill):
    """Run a prompt or skill on a recurring interval (e.g. 5m, 30m, 2h). Defaults to 10m."""

    @property
    def name(self) -> str:
        return "loop"

    @property
    def description(self) -> str:
        return (
            "Run a prompt or skill on a recurring interval. "
            "Use when the user wants to set up a recurring task, poll for status, "
            "or run something repeatedly on an interval (e.g. 'check the deploy every 5 minutes'). "
            "Do NOT invoke for one-off tasks. "
            "Usage: loop [interval] <prompt> — interval defaults to 10m."
        )

    def execute_impl(self, problem: str, **kwargs) -> str:
        trimmed = problem.strip()
        if not trimmed:
            return USAGE_MESSAGE
        return _build_prompt(trimmed)
