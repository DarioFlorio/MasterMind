# -*- coding: utf-8 -*-
"""
MasterMind -- Chainlit WebUI
============================
Install:  pip install chainlit
Run:      cd MasterMind && chainlit run chainlit_app.py --port 8000

Opens at http://localhost:8000
Features: streaming, thinking blocks, tool call display, skill display,
          voice input, file upload, dark mode, chat history.
"""
from __future__ import annotations
import os, sys, asyncio, threading
from pathlib import Path

# Make sure MasterMind root is on the path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import chainlit as cl

# ── MasterMind imports ────────────────────────────────────────────────────────
from config.settings import (
    MODEL_PATH, DIRECT_MODE, LLAMA_SERVER_URL, CONTEXT_SIZE,
    MAX_TOKENS, TEMPERATURE, PERMISSION_MODE, WORKING_DIR,
    N_THREADS, N_THREADS_BATCH, BATCH_SIZE, N_GPU_LAYERS,
)
from utils.model_client import ModelClient
from utils.permissions import PermissionManager
from utils.token_counter import SessionUsage
from agent.query_engine import QueryEngine
from agent.narrator_filter import _StreamingNarratorFilter
from agent.session import Session
from agent.task import AgentTool
from tools.bash_tool import BashTool
from tools.read_file_tool import ReadFileTool
from tools.write_file_tool import WriteFileTool
from tools.edit_file_tool import EditFileTool
from tools.glob_tool import GlobTool
from tools.grep_tool import GrepTool
from tools.list_dir_tool import ListDirTool
from tools.web_search_tool import WebSearchTool
from tools.web_fetch_tool import WebFetchTool
from tools.agent_tool import AgentTool
from tools.todo_tool import TodoWriteTool, TodoReadTool
from tools.memory_tool import MemoryWriteTool, MemoryReadTool
from tools.skill_tool import SkillTool
from tools.pm_tool import PMTool
from tools.scratchpad_tool import ScratchpadTool
from tools.reflect_tool import ReflectTool
from tools.git_tool import GitTool
from tools.export_tool import ExportTool
from tools.journal_tool import JournalTool
from tools.test_runner_tool import TestRunnerTool


# ── Shared model client (loaded once, reused across sessions) ─────────────────
_client: ModelClient | None = None
_client_lock = threading.Lock()

def _get_client() -> ModelClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = ModelClient(
                    base_url=LLAMA_SERVER_URL,
                    direct=DIRECT_MODE,
                    model_path=MODEL_PATH,
                )
    return _client


def _build_tools(cwd: str) -> list:
    return [
        BashTool(working_dir=cwd),
        ReadFileTool(working_dir=cwd),
        WriteFileTool(working_dir=cwd),
        EditFileTool(working_dir=cwd),
        GlobTool(working_dir=cwd),
        GrepTool(working_dir=cwd),
        ListDirTool(working_dir=cwd),
        WebSearchTool(), WebFetchTool(), AgentTool(),
        TodoWriteTool(), TodoReadTool(),
        MemoryWriteTool(), MemoryReadTool(),
        SkillTool(), PMTool(), ScratchpadTool(), ReflectTool(),
        GitTool(), ExportTool(), JournalTool(), TestRunnerTool(),
    ]


# ── Session startup ───────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_start():
    """Called once per browser session. Build the QueryEngine for this user."""
    cwd = WORKING_DIR or str(ROOT)
    client = _get_client()
    perms  = PermissionManager(PERMISSION_MODE)
    usage  = SessionUsage()
    tools  = _build_tools(cwd)

    engine = QueryEngine(
        tools=tools,
        client=client,
        session=Session(model_client=client),
        permission_manager=perms,
        usage=usage,
        working_dir=cwd,
        verbose=False,
        on_tool_start=None,   # replaced per-message below
        on_tool_end=None,
        on_chunk=None,
    )

    # Store engine in Chainlit session (per-user)
    cl.user_session.set("engine", engine)
    cl.user_session.set("usage", usage)

    await cl.Message(
        content=(
            "**MasterMind is ready.**\n\n"
            f"Model: `{Path(MODEL_PATH).stem}` | "
            f"Context: {CONTEXT_SIZE:,} tokens | "
            f"CWD: `{cwd}`\n\n"
            "Type anything to start. I can search the web, write and run code, "
            "manage files, and reason with 36 skills."
        ),
        author="MasterMind",
    ).send()


# ── Main message handler ──────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    engine: QueryEngine = cl.user_session.get("engine")

    # --- Streaming state ---
    # We build three Chainlit objects per turn:
    #   1. thinking_msg  -- the pale reasoning block (collapsible)
    #   2. active_step   -- current tool call (shown as a Step)
    #   3. response_msg  -- the final assistant answer

    thinking_content  = ""
    thinking_msg      = None
    response_msg      = cl.Message(content="", author="MasterMind")
    in_think          = False
    in_tool_use       = False
    think_buf         = ""
    tool_step: cl.Step | None = None
    tool_steps: dict[str, cl.Step] = {}

    # Send the response message immediately so streaming appears
    await response_msg.send()

    # Narrator-bleed streaming filter (per message)
    _cl_narrator_stream = _StreamingNarratorFilter()

    # --- Chunk callback: fires for every token ---
    def on_chunk(chunk: str):
        nonlocal in_think, in_tool_use, think_buf, thinking_msg, thinking_content

        # Accumulate into buffer and parse out <think> and <tool_use> blocks
        think_buf += chunk
        output_buf = ""

        while think_buf:
            if in_tool_use:
                # Swallow everything until </tool_use>
                idx = think_buf.find("</tool_use>")
                if idx == -1:
                    think_buf = ""  # still inside, discard
                    break
                else:
                    think_buf = think_buf[idx + 11:]
                    in_tool_use = False
            elif in_think:
                idx = think_buf.find("</think>")
                if idx == -1:
                    thinking_content += think_buf
                    think_buf = ""
                else:
                    thinking_content += think_buf[:idx]
                    think_buf = think_buf[idx + 8:]
                    in_think = False
            else:
                # Look for whichever tag opens first
                t_idx  = think_buf.find("<think>")
                tu_idx = think_buf.find("<tool_use>")
                # pick earliest
                first = -1
                if t_idx != -1 and (tu_idx == -1 or t_idx < tu_idx):
                    first = t_idx
                    tag_len = 7
                    flag = "think"
                elif tu_idx != -1:
                    first = tu_idx
                    tag_len = 10
                    flag = "tool_use"
                else:
                    flag = None

                if flag is None:
                    output_buf += think_buf
                    think_buf = ""
                else:
                    output_buf += think_buf[:first]
                    think_buf = think_buf[first + tag_len:]
                    if flag == "think":
                        in_think = True
                    else:
                        in_tool_use = True

        # Filter narrator bleed, then stream visible output to UI
        if output_buf:
            filtered = _cl_narrator_stream.feed(output_buf)
            if filtered:
                asyncio.run_coroutine_threadsafe(
                    response_msg.stream_token(filtered),
                    asyncio.get_event_loop(),
                )

    # --- Tool start callback ---
    def on_tool_start(name: str, inp: dict):
        nonlocal tool_step
        # Summarise the tool call in one line
        if   name == "bash":       summary = inp.get("command", "")[:80]
        elif name in ("read_file","write_file","edit_file"): summary = inp.get("path","")
        elif name == "web_search": summary = inp.get("query","")[:60]
        elif name == "web_fetch":  summary = inp.get("url","")[:60]
        elif name == "skill":
            sn   = inp.get("skill","")
            prob = (inp.get("args") or {}).get("problem","")[:50]
            summary = f"{sn} | {prob}" if prob else sn
        elif name == "grep":       summary = f"'{inp.get('pattern','')}'"
        elif name == "glob":       summary = inp.get("pattern","")
        else:                      summary = str(inp)[:60]

        step = cl.Step(name=f"{name}", type="tool", show_input=True)
        step.input = summary
        tool_steps[name] = step
        tool_step = step
        asyncio.run_coroutine_threadsafe(step.__aenter__(), asyncio.get_event_loop())

    # --- Tool end callback ---
    def on_tool_end(name: str, result):
        step = tool_steps.get(name) or tool_step
        if step:
            out = (result.output or "").strip()[:400]
            step.output = out
            step.is_error = result.is_error
            asyncio.run_coroutine_threadsafe(
                step.__aexit__(None, None, None),
                asyncio.get_event_loop(),
            )

    # Attach callbacks
    engine.on_chunk      = on_chunk
    engine.on_tool_start = on_tool_start
    engine.on_tool_end   = on_tool_end

    # --- Run the engine in a thread (it's blocking) ---
    loop = asyncio.get_event_loop()
    reply = await loop.run_in_executor(
        None,
        engine.submit_message,
        message.content,
    )

    # --- Finalise response ---
    # Flush any buffered narrator tail and merge with reply
    narrator_tail = _cl_narrator_stream.flush()
    if narrator_tail:
        reply = reply + narrator_tail if reply else narrator_tail
    response_msg.content = reply
    await response_msg.update()

    # --- Show thinking block if there was one ---
    if thinking_content.strip():
        await cl.Message(
            content=thinking_content.strip(),
            author="thinking",
            parent_id=response_msg.id,
        ).send()