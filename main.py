#!/usr/bin/env python3
"""
main.py — MasterMind: Fully merged agent harness.
MasterMind Harness — unified agent system
  - MCP (Model Context Protocol) servers
  - Plugin system with marketplace
  - Multi-agent swarm (in-process + tmux)
  - Hook system (pre/post events on every action)
  - Telemetry + performance tracing
  - Secure storage (keychain + JSON fallback)
  - Bridge server (remote HTTP/SSE access)
  - Plan mode (propose-only, user-approved execution)
  - Git worktree support for parallel branch work
  - LSP code intelligence (definitions, symbols, diagnostics)
  - Jupyter notebook editing
  - PowerShell execution (cross-platform)
  - Cron scheduling with persistence
  - Task lifecycle management (SQLite-backed)
  - Team/swarm management (in-process + tmux backends)
  - Sandbox execution (bubblewrap on Linux)
  - All original MasterMind reasoning skills (30+)
  - WhatsApp via Baileys (no Twilio, no accounts — scan QR once, works forever)
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

# ── Performance: prevent BLAS libraries from oversubscribing cores ──────
os.environ["OMP_NUM_THREADS"] = str(
    int(os.environ.get("OMP_NUM_THREADS", 1))
)
os.environ["MKL_NUM_THREADS"] = str(
    int(os.environ.get("MKL_NUM_THREADS", 1))
)

ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoinstall import ensure_dependencies
ensure_dependencies()

from config.settings import (
    LLAMA_SERVER_URL, LLAMA_SERVER_PORT, MODEL_PATH, MODEL_DISPLAY,
    MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR,
    CONTEXT_SIZE, MAX_TOKENS, DIRECT_MODE, BASH_TIMEOUT_S,
    UNLIMITED_CONTEXT, IDLE_CONSOLIDATION_S, DAILY_DIGEST_ENABLED,
    # ── New performance knobs ──────────────────────────────────────────
    DRAFT_MODEL_PATH, DRAFT_P_MIN, NO_KV_OFFLOAD, UBATCH_SIZE,
    CPU_MASK, NO_MMAP, USE_MLOCK, NO_PERF,
)

# Core agent
from agent.query_engine import QueryEngine
from agent.session import Session
from agent.ultraplan import UltraPlan, should_ultraplan
from agent.narrator_filter import _StreamingNarratorFilter
from agent.context_budget import ContextBudget

# Tools - original
from tools.pm_tool          import PMTool
from tools.bash_tool        import BashTool
from tools.read_file_tool   import ReadFileTool
from tools.write_file_tool  import WriteFileTool
from tools.edit_file_tool   import EditFileTool
from tools.glob_tool        import GlobTool
from tools.grep_tool        import GrepTool
from tools.list_dir_tool    import ListDirTool
from tools.web_search_tool  import WebSearchTool
from tools.web_fetch_tool   import WebFetchTool
from tools.agent_tool       import AgentTool
from tools.todo_tool        import TodoWriteTool, TodoReadTool
from tools.memory_tool      import MemoryWriteTool, MemoryReadTool
from tools.skill_tool       import SkillTool
from tools.git_tool         import GitTool
from tools.scratchpad_tool  import ScratchpadTool
from tools.reflect_tool     import ReflectTool
from tools.export_tool      import ExportTool, export_session
from tools.journal_tool     import JournalTool
from tools.test_runner_tool import TestRunnerTool
# GAP: User-specific vector store search
from tools.memory_search_tool import MemorySearchTool
from tools.wiki_tool import (
    WikiWriteTool, WikiReadTool, WikiSearchTool,
    WikiListTool, WikiPromoteTool,
)
from skills.self_healing import SelfHealingTool
from skills.code_remediation import CodeRemediationTool

# Tools - Harness extensions
from tools.lsp_tool          import LSPTool
from tools.notebook_tool     import NotebookTool
from tools.powershell_tool   import PowerShellTool
from tools.cron_tool         import CronCreateTool, CronListTool, CronDeleteTool
from tools.ask_user_tool     import AskUserTool
from tools.sleep_tool        import SleepTool
from tools.send_message_tool import SendMessageTool, ReceiveMessageTool
from tools.tool_search_tool  import ToolSearchTool, register_all as _register_tool_search
from tools.mcp_tool          import MCPInvokeTool, MCPListServersTool
from tools.plan_mode_tool    import EnterPlanModeTool, ExitPlanModeTool
from tools.worktree_tool     import EnterWorktreeTool, ExitWorktreeTool, WorktreeListTool
from tools.team_tool         import TeamCreateTool, TeamDeleteTool, TeamStatusTool
from tools.task_tool         import (TaskCreateTool, TaskGetTool, TaskListTool,
                                      TaskUpdateTool, TaskStopTool)
from tools.remote_trigger_tool import RemoteTriggerTool
from tools.sandbox_tool      import SandboxTool
from tools.brief_tool        import BriefTool
from tools.structured_output_tool import StructuredOutputTool
from tools.task_output_tool       import TaskOutputTool
from tools.whatsapp_tool          import WhatsAppSendTool
WhatsAppSendTool.set_owner("447447148024")  # permanent default — overwritten by bridge on connect

# Services (ported from src)
from services.magic_docs        import magic_docs, MagicDocs
from services.session_memory    import session_memory, SessionMemory
from services.away_summary      import AwaySummary
from services.extract_memories  import extract_memories, ExtractMemories
from services.team_memory_sync  import team_memory_sync, TeamMemorySync
from services.tool_use_summary  import generate_tool_use_summary

# Infrastructure
from utils.model_client  import ModelClient, ThinkingStreamParser
from utils.permissions   import PermissionManager
from utils.token_counter import SessionUsage
from utils.episode_log   import ep as _ep
from heartbeat           import Heartbeat
from memory.autodream    import AutoDream
from kairos              import Kairos, write_daemon_script
from hooks.manager       import hook_manager
from mcp.registry        import mcp_registry
from plugins.manager     import plugin_manager
from swarm.team          import team_manager
from telemetry.logger    import TelemetryLogger
from telemetry.tracer    import tracer
from bridge.server       import BridgeServer

_voice_session = None
def _get_voice():
    global _voice_session
    if _voice_session is None:
        from voice import VoiceSession
        _voice_session = VoiceSession()
    return _voice_session

_bridge: BridgeServer | None = None
_telemetry: TelemetryLogger | None = None
_server_proc = None

import logging
def _setup_logging(verbose):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s [%(name)s] %(message)s", stream=sys.stderr)
    if not verbose:
        for n in ("httpx","httpcore","llama_cpp","urllib3"): logging.getLogger(n).setLevel(logging.ERROR)

def _tty(): return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
_C = _tty()
RESET="\033[0m" if _C else ""; BOLD="\033[1m" if _C else ""; DIM="\033[2m" if _C else ""
WHITE="\033[97m" if _C else ""; CYAN="\033[96m" if _C else ""; GREEN="\033[92m" if _C else ""
RED="\033[91m" if _C else ""; YELLOW="\033[93m" if _C else ""; DIM_C="\033[2;36m" if _C else ""
THINK="\033[3;38;5;117m" if _C else ""; THINK_HEAD="\033[38;5;67m" if _C else ""
THINK_SEP="\033[38;5;237m" if _C else ""; TOOL_NAME="\033[1;96m" if _C else ""
TOOL_DIM="\033[38;5;244m" if _C else ""
ICON_THINK="◈"; ICON_TOOL="▸"; ICON_OK="◆"; ICON_ERR="✖"; ICON_PROMPT="❯"; ICON_CACHED="⟲"

class _SS:
    def __init__(self):
        self.first=True; self.spin_done=threading.Event(); self.chunks=0
        self.active=False; self.lock=threading.Lock()
        self.parser=ThinkingStreamParser(); self.in_think=False
        self.step=0; self.step_lock=threading.Lock()
        self.run_log: list[str]=[]
        self.t_step: float=0.0
_ss=_SS(); _spinner_cleared=threading.Event()
_main_narrator_stream=_StreamingNarratorFilter()

# ── Tool category colours ────────────────────────────────────────────────────
_CAT_COL = {
    "shell":  "\033[1;33m"  if _C else "",   # amber  — bash / powershell / sandbox
    "file":   "\033[1;34m"  if _C else "",   # blue   — read/write/edit/glob/grep/list
    "web":    "\033[1;35m"  if _C else "",   # violet — web_search / web_fetch
    "memory": "\033[1;32m"  if _C else "",   # green  — memory / journal / scratchpad
    "skill":  "\033[1;96m"  if _C else "",   # cyan   — skill dispatcher
    "agent":  "\033[1;37m"  if _C else "",   # white  — agent / team / task
    "other":  "\033[1;90m"  if _C else "",   # grey   — everything else
}
_STEP_GLYPHS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

def _step_glyph(n: int) -> str:
    return _STEP_GLYPHS[n-1] if 1 <= n <= len(_STEP_GLYPHS) else f"({n})"

def _tool_cat(name: str) -> str:
    if name in ("bash","powershell","sandbox"):                               return "shell"
    if name in ("read_file","write_file","edit_file","glob","grep",
                "list_dir","git"):                                            return "file"
    if name in ("web_search","web_fetch"):                                    return "web"
    if name in ("memory_write","memory_read","journal","scratchpad","reflect"):return "memory"
    if name == "skill":                                                        return "skill"
    if name in ("agent","team_create","team_delete","team_status",
                "task_create","task_get","task_list","task_update",
                "task_stop"):                                                  return "agent"
    return "other"

def _spinner():
    fr=["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]; i=0
    while not _ss.spin_done.wait(timeout=0.07):
        sys.stdout.write(f"\r{THINK_HEAD}{fr[i%len(fr)]}  reasoning…{RESET}"); sys.stdout.flush(); i+=1
    sys.stdout.write("\r"+" "*40+"\r"); sys.stdout.flush(); _spinner_cleared.set()

_THINK_RULE=f"{THINK_SEP}{'─'*64}{RESET}"

def _on_chunk(chunk):
    with _ss.lock: _ss.chunks+=1; _ss.active=True
    if _ss.first:
        _ss.first=False; _ss.spin_done.set(); _spinner_cleared.wait(timeout=0.15)
    for text,is_think in _ss.parser.feed(chunk):
        if not text: continue
        if is_think:
            if not _ss.in_think:
                sys.stdout.write(f"\n{THINK_HEAD}{ICON_THINK} thinking{RESET}\n{_THINK_RULE}\n{THINK}"); _ss.in_think=True
            sys.stdout.write(f"{THINK}{text}")
        else:
            if _ss.in_think:
                sys.stdout.write(f"{RESET}\n{_THINK_RULE}\n\n{DIM_C}{ICON_PROMPT}{RESET}  "); _ss.in_think=False
            sys.stdout.write(f"{WHITE}{text}{RESET}")
    sys.stdout.flush()

def _flush_parser():
    for text,is_think in _ss.parser.flush():
        if text: sys.stdout.write(f"{THINK if is_think else WHITE}{text}{RESET}")
    if _ss.in_think: sys.stdout.write(f"{RESET}\n{_THINK_RULE}\n"); _ss.in_think=False
    sys.stdout.flush()

def _reset_ss():
    _ss.first=True; _ss.spin_done.clear(); _spinner_cleared.clear()
    _ss.chunks=0; _ss.active=False; _ss.parser=ThinkingStreamParser(); _ss.in_think=False
    with _ss.step_lock: _ss.step=0; _ss.run_log.clear()

def _on_tool_start(name, inp):
    hook_manager.fire("tool_use:pre", name=name, inp=inp)
    if _telemetry: _telemetry.tool_use(tool=name)
    _ep.on_tool_start(name, inp)

    with _ss.step_lock:
        _ss.step += 1
        n = _ss.step
    _ss.t_step = time.time()

    col  = _CAT_COL[_tool_cat(name)]
    icon = _step_glyph(n)

    if name == "bash":
        first = (inp.get("command","").strip().splitlines() or [""])[0][:80]
        detail = f"{TOOL_DIM}{first}{RESET}"
    elif name == "powershell":
        detail = f"{TOOL_DIM}{inp.get('command','')[:80]}{RESET}"
    elif name in ("read_file","write_file","edit_file"):
        op = {"read_file":"read","write_file":"write","edit_file":"edit"}[name]
        detail = f"{TOOL_DIM}{op} {inp.get('path','')}{RESET}"
    elif name in ("glob","grep"):
        detail = f"{TOOL_DIM}{inp.get('pattern','') or inp.get('query','')}{RESET}"
    elif name == "list_dir":
        detail = f"{TOOL_DIM}{inp.get('path','.')}{RESET}"
    elif name == "git":
        detail = f"{TOOL_DIM}{inp.get('command','')[:60]}{RESET}"
    elif name == "web_search":
        detail = f"{TOOL_DIM}{inp.get('query','')[:70]}{RESET}"
    elif name == "web_fetch":
        detail = f"{TOOL_DIM}{inp.get('url','')[:70]}{RESET}"
    elif name == "skill":
        sk   = inp.get("skill","?")
        prob = (inp.get("args",{}).get("problem") or "")[:55]
        detail = f"{TOOL_DIM}{sk}{RESET}" + (f"  {DIM}│ {prob}{RESET}" if prob else "")
    elif name in ("memory_write","memory_read"):
        detail = f"{TOOL_DIM}{inp.get('key','') or inp.get('query','')}{RESET}"
    elif name == "agent":
        detail = f"{TOOL_DIM}{inp.get('prompt','')[:60]}{RESET}"
    elif name == "task_create":
        detail = f"{TOOL_DIM}{inp.get('title','')[:60]}{RESET}"
    elif name == "team_create":
        detail = f"{TOOL_DIM}team={inp.get('name','')} agents={len(inp.get('agents',[]))}{RESET}"
    elif name == "sandbox":
        detail = f"{TOOL_DIM}lang={inp.get('language','')}{RESET}"
    elif name == "mcp":
        detail = f"{TOOL_DIM}{inp.get('server','')}.{inp.get('tool','')}{RESET}"
    else:
        detail = f"{TOOL_DIM}{str(inp)[:60]}{RESET}"

    prefix = ""
    if _ss.in_think:
        prefix = f"{RESET}\n{_THINK_RULE}\n"
        _ss.in_think = False

    print(f"{prefix}\n  {col}{icon} {name}{RESET}  {detail}", flush=True)

    with _ss.step_lock:
        _ss.run_log.append(name)

def _on_tool_end(name, result):
    hook_manager.fire("tool_use:post", name=name, result=result)
    if result.is_error and _telemetry: _telemetry.tool_error(tool=name, error=result.output[:100])
    _ep.on_tool_end(name, result)

    elapsed = time.time() - _ss.t_step
    t_str = f"  {TOOL_DIM}{elapsed:.1f}s{RESET}" if elapsed >= 0.5 else ""
    MAX_SNIP = 280

    if result.is_error:
        snip = (result.output or "").strip()[:MAX_SNIP]
        print(f"  {RED}{ICON_ERR} {snip}{RESET}{t_str}", flush=True)
        return

    out = (result.output or "").strip()
    cached = out.startswith("[cached]")
    if cached:
        out = out[8:].strip()
        ok = f"{TOOL_DIM}{ICON_CACHED}{RESET}"
    else:
        ok = f"{GREEN}{ICON_OK}{RESET}"

    lines = out.splitlines()

    if name == "skill" and len(lines) > 2:
        head = "\n    ".join(lines[:8])
        tail = f"\n    {DIM}… ({len(lines)} lines){RESET}" if len(lines) > 8 else ""
        print(f"  {ok}  {DIM}{head}{tail}{RESET}{t_str}", flush=True)
    elif name == "read_file":
        count = f"{DIM}({len(lines)} lines){RESET}  " if len(lines) > 2 else ""
        preview = "  ".join(lines[:2])[:MAX_SNIP]
        print(f"  {ok}  {count}{DIM}{preview}{RESET}{t_str}", flush=True)
    elif name in ("bash","powershell","sandbox"):
        snip = "\n    ".join(lines[:4])[:MAX_SNIP]
        more = f"\n    {DIM}… ({len(lines)} lines){RESET}" if len(lines) > 4 else ""
        print(f"  {ok}  {DIM}{snip}{more}{RESET}{t_str}", flush=True)
    else:
        snip = "  ".join(lines[:2])[:MAX_SNIP]
        print(f"  {ok}  {DIM}{snip}{RESET}{t_str}", flush=True)

def _find_server():
    for name in ("llama-server","llama-server.exe","server","server.exe"):
        found=shutil.which(name)
        if found: return found
    return None

def _healthy(url=LLAMA_SERVER_URL):
    try:
        import httpx; return httpx.get(f"{url}/health",timeout=2).status_code==200
    except: return False

def _start_server():
    global _server_proc
    if _healthy(): return True
    if not MODEL_PATH or MODEL_PATH.strip().lower() == "auto":
        return False
    binary=_find_server()
    if not binary: print(f"{YELLOW}[server] llama-server not found{RESET}"); return False
    model=Path(MODEL_PATH)
    if not model.exists(): print(f"{RED}[server] Model not found: {model}{RESET}"); return False
    cpu=os.cpu_count() or 4; nt=max(1,cpu//2); ngl=int(os.environ.get("N_GPU_LAYERS","0"))
    mmproj=os.environ.get("MMPROJ_PATH","")
    cmd=[binary,"-m",str(model),"--port",str(LLAMA_SERVER_PORT),"--host","127.0.0.1",
         "--ctx-size",str(CONTEXT_SIZE),"-ngl",str(ngl),"-t",str(nt),"--batch-size","512",
         "--cont-batching","--mmap"]

    # ── New performance flags ────────────────────────────────────────────
    if DRAFT_MODEL_PATH and Path(DRAFT_MODEL_PATH).exists():
        cmd += ["--draft-model", str(DRAFT_MODEL_PATH), "--draft-p-min", str(DRAFT_P_MIN)]
        print(f"{GREEN}[server] Speculative decoding draft: {Path(DRAFT_MODEL_PATH).name}{RESET}")
    if NO_KV_OFFLOAD:
        cmd.append("--no-kv-offload")
    if UBATCH_SIZE:
        cmd += ["--ubatch-size", str(UBATCH_SIZE)]
    if CPU_MASK:
        cmd += ["--cpu-mask", CPU_MASK]
    if NO_MMAP:
        cmd.append("--no-mmap")
        cmd.append("--mlock")  # always pair with mlock to avoid swapping
    else:
        # original mmap is already there, add mlock anyway
        cmd.append("--mlock")
    if NO_PERF:
        cmd.append("--no-perf")
    cmd.append("--simple-io")       # reduce terminal overhead
    cmd.append("--log-disable")     # silence internal logging
    # ──────────────────────────────────────────────────────────────────────

    if mmproj and Path(mmproj).exists():
        cmd+=["--mmproj", mmproj]
        print(f"{GREEN}[server] Vision enabled: {Path(mmproj).name}{RESET}")
    elif mmproj:
        print(f"{RED}[server] ✖ mmproj file not found: {mmproj}{RESET}")
        print(f"{YELLOW}[server] Vision disabled. Download the mmproj from the same HuggingFace repo as your model.{RESET}")
    else:
        print(f"{DIM}[server] No mmproj set — vision disabled. Set MMPROJ_PATH in .env to enable.{RESET}")
    kwargs={"creationflags":subprocess.CREATE_NO_WINDOW} if sys.platform=="win32" else {}
    try: _server_proc=subprocess.Popen(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,**kwargs)
    except Exception as e: print(f"{RED}[server] Launch failed: {e}{RESET}"); return False
    print(f"{DIM}[server] Starting",end="",flush=True)
    deadline=time.time()+90
    while time.time()<deadline:
        time.sleep(1.5); print(".",end="",flush=True)
        if _server_proc.poll() is not None: print(f"\n{RED}[server] Crashed{RESET}"); return False
        if _healthy(): print(f" {GREEN}ready!{RESET}"); return True
    print(f"\n{RED}[server] Timed out{RESET}"); return False

def _stop_server():
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try: _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: _server_proc.kill()
        _server_proc=None

def _build_tools(cwd):
    return [
        BashTool(working_dir=cwd), ReadFileTool(working_dir=cwd), WriteFileTool(working_dir=cwd),
        EditFileTool(working_dir=cwd), GlobTool(working_dir=cwd), GrepTool(working_dir=cwd),
        ListDirTool(working_dir=cwd), WebSearchTool(), WebFetchTool(), AgentTool(), SkillTool(),
        ReflectTool(), ScratchpadTool(), TodoWriteTool(), TodoReadTool(), MemoryWriteTool(),
        MemoryReadTool(), JournalTool(), ExportTool(), GitTool(), TestRunnerTool(), PMTool(),
        LSPTool(), NotebookTool(), PowerShellTool(), SandboxTool(),
        CronCreateTool(), CronListTool(), CronDeleteTool(),
        AskUserTool(), SleepTool(), SendMessageTool(), ReceiveMessageTool(), BriefTool(),
        EnterPlanModeTool(), ExitPlanModeTool(),
        EnterWorktreeTool(), ExitWorktreeTool(), WorktreeListTool(),
        MCPInvokeTool(), MCPListServersTool(),
        TeamCreateTool(), TeamDeleteTool(), TeamStatusTool(),
        TaskCreateTool(), TaskGetTool(), TaskListTool(), TaskUpdateTool(), TaskStopTool(),
        RemoteTriggerTool(), ToolSearchTool(),
        StructuredOutputTool(), TaskOutputTool(),
        MemorySearchTool(working_dir=cwd),
        WikiWriteTool(working_dir=cwd), WikiReadTool(working_dir=cwd),
        WikiSearchTool(working_dir=cwd), WikiListTool(working_dir=cwd),
        WikiPromoteTool(working_dir=cwd),
        SelfHealingTool(working_dir=cwd),
        CodeRemediationTool(working_dir=cwd),
        WhatsAppSendTool(),
    ]

def _make_factory(client, perms, usage, cwd, verbose):
    def factory(max_turns=10, is_subagent=True):
        return QueryEngine(
            tools=_build_tools(cwd), client=client, session=Session(),
            permission_manager=perms, usage=usage, max_turns=max_turns,
            working_dir=cwd, verbose=verbose, is_subagent=is_subagent,
            on_tool_start=_on_tool_start, on_tool_end=_on_tool_end, on_chunk=_on_chunk)
    return factory

def _submit(engine, text):
    _reset_ss(); sys.stdout.write(f"\n{BOLD}>{RESET} "); sys.stdout.flush()
    t=threading.Thread(target=_spinner,daemon=True); t.start()
    result=None; err=None
    def _run():
        nonlocal result,err
        try: result=engine.submit_message(text)
        except Exception as e: err=e
    th=threading.Thread(target=_run,daemon=True); th.start(); th.join()
    _flush_parser(); _ss.spin_done.set(); t.join(timeout=0.5)
    if err: print(f"\n{RED}Error: {err}{RESET}",file=sys.stderr); return ""
    if not _ss.chunks and result: sys.stdout.write(result); sys.stdout.flush()
    with _ss.step_lock:
        log = list(_ss.run_log)
    if log:
        summary_parts: list[str] = []
        i = 0
        while i < len(log):
            name = log[i]; count = 1
            while i + count < len(log) and log[i+count] == name: count += 1
            col = _CAT_COL[_tool_cat(name)]
            part = f"{col}{name}{RESET}"
            if count > 1: part += f"{DIM} ×{count}{RESET}"
            summary_parts.append(part)
            i += count
        steps_word = "step" if len(log) == 1 else "steps"
        print(f"\n{TOOL_DIM}  ╶── ran {len(log)} {steps_word}: {'  '.join(summary_parts)}{RESET}\n", flush=True)
    return result or ""

def _handle_slash(cmd, engine, usage, perms, vs=None):
    low=cmd.strip().lower()
    if low in ("/clear","/reset"):
        engine.session.clear(); engine.invalidate_system_prompt()
        print(f"{DIM}Session cleared.{RESET}"); return True
    if low in ("/status","/cost"):
        print(f"{DIM}{usage.summary()}\nMessages: {len(engine.session)}{RESET}")
        if _telemetry: print(f"{DIM_C}{_telemetry.summary()}{RESET}")
        return True
    if low.startswith("/compact"):
        instruction = cmd[8:].strip()
        try:
            from compaction.compactor import SessionCompactor, CompactionConfig
            msgs = [{"role": m.role, "content": m.content} for m in engine.session._messages]
            if not msgs:
                print(f"{DIM}No messages to compact.{RESET}"); return True
            def _llm_fn(prompt: str, model=None) -> str:
                import threading
                result_box = [None]; err_box = [None]
                def _run():
                    import httpx, json as _json
                    try:
                        r = httpx.post(f"{LLAMA_SERVER_URL}/completion",
                            json={"prompt": prompt, "n_predict": 512, "temperature": 0.3},
                            timeout=60)
                        result_box[0] = r.json().get("content", "")
                    except Exception as e:
                        err_box[0] = e
                t = threading.Thread(target=_run, daemon=True); t.start(); t.join(timeout=90)
                if err_box[0]: raise err_box[0]
                return result_box[0] or ""
            config = CompactionConfig(notify_user=True, min_tail_turns=4)
            compactor = SessionCompactor(llm_fn=_llm_fn, config=config)
            result = compactor.compact(msgs, context_limit=engine.session._context_size or 32000,
                                      instruction=instruction)
            engine.session.clear()
            for m in result.compacted_messages:
                engine.session.add_message(m.get("role", "user"), m.get("content", ""))
            hook_manager.fire("compact")
            print(f"{DIM}Context compacted: ~{result.tokens_before:,} → ~{result.tokens_after:,} tokens "
                  f"({result.reduction_pct:.0f}% reduction){RESET}")
        except Exception as e:
            engine.session._unlimited = True; engine.session._maybe_compress()
            hook_manager.fire("compact"); print(f"{DIM}Context compacted. ({e}){RESET}")
        return True
    if low.startswith("/mode "):
        nm=cmd[6:].strip()
        try: perms.set_mode(nm); print(f"{DIM}Permission mode → {nm}{RESET}")
        except: print(f"{RED}Modes: auto | ask | deny{RESET}")
        return True
    if low.startswith("/save"):
        parts=cmd.split(maxsplit=1); p=Path(parts[1]) if len(parts)>1 else Path("session.json")
        engine.session.save(p); print(f"{DIM}Saved → {p}{RESET}"); return True
    if low.startswith("/load"):
        parts=cmd.split(maxsplit=1)
        if len(parts)<2: print(f"{RED}Usage: /load <file.json>{RESET}"); return True
        p=Path(parts[1])
        if not p.exists(): print(f"{RED}Not found: {p}{RESET}"); return True
        engine.session=Session.load(p,engine.client); engine.invalidate_system_prompt()
        print(f"{DIM}Loaded {p} ({len(engine.session)} msgs){RESET}"); return True
    if low.startswith("/memory"):
        try:
            from memory.manager import load_context; print(load_context() or "(no memories)")
        except Exception as e: print(f"{RED}{e}{RESET}")
        return True
    if low in ("/skills","/skill"):
        st=next((t for t in engine.tools.values() if t.name=="skill"),None)
        if st: print(f"\n{CYAN}{st.execute({}).output}{RESET}")
        return True
    if low.startswith("/skill "):
        rest=cmd[7:].strip(); parts=rest.split(maxsplit=1)
        sname=parts[0] if parts else ""; sprob=parts[1] if len(parts)>1 else ""
        st=next((t for t in engine.tools.values() if t.name=="skill"),None)
        if st:
            r=st.execute({"skill":sname,"args":{"problem":sprob or sname}})
            print(f"\n{RED if r.is_error else ''}{r.output}{RESET}")
        return True
    if low=="/mcp":
        print(f"{DIM}MCP servers: {mcp_registry.list_servers() or ['(none)']}{RESET}"); return True
    if low.startswith("/mcp add "):
        parts=cmd.split(maxsplit=3)
        if len(parts)>=4: mcp_registry.add(parts[2],parts[3]); print(f"{GREEN}MCP: {parts[2]} added{RESET}")
        return True
    if low=="/plugins":
        pl=plugin_manager.list_plugins()
        if pl:
            for p in pl: print(f"  {GREEN}✓{RESET} {p['name']} v{p['version']} — {p['description']}")
        else: print(f"{DIM}No plugins loaded. Place dirs in ~/.mastermind/plugins/{RESET}")
        return True
    if low.startswith("/plugin install "):
        name=cmd.split(maxsplit=2)[2]
        from plugins.marketplace import Marketplace
        ok=Marketplace().install(name)
        print(f"{GREEN}Installed {name}{RESET}" if ok else f"{RED}Failed{RESET}"); return True
    if low=="/teams":
        teams=team_manager.list_teams()
        print(json.dumps(teams,indent=2) if teams else f"{DIM}No active teams{RESET}"); return True
    if low.startswith("/tasks"):
        from tools.task_tool import TaskListTool
        parts=cmd.split(maxsplit=2); status=parts[1] if len(parts)>1 else ""
        r=TaskListTool().execute({"status":status} if status else {}); print(r.output); return True
    if low=="/hooks":
        hooks=hook_manager.list_hooks()
        for ev,fns in hooks.items(): print(f"  {CYAN}{ev}{RESET}: {', '.join(fns)}")
        if not hooks: print(f"{DIM}No hooks registered{RESET}")
        return True
    if low.startswith("/bridge"):
        global _bridge
        parts=cmd.split(); sub=parts[1].lower() if len(parts)>1 else "start"
        port=int(parts[2]) if len(parts)>2 and parts[2].isdigit() else 7777
        if sub=="start":
            if not _bridge:
                _bridge=BridgeServer(engine=engine,port=port); _bridge.start()
                print(f"{GREEN}Bridge on port {port}{RESET}")
            else: print(f"{DIM}Already running{RESET}")
        elif sub=="stop":
            if _bridge: _bridge.stop(); _bridge=None; print(f"{DIM}Bridge stopped{RESET}")
        elif sub=="status": print(f"{DIM}Bridge: {'running' if _bridge else 'stopped'}{RESET}")
        return True
    if low=="/telemetry":
        if _telemetry: print(f"{DIM}{_telemetry.summary()}{RESET}\n{DIM_C}{tracer.report()}{RESET}")
        return True
    if low=="/plan":
        from tools.plan_mode_tool import is_plan_mode,get_plan
        if is_plan_mode():
            print(f"{CYAN}Plan mode ACTIVE{RESET}")
            for i,s in enumerate(get_plan(),1): print(f"  {i}. {s}")
        else: print(f"{DIM}Plan mode inactive{RESET}")
        return True
    if low.startswith("/voice"):
        _vs=vs or _get_voice(); parts=cmd.strip().split()
        sub=parts[1].lower() if len(parts)>1 else "toggle"
        if sub=="on": _vs.enable()
        elif sub=="off": _vs.disable()
        elif sub=="backend" and len(parts)>2: _vs.set_backend(parts[2].lower())
        elif sub=="backends":
            from voice import VoiceSession; print(VoiceSession.list_backends())
        else: _vs.toggle()
        return True
    # ── WhatsApp slash command ──────────────────────────────────────────────
    if low == "/whatsapp":
        _wa.activate()
        return True
    if low == "/whatsapp off":
        _wa.deactivate()
        return True
    if low == "/whatsapp status":
        _wa.status()
        return True
    # ───────────────────────────────────────────────────────────────────────
    _HARNESS = {"/prose":("prose",6),"/memory":("memory",7),"/approve":("approve",8),
                "/discover":("discover",9),"/flow":("flow",5),"/canvas":("canvas",7)}
    for _pfx,(_key,_strip) in _HARNESS.items():
        if low == _pfx or low.startswith(_pfx + " "):
            import asyncio
            from harness_commands import register_harness_commands
            _hc: dict = {}
            register_harness_commands(_hc, engine=engine)
            _result = asyncio.run(_hc[_key](cmd[_strip:].strip(), session=engine.session))
            print(f"{CYAN}{_result}{RESET}")
            return True

    if low in ("/help","/?"):
        print(f"""{DIM}
Commands:
  /clear              Clear session history
  /compact            Compress context window (LLM-summarised)
  /load FILE          Restore session from JSON
  /save [FILE]        Save session to JSON
  /status             Token usage + telemetry
  /mode MODE          auto | ask | deny
  /memory [search|ingest|status|dream|clear]
  /prose [run|compile|examples] [FILE]
  /approve [list|add|log|check|status]
  /discover [SECONDS] mDNS local network scan
  /flow [list|show|cancel|delete] [ID]
  /canvas             A2UI widget demo
  /skills             List all reasoning skills
  /skill NAME [Q]     Run skill directly
  /tasks [STATUS]     List tasks
  /teams              List active agent teams
  /hooks              List registered event hooks
  /mcp                List MCP servers
  /mcp add NAME URL   Register an MCP server
  /plugins            List loaded plugins
  /plugin install X   Install from marketplace
  /bridge [start|stop|status] [PORT]
  /plan               Show current plan
  /telemetry          Show telemetry + trace
  /voice [on|off|toggle|backend B|backends]
  /whatsapp           Activate WhatsApp mirror mode
  /whatsapp off       Deactivate WhatsApp mirror mode
  /whatsapp status    Show WhatsApp bridge status
  /help               This help
  exit / quit         Exit{RESET}""")
        return True
    return False

def _banner(cwd, perm, direct):
    mode="llama-cpp (direct)" if direct else f"llama-server {LLAMA_SERVER_URL}"
    n_skills=len(list((ROOT/"skills").glob("*.py")))-1
    mcp_count=len(mcp_registry.list_servers())
    ctx_str="unlimited (sliding window)" if UNLIMITED_CONTEXT else f"{CONTEXT_SIZE:,} tokens"
    mmproj=os.environ.get("MMPROJ_PATH","")
    if mmproj and Path(mmproj).exists():
        vision_str=f"{GREEN}enabled ({Path(mmproj).name}){RESET}"
    elif mmproj and not Path(mmproj).exists():
        vision_str=f"{RED}mmproj not found: {mmproj}{RESET}"
    else:
        vision_str=f"{YELLOW}disabled — set MMPROJ_PATH in .env{RESET}"
    wa_status = (
        f"{GREEN}connected{RESET}"          if _wa.connected else
        f"{CYAN}waiting for QR scan…{RESET}" if _wa.started   else
        f"{YELLOW}standby — /whatsapp to connect{RESET}"
    )
    print(f"""
{BOLD}{CYAN}  ╭──────────────────────────────────────────────────────────╮{RESET}
{BOLD}{CYAN}  │{RESET}  {WHITE}{BOLD}EVE — Persistent Agent Harness{RESET}  {BOLD}{CYAN}│{RESET}
{BOLD}{CYAN}  ├──────────────────────────────────────────────────────────┤{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}model    {RESET}  {WHITE}{MODEL_DISPLAY}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}backend  {RESET}  {DIM}{mode}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}vision   {RESET}  {vision_str}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}context  {RESET}  {DIM}{ctx_str}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}skills   {RESET}  {GREEN}{n_skills} reasoning skills{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}tools    {RESET}  {GREEN}core + 25 new (MCP/swarm/LSP/cron/tasks…){RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}MCP      {RESET}  {DIM if not mcp_count else GREEN}{mcp_count} server(s) configured{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}memory   {RESET}  {GREEN}hybrid FTS+vector + dreaming{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}prose    {RESET}  {GREEN}workflow engine (.prose scripts){RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}approvals{RESET}  {GREEN}exec approval gating + allowlist{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}perms    {RESET}  {YELLOW}{perm}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}whatsapp {RESET}  {wa_status}
{BOLD}{CYAN}  ╰──────────────────────────────────────────────────────────╯{RESET}

  {DIM}Type a request · /help for commands · /voice to toggle mic · exit to quit{RESET}
""")


import re     as _re
import queue  as _queue
import urllib.request as _urllib

_WA_PORT = 5005

_WA_ACTIVATE_RE = _re.compile(
    r"(let'?s?\s+chat\s+on\s+whatsapp"
    r"|switch\s+to\s+whatsapp"
    r"|reply\s+(on|via|through)\s+whatsapp"
    r"|whatsapp\s+mode"
    r"|talk\s+on\s+whatsapp)",
    _re.IGNORECASE,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  WhatsApp Bridge — Baileys (no Twilio, no accounts, no URLs to paste)
#
#  HOW IT WORKS:
#  1. main.py auto-installs Node.js deps on first run (npm install, ~30s)
#  2. Starts wa_bridge.js as a background subprocess
#  3. First ever run: a QR code appears — scan it with your phone once
#     (WhatsApp → ⋮ → Linked Devices → Link a Device)
#  4. Session saved to wa_session/ — never scan again
#  5. Text EVE on WhatsApp directly, or say "let's chat on whatsapp"
#     in the terminal to mirror replies
#
#  REQUIREMENT: Node.js ≥ 18  →  https://nodejs.org  (LTS, one-time install)
# ═══════════════════════════════════════════════════════════════════════════════

class _WhatsAppBridge:
    def __init__(self):
        self._engine    = None
        self._proc      = None       # wa_bridge.js subprocess
        self._active    = False      # mirror terminal replies to WA
        self._owner_jid = ""         # phone number, learned on first message
        self.started    = False      # bridge process is running
        self.connected  = False      # WA socket is open
        self._inbound   = _queue.Queue()

    # ── setup ──────────────────────────────────────────────────────────────
    def init(self, engine):
        """Attach engine — does NOT start bridge. Use /whatsapp to connect."""
        self._engine = engine
        # Bridge starts on demand only — no subprocess, no QR, no threads here.

    def _check_node(self) -> bool:
        if shutil.which("node") or shutil.which("node.exe"):
            return True
        # ── Auto-install Node.js ──────────────────────────────────────────
        print(f"{CYAN}[whatsapp] Node.js not found — auto-installing…{RESET}")
        try:
            if sys.platform == "win32":
                self._install_node_windows()
            elif sys.platform == "darwin":
                self._install_node_mac()
            else:
                self._install_node_linux()
        except Exception as e:
            print(f"{RED}[whatsapp] Node.js auto-install failed: {e}{RESET}")
            print(f"{YELLOW}           Install manually from https://nodejs.org (LTS){RESET}")
            return False
        # Re-check after install
        if shutil.which("node") or shutil.which("node.exe"):
            print(f"{GREEN}[whatsapp] Node.js installed successfully.{RESET}")
            return True
        # Installer ran but node not on PATH yet — need restart
        print(f"{YELLOW}[whatsapp] Node.js installed. Please restart EVE once for PATH to update.{RESET}")
        return False

    def _install_node_windows(self):
        import urllib.request, tempfile, os
        # Fetch current LTS version number from Node.js release feed
        try:
            with urllib.request.urlopen("https://nodejs.org/dist/index.json", timeout=15) as r:
                import json as _j
                releases = _j.loads(r.read())
            lts = next(x for x in releases if x.get("lts"))
            ver = lts["version"]   # e.g. "v20.11.0"
        except Exception:
            ver = "v20.11.0"       # safe fallback
        arch = "x64" if sys.maxsize > 2**32 else "x86"
        url  = f"https://nodejs.org/dist/{ver}/node-{ver}-{arch}.msi"
        print(f"{DIM}[whatsapp] Downloading Node.js {ver} ({arch})…{RESET}")
        tmp = Path(tempfile.gettempdir()) / f"node_install_{ver}.msi"
        if not tmp.exists():
            urllib.request.urlretrieve(url, tmp)
        print(f"{DIM}[whatsapp] Running installer (UAC prompt may appear)…{RESET}")
        # /quiet = no UI, /norestart = no forced reboot, ADDLOCAL=ALL = full install
        subprocess.run(
            ["msiexec", "/i", str(tmp), "/quiet", "/norestart", "ADDLOCAL=ALL"],
            check=True
        )
        # MSI installs to Program Files — add to current process PATH
        for candidate in [
            r"C:\Program Files\nodejs",
            r"C:\Program Files (x86)\nodejs",
        ]:
            if Path(candidate).exists():
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                break

    def _install_node_mac(self):
        # Try brew first, fall back to pkg installer
        if shutil.which("brew"):
            print(f"{DIM}[whatsapp] Installing Node.js via Homebrew…{RESET}")
            subprocess.run(["brew", "install", "node"], check=True)
        else:
            import urllib.request, tempfile
            try:
                with urllib.request.urlopen("https://nodejs.org/dist/index.json", timeout=15) as r:
                    import json as _j; releases = _j.loads(r.read())
                ver = next(x for x in releases if x.get("lts"))["version"]
            except Exception:
                ver = "v20.11.0"
            url = f"https://nodejs.org/dist/{ver}/node-{ver}.pkg"
            tmp = Path(tempfile.gettempdir()) / f"node_{ver}.pkg"
            urllib.request.urlretrieve(url, tmp)
            subprocess.run(["sudo", "installer", "-pkg", str(tmp), "-target", "/"], check=True)

    def _install_node_linux(self):
        # Try apt, then yum/dnf, then snap
        for cmd in (
            ["sudo", "apt-get", "install", "-y", "nodejs", "npm"],
            ["sudo", "dnf",     "install", "-y", "nodejs", "npm"],
            ["sudo", "yum",     "install", "-y", "nodejs", "npm"],
            ["sudo", "snap",    "install", "node", "--classic"],
        ):
            if shutil.which(cmd[2] if cmd[1] != "snap" else "snap"):
                try:
                    subprocess.run(cmd, check=True); return
                except Exception:
                    continue
        raise RuntimeError("No package manager found. Install Node.js manually.")

    def _ensure_npm_deps(self):
        marker = ROOT / "node_modules" / "@whiskeysockets" / "baileys"
        if marker.exists():
            return
        print(f"{CYAN}[whatsapp] Installing WhatsApp deps (first run only, ~30s)…{RESET}")
        # write package.json if missing
        pkg = ROOT / "package.json"
        if not pkg.exists():
            import json as _j
            pkg.write_text(_j.dumps({
                "name": "eve-wa-bridge", "version": "1.0.0", "private": True,
                "dependencies": {
                    "@whiskeysockets/baileys": "^6.7.0",
                    "@hapi/boom":              "^10.0.1",
                    "qrcode-terminal":         "^0.12.0",
                    "pino":                    "^8.0.0"
                }
            }, indent=2))
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if not npm:
            print(f"{RED}[whatsapp] npm not found — cannot install deps.{RESET}"); return
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        r  = subprocess.run([npm, "install", "--prefer-offline"], cwd=str(ROOT), **kw)
        if r.returncode == 0:
            print(f"{GREEN}[whatsapp] Deps installed.{RESET}")
        else:
            print(f"{RED}[whatsapp] npm install failed — WhatsApp may not work.{RESET}")

    def _kill_stale_bridge(self):
        """Kill any process already holding port 5005 (stale wa_bridge.js from a previous hard-kill)."""
        try:
            if sys.platform == "win32":
                # netstat -ano lists PID in last column; filter for :5005
                out = subprocess.check_output(
                    ["netstat", "-ano"],
                    text=True, creationflags=subprocess.CREATE_NO_WINDOW,
                )
                for line in out.splitlines():
                    if f":{_WA_PORT}" in line and "LISTENING" in line:
                        parts = line.split()
                        pid = parts[-1]
                        if pid.isdigit():
                            subprocess.run(
                                ["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW,
                            )
                            print(f"{DIM}[whatsapp] Killed stale bridge (PID {pid}){RESET}")
            else:
                # lsof / fuser approach on Linux/macOS
                try:
                    out = subprocess.check_output(
                        ["lsof", "-ti", f"tcp:{_WA_PORT}"], text=True
                    ).strip()
                    for pid in out.splitlines():
                        if pid.isdigit():
                            subprocess.run(["kill", "-9", pid], capture_output=True)
                            print(f"{DIM}[whatsapp] Killed stale bridge (PID {pid}){RESET}")
                except FileNotFoundError:
                    # lsof not available — try fuser
                    subprocess.run(["fuser", "-k", f"{_WA_PORT}/tcp"], capture_output=True)
        except Exception as e:
            print(f"{YELLOW}[whatsapp] Could not kill stale bridge: {e}{RESET}")

    def _start_bridge_process(self):
        bridge_js = ROOT / "wa_bridge.js"
        if not bridge_js.exists():
            print(f"{RED}[whatsapp] wa_bridge.js not found — WhatsApp disabled.{RESET}"); return
        node = shutil.which("node") or shutil.which("node.exe")
        if not node:
            return
        # ── Kill any stale wa_bridge.js left from a previous hard-kill ──────
        self._kill_stale_bridge()
        time.sleep(0.5)   # let the OS release the port
        # ────────────────────────────────────────────────────────────────────
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        # Pass UTF-8 encoding so QR block-chars render correctly on Windows
        node_env = {**os.environ}
        if sys.platform == "win32":
            node_env["CHCP"] = "65001"   # hint — actual chcp done inside wa_bridge.js
        self._proc = subprocess.Popen(
            [node, str(bridge_js)], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            env=node_env, **kw,
        )
        self.started = True
        # forward bridge output (QR code + status lines) to terminal
        def _pipe():
            for line in self._proc.stdout:
                line = line.rstrip()
                if line:
                    print(line, flush=True)
        threading.Thread(target=_pipe, daemon=True, name="wa-pipe").start()
        time.sleep(1.5)   # give the process a moment to bind

    def _start_recv_thread(self):
        """Long-poll /recv and push inbound messages to the queue."""
        def _poll():
            while True:
                try:
                    req  = _urllib.Request(
                        f"http://127.0.0.1:{_WA_PORT}/recv",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with _urllib.urlopen(req, timeout=35) as resp:
                        if resp.status == 200:
                            import json as _j
                            data = _j.loads(resp.read())
                            if data.get("text"):
                                self._inbound.put(data)
                except Exception:
                    time.sleep(2)

        threading.Thread(target=_poll, daemon=True, name="wa-recv").start()

        # worker: drain queue → EVE engine → send reply
        def _work():
            while True:
                try:
                    msg = self._inbound.get(timeout=1)
                except _queue.Empty:
                    continue
                if not self._engine:
                    continue
                self._owner_jid = msg.get("from", "")
                self._active    = True
                WhatsAppSendTool.set_owner(self._owner_jid)
                try:
                    reply = self._engine.submit_message(msg["text"])
                    if reply and reply.strip():
                        self._send(reply)
                except Exception as e:
                    self._send(f"⚠️ Error: {e}")

        threading.Thread(target=_work, daemon=True, name="wa-worker").start()

    def _sync_owner(self):
        """Blocking /status fetch right at startup so owner JID is known immediately."""
        import json as _j
        for _ in range(30):  # try for up to 15s (30 × 0.5s)
            try:
                with _urllib.urlopen(
                    f"http://127.0.0.1:{_WA_PORT}/status", timeout=2
                ) as r:
                    data = _j.loads(r.read())
                    self.connected = data.get("ready", False)
                    owner = data.get("owner", "")
                    if owner:
                        self._owner_jid = owner
                        WhatsAppSendTool.set_owner(owner)
                        return
            except Exception:
                pass
            time.sleep(0.5)

    def _start_status_thread(self):
        """Poll /status every 5 s to update self.connected."""
        def _run():
            while True:
                try:
                    with _urllib.urlopen(
                        f"http://127.0.0.1:{_WA_PORT}/status", timeout=3
                    ) as r:
                        import json as _j
                        data = _j.loads(r.read())
                        self.connected = data.get("ready", False)
                        if data.get("owner") and not self._owner_jid:
                            self._owner_jid = data["owner"]
                            WhatsAppSendTool.set_owner(self._owner_jid)
                except Exception:
                    self.connected = False
                time.sleep(5)

        threading.Thread(target=_run, daemon=True, name="wa-status").start()

    # ── send ───────────────────────────────────────────────────────────────
    def _send(self, text: str):
        if not self._owner_jid:
            return
        try:
            import json as _j
            body = _j.dumps({"to": self._owner_jid, "text": text}).encode()
            req  = _urllib.Request(
                f"http://127.0.0.1:{_WA_PORT}/send",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            _urllib.urlopen(req, timeout=10)
        except Exception as e:
            print(f"{YELLOW}[whatsapp] Send failed: {e}{RESET}")

    # ── public hooks called from the chat loop ────────────────────────────
    def on_user_message(self, text: str):
        if _WA_ACTIVATE_RE.search(text):
            self.activate()

    def on_eve_reply(self, reply: str):
        if self._active and reply and reply.strip():
            self._send(reply)

    def activate(self):
        # ── Start bridge on first activation (lazy) ────────────────────────
        if not self.started:
            if not self._check_node():
                print(f"{RED}[whatsapp] Node.js required. Install from https://nodejs.org{RESET}")
                return
            self._ensure_npm_deps()
            self._start_bridge_process()
            self._sync_owner()
            self._start_recv_thread()
            self._start_status_thread()
        self._active = True
        print(f"{CYAN}[whatsapp] Mirror mode ON — replies will also appear on WhatsApp.{RESET}")
        if self._owner_jid:
            self._send("✅ WhatsApp mode activated. I'll reply here.")

    def deactivate(self):
        self._active = False
        print(f"{DIM}[whatsapp] Mirror mode OFF.{RESET}")

    def status(self):
        state = f"{GREEN}connected{RESET}" if self.connected else f"{YELLOW}connecting…{RESET}"
        mirror = f"{GREEN}ON{RESET}" if self._active else f"{DIM}OFF{RESET}"
        print(f"{DIM}[whatsapp] status={state}  mirror={mirror}  number={self._owner_jid or '(scanning QR…)'}{RESET}")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


_wa = _WhatsAppBridge()
# ═══════════════════════════════════════════════════════════════════════════════


def _run_repl(engine, usage, perms):
    for t in engine.tools.values():
        if hasattr(t,"set_session"): t.set_session(engine.session)
    _banner(engine.working_dir, perms.mode.value, DIRECT_MODE)
    hook_manager.fire("session_start", session_id=engine.session.session_id)
    vs=_get_voice()
    while True:
        try:
            prompt=f"\n{BOLD}{WHITE}{ICON_PROMPT}{RESET}  "
            if vs.active:
                user=vs.get_input(fallback_prompt=prompt)
                if user is None: continue
            else: user=input(prompt).strip()
        except (EOFError,KeyboardInterrupt): print(f"\n{DIM}Goodbye.{RESET}"); break
        if not user: continue
        if user.lower() in ("exit","quit","/exit","/quit"): print(f"{DIM}Goodbye.{RESET}"); break
        if user.startswith("/"):
            if _handle_slash(user,engine,usage,perms,vs=vs): continue
        print()
        try:
            hook_manager.fire("message:pre",text=user)
            _wa.on_user_message(user)           # ← WhatsApp: detect activation phrase
            reply=_submit(engine,user)
            hook_manager.fire("message:post",text=reply)
            _wa.on_eve_reply(reply)             # ← WhatsApp: mirror reply if active
            vs.maybe_speak(reply); print()
        except Exception as e:
            print(f"\n  {RED}{ICON_ERR}  {e}{RESET}")
            if VERBOSE: import traceback; traceback.print_exc()

def _run_headless(engine, prompt):
    if prompt=="-": prompt=sys.stdin.read().strip()
    if not prompt: sys.exit(1)
    _submit(engine,prompt); print()

def _parse():
    p=argparse.ArgumentParser(description="MasterMind — Merged agent harness")
    p.add_argument("-p","--print",dest="prompt",nargs="?",const="-")
    p.add_argument("--auto",action="store_true")
    p.add_argument("--deny",action="store_true")
    p.add_argument("--cwd",default=WORKING_DIR)
    p.add_argument("--max-turns",type=int,default=MAX_TURNS)
    p.add_argument("--verbose","-v",action="store_true",default=VERBOSE)
    p.add_argument("--no-server",action="store_true")
    p.add_argument("--http",action="store_true")
    p.add_argument("--bridge",action="store_true")
    p.add_argument("--bridge-port",type=int,default=7777)
    p.add_argument("--skills",action="store_true")
    p.add_argument("--skill",metavar="NAME")
    p.add_argument("--plugins",action="store_true")
    return p.parse_args()

def main():
    global _bridge,_telemetry
    args=_parse(); cwd=str(Path(args.cwd).resolve()); verbose=args.verbose
    perm="auto" if args.auto else ("deny" if args.deny else PERMISSION_MODE)
    direct=False if args.http else DIRECT_MODE
    _setup_logging(verbose)
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(line_buffering=True)

    if args.skills: print(SkillTool().execute({}).output); return
    if args.skill:
        prompt=args.prompt if args.prompt and args.prompt!="-" else sys.stdin.read().strip()
        if not prompt: sys.exit(1)
        r=SkillTool().execute({"skill":args.skill,"args":{"problem":prompt}})
        if r.is_error: print(f"{RED}{r.output}{RESET}",file=sys.stderr); sys.exit(1)
        print(r.output); return

    if not direct and not args.no_server:
        _proxy_mode = not MODEL_PATH or MODEL_PATH.strip().lower() == "auto"
        if not _start_server() and not _proxy_mode:
            print(f"{YELLOW}[warn] Proceeding without server…{RESET}")

    client=ModelClient(base_url=LLAMA_SERVER_URL,direct=direct,model_path=MODEL_PATH)
    if not client.health(): print(f"{RED}Cannot connect to model.{RESET}"); _stop_server(); sys.exit(1)

    import uuid
    _session_uuid = str(uuid.uuid4())
    _telemetry=TelemetryLogger(session_id=_session_uuid[:8])

    _ep.session_start(_session_uuid, cwd, MODEL_PATH or "auto")
    _ep.install_signal_handlers()
    _ep.install_excepthook()
    _resume_msg = _ep.detect_resume(_session_uuid, cwd)
    if _resume_msg:
        print(f"\n{YELLOW}{_resume_msg}{RESET}\n")
    # FIX: keep the resume message so we can inject it into the session after engine init
    _pending_resume_inject = _resume_msg

    perms=PermissionManager(perm); usage=SessionUsage()
    tools=_build_tools(cwd); tools_dict={t.name:t for t in tools}
    _register_tool_search(tools_dict)
    AgentTool.set_factory(_make_factory(client,perms,usage,cwd,verbose))
    team_manager.set_engine_factory(_make_factory(client,perms,usage,cwd,verbose))

    engine=QueryEngine(
        tools=tools, client=client,
        session=Session.resume_or_create(model_client=client),
        permission_manager=perms, usage=usage, max_turns=args.max_turns,
        working_dir=cwd, verbose=verbose,
        on_tool_start=_on_tool_start, on_tool_end=_on_tool_end, on_chunk=_on_chunk)

    # FIX: inject resume context into the session so the model actually sees it.
    if _pending_resume_inject and engine._goal_text:
        _goal_wake = (
            f"[SYSTEM RESUME] {_pending_resume_inject}\n"
            f"Interrupted goal: {engine._goal_text}\n"
            f"Session history has been restored. Pick up exactly where you left off — "
            f"review the last tool results above and continue the task without asking "
            f"the user to repeat themselves."
        )
        engine.session.add_user(_goal_wake)
        engine.session.add_assistant(
            f"Understood. Resuming: {engine._goal_text[:120]}. Continuing now."
        )
    elif _pending_resume_inject:
        engine.session.add_user(_pending_resume_inject)
        engine.session.add_assistant("Session restored. Ready.")

    # ── Init WhatsApp bridge ───────────────────────────────────────────────
    _wa.init(engine)
    # ──────────────────────────────────────────────────────────────────────

    plugin_manager.set_engine(engine)
    if args.plugins:
        n=plugin_manager.load_all()
        if n: print(f"{GREEN}[plugins] Loaded {n} plugin(s){RESET}")

    try:
        from memory_core.embeddings import get_embedding_fn
        from memory_core.manager import get_memory_manager
        _embed_fn, _embed_dims = get_embedding_fn()
        if _embed_fn:
            get_memory_manager().set_embedding_fn(_embed_fn, _embed_dims)
            print(f"{DIM}[memory] Vector search enabled ({_embed_dims}d){RESET}")
        else:
            print(f"{DIM}[memory] Keyword-only search (install sentence-transformers for vector){RESET}")
    except Exception as _emb_err:
        print(f"{YELLOW}[memory] Embedding init failed: {_emb_err}{RESET}")

    if args.bridge:
        _bridge=BridgeServer(engine=engine,port=args.bridge_port); _bridge.start()
        print(f"{GREEN}[bridge] HTTP server on port {args.bridge_port}{RESET}")

    if not args.prompt:
        print(f"{DIM}Warming up...{RESET}",end="",flush=True)
        try:
            from agent.query_engine import _build_system_prompt
            _warm_sys=_build_system_prompt({t.name:t for t in _build_tools(cwd)},cwd)
            import time as _t; _t0=_t.time()
            client.complete([{"role":"user","content":"hi"}],system=_warm_sys,max_tokens=1,stream=False)
            print(f" {DIM}ready in {_t.time()-_t0:.1f}s{RESET}")
        except Exception as _e: print(f" {YELLOW}skipped ({_e}){RESET}")

    try:
        import reflector_agent as _ra
        _ra._REFLECTOR = _ra.build_reflector(poll_interval=0.25, timeout=1.2)
        _ra._REFLECTOR.memory.remember("cwd", cwd)
        _ra._REFLECTOR.memory.remember("model", MODEL_DISPLAY)
        _ra._REFLECTOR.memory.remember("perm_mode", perm)
        print(f"{DIM}[reflector] Sensory brain active (≤1.2s pre-LLM layer){RESET}")
    except Exception as _re:
        print(f"{YELLOW}[reflector] Init failed: {_re}{RESET}")

    session_memory.init()
    extract_memories.init()
    team_memory_sync.start()
    away_summary_svc = AwaySummary(session_memory)

    print(f"{DIM}[services] SessionMemory, ExtractMemories, TeamMemorySync, AwaySummary ready{RESET}")

    if engine._idle_consolidation:
        try:
            def _llm_call(prompt, system="", temperature=0.3, max_tokens=300):
                msgs = [{"role": "user", "content": prompt}]
                r = client.complete(msgs, system=system, max_tokens=max_tokens, stream=False)
                return r if isinstance(r, str) else ""
            engine._idle_consolidation.set_llm(_llm_call)
            print(f"{DIM}[idle_consolidation] Active memory curation ready (idle={IDLE_CONSOLIDATION_S}s){RESET}")
        except Exception as _ic_err:
            print(f"{YELLOW}[idle_consolidation] LLM wire failed: {_ic_err}{RESET}")

    if engine._daily_digest:
        try:
            engine._daily_digest.set_llm(_llm_call if '_llm_call' in dir() else None)
            digest = engine.get_startup_digest()
            if digest:
                print(f"\n{DIM}{'─'*60}{RESET}")
                print(f"{DIM}{digest}{RESET}")
                print(f"{DIM}{'─'*60}{RESET}\n")
            due = engine._daily_digest.get_pending_reminders_for_startup()
            for rem in due:
                print(f"{YELLOW}{rem}{RESET}")
        except Exception as _dd_err:
            if verbose:
                print(f"{YELLOW}[daily_digest] {_dd_err}{RESET}")

    hb=Heartbeat()
    def _autosave():
        try:
            from memory.manager import append_session
            append_session(f"Active — {len(engine.session)} messages")
        except: pass
    hb.register(every=300,task=_autosave); hb.start()

    def _sigint(sig,frame):
        _ss.spin_done.set(); print(f"\n{DIM}Interrupted.{RESET}"); print(f"{DIM}{usage.summary()}{RESET}")
        hook_manager.fire("session_end",session_id=engine.session.session_id)
        if _telemetry: _telemetry.session_end()
        if _bridge: _bridge.stop()
        _wa.stop()
        try:
            from memory.manager import append_session
            append_session(f"Session ended — {len(engine.session)} messages")
        except: pass
        hb.stop(); _stop_server(); sys.exit(0)
    signal.signal(signal.SIGINT,_sigint)
    _telemetry.session_start(cwd=cwd,perm=perm)

    try:
        if args.prompt is not None: _run_headless(engine,args.prompt)
        else: _run_repl(engine,usage,perms)
    finally:
        hb.stop()
        team_memory_sync.stop()
        extract_memories.drain(timeout_s=10)
        try:
            import reflector_agent as _ra
            if hasattr(_ra, "_REFLECTOR"):
                _ra._REFLECTOR.memory.stop()
        except Exception:
            pass
        hook_manager.fire("session_end",session_id=engine.session.session_id)
        if _telemetry: _telemetry.session_end()
        if _bridge: _bridge.stop()
        try:
            from memory.manager import append_session
            append_session(f"Session ended — {len(engine.session)} messages")
        except: pass
        _stop_server()

if __name__=="__main__":
    main()