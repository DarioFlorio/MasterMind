#!/usr/bin/env python3
"""
main.py — PyClaudeCode: Local agentic coding assistant.
Merges Claude Code architecture + EVE features + local llama-cpp inference.
"""
from __future__ import annotations
import argparse, json, os, shutil, signal, subprocess, sys, threading, time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoinstall import ensure_dependencies
ensure_dependencies()

from config.settings import (
    LLAMA_SERVER_URL, LLAMA_SERVER_PORT, MODEL_PATH, MODEL_DISPLAY,
    MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR,
    CONTEXT_SIZE, MAX_TOKENS, DIRECT_MODE, BASH_TIMEOUT_S,
    UNLIMITED_CONTEXT,
)
from tools.pm_tool import PMTool  
from agent.query_engine import QueryEngine
from agent.session import Session
from tools.bash_tool       import BashTool
from tools.read_file_tool  import ReadFileTool
from tools.write_file_tool import WriteFileTool
from tools.edit_file_tool  import EditFileTool
from tools.glob_tool       import GlobTool
from tools.grep_tool       import GrepTool
from tools.list_dir_tool   import ListDirTool
from tools.web_search_tool import WebSearchTool
from tools.web_fetch_tool  import WebFetchTool
from tools.agent_tool      import AgentTool
from tools.todo_tool       import TodoWriteTool, TodoReadTool
from tools.memory_tool     import MemoryWriteTool, MemoryReadTool
from tools.skill_tool      import SkillTool
from tools.git_tool        import GitTool
from tools.scratchpad_tool import ScratchpadTool
from tools.reflect_tool    import ReflectTool
from tools.export_tool     import ExportTool, export_session
from utils.model_client    import ModelClient, ThinkingStreamParser
from utils.permissions     import PermissionManager
from utils.token_counter   import SessionUsage
from heartbeat             import Heartbeat
from agent.context_budget  import ContextBudget
from agent.ultraplan       import UltraPlan, should_ultraplan
from memory.autodream      import AutoDream
from kairos                import Kairos, write_daemon_script

# ── Leak feature globals ─────────────────────────────────────────────────────
_context_budget: ContextBudget | None = None
_autodream:      AutoDream     | None = None
_kairos:         Kairos        | None = None

# ── Logging ───────────────────────────────────────────────────────────────────
import logging

def _setup_logging(verbose: bool) -> None:
    level  = logging.DEBUG if verbose else logging.WARNING
    fmt    = "%(levelname)s [%(name)s] %(message)s"
    logging.basicConfig(level=level, format=fmt, stream=sys.stderr)
    # Silence noisy third-party loggers unless in verbose mode
    if not verbose:
        for noisy in ("httpx", "httpcore", "llama_cpp", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.ERROR)

# ── ANSI ──────────────────────────────────────────────────────────────────────
def _tty() -> bool: return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
_C = _tty()

# Core palette
RESET  = "\033[0m"          if _C else ""
BOLD   = "\033[1m"          if _C else ""
DIM    = "\033[2m"          if _C else ""
ITALIC = "\033[3m"          if _C else ""

# Semantic colors
WHITE  = "\033[97m"         if _C else ""   # bright white — all agent output
CYAN   = "\033[96m"         if _C else ""   # bright cyan  — tool names / headers
GREEN  = "\033[92m"         if _C else ""   # bright green — success
RED    = "\033[91m"         if _C else ""   # bright red   — errors
YELLOW = "\033[93m"         if _C else ""   # bright amber — warnings / prompt
DIM_C  = "\033[2;36m"       if _C else ""   # dim cyan     — subtitles

# Thinking block: italic + icy pale blue (almost white, clearly blue-tinted)
THINK      = "\033[3;38;5;117m"  if _C else ""   # italic + sky blue
THINK_HEAD = "\033[38;5;67m"     if _C else ""   # muted steel blue — headers
THINK_SEP  = "\033[38;5;237m"    if _C else ""   # near-black gray  — separator lines

# Tool display
TOOL_NAME  = "\033[1;96m"        if _C else ""   # bold bright cyan
TOOL_DIM   = "\033[38;5;244m"    if _C else ""   # medium gray      — tool args

# Icons (Windows Terminal renders all of these)
ICON_THINK   = "◈"   # thinking block marker
ICON_TOOL    = "▸"   # tool invocation
ICON_OK      = "◆"   # tool success
ICON_ERR     = "✖"   # tool error
ICON_PROMPT  = "❯"   # output / user prompt
ICON_CACHED  = "⟲"   # cached result

# ── Streaming state ────────────────────────────────────────────────────────────
class _SS:
    def __init__(self):
        self.first      = True
        self.spin_done  = threading.Event()
        self.chunks     = 0
        self.active     = False
        self.lock       = threading.Lock()
        self.parser     = ThinkingStreamParser()
        self.in_think   = False

_ss = _SS()

def _spinner() -> None:
    fr = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    i  = 0
    while not _ss.spin_done.wait(timeout=0.07):
        sys.stdout.write(f"\r{THINK_HEAD}{fr[i%len(fr)]}  reasoning…{RESET}")
        sys.stdout.flush()
        i += 1
    sys.stdout.write("\r" + " "*40 + "\r")
    sys.stdout.flush()

_THINK_RULE = f"{THINK_SEP}{'─'*64}{RESET}"

def _on_chunk(chunk: str) -> None:
    with _ss.lock:
        _ss.chunks += 1
        _ss.active  = True
    if _ss.first:
        _ss.first = False
        _ss.spin_done.set()
        time.sleep(0.01)
    for text, is_think in _ss.parser.feed(chunk):
        if not text: continue
        if is_think:
            if not _ss.in_think:
                # ── Enter thinking block ──────────────────────────────────────
                sys.stdout.write(
                    f"\n{THINK_HEAD}{ICON_THINK} thinking{RESET}\n"
                    f"{_THINK_RULE}\n"
                    f"{THINK}"
                )
                _ss.in_think = True
            sys.stdout.write(f"{THINK}{text}")
        else:
            if _ss.in_think:
                # ── Exit thinking block, enter output ─────────────────────────
                sys.stdout.write(
                    f"{RESET}\n{_THINK_RULE}\n\n"
                    f"{DIM_C}{ICON_PROMPT}{RESET}  "
                )
                _ss.in_think = False
            sys.stdout.write(f"{WHITE}{text}{RESET}")
    sys.stdout.flush()

def _flush_parser() -> None:
    for text, is_think in _ss.parser.flush():
        if text:
            color = THINK if is_think else WHITE
            sys.stdout.write(f"{color}{text}{RESET}")
    if _ss.in_think:
        sys.stdout.write(f"{RESET}\n{_THINK_RULE}\n")
        _ss.in_think = False
    sys.stdout.flush()

def _reset_ss() -> None:
    _ss.first    = True
    _ss.spin_done.clear()
    _ss.chunks   = 0
    _ss.active   = False
    _ss.parser   = ThinkingStreamParser()
    _ss.in_think = False

# ── Tool hooks ────────────────────────────────────────────────────────────────
def _on_tool_start(name: str, inp: dict) -> None:
    if   name == "bash":        s = inp.get("command","")[:72]
    elif name in ("read_file","write_file","edit_file"): s = inp.get("path","")
    elif name == "grep":        s = f"'{inp.get('pattern','')}'"
    elif name == "glob":        s = inp.get("pattern","")
    elif name == "list_dir":    s = inp.get("path",".")
    elif name == "web_search":  s = inp.get("query","")[:60]
    elif name == "web_fetch":   s = inp.get("url","")[:60]
    elif name == "agent":       s = inp.get("task","")[:60]
    elif name == "git":         s = f"{inp.get('op','')} {inp.get('args','')}".strip()[:60]
    elif name == "scratchpad":  s = f"{inp.get('op','')} {inp.get('key','')}"
    elif name == "reflect":     s = f"mode={inp.get('mode','general')}"
    elif name == "skill":
        sn   = inp.get("skill","")
        prob = (inp.get("args",{}).get("problem") or
                inp.get("args",{}).get("query") or "")[:55]
        s = f"{sn}" + (f"  {TOOL_DIM}│ {prob}{RESET}" if prob else "")
    elif name == "todo_write":  s = f"{len(inp.get('todos',[]))} items"
    elif name == "memory_write":s = inp.get("key","")
    else: s = str(inp)[:60]

    # Newline only if not currently printing think block
    prefix = "\n" if not _ss.in_think else "\n\n"
    print(f"{prefix}  {TOOL_NAME}{ICON_TOOL} {name}{RESET}  {TOOL_DIM}{s}{RESET}", flush=True)


def _on_tool_end(name: str, result) -> None:
    MAX_SNIP = 300

    if result.is_error:
        icon  = f"{RED}{ICON_ERR}{RESET}"
        color = RED
        snip  = (result.output or "").strip()[:MAX_SNIP]
    else:
        icon  = f"{GREEN}{ICON_OK}{RESET}"
        color = DIM
        out   = (result.output or "").strip()
        # Detect cached results
        if out.startswith("[cached]"):
            icon = f"{TOOL_DIM}{ICON_CACHED}{RESET}"
            out  = out[8:].strip()
        lines = out.splitlines()
        if name == "skill" and len(lines) > 2:
            snip = "\n    ".join(lines[:6]) + (f"\n    … ({len(lines)} lines)" if len(lines) > 6 else "")
        else:
            snip = " ".join(lines[:2])[:MAX_SNIP]

    print(f"  {icon} {color}{snip}{RESET}", flush=True)


def _find_server() -> str | None:
    override = os.environ.get("LLAMA_SERVER_BIN","").strip()
    if override:
        p = Path(override)
        if p.exists(): return str(p)
        found = shutil.which(override)
        if found: return found
    for name in ("llama-server","llama-server.exe","server","server.exe"):
        found = shutil.which(name)
        if found: return found
    for name in ("llama-server.exe","llama-server","server.exe","server"):
        local = ROOT / "bin" / name
        if local.exists(): return str(local)
    return None

def _healthy(url: str = LLAMA_SERVER_URL) -> bool:
    try:
        import httpx
        return httpx.get(f"{url}/health", timeout=2).status_code == 200
    except: return False

def _start_server() -> bool:
    global _server_proc
    if _healthy(): return True
    binary = _find_server()
    if not binary:
        print(f"{YELLOW}[server] llama-server not found — download from github.com/ggerganov/llama.cpp{RESET}")
        return False
    model = Path(MODEL_PATH)
    if not model.exists():
        print(f"{RED}[server] Model not found: {model}\n → Set MODEL_PATH in .env{RESET}")
        return False
    cpu   = os.cpu_count() or 4
    nt    = int(os.environ.get("N_THREADS","0")) or max(1, cpu//2)
    ngl   = int(os.environ.get("N_GPU_LAYERS","0"))
    cmd   = [binary, "-m", str(model), "--port", str(LLAMA_SERVER_PORT),
             "--host","127.0.0.1","--ctx-size", str(CONTEXT_SIZE),
             "-ngl", str(ngl), "-t", str(nt), "--threads-batch", str(cpu),
             "--batch-size","512","--cont-batching","--mmap"]
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        _server_proc = subprocess.Popen(
            cmd, stdout=None if VERBOSE else subprocess.DEVNULL,
            stderr=None if VERBOSE else subprocess.DEVNULL, **kwargs)
    except Exception as e:
        print(f"{RED}[server] Launch failed: {e}{RESET}"); return False
    print(f"{DIM}[server] Starting", end="", flush=True)
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(1.5); print(".", end="", flush=True)
        if _server_proc.poll() is not None:
            print(f"\n{RED}[server] Crashed{RESET}"); return False
        if _healthy():
            print(f" {GREEN}ready!{RESET}"); return True
    print(f"\n{RED}[server] Timed out{RESET}"); return False

def _stop_server() -> None:
    global _server_proc
    if _server_proc and _server_proc.poll() is None:
        _server_proc.terminate()
        try: _server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: _server_proc.kill()
        _server_proc = None

# ── Build tools ───────────────────────────────────────────────────────────────
def _build_tools(cwd: str) -> list:
    return [
        BashTool(working_dir=cwd),
        ReadFileTool(working_dir=cwd),
        WriteFileTool(working_dir=cwd),
        EditFileTool(working_dir=cwd),
        GlobTool(working_dir=cwd),
        GrepTool(working_dir=cwd),
        ListDirTool(working_dir=cwd),
        WebSearchTool(),
        WebFetchTool(),
        AgentTool(),
        TodoWriteTool(),
        TodoReadTool(),
        MemoryWriteTool(),
        MemoryReadTool(),
        SkillTool(),
        PMTool(),       
        
    ]

def _make_factory(client, perms, usage, cwd, verbose):
    def factory(max_turns=10, is_subagent=True):
        return QueryEngine(
            tools=_build_tools(cwd), client=client, session=Session(),
            permission_manager=perms, usage=usage,
            max_turns=max_turns, working_dir=cwd, verbose=verbose,
            is_subagent=is_subagent,
            on_tool_start=_on_tool_start, on_tool_end=_on_tool_end,
            on_chunk=_on_chunk,
        )
    return factory

# ── Submit one turn ───────────────────────────────────────────────────────────
def _submit(engine: QueryEngine, text: str) -> str:
    _reset_ss()
    sys.stdout.write(f"\n{BOLD}>{RESET} ")
    sys.stdout.flush()
    t = threading.Thread(target=_spinner, daemon=True)
    t.start()
    result = None; err = None
    def _run():
        nonlocal result, err
        try: result = engine.submit_message(text)
        except Exception as e: err = e
    th = threading.Thread(target=_run, daemon=True)
    th.start(); th.join()
    _flush_parser()
    _ss.spin_done.set(); t.join(timeout=0.5)
    if th.is_alive():
        print(f"\n{RED}[timeout]{RESET}", file=sys.stderr); return ""
    if err:
        print(f"\n{RED}Error: {err}{RESET}", file=sys.stderr)
        if VERBOSE: import traceback; traceback.print_exc()
        return ""
    if not _ss.active and result:
        sys.stdout.write(result); sys.stdout.flush()
    return result or ""

# ── Slash commands ────────────────────────────────────────────────────────────
def _handle_slash(cmd: str, engine: QueryEngine, usage: SessionUsage,
                  perms: PermissionManager) -> bool:
    """Handle /commands. Returns True if handled."""
    low = cmd.strip().lower()
    if low in ("/clear", "/reset"):
        engine.session.clear()
        engine.invalidate_system_prompt()
        print(f"{DIM}Session cleared.{RESET}")
        return True
    if low in ("/status", "/cost"):
        files_info = engine.file_tracker.summary()
        print(f"{DIM}{usage.summary()}\nMessages: {len(engine.session)}{RESET}")
        if files_info:
            print(f"{DIM_C}{files_info}{RESET}")
        return True
    if low.startswith("/mode "):
        nm = cmd[6:].strip()
        try:
            perms.set_mode(nm)
            print(f"{DIM}Permission mode → {nm}{RESET}")
        except: print(f"{RED}Modes: auto | ask | deny{RESET}")
        return True
    if low.startswith("/save"):
        parts = cmd.split(maxsplit=1)
        p = Path(parts[1]) if len(parts) > 1 else Path("session.json")
        engine.session.save(p)
        print(f"{DIM}Saved → {p}{RESET}")
        return True
    if low == "/compact":
        engine.session._unlimited = True
        engine.session._maybe_compress()
        print(f"{DIM}Context compacted.{RESET}")
        return True
    if low.startswith("/memory"):
        try:
            from memory.manager import load_context
            print(load_context() or "(no memories)")
        except Exception as e: print(f"{RED}{e}{RESET}")
        return True
    if low in ("/skills", "/skill"):
        # List all available skills with their descriptions
        skill_tool = next((t for t in engine.tools.values()
                           if t.name == "skill"), None)
        if skill_tool is None:
            print(f"{RED}SkillTool not found in engine tools.{RESET}")
            return True
        result = skill_tool.execute({})
        print(f"\n{CYAN}{result.output}{RESET}")
        return True
    if low.startswith("/skill "):
        # /skill <name> <problem text>  — invoke a skill directly
        rest   = cmd[7:].strip()
        parts  = rest.split(maxsplit=1)
        sname  = parts[0] if parts else ""
        sprob  = parts[1] if len(parts) > 1 else ""
        if not sname:
            print(f"{RED}Usage: /skill <skill_name> <problem>{RESET}")
            return True
        skill_tool = next((t for t in engine.tools.values()
                           if t.name == "skill"), None)
        if skill_tool is None:
            print(f"{RED}SkillTool not found.{RESET}")
            return True
        print(f"\n{CYAN}Running skill: {BOLD}{sname}{RESET}", flush=True)
        result = skill_tool.execute({"skill": sname,
                                     "args": {"problem": sprob or sname}})
        if result.is_error:
            print(f"{RED}{result.output}{RESET}")
        else:
            print(f"\n{result.output}")
        return True
    if low in ("/help", "/?"):
        print(f"""{DIM}
Commands:
  /clear           Clear session history
  /compact         Compress context window
  /status          Show token usage
  /mode MODE       Set permission mode (auto|ask|deny)
  /save [FILE]     Save session to JSON
  /memory          Show persistent memory
  /skills          List all 24 reasoning skills with descriptions
  /skill NAME [Q]  Run a skill directly (e.g. /skill bayes_reason monty hall)
  /export [FILE]   Export session to Markdown
  /files           Show files read/written this session
  /tasks [OBJ]     BabyAGI task queue
  /help            Show this help
  exit / quit      Exit{RESET}""")
        return True
    return False

# ── REPL ──────────────────────────────────────────────────────────────────────
def _banner(cwd: str, perm: str, direct: bool) -> None:
    mode = "llama-cpp  (direct)" if direct else f"llama-server  {LLAMA_SERVER_URL}"
    from pathlib import Path as _P
    n_skills = len(list((_P(__file__).parent / "skills").glob("*.py"))) - 1
    ctx_str  = "unlimited  (sliding window)" if UNLIMITED_CONTEXT else f"{CONTEXT_SIZE:,} tokens"
    print(f"""
{BOLD}{CYAN}  ╭─────────────────────────────────────────────────────╮{RESET}
{BOLD}{CYAN}  │{RESET}  {WHITE}{BOLD}PyClaudeCode{RESET}                                         {BOLD}{CYAN}│{RESET}
{BOLD}{CYAN}  ├─────────────────────────────────────────────────────┤{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}model   {RESET}  {WHITE}{MODEL_DISPLAY}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}backend {RESET}  {DIM}{mode}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}cwd     {RESET}  {DIM}{cwd}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}perms   {RESET}  {YELLOW}{perm}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}context {RESET}  {DIM}{ctx_str}{RESET}
{BOLD}{CYAN}  │{RESET}  {TOOL_DIM}skills  {RESET}  {GREEN}{n_skills} reasoning skills{RESET}
{BOLD}{CYAN}  ╰─────────────────────────────────────────────────────╯{RESET}

  {DIM}Type a request  ·  /help for commands  ·  exit to quit{RESET}
  {THINK}{ICON_THINK} reasoning shown in pale blue italic  ·  {RESET}{DIM_C}{ICON_PROMPT}{RESET}  {WHITE}output in white{RESET}
""")

def _run_repl(engine: QueryEngine, usage: SessionUsage, perms: PermissionManager) -> None:
    for t in engine.tools.values():
        if hasattr(t, 'set_session'): t.set_session(engine.session)
    _banner(engine.working_dir, perms.mode.value, DIRECT_MODE)
    while True:
        try:
            user = input(f"\n{BOLD}{WHITE}{ICON_PROMPT}{RESET}  ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM}Goodbye.{RESET}")
            break
        if not user: continue
        if user.lower() in ("exit","quit","/exit","/quit"):
            print(f"{DIM}Goodbye.{RESET}"); break
        if user.startswith("/"):
            if _handle_slash(user, engine, usage, perms): continue
        print()
        try:
            _submit(engine, user)
            print()
        except Exception as e:
            print(f"\n  {RED}{ICON_ERR}  {e}{RESET}")
            if VERBOSE: import traceback; traceback.print_exc()

def _run_headless(engine: QueryEngine, prompt: str) -> None:
    if prompt == "-": prompt = sys.stdin.read().strip()
    if not prompt: sys.exit(1)
    _submit(engine, prompt)
    print()

# ── Args ──────────────────────────────────────────────────────────────────────
def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyClaudeCode — Local agentic coding assistant")
    p.add_argument("-p","--print", dest="prompt", nargs="?", const="-",
                   help="Headless mode: run PROMPT and exit")
    p.add_argument("--auto",      action="store_true", help="Auto-approve all tools")
    p.add_argument("--deny",      action="store_true", help="Deny all destructive tools")
    p.add_argument("--cwd",       default=WORKING_DIR)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--verbose","-v", action="store_true", default=VERBOSE)
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--http",      action="store_true", help="Force HTTP server mode")
    p.add_argument("--skills",    action="store_true",
                   help="List all available reasoning skills and exit")
    p.add_argument("--skill",     metavar="NAME",
                   help="Run a single skill headless: --skill deep_reason -p 'your question'")
    return p.parse_args()

# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args    = _parse()
    cwd     = str(Path(args.cwd).resolve())
    verbose = args.verbose
    perm    = "auto" if args.auto else ("deny" if args.deny else PERMISSION_MODE)
    direct  = False if args.http else DIRECT_MODE

    # Set up logging first so all subsystems can emit debug output
    _setup_logging(verbose)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    # --skills: list all skills and exit (no model needed)
    if args.skills:
        st = SkillTool()
        result = st.execute({})
        print(result.output)
        return

    # --skill NAME: run a single skill headless (no model needed)
    if args.skill:
        prompt = args.prompt if args.prompt and args.prompt != "-" else ""
        if not prompt:
            prompt = sys.stdin.read().strip()
        if not prompt:
            print(f"{RED}Provide a problem via -p 'question' or stdin.{RESET}", file=sys.stderr)
            sys.exit(1)
        st = SkillTool()
        result = st.execute({"skill": args.skill, "args": {"problem": prompt}})
        if result.is_error:
            print(f"{RED}{result.output}{RESET}", file=sys.stderr)
            sys.exit(1)
        print(result.output)
        return

    # Server (HTTP mode only)
    if not direct and not args.no_server:
        if not _start_server():
            print(f"{YELLOW}[warn] Proceeding without confirmed server…{RESET}")

    # Client
    client = ModelClient(base_url=LLAMA_SERVER_URL, direct=direct, model_path=MODEL_PATH)
    if not client.health():
        print(f"{RED}Cannot connect to model. Check MODEL_PATH / llama-server.{RESET}")
        _stop_server(); sys.exit(1)

    if direct:
        print(f"{GREEN}[ready] Model loaded ({MODEL_DISPLAY}){RESET}")

    # Wire up
    perms   = PermissionManager(perm)
    usage   = SessionUsage()
    tools   = _build_tools(cwd)

    AgentTool.set_factory(_make_factory(client, perms, usage, cwd, verbose))

    engine  = QueryEngine(
        tools=tools, client=client,
        session=Session(model_client=client),
        permission_manager=perms, usage=usage,
        max_turns=args.max_turns, working_dir=cwd,
        verbose=verbose, on_tool_start=_on_tool_start,
        on_tool_end=_on_tool_end, on_chunk=_on_chunk,
    )

# ─────────────────────────────────────────────────────────────
    # PRE‑WARM THE MODEL (first inference cold start)
    # ─────────────────────────────────────────────────────────────
    if not args.prompt:   # only in interactive mode
        print(f"{DIM}Pre‑warming model (first inference may take a while)...{RESET}")
        start = time.time()
        try:
            # Dummy query – very short, just to build KV cache
            _ = engine.submit_message("Ping")
            print(f"{DIM}Warmed up in {time.time()-start:.1f}s{RESET}")
        except Exception as e:
            print(f"{YELLOW}Pre‑warm failed: {e}{RESET}")
    # ─────────────────────────────────────────────────────────────
    
    


    # Heartbeat autosave
    hb = Heartbeat()
    def _autosave():
        try:
            from memory.manager import append_session
            append_session(f"Active — {len(engine.session)} messages")
        except: pass
    hb.register(every=300, task=_autosave)
    hb.start()

    def _sigint(sig, frame):
        _ss.spin_done.set()
        print(f"\n{DIM}Interrupted.{RESET}")
        print(f"{DIM}{usage.summary()}{RESET}")
        try:
            from memory.manager import append_session
            append_session(f"Session ended — {len(engine.session)} messages")
        except: pass
        hb.stop(); _stop_server(); sys.exit(0)
    signal.signal(signal.SIGINT, _sigint)

    try:
        if args.prompt is not None:
            _run_headless(engine, args.prompt)
        else:
            _run_repl(engine, usage, perms)
    finally:
        hb.stop()
        try:
            from memory.manager import append_session
            append_session(f"Session ended — {len(engine.session)} messages")
        except: pass
        _stop_server()

if __name__ == "__main__":
    main()