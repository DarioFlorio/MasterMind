from __future__ import annotations

import sys, os, json, threading, queue, subprocess, time, shutil, signal, logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autoinstall import ensure_dependencies
ensure_dependencies()

from config.settings import (
    LLAMA_SERVER_URL, LLAMA_SERVER_PORT, MODEL_PATH, MODEL_DISPLAY,
    MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR,
    CONTEXT_SIZE, MAX_TOKENS, DIRECT_MODE, BASH_TIMEOUT_S, UNLIMITED_CONTEXT,
)
from agent.query_engine    import QueryEngine
from agent.session         import Session
from agent.context_budget  import ContextBudget
from agent.ultraplan       import UltraPlan, should_ultraplan
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

try:
    from tools.meta_harness import MetaHarnessTool
    META_HARNESS_AVAILABLE = True
except ImportError:
    META_HARNESS_AVAILABLE = False
    MetaHarnessTool = None

from utils.model_client    import ModelClient, ThinkingStreamParser
from utils.permissions     import PermissionManager
from utils.token_counter   import SessionUsage
from heartbeat             import Heartbeat
from memory.autodream      import AutoDream
from memory.manager        import load_context, append_session
from kairos                import Kairos, write_daemon_script

from flask import Flask, render_template_string, request, jsonify, Response
from flask_cors import CORS

app  = Flask(__name__)
CORS(app)

def _setup_logging(verbose):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s", stream=sys.stderr)
    if not verbose:
        for n in ("httpx","httpcore","llama_cpp","urllib3"):
            logging.getLogger(n).setLevel(logging.ERROR)
_setup_logging(VERBOSE)

# ── state ─────────────────────────────────────────────────────────────────────
class AppState:
    def __init__(self):
        self.server_proc    = None
        self.server_running = False
        self.engine         = None
        self.client         = None
        self.perms          = None
        self.usage          = None
        self.think_parser   = ThinkingStreamParser()
        self.in_think       = False
        self.output_queue   = queue.Queue()
        self.lock           = threading.Lock()
        self.heartbeat      = None
        self.context_budget = None
        self.autodream      = None
        self.kairos         = None
        self.conversations  = []
        self.active_conv_id = None

state = AppState()

# ── server ────────────────────────────────────────────────────────────────────
def _find_server():
    override = os.environ.get("LLAMA_SERVER_BIN","").strip()
    if override:
        p = Path(override)
        if p.exists(): return str(p)
        f = shutil.which(override)
        if f: return f
    for n in ("llama-server","llama-server.exe","server","server.exe"):
        f = shutil.which(n)
        if f: return f
    for n in ("llama-server.exe","llama-server","server.exe","server"):
        p = PROJECT_ROOT/"bin"/n
        if p.exists(): return str(p)
    return None

def _healthy(url=LLAMA_SERVER_URL, timeout=2.0):
    try:
        import httpx
        return httpx.get(f"{url}/health", timeout=timeout).status_code == 200
    except: return False

def start_llama_server():
    if state.server_proc and state.server_proc.poll() is None:
        state.server_running = True; return True
    if _healthy(timeout=1.0):
        state.server_running = True; return True
    binary = _find_server()
    if not binary: print("[server] llama-server not found!"); return False
    model = Path(MODEL_PATH)
    if not model.exists(): print(f"[server] Model not found: {model}"); return False
    cpu = os.cpu_count() or 4
    nt  = int(os.environ.get("N_THREADS","0"))    or max(1, cpu//2)
    ngl = int(os.environ.get("N_GPU_LAYERS","0"))
    ctx = int(os.environ.get("CONTEXT_SIZE", str(CONTEXT_SIZE)))
    cmd = [binary, "-m", str(model),
           "--port", str(LLAMA_SERVER_PORT), "--host","127.0.0.1",
           "--no-webui",
           "--ctx-size", str(ctx), "-ngl", str(ngl), "-t", str(nt),
           "--threads-batch", str(cpu), "--batch-size","512",
           "--ubatch-size","256","--cont-batching",
           "--cache-reuse","256","--mmap",
           "--cache-type-k","q8_0","--cache-type-v","q8_0"]
    if ngl > 0: cmd += ["--flash-attn","--mlock"]
    try:
        kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform=="win32" else {}
        state.server_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    except Exception as e: print(f"[server] Failed: {e}"); return False
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(1.5)
        if state.server_proc.poll() is not None: return False
        if _healthy(): state.server_running = True; return True
    return False

def stop_llama_server():
    if state.server_proc and state.server_proc.poll() is None:
        state.server_proc.terminate()
        try: state.server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired: state.server_proc.kill()
    state.server_proc = None; state.server_running = False

# ── hooks ─────────────────────────────────────────────────────────────────────
def on_tool_start(name, inp):
    if   name == "bash":         s = inp.get("command","")[:72]
    elif name in ("read_file","write_file","edit_file"): s = inp.get("path","")
    elif name == "grep":         s = f"'{inp.get('pattern','')}'"
    elif name == "glob":         s = inp.get("pattern","")
    elif name == "list_dir":     s = inp.get("path",".")
    elif name == "web_search":   s = inp.get("query","")[:60]
    elif name == "web_fetch":    s = inp.get("url","")[:60]
    elif name == "agent":        s = inp.get("task","")[:60]
    elif name == "skill":
        args = inp.get("args",{})
        prob = (args.get("problem") or args.get("query") or args.get("input",""))[:60]
        s = f"{inp.get('skill','')} | {prob}" if prob else inp.get("skill","")
    elif name == "todo_write":   s = f"{len(inp.get('todos',[]))} items"
    elif name == "memory_write": s = inp.get("key","")
    else:                        s = json.dumps(inp)[:60]
    state.output_queue.put({"type":"tool_start","name":name,"summary":s[:70]})

def on_tool_end(name, result):
    if result.is_error:
        ok, out = False, (result.output or "").strip()[:200]
    else:
        ok    = True
        lines = (result.output or "").splitlines()
        out   = (f"{lines[0][:80]} … ({len(lines)} lines)"
                 if name=="skill" and len(lines)>1
                 else (lines[0][:80] if lines else ""))
    state.output_queue.put({"type":"tool_end","name":name,"success":ok,"output":out})

def on_chunk(chunk):
    for text, is_think in state.think_parser.feed(chunk):
        if not text: continue
        if is_think:
            if not state.in_think:
                state.output_queue.put({"type":"think_start"})
                state.in_think = True
            state.output_queue.put({"type":"think","content":text})
        else:
            if state.in_think:
                state.output_queue.put({"type":"think_end"})
                state.in_think = False
                state.output_queue.put({"type":"answer_start"})
            state.output_queue.put({"type":"answer","content":text})

def flush_think():
    for text, is_think in state.think_parser.flush():
        if text:
            state.output_queue.put({"type":"think" if is_think else "answer","content":text})
    if state.in_think:
        state.output_queue.put({"type":"think_end"})
        state.in_think = False

# ── tools / factory ───────────────────────────────────────────────────────────
def _build_tools(cwd):
    return [BashTool(working_dir=cwd), ReadFileTool(working_dir=cwd),
            WriteFileTool(working_dir=cwd), EditFileTool(working_dir=cwd),
            GlobTool(working_dir=cwd), GrepTool(working_dir=cwd),
            ListDirTool(working_dir=cwd), WebSearchTool(), WebFetchTool(),
            AgentTool(), TodoWriteTool(), TodoReadTool(),
            MemoryWriteTool(), MemoryReadTool(), SkillTool()]

def _make_factory(client, perms, usage, cwd, verbose):
    def factory(max_turns=10, is_subagent=True):
        return QueryEngine(
            tools=_build_tools(cwd), client=client, session=Session(),
            permission_manager=perms, usage=usage, max_turns=max_turns,
            working_dir=cwd, verbose=verbose, is_subagent=is_subagent,
            on_tool_start=on_tool_start, on_tool_end=on_tool_end, on_chunk=on_chunk)
    return factory

# ── agent init ────────────────────────────────────────────────────────────────
def init_agent():
    with state.lock:
        if state.engine: return True
        if not DIRECT_MODE and not _healthy():
            if not start_llama_server(): return False
        state.client = ModelClient(base_url=LLAMA_SERVER_URL, direct=DIRECT_MODE, model_path=MODEL_PATH)
        if not state.client.health(): return False
        state.perms = PermissionManager(mode=PERMISSION_MODE)
        state.usage = SessionUsage()
        tools = _build_tools(WORKING_DIR)
        if META_HARNESS_AVAILABLE and MetaHarnessTool:
            try: tools.append(MetaHarnessTool(model_client=state.client,
                    working_dir=WORKING_DIR, verbose=VERBOSE,
                    default_max_iterations=5, permission_mode="auto"))
            except: pass
        AgentTool.set_factory(_make_factory(state.client, state.perms, state.usage, WORKING_DIR, VERBOSE))
        state.engine = QueryEngine(
            tools=tools, client=state.client,
            session=Session(model_client=state.client),
            permission_manager=state.perms, usage=state.usage,
            max_turns=MAX_TURNS, working_dir=WORKING_DIR, verbose=VERBOSE,
            on_tool_start=on_tool_start, on_tool_end=on_tool_end, on_chunk=on_chunk)
        try: state.context_budget = ContextBudget(state.engine)
        except: pass
        try: state.autodream = AutoDream(state.engine)
        except: pass
        try: state.kairos = Kairos(); write_daemon_script()
        except: pass
        state.heartbeat = Heartbeat()
        def _save():
            try: append_session(f"Active — {len(state.engine.session)} messages")
            except: pass
        state.heartbeat.register(every=300, task=_save)
        state.heartbeat.start()
        _new_conv()
        return True

def _new_conv():
    import uuid
    cid = str(uuid.uuid4())[:8]
    state.conversations.append({"id":cid,"title":"New chat","messages":[]})
    state.active_conv_id = cid
    if state.engine:
        state.engine.session.clear()
        state.engine.invalidate_system_prompt()
    return cid

def shutdown_agent():
    if state.heartbeat: state.heartbeat.stop()
    try: append_session(f"Session ended — {len(state.engine.session)} messages")
    except: pass
    stop_llama_server()

def _model_parts():
    p = MODEL_DISPLAY.replace(".gguf","").rsplit("-", 2)
    family = p[0] if len(p)>=1 else MODEL_DISPLAY
    size   = p[1] if len(p)>=2 else ""
    quant  = p[2]+".gguf" if len(p)>=3 else ""
    return family, size, quant

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>llama.cpp</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0a0a;color:#e8e8e8;height:100vh;display:flex;overflow:hidden}
#sidebar{width:260px;min-width:260px;background:#111;display:flex;flex-direction:column;border-right:1px solid #1e1e1e;transition:width .2s,min-width .2s}
#sidebar.collapsed{width:0;min-width:0;overflow:hidden}
.sb-logo{padding:16px 18px;font-size:17px;font-weight:700;color:#fff;border-bottom:1px solid #1e1e1e;display:flex;align-items:center;gap:10px;flex-shrink:0}
.sb-actions{padding:8px 8px 4px;flex-shrink:0}
.sb-btn{display:flex;align-items:center;gap:10px;padding:9px 12px;border-radius:8px;cursor:pointer;color:#bbb;font-size:14px;background:none;border:none;width:100%;text-align:left}
.sb-btn:hover{background:#1a1a1a;color:#fff}
.sb-section{padding:12px 18px 4px;font-size:11px;color:#444;text-transform:uppercase;letter-spacing:.08em;flex-shrink:0}
.conv-list{flex:1;overflow-y:auto;padding:4px 8px 8px}
.conv-item{padding:8px 12px;border-radius:8px;cursor:pointer;font-size:13px;color:#999;display:flex;justify-content:space-between;align-items:center;gap:6px}
.conv-item:hover{background:#1a1a1a;color:#fff}
.conv-item.active{background:#1e1e1e;color:#fff}
.conv-title{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.conv-del{opacity:0;font-size:11px;color:#555;border:none;background:none;cursor:pointer;padding:2px 5px;border-radius:4px;flex-shrink:0}
.conv-item:hover .conv-del{opacity:1}
.conv-del:hover{color:#ef4444}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;border-bottom:1px solid #1a1a1a;flex-shrink:0;gap:10px}
.tb-left{display:flex;align-items:center;gap:8px}
.tb-right{display:flex;align-items:center;gap:6px}
.tb-icon{background:none;border:none;color:#666;cursor:pointer;padding:6px;border-radius:6px;display:flex;align-items:center}
.tb-icon:hover{background:#1a1a1a;color:#ddd}
.perm-group{display:flex;background:#141414;border:1px solid #222;border-radius:20px;padding:2px;gap:1px}
.perm-opt{cursor:pointer;padding:3px 10px;border-radius:16px;font-size:11px;color:#666;transition:all .15s;user-select:none}
.perm-opt:hover{color:#ccc}
.perm-opt.active{background:#2563eb;color:#fff}
#chatWrap{flex:1;overflow-y:auto;scroll-behavior:smooth}
#chatInner{max-width:760px;margin:0 auto;padding:24px 20px 8px}
#emptyState{display:flex;flex-direction:column;align-items:center;justify-content:center;height:calc(100vh - 180px);gap:12px}
#emptyState h2{font-size:28px;font-weight:600;color:#fff}
#emptyState p{color:#555;font-size:14px}
.msg-wrap{margin-bottom:24px}
.msg-user{display:flex;justify-content:flex-end;margin-bottom:4px}
.bubble{background:#1c1c1c;border:1px solid #282828;border-radius:18px 18px 4px 18px;padding:11px 16px;font-size:14px;max-width:72%;line-height:1.55;color:#ddd;white-space:pre-wrap;word-break:break-word}
.reason-block{border:1px solid #222;border-radius:10px;margin-bottom:10px;overflow:hidden}
.reason-hdr{display:flex;align-items:center;gap:8px;padding:10px 14px;cursor:pointer;background:#131313;font-size:13px;color:#888;user-select:none}
.reason-hdr:hover{background:#181818}
.reason-chev{margin-left:auto;transition:transform .2s;color:#555}
.reason-chev.open{transform:rotate(180deg)}
.reason-body{padding:12px 16px;font-size:13px;color:#777;line-height:1.6;font-style:italic;background:#0d0d0d;border-top:1px solid #1a1a1a;max-height:380px;overflow-y:auto;white-space:pre-wrap}
.tool-row{display:flex;align-items:center;gap:8px;padding:7px 12px;border-radius:8px;margin-bottom:4px;font-size:13px;background:#111;border:1px solid #1e1e1e}
.tool-row.ok{border-color:#14291a;background:#0b1a0e}
.tool-row.err{border-color:#2a1212;background:#160c0c}
.t-icon{width:16px;height:16px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:9px;flex-shrink:0;font-weight:700}
.t-icon.pending{background:#252525;color:#666;border:1px solid #333}
.t-icon.ok{background:#14532d;color:#4ade80}
.t-icon.err{background:#7f1d1d;color:#f87171}
.t-name{font-weight:600;color:#bbb;flex-shrink:0}
.t-sum{color:#555;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;font-size:12px}
.answer-text{font-size:14px;line-height:1.7;color:#ddd;word-break:break-word}
.answer-text p{margin:.3em 0}
.answer-text h1,.answer-text h2,.answer-text h3{color:#fff;margin:.7em 0 .3em;font-weight:600}
.answer-text h3{font-size:15px}
.answer-text ul,.answer-text ol{padding-left:22px;margin:.3em 0}
.answer-text li{margin:.2em 0}
.answer-text code{background:#1a1a1a;border:1px solid #252525;border-radius:4px;padding:1px 5px;font-family:Consolas,monospace;font-size:12px;color:#c9d1d9}
.answer-text pre{background:#111;border:1px solid #222;border-radius:8px;padding:14px;overflow-x:auto;margin:.5em 0}
.answer-text pre code{background:none;border:none;padding:0;font-size:13px}
.answer-text strong{color:#fff}
.answer-text a{color:#58a6ff;text-decoration:none}
.answer-text a:hover{text-decoration:underline}
.msg-meta{display:flex;align-items:center;gap:8px;margin-top:10px;flex-wrap:wrap}
.meta-badge{background:#131313;border:1px solid #1e1e1e;border-radius:20px;padding:3px 10px;font-size:11px;color:#777;display:flex;align-items:center;gap:5px}
.pill-name{color:#bbb}
.pill-size{background:#1d4ed8;color:#fff;border-radius:10px;padding:1px 6px;font-size:10px;font-weight:700}
.pill-quant{color:#555}
.meta-stat{font-size:11px;color:#444}
.msg-actions{display:flex;gap:2px;margin-top:6px;opacity:0;transition:opacity .15s}
.msg-wrap:hover .msg-actions{opacity:1}
.act-btn{background:none;border:none;color:#555;cursor:pointer;padding:5px;border-radius:6px;display:flex}
.act-btn:hover{background:#1a1a1a;color:#bbb}
#bottom{padding:10px 16px 14px;flex-shrink:0}
#inputWrap{max-width:760px;margin:0 auto;background:#111;border:1px solid #222;border-radius:14px;overflow:hidden;transition:border-color .15s}
#inputWrap:focus-within{border-color:#333}
#msgInput{width:100%;background:none;border:none;color:#e0e0e0;font-size:14px;padding:13px 16px 6px;resize:none;outline:none;min-height:46px;max-height:160px;font-family:inherit;line-height:1.5}
#msgInput::placeholder{color:#444}
.inp-footer{display:flex;align-items:center;padding:5px 10px 9px;gap:8px}
.model-pill{margin-left:auto;background:#141414;border:1px solid #222;border-radius:20px;padding:4px 10px;font-size:12px;display:flex;align-items:center;gap:5px}
#sendBtn{background:#e0e0e0;border:none;border-radius:50%;width:30px;height:30px;display:flex;align-items:center;justify-content:center;cursor:pointer;color:#0a0a0a;flex-shrink:0;transition:background .15s}
#sendBtn:hover{background:#fff}
#sendBtn:disabled{background:#222;color:#444;cursor:not-allowed}
#sendBtn.stop{background:#dc2626;color:#fff}
#statsBar{max-width:760px;margin:6px auto 0;display:flex;justify-content:center;gap:20px;font-size:11px;color:#333}
#scrollBtn{position:fixed;bottom:110px;right:22px;background:#181818;border:1px solid #2a2a2a;border-radius:50%;width:34px;height:34px;display:none;align-items:center;justify-content:center;cursor:pointer;color:#777}
#scrollBtn:hover{background:#222;color:#fff}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#222;border-radius:3px}
</style>
</head>
<body>
<div id="sidebar">
  <div class="sb-logo">
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4l3 3"/></svg>
    llama.cpp
  </div>
  <div class="sb-actions">
    <button class="sb-btn" onclick="newChat()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14M5 12h14"/></svg>New chat
    </button>
    <button class="sb-btn" onclick="document.getElementById('msgInput').focus()">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>Search
    </button>
    <button class="sb-btn" onclick="runSlash('/skills')">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>Skills &amp; Tools
    </button>
  </div>
  <div class="sb-section">Conversations</div>
  <div class="conv-list" id="convList"></div>
</div>

<div id="main">
  <div id="topbar">
    <div class="tb-left">
      <button class="tb-icon" onclick="document.getElementById('sidebar').classList.toggle('collapsed')" title="Toggle sidebar">
        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 3v18"/></svg>
      </button>
    </div>
    <div class="tb-right">
      <div class="perm-group">
        <span class="perm-opt" id="perm-auto" onclick="setPerm('auto')">AUTO</span>
        <span class="perm-opt" id="perm-ask"  onclick="setPerm('ask')">ASK</span>
        <span class="perm-opt" id="perm-deny" onclick="setPerm('deny')">DENY</span>
      </div>
      <button class="tb-icon" onclick="clearChat()" title="Clear session">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/></svg>
      </button>
      <button class="tb-icon" onclick="saveSession()" title="Save">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>
      </button>
      <button class="tb-icon" onclick="openSettings()" title="Settings">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
      </button>
    </div>
  </div>

  <div id="chatWrap" onscroll="onScroll()">
    <div id="chatInner">
      <div id="emptyState">
        <h2>llama.cpp</h2>
        <p>Type a message or upload files to get started</p>
      </div>
    </div>
  </div>

  <div id="bottom">
    <div id="inputWrap">
      <textarea id="msgInput" rows="1" placeholder="Type a message..."
        onkeydown="onKey(event)" oninput="grow(this)"></textarea>
      <div class="inp-footer">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#555" stroke-width="2" style="cursor:pointer" title="Attach"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
        <div class="model-pill">
          <span class="pill-name">{{ model_family }}</span>
          <span class="pill-size">{{ model_size }}</span>
          <span class="pill-quant">{{ model_quant }}</span>
        </div>
        <button id="sendBtn" onclick="send()">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>
        </button>
      </div>
    </div>
    <div id="statsBar">
      <span id="sCtx">Context: 0/{{ ctx_size }} (0%)</span>
      <span id="sOut">Output: 0/∞</span>
      <span id="sSpd"></span>
    </div>
    <div style="text-align:center;margin-top:6px;font-size:11px;color:#2a2a2a">
      Press <kbd style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:3px;padding:1px 5px;color:#555">Enter</kbd> to send,
      <kbd style="background:#1a1a1a;border:1px solid #2a2a2a;border-radius:3px;padding:1px 5px;color:#555">Shift+Enter</kbd> for new line
    </div>
  </div>
</div>

<button id="scrollBtn" onclick="scrollBot()">
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
</button>

<script>
let busy=false, es=null, convs=[], activeId=null, currentPerm='{{ perm }}';
let _tC=0,_pendId=null,_pendName=null;
let _wrap=null,_thEl=null,_anEl=null,_tMap={},_ans='',_t0=0;
const CTX={{ctx_size}};

// ── sidebar ──────────────────────────────────────────────────────────────────
function renderSB(){
  const el=document.getElementById('convList'); el.innerHTML='';
  [...convs].reverse().forEach(c=>{
    const d=document.createElement('div');
    d.className='conv-item'+(c.id===activeId?' active':'');
    d.innerHTML=`<span class="conv-title">${esc(c.title)}</span><button class="conv-del" onclick="delC(event,'${c.id}')">✕</button>`;
    d.onclick=e=>{if(!e.target.classList.contains('conv-del'))switchC(c.id);};
    el.appendChild(d);
  });
}

function switchC(id){
  activeId=id;
  fetch('/api/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id})});
  renderSB();
  const c=convs.find(x=>x.id===id);
  rebuildChat(c?c.messages:[]);
}

function delC(e,id){
  e.stopPropagation(); convs=convs.filter(c=>c.id!==id);
  if(activeId===id){convs.length?switchC(convs[convs.length-1].id):newChat();}
  else renderSB();
}

async function newChat(){
  const r=await fetch('/api/new_chat',{method:'POST'}); const d=await r.json();
  convs.push({id:d.id,title:'New chat',messages:[]});
  activeId=d.id; renderSB(); rebuildChat([]); showEmpty(true);
}

// ── chat ─────────────────────────────────────────────────────────────────────
function showEmpty(v){document.getElementById('emptyState').style.display=v?'flex':'none';}

function rebuildChat(msgs){
  const ci=document.getElementById('chatInner');
  ci.innerHTML='<div id="emptyState" style="display:none;flex-direction:column;align-items:center;justify-content:center;height:calc(100vh - 180px);gap:12px"><h2>llama.cpp</h2><p>Type a message or upload files to get started</p></div>';
  msgs.forEach(m=>{ if(m.role==='user') addUser(m.content); else addAgent(m); });
  if(!msgs.length) showEmpty(true);
}

function addUser(txt){
  showEmpty(false);
  const w=mk('div','msg-wrap'),r=mk('div','msg-user'),b=mk('div','bubble');
  b.textContent=txt; r.appendChild(b); w.appendChild(r);
  ci().appendChild(w); scrollBot(); return w;
}

function addAgent(d){
  showEmpty(false);
  const w=mk('div','msg-wrap');
  if(d.thinking&&d.thinking.trim()){
    const rb=mk('div','reason-block');
    const rh=mk('div','reason-hdr');
    rh.innerHTML=`<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.89A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.89A2.5 2.5 0 0 0 14.5 2Z"/></svg>Reasoning<svg class="reason-chev open" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>`;
    const rb2=mk('div','reason-body'); rb2.textContent=d.thinking;
    rh.onclick=()=>{const o=rb2.style.display!=='none';rb2.style.display=o?'none':'';rh.querySelector('.reason-chev').classList.toggle('open',!o);};
    rb.appendChild(rh); rb.appendChild(rb2); w.appendChild(rb);
  }
  (d.tools||[]).forEach(t=>{
    const row=mk('div','tool-row '+(t.ok?'ok':'err'));
    const ic=mk('span','t-icon '+(t.ok?'ok':'err')); ic.textContent=t.ok?'✓':'✕';
    const nm=mk('span','t-name'); nm.textContent=t.name;
    const sm=mk('span','t-sum');  sm.textContent=t.output||t.summary||'';
    row.appendChild(ic); row.appendChild(nm); row.appendChild(sm); w.appendChild(row);
  });
  const at=mk('div','answer-text'); at.innerHTML=md(d.answer||''); w.appendChild(at);
  appendMeta(w,d); ci().appendChild(w); scrollBot(); return w;
}

function appendMeta(w,d){
  const meta=mk('div','msg-meta');
  const badge=mk('div','meta-badge');
  badge.innerHTML=`<span class="pill-name">${q('pillName')}</span><span class="pill-size">${q('pillSize')}</span><span class="pill-quant">${q('pillQuant')}</span>`;
  meta.appendChild(badge);
  if(d&&d.elapsed){const s=mk('span','meta-stat');s.textContent=`⏱ ${d.elapsed}s`;meta.appendChild(s);}
  const acts=mk('div','msg-actions');
  acts.innerHTML=`<button class="act-btn" onclick="cpText(this)" title="Copy"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`;
  w.appendChild(meta); w.appendChild(acts);
}

// ── stream helpers ────────────────────────────────────────────────────────────
function startStream(){
  _wrap=null;_thEl=null;_anEl=null;_tMap={};_ans='';_t0=Date.now();
  showEmpty(false);
}

function ensureWrap(){if(!_wrap){_wrap=mk('div','msg-wrap');ci().appendChild(_wrap);}}

function streamThinkStart(){
  ensureWrap();
  const rb=mk('div','reason-block');
  const rh=mk('div','reason-hdr');
  rh.innerHTML=`<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.89A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.89A2.5 2.5 0 0 0 14.5 2Z"/></svg>Reasoning<svg class="reason-chev open" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="18 15 12 9 6 15"/></svg>`;
  _thEl=mk('div','reason-body');
  rh.onclick=()=>{const o=_thEl.style.display!=='none';_thEl.style.display=o?'none':'';rh.querySelector('.reason-chev').classList.toggle('open',!o);};
  rb.appendChild(rh); rb.appendChild(_thEl); _wrap.appendChild(rb); scrollBot();
}

function streamTool(name,summary,id){
  ensureWrap();
  const row=mk('div','tool-row'); row.id='t'+id;
  const ic=mk('span','t-icon pending'); ic.textContent='…';
  const nm=mk('span','t-name'); nm.textContent=name;
  const sm=mk('span','t-sum');  sm.textContent=summary;
  row.appendChild(ic); row.appendChild(nm); row.appendChild(sm);
  _wrap.appendChild(row); _tMap[id]=row; scrollBot();
}

function streamToolDone(id,ok,out){
  const row=_tMap[id]; if(!row) return;
  row.className='tool-row '+(ok?'ok':'err');
  const ic=row.querySelector('.t-icon');
  ic.className='t-icon '+(ok?'ok':'err'); ic.textContent=ok?'✓':'✕';
  const sm=row.querySelector('.t-sum'); if(sm) sm.textContent=out||'';
}

function streamAnswer(txt){
  if(!_anEl){ensureWrap();_anEl=mk('div','answer-text');_wrap.appendChild(_anEl);}
  _ans+=txt; _anEl.innerHTML=md(_ans); scrollBot();
}

function finishStream(){
  if(!_wrap) return;
  const el=((Date.now()-_t0)/1000).toFixed(1);
  appendMeta(_wrap,{elapsed:el}); scrollBot();
}

// ── send ──────────────────────────────────────────────────────────────────────
function send(){
  if(busy){stopGen();return;}
  const inp=document.getElementById('msgInput');
  const txt=inp.value.trim(); if(!txt) return;
  inp.value=''; grow(inp); setBusy(true); addUser(txt);
  const c=convs.find(x=>x.id===activeId);
  if(c&&c.title==='New chat') c.title=txt.slice(0,42);
  renderSB(); startStream();
  es=new EventSource('/api/stream?message='+encodeURIComponent(txt));
  es.onmessage=handleSSE; es.onerror=()=>{streamAnswer('\n❌ Connection error');finishStream();setBusy(false);es.close();};
}

function runSlash(cmd){
  if(busy) return;
  setBusy(true); startStream();
  es=new EventSource('/api/stream?message='+encodeURIComponent(cmd));
  es.onmessage=handleSSE; es.onerror=()=>{finishStream();setBusy(false);es.close();};
}

function handleSSE(ev){
  const d=JSON.parse(ev.data);
  if(d.type==='tool_start'){streamTool(d.name,d.summary,++_tC);_pendId=_tC;}
  else if(d.type==='tool_end'){if(_pendId)streamToolDone(_pendId,d.success,d.output);_pendId=null;}
  else if(d.type==='think_start'){streamThinkStart();}
  else if(d.type==='think'){if(_thEl){_thEl.textContent+=d.content;scrollBot();}}
  else if(d.type==='think_end'){}
  else if(d.type==='answer_start'||d.type==='answer'){streamAnswer(d.content||'');}
  else if(d.type==='done'){finishStream();setBusy(false);es.close();updateStats();}
  else if(d.type==='error'){streamAnswer('\n❌ '+d.content);finishStream();setBusy(false);es.close();}
}

function stopGen(){if(es)es.close();setBusy(false);finishStream();}
function setBusy(b){
  busy=b;
  const btn=document.getElementById('sendBtn'); btn.disabled=false;
  btn.className=b?'stop':'';
  btn.innerHTML=b
    ?'<svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>'
    :'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>';
}

// ── controls ──────────────────────────────────────────────────────────────────
async function setPerm(m){
  await fetch('/api/permission',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:m})});
  currentPerm=m;
  document.querySelectorAll('.perm-opt').forEach(e=>e.classList.remove('active'));
  document.getElementById('perm-'+m).classList.add('active');
}
async function clearChat(){
  await fetch('/api/clear',{method:'POST'});
  const c=convs.find(x=>x.id===activeId); if(c) c.messages=[];
  rebuildChat([]); showEmpty(true);
}
async function saveSession(){const r=await fetch('/api/save',{method:'POST'});const d=await r.json();alert(d.message);}
function openSettings(){const m=prompt('Permission (auto/ask/deny):',currentPerm);if(m)setPerm(m.trim().toLowerCase());}
async function updateStats(){
  const r=await fetch('/api/stats'); const d=await r.json();
  const pct=Math.min(100,Math.round((d.tokens/CTX)*100));
  document.getElementById('sCtx').textContent=`Context: ${d.tokens}/${CTX} (${pct}%)`;
  document.getElementById('sOut').textContent=`Output: ${d.out||0}/∞`;
}

// ── utils ─────────────────────────────────────────────────────────────────────
function onKey(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}}
function grow(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,160)+'px';}
function scrollBot(){const w=document.getElementById('chatWrap');w.scrollTop=w.scrollHeight;}
function onScroll(){const w=document.getElementById('chatWrap');document.getElementById('scrollBtn').style.display=w.scrollHeight-w.scrollTop-w.clientHeight>80?'flex':'none';}
function mk(t,c){const e=document.createElement(t);if(c)e.className=c;return e;}
function ci(){return document.getElementById('chatInner');}
function q(id){const e=document.querySelector('.model-pill .'+id.replace('pill','pill-').toLowerCase());return e?e.textContent:'';}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function cpText(btn){navigator.clipboard.writeText(_ans).catch(()=>{});}

// pill helpers
function q(cls){const e=document.querySelector('.inp-footer .'+cls);return e?e.textContent:'';}
function pillName(){return document.querySelector('.inp-footer .pill-name').textContent;}
function pillSize(){return document.querySelector('.inp-footer .pill-size').textContent;}
function pillQuant(){return document.querySelector('.inp-footer .pill-quant').textContent;}
function appendMeta(w,d){
  const meta=mk('div','msg-meta');
  const badge=mk('div','meta-badge');
  badge.innerHTML=`<span class="pill-name">${pillName()}</span><span class="pill-size">${pillSize()}</span><span class="pill-quant">${pillQuant()}</span>`;
  meta.appendChild(badge);
  if(d&&d.elapsed){const s=mk('span','meta-stat');s.textContent=`⏱ ${d.elapsed}s`;meta.appendChild(s);}
  const acts=mk('div','msg-actions');
  acts.innerHTML=`<button class="act-btn" onclick="navigator.clipboard.writeText(this.closest('.msg-wrap').querySelector('.answer-text').textContent)" title="Copy"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>`;
  w.appendChild(meta); w.appendChild(acts);
}

function md(s){
  s=s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  s=s.replace(/```(\w*)\n?([\s\S]*?)```/g,(_,l,c)=>`<pre><code>${c.trim()}</code></pre>`);
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/^### (.+)$/gm,'<h3>$1</h3>');
  s=s.replace(/^## (.+)$/gm,'<h2>$1</h2>');
  s=s.replace(/^# (.+)$/gm,'<h1>$1</h1>');
  s=s.replace(/^\s*[-*]\s+(.+)$/gm,'<li>$1</li>');
  s=s.replace(/(<li>[\s\S]+?<\/li>)/g,'<ul>$1</ul>');
  s=s.replace(/\n\n+/g,'</p><p>');
  s=s.replace(/\n/g,'<br>');
  return '<p>'+s+'</p>';
}

// ── init ──────────────────────────────────────────────────────────────────────
(async()=>{
  const r=await fetch('/api/convs'); const d=await r.json();
  convs=d.convs||[]; activeId=d.active||null;
  if(!convs.length) await newChat(); else{renderSB();showEmpty(true);}
  document.getElementById('perm-{{ perm }}').classList.add('active');
  document.getElementById('msgInput').focus();
  updateStats();
})();
</script>
</body>
</html>"""

# ── routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    fam, sz, qt = _model_parts()
    return render_template_string(HTML,
        model_family=fam, model_size=sz, model_quant=qt,
        ctx_size=CONTEXT_SIZE,
        perm=state.perms.mode.value if state.perms else PERMISSION_MODE)

@app.route('/api/convs')
def get_convs():
    return jsonify({"convs":[{"id":c["id"],"title":c["title"]} for c in state.conversations],
                    "active": state.active_conv_id})

@app.route('/api/new_chat', methods=['POST'])
def api_new_chat():
    return jsonify({"id": _new_conv()})

@app.route('/api/switch', methods=['POST'])
def api_switch():
    cid = request.json.get("id")
    c   = next((x for x in state.conversations if x["id"]==cid), None)
    if c and state.engine:
        state.active_conv_id = cid
        state.engine.session.clear()
        state.engine.invalidate_system_prompt()
        for m in c.get("messages",[]):
            if m["role"]=="user": state.engine.session.add_user(m["content"])
            else: state.engine.session.add_assistant(m["content"])
    return jsonify({"ok":True})

@app.route('/api/stream')
def stream():
    message = request.args.get('message','')
    if not message or not state.engine:
        def _e():
            yield f"data: {json.dumps({'type':'error','content':'Engine not ready'})}\n\n"
        return Response(_e(), mimetype='text/event-stream')

    def generate():
        state.think_parser = ThinkingStreamParser()
        state.in_think     = False
        while not state.output_queue.empty():
            try: state.output_queue.get_nowait()
            except queue.Empty: break

        def run():
            try:
                if message.startswith('/'):
                    cmd=message.strip(); low=cmd.lower()
                    if low in('/clear','/reset'):
                        state.engine.session.clear(); state.engine.invalidate_system_prompt()
                        state.output_queue.put({"type":"answer","content":"✅ Session cleared."})
                    elif low=='/compact':
                        state.engine.session._unlimited=True; state.engine.session._maybe_compress()
                        state.output_queue.put({"type":"answer","content":"📦 Context compacted."})
                    elif low in('/status','/cost'):
                        state.output_queue.put({"type":"answer","content":state.usage.summary()+f"\nMessages: {len(state.engine.session)}"})
                    elif low.startswith('/mode '):
                        nm=cmd[6:].strip()
                        try: state.perms.set_mode(nm); state.output_queue.put({"type":"answer","content":f"Mode → {nm}"})
                        except: state.output_queue.put({"type":"answer","content":"Modes: auto|ask|deny"})
                    elif low.startswith('/save'):
                        parts=cmd.split(maxsplit=1); p=Path(parts[1]) if len(parts)>1 else PROJECT_ROOT/"session.json"
                        state.engine.session.save(p); state.output_queue.put({"type":"answer","content":f"💾 Saved → {p}"})
                    elif low=='/memory':
                        state.output_queue.put({"type":"answer","content":load_context() or "(no memories)"})
                    elif low in('/skills','/skill'):
                        res=SkillTool().execute({}); state.output_queue.put({"type":"answer","content":res.output})
                    elif low.startswith('/skill '):
                        rest=cmd[7:].strip(); parts=rest.split(maxsplit=1)
                        sname=parts[0] if parts else ""; sprob=parts[1] if len(parts)>1 else ""
                        if sname:
                            res=SkillTool().execute({"skill":sname,"args":{"problem":sprob or sname}})
                            state.output_queue.put({"type":"answer","content":res.output})
                    elif low in('/help','/?'):
                        state.output_queue.put({"type":"answer","content":"/clear /compact /status /mode /save /memory /skills /skill NAME [Q]"})
                    else:
                        state.output_queue.put({"type":"answer","content":f"Unknown: {cmd}"})
                    state.output_queue.put({"type":"done"}); return

                if should_ultraplan(message):
                    try: UltraPlan(state.engine).run(message)
                    except: state.engine.submit_message(message)
                else:
                    state.engine.submit_message(message)
                flush_think()
                state.output_queue.put({"type":"done"})
            except Exception as e:
                state.output_queue.put({"type":"error","content":str(e)})

        threading.Thread(target=run, daemon=True).start()
        while True:
            try:
                item=state.output_queue.get(timeout=0.1)
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("type") in("done","error"): break
            except queue.Empty: continue

    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/permission', methods=['POST'])
def set_perm():
    d=request.json
    if state.perms and 'mode' in d: state.perms.set_mode(d['mode'])
    return jsonify({"ok":True})

@app.route('/api/clear', methods=['POST'])
def clear():
    if state.engine: state.engine.session.clear(); state.engine.invalidate_system_prompt()
    return jsonify({"ok":True})

@app.route('/api/save', methods=['POST'])
def save():
    if state.engine:
        p=PROJECT_ROOT/"session.json"; state.engine.session.save(p)
        return jsonify({"message":f"Saved → {p}"})
    return jsonify({"message":"Engine not ready"})

@app.route('/api/stats')
def stats():
    return jsonify({"tokens":state.usage.total_tokens if state.usage else 0,
                    "out":   state.usage.out_tokens   if state.usage else 0})

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*58}")
    print("  🦙 Local Code Agent — WEB GUI")
    print(f"{'='*58}")
    print(f"  Model  : {MODEL_DISPLAY}")
    print(f"  Server : {LLAMA_SERVER_URL}")
    print(f"  CWD    : {WORKING_DIR}")
    print(f"  Context: {'unlimited' if UNLIMITED_CONTEXT else str(CONTEXT_SIZE)+' tokens'}")
    try: n=len(list((PROJECT_ROOT/"skills").glob("*.py")))-1
    except: n=0
    print(f"  Skills : {n} reasoning skills loaded")
    print(f"{'='*58}")
    if start_llama_server(): print("  ✅ llama.cpp server running")
    else:                    print("  ⚠️  server not started")
    if init_agent():         print("  ✅ Agent initialized")
    else:                    print("  ❌ Agent init failed"); return
    print("\n  🌐 http://localhost:5000\n")
    import flask.cli as cli; cli.show_server_banner=lambda*a,**k:None
    def _sig(s,f): print("\n🛑 Shutting down…"); shutdown_agent(); sys.exit(0)
    signal.signal(signal.SIGINT, _sig)
    try: app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
    except KeyboardInterrupt: shutdown_agent()

if __name__ == "__main__":
    main()