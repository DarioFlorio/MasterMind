# -*- coding: utf-8 -*-
"""
MasterMind WebUI -- FastAPI + SSE  (v4)
========================================
v4 additions (continuing from v3):
  - File upload: paperclip button injects file content into message
  - Stop: red stop button aborts the current stream mid-flight
  - Pause/Resume: freeze rendering without closing the connection
  - CPU perf: settings.py + model_client.py updated (see those files)

Install:  pip install fastapi uvicorn python-multipart
Run:      cd MasterMind && python webui.py
Opens at: http://localhost:7860
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import AsyncIterator

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import utils.db as db

from config.settings import (
    MODEL_PATH, DIRECT_MODE, LLAMA_SERVER_URL,
    CONTEXT_SIZE, MAX_TOKENS, TEMPERATURE, PERMISSION_MODE, WORKING_DIR,
)
from utils.model_client import ModelClient
from utils.permissions import PermissionManager
from utils.token_counter import SessionUsage
from agent.query_engine import QueryEngine
from agent.session import Session
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
# GAP: User-specific vector store search
from tools.memory_search_tool import MemorySearchTool
from tools.wiki_tool import (
    WikiWriteTool, WikiReadTool, WikiSearchTool,
    WikiListTool, WikiPromoteTool,
)
from skills.self_healing import SelfHealingTool
from skills.code_remediation import CodeRemediationTool

# ── Connectors (WhatsApp, Email — self-register on import) ────────────────────
import connectors  # loads the registry
import connectors.whatsapp        # self-registers _wa_connector
import connectors.email_connector  # self-registers _email_connector
from connectors import registry as _connector_registry

# ── Uploaded-models directory (user can drop .gguf files here) ────────────────
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="MasterMind WebUI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Initialise SQLite on startup
db.init()

# ── Shared model client ───────────────────────────────────────────────────────
_client: ModelClient | None = None
_client_lock = threading.Lock()

def get_client() -> ModelClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                print("[webui] Loading model...", flush=True)
                # Pull model path + mmproj from persisted settings if present
                stored = db.get_settings()
                mdl    = stored.get("model", {})
                m_path = mdl.get("path") or MODEL_PATH
                mmproj = mdl.get("mmproj", "")
                _client = ModelClient(
                    base_url    = LLAMA_SERVER_URL,
                    direct      = DIRECT_MODE,
                    model_path  = m_path,
                    mmproj_path = mmproj,
                )
                print("[webui] Model ready.", flush=True)
    return _client

def build_tools(cwd: str) -> list:
    return [
        BashTool(working_dir=cwd), ReadFileTool(working_dir=cwd),
        WriteFileTool(working_dir=cwd), EditFileTool(working_dir=cwd),
        GlobTool(working_dir=cwd), GrepTool(working_dir=cwd),
        ListDirTool(working_dir=cwd), WebSearchTool(), WebFetchTool(),
        AgentTool(), TodoWriteTool(), TodoReadTool(),
        MemoryWriteTool(), MemoryReadTool(), SkillTool(), PMTool(),
        ScratchpadTool(), ReflectTool(), GitTool(), ExportTool(),
        JournalTool(), TestRunnerTool(),
        WikiWriteTool(working_dir=cwd), WikiReadTool(working_dir=cwd),
        WikiSearchTool(working_dir=cwd), WikiListTool(working_dir=cwd),
        WikiPromoteTool(working_dir=cwd),
        SelfHealingTool(working_dir=cwd),
        CodeRemediationTool(working_dir=cwd),
        MemorySearchTool(working_dir=cwd),
    ]


# ── Sessions ──────────────────────────────────────────────────────────────────
_sessions: dict[str, QueryEngine] = {}
_sessions_lock = threading.Lock()
_connectors_inited = False


def get_engine(session_id: str) -> QueryEngine:
    global _connectors_inited
    with _sessions_lock:
        if session_id not in _sessions:
            stored  = db.get_settings()
            inf     = stored.get("inference", {})
            cwd     = WORKING_DIR or str(ROOT)
            client  = get_client()
            tools   = build_tools(cwd)
            engine  = QueryEngine(
                tools              = tools,
                client             = client,
                session            = Session.resume_or_create(model_client=client),
                permission_manager = PermissionManager(
                    inf.get("perm_mode", PERMISSION_MODE)
                ),
                usage              = SessionUsage(),
                max_turns          = int(inf.get("max_turns", 200)),
                working_dir        = cwd,
                verbose            = False,
                is_subagent        = False,
            )
            _sessions[session_id] = engine

            # Initialise connectors once (on first engine creation)
            if not _connectors_inited:
                _connectors_inited = True
                _restore_connector_settings(stored)
                threading.Thread(
                    target=_connector_registry.init_all,
                    args=(engine,),
                    daemon=True,
                    name="connectors-init",
                ).start()

    return _sessions[session_id]


def _restore_connector_settings(stored: dict) -> None:
    """Restore saved connector enabled/config state from DB on startup."""
    conn_cfg = stored.get("connectors", {})
    for cid, c in _connector_registry.all().items():
        cfg = conn_cfg.get(cid, {})
        if cfg.get("config"):
            c.configure(cfg["config"])
        if cfg.get("enabled", False):
            try:
                c.enable(cfg.get("config"))
            except Exception as e:
                print(f"[connectors] restore {cid}: {e}", flush=True)

# ── Stop flags (per active request) ──────────────────────────────────────────
_stop_flags: dict[str, threading.Event] = {}
_stop_lock = threading.Lock()

# ── SSE helper ────────────────────────────────────────────────────────────────
def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

# ── Streaming XML tag filter ──────────────────────────────────────────────────
# All tag forms that open/close a tool-call block.
# Covers proper format (<tool_use>) AND model variants (<tooluse>, <toolcall>)
_TOOL_OPEN_TAGS  = {"<tool_use>", "<tooluse>", "<toolcall>", "<tool>"}
_TOOL_CLOSE_TAGS = {"</tool_use>", "</tooluse>", "</toolcall>", "</tool>"}
# Tags that are structural-only and should be silently dropped (not emitted)
_SILENT_TAGS = {
    "<n>", "</n>", "<name>", "</name>",
    "<input>", "</input>",
    "<arg_key>", "</arg_key>",
    "<arguments>", "</arguments>",
    "<parameters>", "</parameters>",
    "<tool_result>", "</tool_result>",
    "<o>", "</o>",
}
# Updated dangling-tag stripper used at flush() time
_DANGLING_RE = re.compile(
    r"</?(?:tool_use|tooluse|toolcall|tool|think|n|name|input|"
    r"arg_key|arguments|parameters|tool_result|o)(?:\s[^>]*)?>",
    re.IGNORECASE,
)
_TOOL_BLOCK_RE  = re.compile(r"<tool_use>.*?</tool_use>", re.DOTALL)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>",      re.DOTALL)

# [F6] Streaming system-prompt-leak detector — same phrases as query_engine
_STREAM_LEAK_PHRASES = [
    r"Never output plain text for tool instructions",
    r"Always use the <tool_use> format",
    r"Never reveal tool names or XML tags",
    r"When you have a result.*speak naturally",
    r"Never output the file content to the chat",
    r"Do not reveal tool names or XML tags",
    r"TOOLS vs SKILLS",
    r"CRITICAL: write_file, bash",
    r"CRITICAL: If a skill name is NOT",
    r"CRITICAL: Never fabricate tool results",
    r"You are MasterMind.*agentic AI",
    r"OUTPUT RULES",
    r"DIRECT TOOLS",
    # ── Model narrating its own tool instructions as prose ─────────────────
    r"I need to (?:instruct|determine|explain) (?:the user|how) (?:to use|to run)",
    r"Based on the given information.*web",
    r"web_fetch\(url\*",
    r"web_search\(query\*",
    r"use `?web_fetch`? to search for",
    r"use `?web_search`? to verify",
    r"After retrieving the results.*use web_",
    r"This process (?:ensures|allows) (?:that )?you (?:first )?fetch",
    r"To execute the web search and verify",
    r"perform the following steps",
]
_STREAM_LEAK_RE = re.compile(
    "|".join(f"(?:{p})" for p in _STREAM_LEAK_PHRASES),
    re.IGNORECASE | re.DOTALL,
)

class _StreamTagFilter:
    # ── CRITICAL FIX: was 2, causing any tag > 4 chars that arrived split
    # across chunks to have its '<' emitted immediately to the browser.
    # E.g. <tool_use> arriving as <tool_us (8 chars) → 8 > 4 → flush('<')!
    # Set to 40: safely holds any partial XML tag without leaking content.
    _MAX_TAG_LEN = 40

    def __init__(self, emit_chunk, emit_think):
        self._emit  = emit_chunk
        self._think = emit_think
        self._buf      = ""
        self._in_think = False
        self._in_tool  = False

    def feed(self, chunk: str):
        self._buf += chunk
        self._process()

    def flush(self):
        remaining = _DANGLING_RE.sub("", self._buf)
        remaining = re.sub(r"\n{3,}", "\n\n", remaining).strip()
        if remaining:
            if self._in_think:
                self._think(remaining)
            elif not self._in_tool:
                self._emit(remaining)
        self._buf      = ""
        self._in_think = False
        self._in_tool  = False

    def _process(self):
        while self._buf:
            tag_start = self._buf.find("<")
            if tag_start == -1:
                # No '<' — safe to emit everything except the last few chars
                # (hold back _MAX_TAG_LEN in case a '<' arrives next chunk)
                safe = len(self._buf) - self._MAX_TAG_LEN
                if safe > 0:
                    self._flush_text(self._buf[:safe])
                    self._buf = self._buf[safe:]
                break

            if tag_start > 0:
                self._flush_text(self._buf[:tag_start])
                self._buf = self._buf[tag_start:]

            end = self._buf.find(">")
            if end == -1:
                # Incomplete tag in buffer — wait for more data.
                # Only give up on it if the buffer is larger than any
                # conceivable XML tag (>40 chars and still no '>').
                if len(self._buf) > self._MAX_TAG_LEN:
                    # Not a tag — emit the '<' and continue parsing
                    self._flush_text("<")
                    self._buf = self._buf[1:]
                break

            tag = self._buf[:end + 1].lower()
            self._buf = self._buf[end + 1:]

            # ── Think block ────────────────────────────────────────────────
            if tag == "<think>":
                self._in_think = True
            elif tag == "</think>":
                self._in_think = False
            # ── Tool-call block (all variant spellings) ────────────────────
            elif tag in _TOOL_OPEN_TAGS:
                self._in_tool = True
            elif tag in _TOOL_CLOSE_TAGS:
                self._in_tool = False
            # ── Structural-only tags — drop silently ───────────────────────
            elif tag in _SILENT_TAGS:
                pass
            # ── Any <tool…> prefix tag not in the above sets ──────────────
            # (catches <tool_input>, <tool_name>, <toolname>, etc.)
            elif tag.startswith("<tool") or tag.startswith("</tool"):
                if tag.startswith("<tool") and not tag.startswith("</tool"):
                    self._in_tool = True   # unknown tool-open → suppress
                # else: unknown tool-close → leave _in_tool as-is
            else:
                self._flush_text(tag)

    def _flush_text(self, text: str):
        if not text:
            return
        # [F6] suppress system-prompt leaks before they reach the browser
        if _STREAM_LEAK_RE.search(text):
            return
        if self._in_think:
            self._think(text)
        elif not self._in_tool:
            self._emit(text)

# ── Upload endpoint ───────────────────────────────────────────────────────────
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Receive a file and return its text content to inject into the chat."""
    raw = await file.read()
    # Try UTF-8 first, fall back to latin-1
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        return {"error": "Could not decode file as text", "filename": file.filename}

    return {
        "filename": file.filename,
        "content": text,
        "size": len(text),
        "lines": text.count("\n") + 1,
    }

# ── Stop endpoint ─────────────────────────────────────────────────────────────
@app.post("/stop/{session_id}")
async def stop_session(session_id: str):
    """Signal the active stream for this session to stop."""
    with _stop_lock:
        ev = _stop_flags.get(session_id)
    if ev:
        ev.set()
    return {"ok": True}

# ── Chat endpoint ─────────────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: Request):
    body       = await request.json()
    user_msg   = body.get("message", "").strip()
    session_id = body.get("session_id", "default")

    if not user_msg:
        return {"error": "empty message"}

    # Register stop flag for this request
    stop_ev = threading.Event()
    with _stop_lock:
        _stop_flags[session_id] = stop_ev

    q: queue.Queue = queue.Queue()

    filt = _StreamTagFilter(
        emit_chunk=lambda text: q.put(("chunk", text)),
        emit_think=lambda text: q.put(("think", text)),
    )

    def on_chunk(chunk: str):
        filt.feed(chunk)

    def on_tool_start(name: str, inp: dict):
        if name == "bash":
            cmd = inp.get("command", "").strip()
            s = cmd[:72] + ("…" if len(cmd) > 72 else "")
        elif name == "web_search":
            s = inp.get("query", "")[:60]
        elif name == "web_fetch":
            url = inp.get("url", "")
            s = url.split("?")[0][:60]
        elif name in ("read_file", "write_file", "edit_file"):
            path = inp.get("path", inp.get("file_path", ""))
            s = Path(path).name if path else ""
        elif name == "glob":
            s = inp.get("pattern", "")[:50]
        elif name == "grep":
            s = f'"{inp.get("pattern","")[:30]}"'
        elif name == "list_dir":
            s = inp.get("path", ".")[:50]
        elif name == "git":
            s = inp.get("command", "")[:50]
        elif name == "skill":
            skill_name = inp.get("skill", "")
            prob = (inp.get("args") or {}).get("problem", "")[:40]
            skill_label = skill_name.replace("_", " ").title()
            s = f"{skill_label}" + (f" — {prob}" if prob else "")
        elif name == "memory_write":
            s = "saving to memory"
        elif name == "memory_read":
            s = "reading memory"
        elif name == "todo_write":
            s = "updating tasks"
        elif name == "todo_read":
            s = "reading tasks"
        elif name == "scratchpad":
            s = "scratchpad"
        elif name == "reflect":
            s = "self-reflection"
        elif name == "export_session":
            s = "exporting session"
        elif name == "test_runner":
            s = inp.get("command", "")[:50]
        elif name == "pm":
            s = inp.get("action", "")[:40]
        else:
            s = ""
        q.put(("tool_start", {"name": name, "summary": s}))

    def on_tool_end(name: str, result):
        out = (result.output or "").strip()
        if out.startswith("[cached]") or out.startswith("Hint:"):
            out = ""
        q.put(("tool_end", {
            "name": name,
            "output": out[:400],
            "is_error": result.is_error,
        }))

    def on_plan(event_type: str, data):
        q.put(("plan", {"type": event_type, "data": data}))

    def run():
        try:
            engine = get_engine(session_id)
            engine.on_chunk      = on_chunk
            engine.on_tool_start = on_tool_start
            engine.on_tool_end   = on_tool_end
            engine.on_plan       = on_plan
            # [F7] Vision guard: if image attached but mmproj not loaded, short-circuit
            if "[IMG:" in user_msg:
                cl = get_client()
                if not getattr(cl, "_vision_enabled", False):
                    stored = db.get_settings()
                    mmproj = stored.get("model", {}).get("mmproj", "").strip()
                    if mmproj and not Path(mmproj).exists():
                        notice = (f"[Vision unavailable — mmproj file not found at: {mmproj}\n\n"
                                  "Download it from the same HuggingFace repo as your model "
                                  "(look for mmproj-gemma-4-E2B-it-f16.gguf) and place it at that path.]")
                    else:
                        notice = ("[Vision unavailable — no mmproj file configured.\n\n"
                                  "Open Settings → paste the mmproj .gguf path → Apply & Save.]")
                    q.put(("chunk", notice))
                    q.put(("done", {}))
                    return
            reply = engine.submit_message(user_msg)
            # Mirror reply to all enabled connectors (WhatsApp, Email, etc.)
            if reply and reply.strip():
                _connector_registry.on_eve_reply(reply)
        except Exception as e:
            q.put(("error", {"message": str(e)}))
        finally:
            filt.flush()
            q.put(("done", {}))

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    async def stream() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        try:
            while True:
                # Check stop flag first
                if stop_ev.is_set():
                    yield sse("done", {"stopped": True})
                    break

                try:
                    kind, data = await loop.run_in_executor(None, q.get, True, 0.05)
                    if kind == "chunk":
                        yield sse("chunk", {"text": data})
                    elif kind == "think":
                        yield sse("think", {"text": data})
                    elif kind == "plan":
                        yield sse("plan", data)
                    elif kind == "tool_start":
                        yield sse("tool_start", data)
                    elif kind == "tool_end":
                        yield sse("tool_end", data)
                    elif kind == "error":
                        yield sse("error", data)
                        break
                    elif kind == "done":
                        yield sse("done", {})
                        break
                except queue.Empty:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(0.05)
        finally:
            with _stop_lock:
                _stop_flags.pop(session_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Conversation endpoints ────────────────────────────────────────────────────

@app.get("/conversations")
async def list_convs():
    return db.list_conversations()

@app.post("/conversations")
async def create_conv(request: Request):
    body = await request.json()
    return db.create_conversation(
        id    = body.get("id", f"c{int(time.time()*1000)}"),
        title = body.get("title", "New conversation"),
        model = body.get("model", Path(MODEL_PATH).stem),
    )

@app.delete("/conversations/{conv_id}")
async def delete_conv(conv_id: str):
    db.delete_conversation(conv_id)
    return {"ok": True}

@app.patch("/conversations/{conv_id}")
async def patch_conv(conv_id: str, request: Request):
    body = await request.json()
    db.update_conversation(conv_id,
        title = body.get("title"),
        model = body.get("model"),
    )
    return {"ok": True}

@app.get("/conversations/{conv_id}/messages")
async def get_conv_messages(conv_id: str):
    return db.get_messages(conv_id)

@app.post("/conversations/{conv_id}/messages")
async def add_conv_message(conv_id: str, request: Request):
    body = await request.json()
    # Ensure conversation row exists
    db.create_conversation(conv_id, model=Path(MODEL_PATH).stem)
    msg = db.add_message(
        conv_id        = conv_id,
        role           = body.get("role", "user"),
        text           = body.get("text", ""),
        think          = body.get("think", ""),
        image_data_url = body.get("imageDataUrl", ""),
        tools          = body.get("tools", []),
    )
    return msg

# ── Settings endpoints ────────────────────────────────────────────────────────

@app.get("/settings")
async def get_settings_route():
    return db.get_settings()

@app.post("/settings")
async def save_settings_route(request: Request):
    body = await request.json()
    db.save_settings(body)
    # Apply live-applicable settings immediately
    _apply_live_settings(body)
    return {"ok": True}

@app.post("/reload")
async def reload_model():
    """Discard current model client — next request triggers a fresh load."""
    global _client
    with _client_lock:
        _client = None
    with _sessions_lock:
        _sessions.clear()
    return {"ok": True, "message": "Model will reload on next request"}

# ── Model discovery ───────────────────────────────────────────────────────────

def _scan_gguf_dirs() -> list[dict]:
    """Scan common cache directories for .gguf model files."""
    import glob as _glob
    search_roots: list[Path] = []

    # From env var (colon/semicolon separated)
    env_dirs = os.environ.get("GGUF_SEARCH_DIRS", "")
    for d in re.split(r"[;:]", env_dirs):
        if d.strip():
            search_roots.append(Path(d.strip()))

    # Current model's parent directory
    search_roots.append(Path(MODEL_PATH).parent)

    home = Path.home()
    search_roots += [
        home / ".cache" / "eve" / "GGUF",
        home / ".cache" / "huggingface" / "hub",
        home / ".lm_studio" / "models",
        home / "lm_studio" / "models",
        home / ".cache" / "lm-studio" / "models",
    ]
    # Windows AppData
    appdata = os.environ.get("LOCALAPPDATA", "")
    if appdata:
        search_roots.append(Path(appdata) / "LM Studio" / "models")

    found: list[dict] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.gguf"):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            size_gb = p.stat().st_size / (1024**3)
            found.append({
                "path": str(p),
                "name": p.stem,
                "filename": p.name,
                "size_gb": round(size_gb, 2),
                "dir": str(p.parent),
            })
    found.sort(key=lambda x: x["name"].lower())
    return found

@app.get("/models")
async def list_models():
    """All discoverable .gguf models (cache dirs + uploaded)."""
    found = _scan_gguf_dirs()
    # Also include anything the user uploaded to models/
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        key = str(p.resolve())
        if not any(m["path"] == key for m in found):
            found.append({
                "path":     str(p),
                "name":     p.stem,
                "filename": p.name,
                "size_gb":  round(p.stat().st_size / (1024 ** 3), 2),
                "dir":      str(p.parent),
                "uploaded": True,
            })
    return found


@app.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """
    Accept a .gguf upload from the browser.
    The file is streamed to models/<filename> — no 2 GB RAM spike.
    """
    if not file.filename.lower().endswith(".gguf"):
        return {"error": "Only .gguf files are accepted"}
    dest = MODELS_DIR / file.filename
    try:
        with dest.open("wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                f.write(chunk)
        size_gb = round(dest.stat().st_size / (1024 ** 3), 2)
        return {
            "ok":       True,
            "path":     str(dest),
            "name":     dest.stem,
            "filename": dest.name,
            "size_gb":  size_gb,
        }
    except Exception as e:
        dest.unlink(missing_ok=True)
        return {"error": str(e)}


@app.delete("/models/upload/{filename}")
async def delete_uploaded_model(filename: str):
    """Delete a previously uploaded model from models/."""
    dest = MODELS_DIR / filename
    if dest.exists() and dest.suffix == ".gguf":
        dest.unlink()
        return {"ok": True}
    return {"error": "File not found"}


# ── Connector management ──────────────────────────────────────────────────────

@app.get("/connectors")
async def list_connectors():
    """Return status of all registered connectors."""
    return _connector_registry.status_all()


@app.post("/connectors/{cid}/enable")
async def enable_connector(cid: str, request: Request):
    """Enable a connector, optionally supplying config in the body."""
    c = _connector_registry.get(cid)
    if not c:
        return {"error": f"Unknown connector: {cid}"}
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass
    try:
        c.enable(body or None)
        # Persist
        stored = db.get_settings()
        conn_cfg = stored.setdefault("connectors", {})
        conn_cfg[cid] = {"enabled": True, "config": body}
        db.save_settings(stored)
        return {"ok": True, "status": c.status()}
    except Exception as e:
        return {"error": str(e)}


@app.post("/connectors/{cid}/disable")
async def disable_connector(cid: str):
    """Disable a connector."""
    c = _connector_registry.get(cid)
    if not c:
        return {"error": f"Unknown connector: {cid}"}
    c.disable()
    stored = db.get_settings()
    conn_cfg = stored.setdefault("connectors", {})
    conn_cfg.setdefault(cid, {})["enabled"] = False
    db.save_settings(stored)
    return {"ok": True, "status": c.status()}


@app.post("/connectors/{cid}/config")
async def configure_connector(cid: str, request: Request):
    """Update a connector's config without enabling/disabling it."""
    c = _connector_registry.get(cid)
    if not c:
        return {"error": f"Unknown connector: {cid}"}
    try:
        body = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}
    c.configure(body)
    stored = db.get_settings()
    conn_cfg = stored.setdefault("connectors", {})
    conn_cfg.setdefault(cid, {})["config"] = body
    db.save_settings(stored)
    return {"ok": True, "status": c.status()}

# ── Live settings application ─────────────────────────────────────────────────

def _apply_live_settings(settings: dict) -> None:
    """Apply settings that take effect without a model reload."""
    global TEMPERATURE, MAX_TOKENS, CONTEXT_SIZE
    inf = settings.get("inference", {})
    if "temperature" in inf:
        TEMPERATURE = float(inf["temperature"])
    if "max_tokens" in inf:
        MAX_TOKENS = int(inf["max_tokens"])

# ── Static HTML ───────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MasterMind</title>
<style>
:root {
  --bg:#ffffff; --bg2:#f9f9f8; --bg3:#f0efed; --bg4:#e8e6e3;
  --border:rgba(0,0,0,0.08); --border2:rgba(0,0,0,0.14);
  --text:#1a1a1a; --text2:#5c5c5c; --text3:#999;
  --accent:#d97706; --accent-bg:rgba(217,119,6,0.08);
  --think:#0891b2; --think-bg:rgba(8,145,178,0.07);
  --tool:#059669; --tool-bg:rgba(5,150,105,0.07);
  --err:#dc2626; --err-bg:rgba(220,38,38,0.06);
  --user-bg:#1a1a1a; --user-text:#ffffff;
  --sidebar:260px; --r:16px; --r2:10px;
  --font:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono:'Cascadia Code','Fira Code',ui-monospace,monospace;
  --shadow:0 4px 24px rgba(0,0,0,0.08);
}
[data-dark] {
  --bg:#212121; --bg2:#171717; --bg3:#2a2a2a; --bg4:#333;
  --border:rgba(255,255,255,0.09); --border2:rgba(255,255,255,0.16);
  --text:#ececec; --text2:#ababab; --text3:#666;
  --user-bg:#2f2f2f; --user-text:#ececec;
  --think-bg:rgba(8,145,178,0.09); --tool-bg:rgba(5,150,105,0.09);
  --err-bg:rgba(220,38,38,0.09);
  --shadow:0 4px 24px rgba(0,0,0,0.4);
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;font-family:var(--font);color:var(--text);background:var(--bg);overflow:hidden}

/* ── Shell ── */
#shell{display:flex;height:100vh}

/* ── Sidebar ── */
#sidebar{
  width:var(--sidebar);flex-shrink:0;
  background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;
  transition:width .2s ease;
}
#sidebar.hide{width:0;border-right:none}
#sb-top{padding:12px 10px 8px;flex-shrink:0}
#sb-top h2{font-size:14px;font-weight:600;padding:6px 10px 10px;color:var(--text)}
#new-btn{
  display:flex;align-items:center;gap:8px;width:100%;
  padding:8px 12px;border:1px solid var(--border2);border-radius:var(--r2);
  background:none;color:var(--text);font-size:13px;font-family:var(--font);
  cursor:pointer;transition:background .12s;
}
#new-btn:hover{background:var(--bg3)}
#new-btn svg{width:14px;height:14px;flex-shrink:0}
#conv-list{flex:1;overflow-y:auto;padding:4px 8px 16px}
#conv-list::-webkit-scrollbar{width:3px}
#conv-list::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.date-label{font-size:11px;color:var(--text3);padding:10px 8px 3px;font-weight:500;letter-spacing:.04em}
.conv-row{
  display:flex;align-items:center;gap:8px;
  padding:7px 10px;border-radius:var(--r2);
  cursor:pointer;transition:background .1s;min-width:0;
}
.conv-row:hover{background:var(--bg3)}
.conv-row.active{background:var(--bg4)}
.conv-row svg{width:13px;height:13px;color:var(--text3);flex-shrink:0}
.conv-title{font-size:13px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1}

/* ── Main ── */
#main{flex:1;display:flex;flex-direction:column;min-width:0}

/* ── Header ── */
#hdr{
  height:52px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;padding:0 16px;gap:10px;flex-shrink:0;
}
.hdr-btn{
  background:none;border:none;color:var(--text2);cursor:pointer;
  padding:6px;border-radius:8px;display:flex;transition:background .1s;
}
.hdr-btn:hover{background:var(--bg3)}
.hdr-btn svg{width:18px;height:18px}
#hdr-title{flex:1;font-size:14px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#model-badge{font-size:12px;color:var(--text3);background:var(--bg3);padding:3px 9px;border-radius:20px}

/* ── Messages ── */
#msgs-wrap{flex:1;overflow-y:auto}
#msgs-wrap::-webkit-scrollbar{width:5px}
#msgs-wrap::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
#msgs{max-width:740px;margin:0 auto;padding:24px 20px 12px;display:flex;flex-direction:column;gap:2px}
#msgs:empty::after{
  content:'Ask me anything — I can search the web, write and run code, manage files, and reason with 36 skills.';
  display:block;color:var(--text3);text-align:center;margin:80px auto 0;font-size:14px;
  max-width:360px;line-height:1.7;
}

/* Message rows */
.mrow{display:flex;flex-direction:column;padding:3px 0}
.mrow.user{align-items:flex-end}
.mrow.assistant{align-items:flex-start}

.bubble{max-width:88%;padding:11px 15px;border-radius:var(--r);line-height:1.7;font-size:15px}
.mrow.user .bubble{
  background:var(--user-bg);color:var(--user-text);
  border-radius:var(--r) var(--r) 4px var(--r);
}
.mrow.assistant .bubble{
  background:transparent;color:var(--text);padding:0;max-width:100%;
}

/* Bubble formatting */
.bubble p{margin:5px 0}.bubble p:first-child{margin-top:0}.bubble p:last-child{margin-bottom:0}
.bubble pre{background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:13px 15px;overflow-x:auto;font-family:var(--mono);font-size:13px;margin:10px 0}
.bubble code{font-family:var(--mono);font-size:13px;background:var(--bg3);padding:1px 5px;border-radius:4px}
.bubble pre code{background:none;padding:0}
.bubble strong{font-weight:600}
.bubble ul,.bubble ol{padding-left:22px;margin:5px 0}.bubble li{margin:2px 0}
.bubble a{color:var(--accent);text-decoration:none}.bubble a:hover{text-decoration:underline}
.bubble h1,.bubble h2,.bubble h3{font-weight:600;margin:12px 0 4px}
.bubble h1{font-size:17px}.bubble h2{font-size:16px}.bubble h3{font-size:15px}

/* Message actions */
.mactions{display:flex;gap:2px;margin-top:4px;opacity:0;transition:opacity .15s}
.mrow:hover .mactions{opacity:1}
.mact-btn{
  background:none;border:none;color:var(--text3);cursor:pointer;
  padding:4px 8px;border-radius:6px;font-size:12px;font-family:var(--font);
  display:flex;align-items:center;gap:4px;transition:background .1s,color .1s;
}
.mact-btn:hover{background:var(--bg3);color:var(--text2)}
.mact-btn svg{width:12px;height:12px}

/* Think block */
.think-block{margin:4px 0 6px;max-width:640px}
.think-hdr{
  display:flex;align-items:center;gap:6px;padding:5px 11px;
  background:var(--think-bg);border:1px solid rgba(8,145,178,0.18);
  border-radius:8px 8px 0 0;cursor:pointer;font-size:12px;color:var(--think);
  font-weight:500;user-select:none;transition:background .1s;
}
.think-hdr:hover{background:rgba(8,145,178,0.11)}
.think-chevron{font-size:9px;transition:transform .18s}
.think-hdr.open .think-chevron{transform:rotate(90deg)}
.think-body{
  padding:9px 13px;background:var(--think-bg);
  border:1px solid rgba(8,145,178,0.18);border-top:none;
  border-radius:0 0 8px 8px;font-size:12.5px;color:var(--think);
  line-height:1.65;white-space:pre-wrap;word-break:break-word;display:none;
}
.think-body.open{display:block}

/* Plan block */
.plan-block{margin:4px 0 8px;max-width:680px;border-radius:8px;overflow:hidden;
  border:1px solid rgba(139,92,246,0.25);}
.plan-hdr{
  display:flex;align-items:center;gap:7px;padding:6px 12px;
  background:rgba(139,92,246,0.07);cursor:pointer;font-size:12px;
  color:#8b5cf6;font-weight:600;user-select:none;transition:background .1s;
}
.plan-hdr:hover{background:rgba(139,92,246,0.12)}
.plan-chevron{font-size:9px;transition:transform .18s;color:#8b5cf6}
.plan-hdr.open .plan-chevron{transform:rotate(90deg)}
.plan-complexity{
  margin-left:auto;font-size:10px;font-weight:500;padding:1px 7px;
  border-radius:10px;background:rgba(139,92,246,0.15);color:#8b5cf6;
}
.plan-body{display:none;padding:6px 0;background:rgba(139,92,246,0.04);}
.plan-body.open{display:block}
.plan-phase{
  display:flex;align-items:flex-start;gap:8px;padding:6px 14px;
  border-bottom:1px solid rgba(139,92,246,0.08);font-size:12.5px;
  animation:plan-phase-in .2s ease;
}
.plan-phase:last-child{border-bottom:none}
@keyframes plan-phase-in{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.plan-phase-id{
  min-width:22px;height:22px;border-radius:50%;background:rgba(139,92,246,0.15);
  color:#8b5cf6;font-size:10px;font-weight:700;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;margin-top:1px;
}
.plan-phase-body{flex:1}
.plan-phase-title{color:var(--text1);font-weight:600;line-height:1.4}
.plan-phase-obj{color:var(--text3);font-size:11.5px;margin-top:1px;line-height:1.4}
.plan-phase-meta{display:flex;gap:6px;margin-top:3px;flex-wrap:wrap}
.plan-tag{
  font-size:10px;padding:1px 6px;border-radius:8px;font-weight:500;
  background:rgba(139,92,246,0.1);color:#8b5cf6;
}
.plan-tag.risk-high{background:rgba(239,68,68,0.1);color:#ef4444}
.plan-tag.risk-medium{background:rgba(245,158,11,0.1);color:#f59e0b}
.plan-tag.risk-low{background:rgba(16,185,129,0.1);color:#10b981}
.plan-footer{
  padding:6px 14px;font-size:11px;color:var(--text3);
  border-top:1px solid rgba(139,92,246,0.1);display:flex;gap:12px;align-items:center;
}


.tool-pill{
  display:inline-flex;align-items:center;gap:7px;
  padding:5px 11px;margin:3px 0;
  background:var(--tool-bg);border:1px solid rgba(5,150,105,0.2);
  border-radius:20px;font-size:12.5px;color:var(--tool);
  font-family:var(--font);transition:border-color .15s;
}
.tool-pill.running{animation:pill-pulse 1.3s ease-in-out infinite}
.tool-pill.done{opacity:.75}
.tool-pill.err{background:var(--err-bg);border-color:rgba(220,38,38,0.25);color:var(--err)}
@keyframes pill-pulse{0%,100%{opacity:1}50%{opacity:.45}}
.pill-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
.pill-label{font-weight:500;white-space:nowrap}
.pill-summary{color:var(--text3);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:260px}
.pill-output{
  display:none;margin-top:2px;margin-left:10px;
  font-size:12px;color:var(--text2);font-family:var(--mono);
  white-space:pre-wrap;word-break:break-word;
  max-height:120px;overflow-y:auto;
  border-left:2px solid rgba(5,150,105,0.2);padding-left:8px;
}
.tool-pill.done .pill-output.has-out{display:block}
.tool-pill.err .pill-output{display:block}

/* Typing dots */
.typing{display:inline-flex;gap:4px;padding:12px 2px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--text3);animation:bounce .9s ease-in-out infinite}
.dot:nth-child(2){animation-delay:.2s}.dot:nth-child(3){animation-delay:.4s}
@keyframes bounce{0%,100%{transform:translateY(0)}45%{transform:translateY(-6px)}}

/* ── Input ── */
#inp-area{padding:10px 20px 16px;flex-shrink:0;max-width:740px;margin:0 auto;width:100%}
#inp-box{
  background:var(--bg);border:1px solid var(--border2);border-radius:14px;
  padding:11px 13px;transition:border-color .15s,box-shadow .15s;
}
#inp-box:focus-within{border-color:rgba(0,0,0,0.28);box-shadow:0 0 0 3px rgba(0,0,0,0.05)}
[data-dark] #inp-box:focus-within{border-color:rgba(255,255,255,0.25);box-shadow:0 0 0 3px rgba(255,255,255,0.04)}

/* Image in user bubble */
.msg-img{
  display:block;max-width:280px;max-height:220px;
  border-radius:10px;margin-bottom:6px;
  object-fit:cover;cursor:pointer;
  transition:opacity .15s;
}
.msg-img:hover{opacity:.88}

/* File chip — shown when a file is attached */
#file-chip{
  display:none;align-items:center;gap:6px;
  background:var(--bg3);border:1px solid var(--border2);border-radius:8px;
  padding:5px 9px;margin-bottom:7px;font-size:12.5px;color:var(--text2);
  max-width:100%;
}
#file-chip.visible{display:flex}
#file-chip-name{flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-family:var(--mono)}
#file-chip-rm{
  background:none;border:none;color:var(--text3);cursor:pointer;
  padding:1px 3px;border-radius:4px;font-size:14px;line-height:1;
  transition:color .1s;flex-shrink:0;
}
#file-chip-rm:hover{color:var(--err)}

#input{width:100%;background:none;border:none;color:var(--text);font-family:var(--font);font-size:15px;resize:none;outline:none;max-height:200px;line-height:1.55}
#input::placeholder{color:var(--text3)}
#inp-foot{display:flex;align-items:center;gap:6px;margin-top:8px}
.ifoot-btn{background:none;border:none;color:var(--text3);cursor:pointer;padding:5px;border-radius:7px;display:flex;transition:color .1s,background .1s}
.ifoot-btn:hover{color:var(--text2);background:var(--bg3)}
.ifoot-btn svg{width:17px;height:17px}
#ifoot-space{flex:1}

/* Stop button */
#stop-btn{
  display:none;background:var(--err);border:none;color:#fff;
  padding:0 13px;height:32px;border-radius:9px;cursor:pointer;
  font-size:13px;font-family:var(--font);font-weight:500;
  align-items:center;gap:6px;
  transition:opacity .15s,transform .1s;
}
#stop-btn:hover{opacity:.85}
#stop-btn:active{transform:scale(.93)}
#stop-btn svg{width:13px;height:13px}

/* Pause button */
#pause-btn{
  display:none;background:none;border:1px solid var(--border2);color:var(--text2);
  padding:0 11px;height:32px;border-radius:9px;cursor:pointer;
  font-size:13px;font-family:var(--font);
  align-items:center;gap:6px;
  transition:background .12s,color .12s;
}
#pause-btn:hover{background:var(--bg3)}
#pause-btn.paused{border-color:var(--accent);color:var(--accent)}
#pause-btn svg{width:13px;height:13px}

/* Send button */
#send-btn{
  background:var(--text);border:none;color:var(--bg);
  width:32px;height:32px;border-radius:9px;cursor:pointer;
  display:flex;align-items:center;justify-content:center;
  transition:opacity .15s,transform .1s;
}
#send-btn:hover{opacity:.8}
#send-btn:active{transform:scale(.93)}
#send-btn:disabled{opacity:.22;cursor:default}
#send-btn svg{width:15px;height:15px}
#hint{font-size:11.5px;color:var(--text3);text-align:center;margin-top:9px}

/* Responsive */
@media(max-width:700px){
  #sidebar{position:fixed;top:0;left:0;height:100vh;z-index:50}
  #sidebar.hide{transform:translateX(-100%);width:var(--sidebar)}
  #ovl{display:block!important}
}
#ovl{display:none;position:fixed;inset:0;background:rgba(0,0,0,0.22);z-index:49}

/* ── Settings drawer ── */
#sett-overlay{
  display:none;position:fixed;inset:0;background:rgba(0,0,0,0.25);z-index:200;
}
#sett-overlay.open{display:block}
#sett-drawer{
  position:fixed;top:0;right:0;height:100vh;width:400px;max-width:95vw;
  background:var(--bg);border-left:1px solid var(--border2);
  z-index:201;display:flex;flex-direction:column;
  transform:translateX(100%);transition:transform .22s ease;
  overflow:hidden;
}
#sett-drawer.open{transform:translateX(0)}
#sett-hdr{
  display:flex;align-items:center;padding:14px 18px;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
#sett-hdr h2{flex:1;font-size:15px;font-weight:600}
.sett-close{
  background:none;border:none;color:var(--text3);cursor:pointer;
  padding:5px;border-radius:7px;font-size:18px;line-height:1;
  transition:color .1s,background .1s;
}
.sett-close:hover{color:var(--text);background:var(--bg3)}
#sett-body{flex:1;overflow-y:auto;padding:8px 0 100px}
#sett-body::-webkit-scrollbar{width:3px}
#sett-body::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.sett-section{border-bottom:1px solid var(--border)}
.sett-sect-hdr{
  display:flex;align-items:center;gap:8px;padding:12px 18px;
  cursor:pointer;user-select:none;font-size:11px;font-weight:600;
  letter-spacing:.06em;color:var(--text3);
  transition:background .1s;
}
.sett-sect-hdr:hover{background:var(--bg3)}
.sett-chevron{font-size:9px;transition:transform .16s;flex-shrink:0}
.sett-sect-hdr.open .sett-chevron{transform:rotate(90deg)}
.sett-sect-body{display:none;padding:4px 18px 14px}
.sett-sect-body.open{display:block}
.sett-row{margin:10px 0}
.sett-label{
  display:flex;justify-content:space-between;align-items:center;
  font-size:13px;color:var(--text2);margin-bottom:5px;
}
.sett-label span{font-size:11px;color:var(--text3);font-family:var(--mono)}
.sett-select,.sett-input{
  width:100%;background:var(--bg3);border:1px solid var(--border2);
  color:var(--text);border-radius:8px;padding:7px 10px;
  font-size:13px;font-family:var(--font);outline:none;
  transition:border-color .15s;
}
.sett-select:focus,.sett-input:focus{border-color:var(--accent)}
.sett-range-wrap{display:flex;align-items:center;gap:10px}
.sett-range{
  flex:1;-webkit-appearance:none;appearance:none;
  height:4px;background:var(--bg4);border-radius:2px;outline:none;
}
.sett-range::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--accent);cursor:pointer;
}
.sett-range-val{
  min-width:36px;text-align:right;font-size:12px;
  font-family:var(--mono);color:var(--text2);
}
.sett-note{font-size:11.5px;color:var(--text3);margin-top:6px;line-height:1.5}
.sett-note.warn{color:var(--accent)}
#sett-footer{
  position:absolute;bottom:0;left:0;right:0;
  padding:12px 18px;background:var(--bg);
  border-top:1px solid var(--border);
  display:flex;gap:8px;
}
.sett-btn{
  flex:1;padding:9px;border-radius:9px;font-size:13px;font-family:var(--font);
  font-weight:500;cursor:pointer;border:none;transition:opacity .15s;
}
.sett-btn:hover{opacity:.85}
#sett-cancel{background:var(--bg3);color:var(--text2)}
#sett-apply{background:var(--accent);color:#fff}
#sett-apply:disabled{opacity:.4;cursor:default}
.model-badge-row{
  display:flex;align-items:center;gap:8px;margin-bottom:10px;
  padding:8px 10px;background:var(--bg3);border-radius:8px;
}
.model-badge-dot{width:7px;height:7px;border-radius:50%;background:var(--tool);flex-shrink:0}
.model-badge-name{font-size:12.5px;font-family:var(--mono);color:var(--text2);flex:1;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.conv-del{
  display:none;background:none;border:none;color:var(--text3);
  cursor:pointer;padding:2px 5px;border-radius:4px;font-size:13px;
  transition:color .1s;flex-shrink:0;
}
.conv-row:hover .conv-del{display:block}
.conv-del:hover{color:var(--err)}

/* Vision-unavailable notice badge */
.vision-notice{
  display:flex;align-items:flex-start;gap:8px;
  padding:9px 13px;margin-bottom:10px;
  background:rgba(217,119,6,0.09);border:1px solid rgba(217,119,6,0.28);
  border-radius:10px;font-size:13px;color:var(--accent);line-height:1.5;
  word-break:break-word;
}

/* Server command block in settings */
.sett-cmd{
  background:var(--bg2);border:1px solid var(--border2);border-radius:8px;
  padding:10px 12px;font-size:11.5px;font-family:var(--mono);
  color:var(--text2);white-space:pre-wrap;word-break:break-all;
  margin-top:6px;position:relative;
}
.sett-cmd-copy{
  position:absolute;top:6px;right:8px;background:var(--bg3);
  border:1px solid var(--border);border-radius:6px;
  padding:2px 8px;font-size:11px;cursor:pointer;color:var(--text3);
  transition:color .1s;font-family:var(--font);
}
.sett-cmd-copy:hover{color:var(--text)}
</style>
</head>
<body>
<div id="shell">

<!-- Sidebar -->
<nav id="sidebar">
  <div id="sb-top">
    <h2>MasterMind</h2>
    <button id="new-btn" onclick="newChat()">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
      New conversation
    </button>
  </div>
  <div id="conv-list"></div>
</nav>

<!-- Main -->
<div id="main">
  <header id="hdr">
    <button class="hdr-btn" onclick="toggleSidebar()" title="Sidebar">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/></svg>
    </button>
    <div id="hdr-title">New conversation</div>
    <span id="model-badge">—</span>
    <button class="hdr-btn" onclick="toggleDark()" title="Theme">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
    </button>
    <button class="hdr-btn" onclick="openSettings()" title="Settings">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
    </button>
  </header>

  <div id="msgs-wrap"><div id="msgs"></div></div>

  <div id="inp-area">
    <div id="inp-box">
      <!-- File attachment chip -->
      <div id="file-chip">
        <img id="file-chip-img" style="display:none;width:28px;height:28px;object-fit:cover;border-radius:4px;flex-shrink:0">
        <svg id="file-chip-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width:13px;height:13px;flex-shrink:0"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span id="file-chip-name"></span>
        <button id="file-chip-rm" onclick="removeAttachment()" title="Remove file">&#x2715;</button>
      </div>

      <textarea id="input" placeholder="Message MasterMind…" rows="1"></textarea>

      <!-- Hidden file input — accepts images + common text/code types -->
      <input type="file" id="file-input" style="display:none"
        accept="image/*,text/*,.py,.js,.ts,.jsx,.tsx,.md,.json,.yaml,.yml,.toml,.csv,.sh,.bash,.rs,.go,.java,.c,.cpp,.h"
        onchange="handleFileSelect(event)">

      <div id="inp-foot">
        <!-- Upload button -->
        <button class="ifoot-btn" id="upload-btn" title="Attach file" onclick="document.getElementById('file-input').click()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
        </button>
        <!-- Voice button -->
        <button class="ifoot-btn" id="voice-btn" title="Voice">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>
        </button>
        <div id="ifoot-space"></div>
        <!-- Pause button (shown while streaming) -->
        <button id="pause-btn" onclick="togglePause()" title="Pause">
          <svg id="pause-icon" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
          <svg id="resume-icon" viewBox="0 0 24 24" fill="currentColor" style="display:none"><polygon points="5 3 19 12 5 21 5 3"/></svg>
          <span id="pause-label">Pause</span>
        </button>
        <!-- Stop button (shown while streaming) -->
        <button id="stop-btn" onclick="stopStream()">
          <svg viewBox="0 0 24 24" fill="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/></svg>
          Stop
        </button>
        <!-- Send button (shown when idle) -->
        <button id="send-btn" onclick="sendMsg()" disabled>
          <svg viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
        </button>
      </div>
    </div>
    <div id="hint">Enter to send · Shift+Enter for newline · Paperclip to attach a file</div>
  </div>
</div>

<div id="ovl" onclick="closeSidebarMobile()"></div>

<!-- Settings drawer -->
<div id="sett-overlay" onclick="closeSettings()"></div>
<div id="sett-drawer">
  <div id="sett-hdr">
    <h2>Settings</h2>
    <button class="sett-close" onclick="closeSettings()">&#x2715;</button>
  </div>
  <div id="sett-body">

    <!-- MODEL -->
    <div class="sett-section">
      <div class="sett-sect-hdr open" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> MODEL
      </div>
      <div class="sett-sect-body open">
        <div class="model-badge-row">
          <div class="model-badge-dot"></div>
          <span class="model-badge-name" id="sett-current-model">—</span>
        </div>
        <div class="sett-row">
          <div class="sett-label">Select model <span id="sett-model-size"></span></div>
          <select class="sett-select" id="sett-model-path" onchange="onModelChange()">
            <option value="">Scanning…</option>
          </select>
        </div>
        <div class="sett-row">
          <div class="sett-label">Context window <span id="sett-ctx-val"></span></div>
          <select class="sett-select" id="sett-ctx" onchange="onModelChange()">
            <option value="2048">2 048</option>
            <option value="4096">4 096</option>
            <option value="8192">8 192</option>
            <option value="16384" selected>16 384</option>
            <option value="32768">32 768</option>
            <option value="65536">65 536</option>
          </select>
        </div>
        <p class="sett-note warn" id="sett-reload-note" style="display:none">
          &#x26A0;&#xFE0F; Model or context changed — click Apply to reload.
        </p>
        <div class="sett-row">
          <div class="sett-label">Vision projector — mmproj
            <span style="font-size:10px;color:var(--text3)">for image support</span>
          </div>
          <input type="text" class="sett-input" id="sett-mmproj"
            placeholder="/path/to/mmproj.gguf  (leave blank if none)"
            oninput="onModelChange();updateServerCmd()">
        </div>
        <p class="sett-note">
          Gemma 4 mmproj: download
          <code style="font-size:11px">mmproj-gemma-4-E2B-it-f16.gguf</code>
          from the same HuggingFace repo as your GGUF, then paste the path above.
        </p>
        <div class="sett-row">
          <div class="sett-label">llama-server startup command</div>
          <div class="sett-cmd" id="sett-server-cmd">—
            <button class="sett-cmd-copy" onclick="copyServerCmd()">copy</button>
          </div>
        </div>
      </div>
    </div>

    <!-- INFERENCE -->
    <div class="sett-section">
      <div class="sett-sect-hdr open" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> INFERENCE
      </div>
      <div class="sett-sect-body open">
        <div class="sett-row">
          <div class="sett-label">Temperature <span id="sett-temp-val">0.7</span></div>
          <div class="sett-range-wrap">
            <span style="font-size:11px;color:var(--text3)">0</span>
            <input type="range" class="sett-range" id="sett-temp"
              min="0" max="2" step="0.05" value="0.7"
              oninput="document.getElementById('sett-temp-val').textContent=parseFloat(this.value).toFixed(2)">
            <span style="font-size:11px;color:var(--text3)">2</span>
          </div>
        </div>
        <div class="sett-row">
          <div class="sett-label">Max tokens <span></span></div>
          <input type="number" class="sett-input" id="sett-max-tokens"
            min="128" max="32768" step="256" value="4096">
        </div>
      </div>
    </div>

    <!-- CPU / GPU -->
    <div class="sett-section">
      <div class="sett-sect-hdr" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> CPU / GPU PERFORMANCE
      </div>
      <div class="sett-sect-body">
        <div class="sett-row">
          <div class="sett-label">Inference threads <span id="sett-threads-hint" style="font-size:11px;color:var(--text3)"></span></div>
          <input type="number" class="sett-input" id="sett-threads"
            min="1" max="128" step="1" value="4" oninput="updateServerCmd()">
        </div>
        <div class="sett-row">
          <div class="sett-label">Batch size (prefill speed)</div>
          <select class="sett-select" id="sett-batch" onchange="updateServerCmd()">
            <option value="512">512</option>
            <option value="1024">1 024</option>
            <option value="2048" selected>2 048</option>
            <option value="4096">4 096</option>
          </select>
        </div>
        <div class="sett-row">
          <div class="sett-label">GPU layers <span id="sett-gpu-hint" style="font-size:11px;color:var(--text3)">0 = CPU only</span></div>
          <input type="number" class="sett-input" id="sett-gpu-layers"
            min="0" max="999" step="1" value="0" oninput="updateServerCmd()">
        </div>
        <p class="sett-note">Threads: use physical core count (logical/2) for best speed. Batch: higher = faster prompt processing.</p>
      </div>
    </div>

    <!-- WEB SEARCH -->
    <div class="sett-section">
      <div class="sett-sect-hdr" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> WEB SEARCH
      </div>
      <div class="sett-sect-body">
        <div class="sett-row">
          <div class="sett-label">Crawl depth <span id="sett-depth-val">1</span></div>
          <div class="sett-range-wrap">
            <span style="font-size:11px;color:var(--text3)">0</span>
            <input type="range" class="sett-range" id="sett-depth"
              min="0" max="3" step="1" value="1"
              oninput="document.getElementById('sett-depth-val').textContent=this.value">
            <span style="font-size:11px;color:var(--text3)">3</span>
          </div>
        </div>
        <div class="sett-row">
          <div class="sett-label">Max search results</div>
          <input type="number" class="sett-input" id="sett-max-results"
            min="1" max="12" step="1" value="6">
        </div>
      </div>
    </div>

    <!-- UPLOAD MODEL -->
    <div class="sett-section">
      <div class="sett-sect-hdr open" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> UPLOAD MODEL
      </div>
      <div class="sett-sect-body open">
        <p class="sett-note" style="margin-bottom:10px">Drop a .gguf file here or click to browse. Saved to <code>models/</code> and immediately available in the model picker above.</p>
        <div id="model-drop-zone"
          style="border:2px dashed var(--border2);border-radius:10px;padding:24px 16px;text-align:center;cursor:pointer;transition:background .15s;font-size:13px;color:var(--text3)"
          onclick="document.getElementById('model-upload-input').click()"
          ondragover="event.preventDefault();this.style.background='var(--accent-bg)'"
          ondragleave="this.style.background=''"
          ondrop="handleModelDrop(event)">
          <div style="font-size:28px;margin-bottom:6px">📦</div>
          <div>Drag &amp; drop .gguf here</div>
          <div style="font-size:11px;margin-top:4px">or click to browse</div>
        </div>
        <input type="file" id="model-upload-input" accept=".gguf" style="display:none" onchange="handleModelUpload(event)">
        <div id="model-upload-progress" style="display:none;margin-top:10px">
          <div style="height:4px;background:var(--bg4);border-radius:2px;overflow:hidden">
            <div id="model-upload-bar" style="height:100%;background:var(--accent);width:0%;transition:width .3s"></div>
          </div>
          <div id="model-upload-label" style="font-size:12px;color:var(--text3);margin-top:5px;text-align:center">Uploading…</div>
        </div>
        <div id="uploaded-models-list" style="margin-top:10px"></div>
      </div>
    </div>

    <!-- CONNECTORS -->
    <div class="sett-section">
      <div class="sett-sect-hdr open" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> CONNECTORS
      </div>
      <div class="sett-sect-body open">
        <p class="sett-note" style="margin-bottom:12px">Activate connectors to let EVE send and receive messages on external channels.</p>
        <div id="connectors-list">
          <!-- populated by loadConnectors() -->
          <div style="color:var(--text3);font-size:13px">Loading…</div>
        </div>
      </div>
    </div>

    <!-- ABOUT -->
    <div class="sett-section">
      <div class="sett-sect-hdr" onclick="toggleSect(this)">
        <span class="sett-chevron">▶</span> ABOUT
      </div>
      <div class="sett-sect-body">
        <p class="sett-note">EVE WebUI · FastAPI + SSE · SQLite-backed</p>
        <p class="sett-note" id="sett-db-path" style="margin-top:6px;word-break:break-all"></p>
      </div>
    </div>

  </div>
  <div id="sett-footer">
    <button class="sett-btn" id="sett-cancel" onclick="closeSettings()">Cancel</button>
    <button class="sett-btn" id="sett-apply" onclick="applySettings()">Apply &amp; Save</button>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════
let activeId   = null;
let isStreaming = false;
let fileAttachment    = null;
let isPaused          = false;
let pauseBuffer       = [];
let abortController   = null;
let _needsReload      = false;   // true when model/cpu settings changed

// ═══════════════════════════════════════════════════════════
// Init
// ═══════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', async () => {
  if (localStorage.getItem('mm_dark') === '1')
    document.documentElement.setAttribute('data-dark', '');

  fetch('/info').then(r => r.json()).then(d => {
    const badge = document.getElementById('model-badge');
    badge.textContent = (d.model || 'local') + (d.vision ? '  👁' : '');
    if (d.vision) {
      badge.title = 'Vision enabled';
    } else if (d.mmproj_missing) {
      badge.title = '⚠️ mmproj file not found at: ' + (d.mmproj || '?') + ' — check Settings';
      badge.style.color = 'var(--accent)';
    } else {
      badge.title = 'Vision disabled — add mmproj path in Settings';
    }
  }).catch(() => {});

  const ta = document.getElementById('input');
  ta.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
  });
  ta.addEventListener('input', () => {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
    document.getElementById('send-btn').disabled =
      !ta.value.trim() && !fileAttachment;
  });

  setupVoice();
  await loadConvsFromServer();
});

// ═══════════════════════════════════════════════════════════
// Server-backed conversations
// ═══════════════════════════════════════════════════════════
async function loadConvsFromServer() {
  try {
    const convs = await fetch('/conversations').then(r => r.json());
    renderSidebar(convs);
    if (convs.length) await switchConv(convs[0].id);
    else await newChat();
  } catch (e) {
    console.error('Failed to load conversations:', e);
    await newChat();
  }
}

async function newChat() {
  const id = 'c' + Date.now();
  await fetch('/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id, title: 'New conversation' }),
  });
  activeId = id;
  document.getElementById('hdr-title').textContent = 'New conversation';
  document.getElementById('msgs').innerHTML = '';
  await refreshSidebar();
}

async function switchConv(id) {
  activeId = id;
  const msgs = document.getElementById('msgs');
  msgs.innerHTML = '';
  try {
    const messages = await fetch(`/conversations/${id}/messages`).then(r => r.json());
    messages.forEach(m => renderStoredMsg(m));
  } catch (e) {
    console.error('Failed to load messages:', e);
  }
  document.querySelectorAll('.conv-row').forEach(el =>
    el.classList.toggle('active', el.dataset.id === id)
  );
  // Update header title from sidebar
  const row = document.querySelector(`.conv-row[data-id="${id}"] .conv-title`);
  document.getElementById('hdr-title').textContent =
    row ? row.textContent : 'MasterMind';
  scrollBottom();
}

async function deleteConv(id, e) {
  e.stopPropagation();
  if (!confirm('Delete this conversation?')) return;
  await fetch(`/conversations/${id}`, { method: 'DELETE' });
  if (activeId === id) {
    const next = document.querySelector(`.conv-row:not([data-id="${id}"])`);
    if (next) await switchConv(next.dataset.id);
    else await newChat();
  }
  await refreshSidebar();
}

async function autoTitle(id, text) {
  const row = document.querySelector(`.conv-row[data-id="${id}"] .conv-title`);
  if (row && row.textContent !== 'New conversation') return;
  const title = text.slice(0, 58);
  await fetch(`/conversations/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  document.getElementById('hdr-title').textContent = title;
  if (row) row.textContent = title;
}

async function saveMessage(convId, msg) {
  try {
    await fetch(`/conversations/${convId}/messages`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(msg),
    });
  } catch (e) {
    console.error('Failed to save message:', e);
  }
}

async function refreshSidebar() {
  const convs = await fetch('/conversations').then(r => r.json());
  renderSidebar(convs);
}

// ═══════════════════════════════════════════════════════════
// Sidebar render
// ═══════════════════════════════════════════════════════════
function renderSidebar(convs) {
  const list = document.getElementById('conv-list');
  list.innerHTML = '';
  const now = Date.now(), DAY = 86400000;
  const groups = [
    { label: 'Today',           f: c => now - c.updated_at < DAY },
    { label: 'Yesterday',       f: c => now - c.updated_at >= DAY && now - c.updated_at < 2*DAY },
    { label: 'Previous 7 days', f: c => now - c.updated_at >= 2*DAY && now - c.updated_at < 8*DAY },
    { label: 'Older',           f: c => now - c.updated_at >= 8*DAY },
  ];
  groups.forEach(g => {
    const items = convs.filter(g.f);
    if (!items.length) return;
    const lbl = document.createElement('div');
    lbl.className = 'date-label'; lbl.textContent = g.label;
    list.appendChild(lbl);
    items.forEach(c => {
      const row = document.createElement('div');
      row.className = 'conv-row' + (c.id === activeId ? ' active' : '');
      row.dataset.id = c.id;
      row.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="width:13px;height:13px;flex-shrink:0"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
        <span class="conv-title">${esc(c.title)}</span>
        <button class="conv-del" title="Delete" onclick="deleteConv('${c.id}',event)">&#x2715;</button>`;
      row.onclick = e => {
        if (e.target.classList.contains('conv-del')) return;
        switchConv(c.id); closeSidebarMobile();
      };
      list.appendChild(row);
    });
  });
}

// ═══════════════════════════════════════════════════════════
// Render stored message (from DB)
// ═══════════════════════════════════════════════════════════
function renderStoredMsg(m) {
  const msgs = document.getElementById('msgs');
  const row = document.createElement('div');
  row.className = 'mrow ' + m.role;
  if (m.role === 'user') {
    const imgHtml = m.image_data_url
      ? `<img class="msg-img" src="${m.image_data_url}" onclick="this.style.maxWidth=this.style.maxWidth?'':'100%'" title="Click to expand">`
      : '';
    row.innerHTML = `
      <div class="bubble">${imgHtml}${esc(m.text).replace(/\n/g,'<br>')}</div>
      <div class="mactions">${copyBtn()}</div>`;
    row.querySelector('.mact-btn').onclick = () => copyText(m.text);
  } else {
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    if (m.think) bubble.appendChild(makeThinkBlock(m.think));
    (m.tools || []).forEach(t => bubble.appendChild(makeToolPill(t)));
    if (m.text) {
      const td = document.createElement('div');
      td.innerHTML = renderMd(m.text);
      bubble.appendChild(td);
    }
    const acts = document.createElement('div');
    acts.className = 'mactions';
    acts.innerHTML = copyBtn();
    acts.querySelector('.mact-btn').onclick = () => copyText(m.text || '');
    row.appendChild(bubble);
    row.appendChild(acts);
  }
  msgs.appendChild(row);
}

// ═══════════════════════════════════════════════════════════
// Streaming helpers
// ═══════════════════════════════════════════════════════════
function setStreaming(on) {
  isStreaming = on;
  document.getElementById('send-btn').style.display  = on ? 'none' : 'flex';
  document.getElementById('stop-btn').style.display  = on ? 'flex' : 'none';
  document.getElementById('pause-btn').style.display = on ? 'flex' : 'none';
  if (!on) {
    isPaused = false; pauseBuffer = []; abortController = null;
    const pb = document.getElementById('pause-btn');
    pb.classList.remove('paused');
    document.getElementById('pause-icon').style.display  = '';
    document.getElementById('resume-icon').style.display = 'none';
    document.getElementById('pause-label').textContent   = 'Pause';
  }
}

async function stopStream() {
  if (abortController) abortController.abort();
  await fetch('/stop/' + activeId, { method: 'POST' }).catch(() => {});
}

function togglePause() {
  isPaused = !isPaused;
  const pb = document.getElementById('pause-btn');
  pb.classList.toggle('paused', isPaused);
  document.getElementById('pause-icon').style.display  = isPaused ? 'none' : '';
  document.getElementById('resume-icon').style.display = isPaused ? '' : 'none';
  document.getElementById('pause-label').textContent   = isPaused ? 'Resume' : 'Pause';
  if (!isPaused && pauseBuffer.length) {
    const buf = pauseBuffer.splice(0);
    for (const [t, d] of buf) {
      if      (t === 'chunk')      _onChunk(d.text);
      else if (t === 'think')      _onThink(d.text);
      else if (t === 'tool_start') _onToolStart(d);
      else if (t === 'tool_end')   _onToolEnd(d);
    }
  }
}

// ═══════════════════════════════════════════════════════════
// File upload
// ═══════════════════════════════════════════════════════════
async function resizeImage(file, maxDim, quality) {
  return new Promise((resolve, reject) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(url);
      let {naturalWidth: w, naturalHeight: h} = img;
      if (w > maxDim || h > maxDim) {
        if (w >= h) { h = Math.round(h * maxDim / w); w = maxDim; }
        else        { w = Math.round(w * maxDim / h); h = maxDim; }
      }
      const c = document.createElement('canvas');
      c.width = w; c.height = h;
      c.getContext('2d').drawImage(img, 0, 0, w, h);
      resolve(c.toDataURL('image/jpeg', quality));
    };
    img.onerror = reject;
    img.src = url;
  });
}

async function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;
  if (file.type.startsWith('image/')) {
    if (file.size > 20 * 1024 * 1024) { toast('Image too large (> 20 MB).'); e.target.value=''; return; }
    try {
      const dataUrl = await resizeImage(file, 768, 0.75);
      fileAttachment = { name: file.name, type: 'image', dataUrl };
      const ci = document.getElementById('file-chip-img');
      ci.src = dataUrl; ci.style.display = 'block';
      document.getElementById('file-chip-icon').style.display = 'none';
      document.getElementById('file-chip-name').textContent = file.name;
      document.getElementById('file-chip').classList.add('visible');
      document.getElementById('input').focus();
    } catch (err) { toast('Could not load image: ' + err.message); }
    return;
  }
  const isBinary = /\.(pdf|zip|gz|tar|exe|dll|bin|wasm|mp[34]|mkv)$/i.test(file.name);
  if (isBinary) { toast('Binary files not supported. Attach a text/code/image file.'); e.target.value=''; return; }
  if (file.size > 500 * 1024) { toast('File too large (> 500 KB). Use bash for big files.'); e.target.value=''; return; }
  const reader = new FileReader();
  reader.onload = ev => {
    if ((ev.target.result.slice(0,512).match(/\0/g)||[]).length > 8) {
      toast('Looks binary. Attach a text/code file.'); e.target.value=''; return;
    }
    fileAttachment = { name: file.name, type: 'text', content: ev.target.result };
    const lines = (ev.target.result.match(/\n/g)||[]).length + 1;
    document.getElementById('file-chip-img').style.display = 'none';
    document.getElementById('file-chip-icon').style.display = '';
    document.getElementById('file-chip-name').textContent = file.name + '  (' + lines + ' lines)';
    document.getElementById('file-chip').classList.add('visible');
    document.getElementById('input').focus();
  };
  reader.onerror = () => toast('Could not read file.');
  reader.readAsText(file);
}

function removeAttachment() {
  fileAttachment = null;
  document.getElementById('file-chip').classList.remove('visible');
  document.getElementById('file-chip-name').textContent = '';
  document.getElementById('file-chip-img').src = '';
  document.getElementById('file-chip-img').style.display = 'none';
  document.getElementById('file-chip-icon').style.display = '';
  document.getElementById('file-input').value = '';
}

// ═══════════════════════════════════════════════════════════
// Send
// ═══════════════════════════════════════════════════════════
async function sendMsg() {
  if (isStreaming) return;
  const ta   = document.getElementById('input');
  const text = ta.value.trim();
  if (!text && !fileAttachment) return;

  let fullMsg = text, attachedName = null, attachedImageDataUrl = null;
  if (fileAttachment) {
    attachedName = fileAttachment.name;
    if (fileAttachment.type === 'image') {
      attachedImageDataUrl = fileAttachment.dataUrl;
      fullMsg = (text || 'Describe this image.') + '\n[IMG:' + fileAttachment.dataUrl + ']';
    } else {
      fullMsg = (text ? text + '\n\n' : '') + '--- ' + fileAttachment.name + ' ---\n' + fileAttachment.content;
    }
    removeAttachment();
  }

  if (navigator.vibrate) navigator.vibrate(10);
  ta.value = ''; ta.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;
  setStreaming(true);

  const displayText = text
    ? (attachedName ? text + '  📎 ' + attachedName : text)
    : '📎 ' + (attachedName || '');

  // Save user message to DB + render
  await saveMessage(activeId, {
    role: 'user', text: displayText,
    imageDataUrl: attachedImageDataUrl || '',
  });
  await autoTitle(activeId, displayText);
  renderStoredMsg({ role: 'user', text: displayText,
                    image_data_url: attachedImageDataUrl });
  await refreshSidebar();

  // Typing indicator
  const typingRow = document.createElement('div');
  typingRow.className = 'mrow assistant'; typingRow.id = 'typing';
  typingRow.innerHTML = '<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  document.getElementById('msgs').appendChild(typingRow);
  scrollBottom();

  // ── Streaming state ──
  let rawText = '', thinkRaw = '', thinkBodyEl = null;
  let responseBubble = null, responseTextDiv = null, toolPills = {};
  let assistantTools = [];

  function getOrCreateAssistantRow() {
    let row = document.getElementById('streaming-row');
    if (!row) {
      const t = document.getElementById('typing'); if (t) t.remove();
      row = document.createElement('div');
      row.className = 'mrow assistant'; row.id = 'streaming-row';
      responseBubble = document.createElement('div');
      responseBubble.className = 'bubble';
      row.appendChild(responseBubble);
      document.getElementById('msgs').appendChild(row);
    }
    return row;
  }
  function getOrCreateTextDiv() {
    getOrCreateAssistantRow();
    if (!responseTextDiv) {
      responseTextDiv = document.createElement('div');
      responseTextDiv.id = 'streaming-text';
      responseBubble.appendChild(responseTextDiv);
    }
    return responseTextDiv;
  }

  // ── Plan block state ──
  let planBlockEl = null, planBodyEl = null, planFooterEl = null;

  function _onPlan(evType, data) {
    if (evType === 'start') {
      // Create the collapsible plan block
      getOrCreateAssistantRow();
      planBlockEl = document.createElement('div');
      planBlockEl.className = 'plan-block';
      const complexity = (data.complexity || 'medium');
      const complexLabel = complexity.toUpperCase();
      planBlockEl.innerHTML =
        `<div class="plan-hdr open" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">` +
        `<span class="plan-chevron">▶</span>` +
        `<span>📋 Building plan…</span>` +
        `<span class="plan-complexity">${esc(complexLabel)}</span>` +
        `</div>` +
        `<div class="plan-body open"></div>`;
      planBodyEl = planBlockEl.querySelector('.plan-body');
      responseBubble.insertBefore(planBlockEl, responseTextDiv || null);
      scrollBottom();

    } else if (evType === 'phase') {
      if (!planBodyEl) return;
      const ph = data;
      const riskCls = `risk-${ph.risk || 'low'}`;
      const riskIcon = ph.risk === 'high' ? '🔴' : ph.risk === 'medium' ? '🟡' : '🟢';
      const stepCount = (ph.steps || []).length;
      const estMin = (ph.steps || []).reduce((a, s) => a + (s.estimated_m || 1), 0);
      const dep = (ph.depends_on || []).length
        ? `<span class="plan-tag">after: ${esc(ph.depends_on.join(', '))}</span>` : '';
      const phDiv = document.createElement('div');
      phDiv.className = 'plan-phase';
      phDiv.innerHTML =
        `<div class="plan-phase-id">${esc(ph.id)}</div>` +
        `<div class="plan-phase-body">` +
        `  <div class="plan-phase-title">${esc(ph.title)}</div>` +
        `  <div class="plan-phase-obj">${esc(ph.objective)}</div>` +
        `  <div class="plan-phase-meta">` +
        `    <span class="plan-tag ${riskCls}">${riskIcon} ${ph.risk || 'low'} risk</span>` +
        `    <span class="plan-tag">${stepCount} step${stepCount !== 1 ? 's' : ''}</span>` +
        `    <span class="plan-tag">~${Math.round(estMin)}m</span>` +
        `    ${dep}` +
        `  </div>` +
        `</div>`;
      planBodyEl.appendChild(phDiv);
      scrollBottom();

    } else if (evType === 'done') {
      if (!planBlockEl) return;
      // Update header label
      const hdr = planBlockEl.querySelector('.plan-hdr');
      const totalPhases = data.total_phases || 0;
      const totalMin    = Math.round(data.total_est_min || 0);
      if (hdr) {
        hdr.querySelector('span:nth-child(2)').textContent =
          `📋 Plan ready — ${totalPhases} phase${totalPhases !== 1 ? 's' : ''}, ~${totalMin} min`;
      }
      // Add footer summary
      const footer = document.createElement('div');
      footer.className = 'plan-footer';
      footer.innerHTML =
        `<span>⏱ Est. total: <b>${totalMin} min</b></span>` +
        `<span>📦 ${totalPhases} phases</span>` +
        (data.elapsed_s ? `<span style="margin-left:auto;opacity:.6">planned in ${data.elapsed_s}s</span>` : '');
      planBodyEl.appendChild(footer);
      // Auto-collapse after a short delay so user sees it before execution starts
      setTimeout(() => {
        const h = planBlockEl && planBlockEl.querySelector('.plan-hdr');
        const b = planBlockEl && planBlockEl.querySelector('.plan-body');
        if (h && b) { h.classList.remove('open'); b.classList.remove('open'); }
      }, 2200);
      scrollBottom();
    }
  }
  function _onThink(text) {
    thinkRaw += text;
    if (!thinkBodyEl) {
      getOrCreateAssistantRow();
      const block = makeThinkBlock('');
      block.querySelector('.think-hdr').classList.add('open');
      thinkBodyEl = block.querySelector('.think-body');
      thinkBodyEl.classList.add('open');
      responseBubble.insertBefore(block, responseTextDiv || null);
    }
    thinkBodyEl.textContent = thinkRaw;
    scrollBottom();
  }
  function _onChunk(text) {
    rawText += text;
    const div = getOrCreateTextDiv();
    // Vision-unavailable notice comes as the first chunk — style it as a badge
    if (rawText.startsWith('[Vision unavailable')) {
      const sep = rawText.indexOf('\n\n');
      const notice = sep > -1 ? rawText.slice(0, sep) : rawText;
      const rest   = sep > -1 ? rawText.slice(sep + 2) : '';
      div.innerHTML =
        `<div class="vision-notice">&#x26A0;&#xFE0F; ${esc(notice.replace(/^\[|\]$/g,''))}</div>` +
        (rest ? `<span>${esc(rest)}&#x2587;</span>` : '&#x2587;');
    } else {
      div.textContent = rawText + '▋';
    }
    scrollBottom();
  }
  function _onToolStart(data) {
    const t = document.getElementById('typing'); if (t) t.remove();
    getOrCreateAssistantRow();
    const pill = makeToolPill({ name: data.name, summary: data.summary, running: true });
    responseBubble.insertBefore(pill, responseTextDiv || null);
    toolPills[data.name] = { pill, outputEl: pill.querySelector('.pill-output') };
    scrollBottom();
  }
  function _onToolEnd(data) {
    const entry = toolPills[data.name];
    if (!entry) return;
    entry.pill.classList.remove('running');
    entry.pill.classList.add(data.is_error ? 'err' : 'done');
    if (data.output) {
      entry.outputEl.textContent = data.output;
      entry.outputEl.classList.add('has-out');
    }
    assistantTools.push({ name: data.name, summary: data.summary || '',
                          output: data.output || '', error: data.is_error });
    scrollBottom();
  }

  async function onDone() {
    if (pauseBuffer.length) {
      const buf = pauseBuffer.splice(0);
      for (const [t, d] of buf) {
        if      (t === 'chunk')      _onChunk(d.text);
        else if (t === 'think')      _onThink(d.text);
        else if (t === 'plan')       _onPlan(d.type, d.data);
        else if (t === 'tool_start') _onToolStart(d);
        else if (t === 'tool_end')   _onToolEnd(d);
      }
    }
    if (!rawText && !thinkRaw && !Object.keys(toolPills).length) {
      const div = getOrCreateTextDiv();
      if (!div.innerHTML) div.innerHTML = '<span style="color:var(--text3);font-size:13px;font-style:italic">— stopped —</span>';
    }
    if (responseTextDiv && rawText) {
      if (rawText.startsWith('[Vision unavailable')) {
        const sep    = rawText.indexOf('\n\n');
        const notice = sep > -1 ? rawText.slice(0, sep) : rawText;
        const rest   = sep > -1 ? rawText.slice(sep + 2) : '';
        responseTextDiv.innerHTML =
          `<div class="vision-notice">&#x26A0;&#xFE0F; ${esc(notice.replace(/^\[|\]$/g,''))}</div>` +
          (rest ? renderMd(rest) : '');
      } else {
        responseTextDiv.innerHTML = renderMd(rawText);
      }
    }
    if (thinkBodyEl) {
      const hdr = thinkBodyEl.previousElementSibling;
      if (hdr) { hdr.classList.remove('open'); thinkBodyEl.classList.remove('open'); }
    }
    const row = document.getElementById('streaming-row');
    if (row) {
      row.id = '';
      const acts = document.createElement('div');
      acts.className = 'mactions'; acts.innerHTML = copyBtn();
      acts.querySelector('.mact-btn').onclick = () => copyText(rawText);
      row.appendChild(acts);
    }
    // Persist assistant message to DB
    await saveMessage(activeId, {
      role: 'assistant', text: rawText, think: thinkRaw,
      tools: assistantTools, imageDataUrl: '',
    });
    setStreaming(false);
    document.getElementById('send-btn').disabled = !document.getElementById('input').value.trim();
    scrollBottom();
  }

  // SSE stream
  abortController = new AbortController();
  let resp;
  try {
    resp = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: fullMsg, session_id: activeId }),
      signal: abortController.signal,
    });
  } catch {
    const ty = document.getElementById('typing'); if (ty) ty.remove();
    setStreaming(false);
    document.getElementById('send-btn').disabled = false;
    return;
  }

  const reader = resp.body.getReader(), dec = new TextDecoder();
  let sseBuf = '';
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      sseBuf += dec.decode(value, { stream: true });
      const lines = sseBuf.split('\n');
      sseBuf = lines.pop();
      let evType = null, evData = null;
      for (const line of lines) {
        if (line.startsWith('event: ')) evType = line.slice(7).trim();
        else if (line.startsWith('data: ')) evData = line.slice(6);
        else if (line === '' && evType && evData) {
          const d = JSON.parse(evData);
          if (evType === 'done')  { await onDone(); break; }
          if (evType === 'error') {
            getOrCreateTextDiv().innerHTML =
              `<span style="color:var(--err)">&#x26A0;&#xFE0F; ${esc(d.message||'Unknown error')}</span>`;
            setStreaming(false); break;
          }
          if (isPaused) { pauseBuffer.push([evType, d]); }
          else {
            if      (evType === 'chunk')      _onChunk(d.text);
            else if (evType === 'think')      _onThink(d.text);
            else if (evType === 'plan')       _onPlan(d.type, d.data);
            else if (evType === 'tool_start') _onToolStart(d);
            else if (evType === 'tool_end')   _onToolEnd(d);
          }
          evType = evData = null;
        }
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') console.error('Stream error:', err);
    await onDone();
  }
  const ty = document.getElementById('typing'); if (ty) ty.remove();
}

// ═══════════════════════════════════════════════════════════
// Settings panel
// ═══════════════════════════════════════════════════════════
let _loadedSettings = {};

async function openSettings() {
  document.getElementById('sett-overlay').classList.add('open');
  document.getElementById('sett-drawer').classList.add('open');
  await Promise.all([
    loadSettingsFromServer(),
    loadModelsFromServer(),
    loadConnectors(),
    loadUploadedModels(),
  ]);
}

function closeSettings() {
  document.getElementById('sett-overlay').classList.remove('open');
  document.getElementById('sett-drawer').classList.remove('open');
  _needsReload = false;
}

function toggleSect(hdr) {
  hdr.classList.toggle('open');
  hdr.nextElementSibling.classList.toggle('open');
}

async function loadSettingsFromServer() {
  try {
    _loadedSettings = await fetch('/settings').then(r => r.json());
    const inf = _loadedSettings.inference || {};
    const cpu = _loadedSettings.cpu || {};
    const mdl = _loadedSettings.model || {};
    const srch = _loadedSettings.search || {};

    _setVal('sett-temp',        inf.temperature ?? 0.7);
    document.getElementById('sett-temp-val').textContent =
      parseFloat(inf.temperature ?? 0.7).toFixed(2);
    _setVal('sett-max-tokens',  inf.max_tokens  ?? 4096);
    _setVal('sett-ctx',         inf.context_size ?? 16384);
    _setVal('sett-threads',     cpu.n_threads    ?? 4);
    _setVal('sett-batch',       cpu.batch_size   ?? 2048);
    _setVal('sett-gpu-layers',  cpu.gpu_layers   ?? 0);
    _setVal('sett-depth',       srch.depth       ?? 1);
    document.getElementById('sett-depth-val').textContent = srch.depth ?? 1;
    _setVal('sett-max-results', srch.max_results ?? 6);

    if (mdl.path) {
      document.getElementById('sett-current-model').textContent = mdl.name || mdl.path;
    }
    // mmproj path
    _setVal('sett-mmproj', mdl.mmproj || '');
    // Info
    const info = await fetch('/info').then(r => r.json()).catch(() => ({}));
    document.getElementById('sett-current-model').textContent = info.model || mdl.name || '—';
    document.getElementById('sett-db-path').textContent = 'DB: data/mastermind.db';
    // Show mmproj warning if set but file missing
    const mmprojEl = document.getElementById('sett-mmproj');
    if (info.mmproj_missing && mmprojEl) {
      mmprojEl.style.borderColor = 'var(--err)';
      mmprojEl.title = '⚠️ File not found — download mmproj-gemma-4-E2B-it-f16.gguf from HuggingFace';
    } else if (mmprojEl) {
      mmprojEl.style.borderColor = '';
      mmprojEl.title = '';
    }
    updateServerCmd();
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
}

async function loadModelsFromServer() {
  const sel = document.getElementById('sett-model-path');
  sel.innerHTML = '<option value="">— keep current —</option>';
  try {
    const models = await fetch('/models').then(r => r.json());
    const current = (_loadedSettings.model || {}).path || '';
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.path;
      opt.textContent = `${m.name}  (${m.size_gb} GB)`;
      if (m.path === current) opt.selected = true;
      sel.appendChild(opt);
    });
    if (models.length === 0) {
      sel.innerHTML = '<option value="">No .gguf files found — check GGUF_SEARCH_DIRS</option>';
    }
  } catch (e) {
    sel.innerHTML = '<option value="">Could not scan models</option>';
  }
}

function onModelChange() {
  _needsReload = true;
  document.getElementById('sett-reload-note').style.display = '';
  updateServerCmd();
}

async function applySettings() {
  const btn = document.getElementById('sett-apply');
  btn.disabled = true; btn.textContent = 'Saving…';

  const modelPath = document.getElementById('sett-model-path').value;
  const mmprojPath = document.getElementById('sett-mmproj').value.trim();
  const settings = {
    inference: {
      temperature:  parseFloat(document.getElementById('sett-temp').value),
      max_tokens:   parseInt(document.getElementById('sett-max-tokens').value),
      context_size: parseInt(document.getElementById('sett-ctx').value),
    },
    cpu: {
      n_threads:  parseInt(document.getElementById('sett-threads').value),
      batch_size: parseInt(document.getElementById('sett-batch').value),
      gpu_layers: parseInt(document.getElementById('sett-gpu-layers').value),
    },
    search: {
      depth:       parseInt(document.getElementById('sett-depth').value),
      max_results: parseInt(document.getElementById('sett-max-results').value),
    },
  };
  if (modelPath || mmprojPath) {
    settings.model = {
      path:   modelPath || ((_loadedSettings.model || {}).path || ''),
      name:   modelPath ? modelPath.split(/[\\/]/).pop().replace('.gguf','') : ((_loadedSettings.model || {}).name || ''),
      mmproj: mmprojPath,
    };
  }

  await fetch('/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  });

  if (_needsReload || modelPath) {
    btn.textContent = 'Reloading…';
    await fetch('/reload', { method: 'POST' });
    toast('Model reloading — next message will use new settings');
    document.getElementById('sett-reload-note').style.display = 'none';
    // Refresh model badge
    setTimeout(() => fetch('/info').then(r=>r.json()).then(d=>{
      const badge = document.getElementById('model-badge');
      badge.textContent = (d.model || 'local') + (d.vision ? '  👁' : '');
      badge.title = d.vision ? 'Vision enabled' : 'Vision disabled — add mmproj in Settings';
    }), 2000);
  } else {
    toast('Settings saved');
  }

  btn.disabled = false; btn.textContent = 'Apply & Save';
  closeSettings();
}

function _setVal(id, v) {
  const el = document.getElementById(id);
  if (el) el.value = v;
}

function updateServerCmd() {
  const el = document.getElementById('sett-server-cmd');
  if (!el) return;
  const mdl    = (_loadedSettings.model || {});
  const selPath = (document.getElementById('sett-model-path') || {}).value;
  const mpath  = selPath || mdl.path || '/path/to/model.gguf';
  const mmproj = (document.getElementById('sett-mmproj') || {}).value || mdl.mmproj || '';
  const ctx    = (document.getElementById('sett-ctx') || {}).value || 16384;
  const nt     = (document.getElementById('sett-threads') || {}).value || 4;
  const ntb    = Math.min(parseInt(nt) * 2, navigator.hardwareConcurrency || 8);
  const batch  = (document.getElementById('sett-batch') || {}).value || 2048;
  const gpu    = (document.getElementById('sett-gpu-layers') || {}).value || 0;
  const port   = 8080;
  let cmd = `llama-server \\\n  -m "${mpath}" \\`;
  if (mmproj) cmd += `\n  --mmproj "${mmproj}" \\`;
  cmd += `\n  -c ${ctx} \\`;
  cmd += `\n  -t ${nt} -tb ${ntb} \\`;
  cmd += `\n  -b ${batch} \\`;
  if (parseInt(gpu) > 0) cmd += `\n  -ngl ${gpu} \\`;
  cmd += `\n  --port ${port}`;
  // Replace cmd text but keep copy button
  const copyBtn = el.querySelector('.sett-cmd-copy');
  el.textContent = cmd;
  if (copyBtn) el.appendChild(copyBtn);
}

function copyServerCmd() {
  const el = document.getElementById('sett-server-cmd');
  const text = el.textContent.replace('copy','').trim();
  navigator.clipboard.writeText(text).then(() => toast('Command copied'));
}

// ═══════════════════════════════════════════════════════════
// UI builders
// ═══════════════════════════════════════════════════════════
function makeThinkBlock(text) {
  const div = document.createElement('div');
  div.className = 'think-block';
  div.innerHTML = `
    <div class="think-hdr" onclick="this.classList.toggle('open');this.nextElementSibling.classList.toggle('open')">
      <span class="think-chevron">▶</span> Thinking
    </div>
    <div class="think-body">${esc(text)}</div>`;
  return div;
}

function makeToolPill(t) {
  const labels = {
    bash:'Running command', web_search:'Searching web', web_fetch:'Fetching page',
    read_file:'Reading file', write_file:'Writing file', edit_file:'Editing file',
    glob:'Finding files', grep:'Searching files', list_dir:'Listing directory',
    git:'Git', memory_write:'Saving memory', memory_read:'Reading memory',
    scratchpad:'Scratchpad', reflect:'Reflecting', export_session:'Exporting',
    todo_write:'Updating tasks', todo_read:'Reading tasks', test_runner:'Running tests',
    pm:'Project management', agent:'Sub-agent',
  };
  const name = t.name || '';
  let label = labels[name] || (name === 'skill' ? 'Reasoning' : name.replace(/_/g,' '));
  if (name === 'skill' && t.summary) label = t.summary;
  const div = document.createElement('div');
  div.className = 'tool-pill' + (t.running ? ' running' : '') + (t.error ? ' err' : '');
  div.innerHTML = `
    <span class="pill-dot"></span>
    <span class="pill-label">${esc(label)}</span>
    ${(t.summary && name !== 'skill') ? `<span class="pill-summary">${esc(t.summary)}</span>` : ''}
    <div class="pill-output">${esc(t.output || '')}</div>`;
  return div;
}

function copyBtn() {
  return `<button class="mact-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>Copy</button>`;
}

function copyText(text) {
  navigator.clipboard.writeText(text || '').then(() => toast('Copied'));
}

function toast(msg) {
  const t = document.createElement('div');
  Object.assign(t.style, {
    position:'fixed', bottom:'70px', left:'50%', transform:'translateX(-50%)',
    background:'var(--text)', color:'var(--bg)', padding:'5px 14px',
    borderRadius:'20px', fontSize:'13px', zIndex:999, pointerEvents:'none',
    opacity:'0', transition:'opacity .18s',
  });
  t.textContent = msg; document.body.appendChild(t);
  requestAnimationFrame(() => {
    t.style.opacity = '1';
    setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 300); }, 1600);
  });
}

function renderMd(text) {
  let h = esc(text)
    .replace(/```(\w*)\n([\s\S]*?)```/g, (_,l,c)=>`<pre><code>${c.trimEnd()}</code></pre>`)
    .replace(/`([^`]+)`/g,'<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,'<em>$1</em>')
    .replace(/^### (.+)$/gm,'<h3>$1</h3>')
    .replace(/^## (.+)$/gm,'<h2>$1</h2>')
    .replace(/^# (.+)$/gm,'<h1>$1</h1>')
    .replace(/^\s*[-*] (.+)$/gm,'<li>$1</li>')
    .replace(/((<li>[^]*?<\/li>\n?)+)/g,'<ul>$1</ul>')
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g,'<a href="$2" target="_blank">$1</a>')
    .replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>');
  return h.startsWith('<') ? h : '<p>'+h+'</p>';
}

function esc(s) { return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
const _esc = esc;   // alias used by _buildConnectorCard and loadUploadedModels
function scrollBottom() { const w=document.getElementById('msgs-wrap'); w.scrollTop=w.scrollHeight; }

function toggleDark() {
  const el = document.documentElement;
  if (el.hasAttribute('data-dark')) { el.removeAttribute('data-dark'); localStorage.setItem('mm_dark','0'); }
  else { el.setAttribute('data-dark',''); localStorage.setItem('mm_dark','1'); }
}
function toggleSidebar() { document.getElementById('sidebar').classList.toggle('hide'); }
function closeSidebarMobile() { if (window.innerWidth <= 700) document.getElementById('sidebar').classList.add('hide'); }

function setupVoice() {
  const btn = document.getElementById('voice-btn');
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { btn.style.opacity = '.3'; return; }
  const rec = new SR();
  rec.continuous = false; rec.interimResults = true; rec.lang = 'en-US';
  rec.onresult = e => {
    const t = Array.from(e.results).map(r => r[0].transcript).join('');
    const ta = document.getElementById('input'); ta.value = t;
    document.getElementById('send-btn').disabled = !t.trim();
  };
  rec.onend = () => btn.style.color = '';
  btn.onclick = () => {
    if (btn.style.color === 'var(--accent)') { rec.stop(); btn.style.color = ''; }
    else { rec.start(); btn.style.color = 'var(--accent)'; if (navigator.vibrate) navigator.vibrate(15); }
  };
}// ═══════════════════════════════════════════════════════════
// Connectors
// ═══════════════════════════════════════════════════════════

// Connector config fields per connector id
const _CONNECTOR_FIELDS = {
  whatsapp: [
    { id:'wa-owner', label:'Your WhatsApp number', placeholder:'+393XXXXXXXXX', type:'text', key:'owner' },
  ],
  email: [
    { id:'em-addr',  label:'Email address',  placeholder:'you@gmail.com',    type:'email',    key:'email'     },
    { id:'em-pw',    label:'App password',   placeholder:'••••••••••••',     type:'password', key:'password'  },
    { id:'em-imap',  label:'IMAP host',      placeholder:'imap.gmail.com',   type:'text',     key:'imap_host' },
    { id:'em-smtp',  label:'SMTP host',      placeholder:'smtp.gmail.com',   type:'text',     key:'smtp_host' },
  ],
};

async function loadConnectors() {
  const list = document.getElementById('connectors-list');
  if (!list) return;
  try {
    const data = await fetch('/connectors').then(r => r.json());
    list.innerHTML = '';
    if (!Object.keys(data).length) {
      list.innerHTML = '<div style="color:var(--text3);font-size:13px">No connectors registered.</div>';
      return;
    }
    for (const [cid, st] of Object.entries(data)) {
      list.appendChild(_buildConnectorCard(cid, st));
    }
  } catch(e) {
    list.innerHTML = `<div style="color:var(--err);font-size:13px">Failed to load connectors: ${e}</div>`;
  }
}

function _buildConnectorCard(cid, st) {
  const card = document.createElement('div');
  card.id = `conn-card-${cid}`;
  card.style.cssText = `
    border:1px solid var(--border2);border-radius:10px;
    padding:12px 14px;margin-bottom:10px;background:var(--bg2);
  `;

  const dotColor = st.connected ? 'var(--tool)' : st.enabled ? '#d97706' : 'var(--text3)';
  const statusText = st.connected ? 'Connected' : st.enabled ? 'Enabled (connecting…)' : 'Disabled';

  let fieldsHtml = '';
  const fields = _CONNECTOR_FIELDS[cid] || [];
  for (const f of fields) {
    const val = (st.config && st.config[f.key]) ? _esc(st.config[f.key]) : '';
    fieldsHtml += `
      <div style="margin-top:8px">
        <div style="font-size:11.5px;color:var(--text3);margin-bottom:3px">${_esc(f.label)}</div>
        <input type="${f.type}" id="${f.id}" placeholder="${_esc(f.placeholder)}"
          value="${val}"
          style="width:100%;background:var(--bg3);border:1px solid var(--border2);
                 color:var(--text);border-radius:7px;padding:6px 9px;font-size:13px;
                 font-family:var(--font);outline:none">
      </div>`;
  }

  card.innerHTML = `
    <div style="display:flex;align-items:center;gap:10px">
      <span style="font-size:20px">${_esc(st.icon||'🔌')}</span>
      <div style="flex:1">
        <div style="font-size:13px;font-weight:600">${_esc(st.label||cid)}</div>
        <div style="font-size:11.5px;color:var(--text3);display:flex;align-items:center;gap:5px;margin-top:2px">
          <span style="width:7px;height:7px;border-radius:50%;background:${dotColor};display:inline-block;flex-shrink:0"></span>
          <span id="conn-status-${cid}">${statusText}</span>
        </div>
      </div>
      <label class="toggle" title="${st.enabled ? 'Click to disable' : 'Click to enable'}">
        <input type="checkbox" id="conn-tog-${cid}" ${st.enabled ? 'checked' : ''}
          onchange="toggleConnector('${cid}', this.checked)">
        <span class="tog-sl"></span>
      </label>
    </div>
    ${fieldsHtml}
    ${st.owner ? `<div style="font-size:11px;color:var(--text3);margin-top:6px">Number: ${_esc(st.owner)}</div>` : ''}
    ${cid === 'whatsapp' && st.started && !st.connected
      ? '<div style="font-size:11.5px;color:#d97706;margin-top:6px">📷 Scan QR code in terminal to connect</div>'
      : ''}
  `;
  return card;
}

async function toggleConnector(cid, enable) {
  const statusEl = document.getElementById(`conn-status-${cid}`);
  if (statusEl) statusEl.textContent = enable ? 'Enabling…' : 'Disabling…';

  // Collect config fields for this connector
  const fields = _CONNECTOR_FIELDS[cid] || [];
  const cfg = {};
  for (const f of fields) {
    const el = document.getElementById(f.id);
    if (el && el.value.trim()) cfg[f.key] = el.value.trim();
  }

  try {
    const url = enable ? `/connectors/${cid}/enable` : `/connectors/${cid}/disable`;
    const res = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: enable ? JSON.stringify(cfg) : '{}',
    }).then(r => r.json());

    if (res.error) {
      toast(`❌ ${res.error}`);
      if (statusEl) statusEl.textContent = 'Error';
      return;
    }
    const st = res.status || {};
    const dotColor = st.connected ? 'var(--tool)' : st.enabled ? '#d97706' : 'var(--text3)';
    if (statusEl) {
      statusEl.textContent = st.connected ? 'Connected' : st.enabled ? 'Enabled (connecting…)' : 'Disabled';
    }
    toast(enable ? `✅ ${cid} enabled` : `🔇 ${cid} disabled`);
    // Refresh the sidebar connector dots
    _pollStatus();
  } catch(e) {
    toast(`❌ ${e}`);
  }
}

// ═══════════════════════════════════════════════════════════
// Model upload
// ═══════════════════════════════════════════════════════════

async function loadUploadedModels() {
  const list = document.getElementById('uploaded-models-list');
  if (!list) return;
  try {
    const models = await fetch('/models').then(r => r.json());
    const uploaded = models.filter(m => m.uploaded);
    if (!uploaded.length) { list.innerHTML = ''; return; }
    list.innerHTML = '<div style="font-size:11.5px;color:var(--text3);margin-bottom:5px">Uploaded models:</div>' +
      uploaded.map(m => `
        <div style="display:flex;align-items:center;gap:8px;padding:5px 8px;
                    background:var(--bg3);border-radius:7px;margin-bottom:4px;font-size:12.5px">
          <span style="flex:1;font-family:var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
            title="${_esc(m.path)}">${_esc(m.name)}</span>
          <span style="color:var(--text3);flex-shrink:0">${m.size_gb} GB</span>
          <button onclick="deleteUploadedModel('${_esc(m.filename)}')"
            style="background:none;border:none;color:var(--text3);cursor:pointer;
                   padding:2px 6px;border-radius:5px;font-size:13px"
            title="Delete">✕</button>
        </div>`).join('');
  } catch(e) {}
}

async function handleModelUpload(evt) {
  const file = evt.target.files[0];
  if (file) await _doUploadModel(file);
  evt.target.value = '';
}

async function handleModelDrop(evt) {
  evt.preventDefault();
  evt.currentTarget.style.background = '';
  const file = evt.dataTransfer.files[0];
  if (file) await _doUploadModel(file);
}

async function _doUploadModel(file) {
  if (!file.name.toLowerCase().endsWith('.gguf')) {
    toast('❌ Only .gguf files are accepted'); return;
  }
  const prog  = document.getElementById('model-upload-progress');
  const bar   = document.getElementById('model-upload-bar');
  const label = document.getElementById('model-upload-label');
  prog.style.display = 'block';
  bar.style.width = '0%';
  label.textContent = `Uploading ${file.name} (${(file.size/1024/1024/1024).toFixed(2)} GB)…`;

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Use XHR for upload progress events
    await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/models/upload');
      xhr.upload.onprogress = e => {
        if (e.lengthComputable) {
          const pct = (e.loaded / e.total * 100).toFixed(0);
          bar.style.width = pct + '%';
          label.textContent = `Uploading… ${pct}%`;
        }
      };
      xhr.onload = () => {
        const res = JSON.parse(xhr.responseText);
        if (res.error) { toast(`❌ ${res.error}`); reject(res.error); return; }
        bar.style.width = '100%';
        label.textContent = `✅ ${res.name} uploaded (${res.size_gb} GB)`;
        toast(`✅ Model uploaded: ${res.name}`);
        // Refresh model picker
        loadModelsFromServer();
        loadUploadedModels();
        setTimeout(() => { prog.style.display = 'none'; }, 3000);
        resolve(res);
      };
      xhr.onerror = () => { toast('❌ Upload failed'); reject('network error'); };
      xhr.send(formData);
    });
  } catch(e) {
    bar.style.width = '0%';
    label.textContent = 'Upload failed';
  }
}

async function deleteUploadedModel(filename) {
  if (!confirm(`Delete ${filename}?`)) return;
  try {
    const res = await fetch(`/models/upload/${encodeURIComponent(filename)}`,
      { method: 'DELETE' }).then(r => r.json());
    if (res.ok) { toast('Deleted'); loadUploadedModels(); loadModelsFromServer(); }
    else toast(`❌ ${res.error}`);
  } catch(e) { toast(`❌ ${e}`); }
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML

@app.get("/info")
async def info():
    stored  = db.get_settings()
    mdl     = stored.get("model", {})
    mmproj  = mdl.get("mmproj", "").strip()
    m_path  = mdl.get("path") or MODEL_PATH
    vision_ready = bool(mmproj) and Path(mmproj).exists()
    if _client is not None:
        vision_ready = getattr(_client, "_vision_enabled", vision_ready)
    conn_status = _connector_registry.status_all()
    return {
        "model":          Path(m_path).stem if m_path else "no model set",
        "model_path":     m_path,
        "context":        CONTEXT_SIZE,
        "backend":        "direct" if DIRECT_MODE else "server",
        "vision":         vision_ready,
        "mmproj":         mmproj,
        "mmproj_missing": bool(mmproj) and not Path(mmproj).exists(),
        "connectors":     conn_status,
        "models_dir":     str(MODELS_DIR),
    }

def _free_port(preferred=7860):
    import socket as _s
    for p in range(preferred, preferred + 20):
        try:
            sock = _s.socket(); sock.bind(("0.0.0.0", p)); sock.close(); return p
        except OSError: continue
    return preferred

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 0)) or _free_port(7860)
    print(f"\n  MasterMind WebUI v4")
    print(f"  Open: http://localhost:{port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")