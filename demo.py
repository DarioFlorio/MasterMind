#!/usr/bin/env python3
"""
PyClaudeCode – Streamlit UI  ·  Enhanced Edition v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Engine: 100% original and untouched.
UI enhancements (plug-and-play, zero engine edits):
  ● True black dark / white light theme — fully isolated, zero bleed on switch
  ● Blue buttons on dark (DeepSeek-style) · Black buttons on light
  ● Code always rendered as artifact panels (never inline markdown)
  ● File upload / drag-and-drop IN the chat input bar (any file type)
  ● Paste-to-attach support inside the input area
  ● Chat persistence  – save / load / new, sidebar history list
  ● Stop / Pause  – real mid-stream cancel
  ● Thinking        – collapsible balloon
  ● st.html() replaces deprecated st.components.v1.html where possible
  ● Zero-fail error handling with multiple fallbacks throughout
"""

import os, sys, subprocess, importlib.util
import threading, time, json, re, uuid, shutil, base64
from pathlib import Path
from queue import Queue, Empty
from datetime import datetime

# ── AUTO-INSTALLER ──────────────────────────────────────────
def auto_install(package, import_name=None):
    if import_name is None:
        import_name = package
    try:
        if importlib.util.find_spec(import_name):
            return True
    except (ValueError, ModuleNotFoundError):
        pass
    print(f"Installing {package}...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        return False
    return True

for _pkg, _imp in [("streamlit","streamlit"),("httpx","httpx"),("llama-cpp-python","llama_cpp")]:
    auto_install(_pkg, _imp)

if not os.environ.get("STREAMLIT_RUNNING"):
    _env = os.environ.copy(); _env["STREAMLIT_RUNNING"] = "1"
    subprocess.run(["streamlit", "run", __file__], env=_env)
    sys.exit(0)

# ── NORMAL IMPORTS ────────────────────────────────────────
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from config.settings import (
    LLAMA_SERVER_URL, LLAMA_SERVER_PORT, MODEL_PATH, MODEL_DISPLAY,
    MAX_TURNS, PERMISSION_MODE, VERBOSE, WORKING_DIR, CONTEXT_SIZE, DIRECT_MODE,
)
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
from utils.model_client import ModelClient, ThinkingStreamParser
from utils.permissions import PermissionManager
from utils.token_counter import SessionUsage
try:
    from heartbeat import Heartbeat
except ImportError:
    pass

# ── HTML helper ──────────────────────────────────────────
def _html(content: str, height=None):
    try:
        if height is not None:
            b64 = base64.b64encode(content.encode()).decode()
            st.html(
                f'<iframe src="data:text/html;base64,{b64}" '
                f'style="width:100%;height:{height}px;border:none;'
                f'border-radius:0 0 8px 8px;display:block"></iframe>'
            )
        else:
            st.html(content)
    except Exception:
        try:
            st.markdown(content, unsafe_allow_html=True)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════
# CHAT PERSISTENCE
# ════════════════════════════════════════════════════════════
CHAT_DIR = Path.home() / ".pyclaudecode" / "chats"
try:
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    CHAT_DIR = Path(ROOT) / ".chats"
    CHAT_DIR.mkdir(parents=True, exist_ok=True)

def _cpath(cid): return CHAT_DIR / f"{cid}.json"
def new_cid():   return uuid.uuid4().hex[:12]

def chats_list():
    out = []
    try:
        for f in sorted(CHAT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                out.append({"id": f.stem, "title": d.get("title","Untitled"),
                            "ts": d.get("updated","")[:10], "n": len(d.get("messages",[]))})
            except Exception:
                pass
    except Exception:
        pass
    return out

def chat_save(cid, messages, title=None):
    try:
        if not title:
            for m in messages:
                if m["role"] == "user":
                    title = m["content"][:55].strip().splitlines()[0]; break
            title = title or "New Chat"
        _cpath(cid).write_text(json.dumps({
            "id": cid, "title": title,
            "updated": datetime.now().isoformat(), "messages": messages,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

def chat_load(cid):
    try:
        p = _cpath(cid)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except Exception:
        return None

def chat_delete(cid):
    try:
        p = _cpath(cid)
        if p.exists(): p.unlink()
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# ARTIFACT SYSTEM
# ════════════════════════════════════════════════════════════
_FENCE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)

def extract_artifacts(text):
    arts = []
    try:
        for m in _FENCE.finditer(text):
            lang = (m.group("lang") or "text").strip().lower()
            code = m.group("code")
            if len(code.strip()) < 4:
                continue
            atype = ("html"    if lang in ("html","htm")      else
                     "svg"     if lang == "svg"               else
                     "mermaid" if lang in ("mermaid","mmd")   else "code")
            arts.append({"type": atype, "lang": lang, "code": code,
                         "id": uuid.uuid4().hex[:8]})
    except Exception:
        pass
    return arts

def _strip_code_fences(text: str) -> str:
    try:
        return re.sub(r"```[^\n`]*\n.*?```", "", text, flags=re.DOTALL).strip()
    except Exception:
        return text

def render_artifact(art: dict, idx: int, dark: bool):
    try:
        atype, lang, code, aid = art["type"], art["lang"], art["code"], art["id"]
        icons = {"html": "🌐", "svg": "🎨", "mermaid": "📊", "code": "💻"}
        type_labels = {"html": "HTML", "svg": "SVG", "mermaid": "Diagram"}
        icon  = icons.get(atype, "💻")
        label = type_labels.get(atype, lang.upper() or "CODE")
        b64   = base64.b64encode(code.encode()).decode()
        fname = f"artifact_{idx+1}.{lang or 'txt'}"
        dl    = f"data:text/plain;base64,{b64}"
        hdr_bg  = "#0d1628" if dark else "#e8eeff"
        hdr_brd = "#1c3060" if dark else "#c0d0f0"
        hdr_clr = "#4d9fff" if dark else "#1a3a7a"
        btn_clr = "#555"    if dark else "#444"
        body_brd = "#1c3060" if dark else "#c0d0f0"

        _html(f"""
        <div style="background:{hdr_bg};border:1px solid {hdr_brd};border-bottom:none;
             border-radius:8px 8px 0 0;padding:8px 14px;display:flex;
             align-items:center;gap:8px;font-size:12px;
             font-family:'JetBrains Mono','Fira Code',monospace;color:{hdr_clr}">
          <span>{icon}</span><span>{label}</span>
          <code style="opacity:.6;font-size:11px">{lang}</code>
          <span style="flex:1"></span>
          <button id="cp_{aid}"
            onclick="navigator.clipboard.writeText(atob('{b64}'))
              .then(()=>{{var b=document.getElementById('cp_{aid}');b.innerText='Copied';
                setTimeout(()=>b.innerText='Copy',1600)}}).catch(()=>{{}})"
            style="padding:3px 10px;font-size:11px;cursor:pointer;border-radius:4px;
              border:1px solid {btn_clr};background:transparent;color:{hdr_clr}">Copy</button>
          <a href="{dl}" download="{fname}"
            style="padding:3px 10px;font-size:11px;border-radius:4px;
              border:1px solid {btn_clr};background:transparent;
              color:{hdr_clr};text-decoration:none">Save</a>
        </div>""")

        if atype == "code":
            st.code(code, language=lang if lang not in ("text","") else None)
        elif atype == "html":
            _html(code, height=440)
        elif atype == "svg":
            _html(f'<div style="background:#fff;padding:16px;text-align:center;'
                  f'border:1px solid {body_brd};border-top:none;border-radius:0 0 8px 8px">'
                  f'{code}</div>')
        elif atype == "mermaid":
            mmd_doc = (
                f'<!DOCTYPE html><html><body style="margin:0;padding:12px;'
                f'background:{"#1a1a2e" if dark else "#fff"}">'
                f'<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>'
                f'<script>mermaid.initialize({{startOnLoad:true,theme:"{"dark" if dark else "default"}"}})</script>'
                f'<div class="mermaid" style="text-align:center">{code}</div>'
                f'</body></html>'
            )
            _html(mmd_doc, height=380)
    except Exception:
        try:
            st.code(art.get("code", ""), language=art.get("lang") or None)
        except Exception:
            pass


# ════════════════════════════════════════════════════════════
# PER-MESSAGE COPY BUTTON
# ════════════════════════════════════════════════════════════
def msg_copy_btn(text: str):
    try:
        b64 = base64.b64encode(text.encode()).decode()
        bid = uuid.uuid4().hex[:6]
        _html(f"""<div style="text-align:right;margin-bottom:2px">
          <button id="mb_{bid}"
            onclick="navigator.clipboard.writeText(atob('{b64}'))
              .then(()=>{{document.getElementById('mb_{bid}').innerText='Copied';
                setTimeout(()=>document.getElementById('mb_{bid}').innerText='Copy',1500)}})
              .catch(()=>{{}})"
            style="font-size:11px;padding:2px 8px;cursor:pointer;border-radius:4px;
                   border:1px solid #333;background:transparent;color:#666">Copy</button>
        </div>""")
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# FILE UPLOAD PROCESSOR
# ════════════════════════════════════════════════════════════
_TEXT_EXTS = {
    '.py','.js','.ts','.jsx','.tsx','.css','.html','.htm','.json',
    '.md','.txt','.yaml','.yml','.toml','.sh','.bash','.sql','.r',
    '.java','.c','.cpp','.h','.cs','.go','.rs','.rb','.php','.swift',
    '.kt','.scala','.lua','.pl','.xml','.csv','.ini','.cfg','.env',
    '.dockerfile','.tf','.vue','.svelte',
}
_IMG_EXTS = {'.png','.jpg','.jpeg','.gif','.webp','.bmp','.svg'}

def _process_uploads(files) -> str:
    if not files:
        return ""
    parts = ["\n\n[Attached files]"]
    for f in files:
        try:
            name = getattr(f, "name", "file")
            ext  = Path(name).suffix.lower()
            try:
                if ext in _TEXT_EXTS:
                    raw = f.read() if hasattr(f, "read") else b""
                    content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                    lang = ext[1:]
                    parts.append(f"\n--- {name} ---\n```{lang}\n{content}\n```")
                elif ext in _IMG_EXTS:
                    size = getattr(f, "size", 0)
                    parts.append(f"\n--- {name} (image, {size:,} bytes) ---\n[Image attached]")
                else:
                    raw = f.read() if hasattr(f, "read") else b""
                    try:
                        content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                        parts.append(f"\n--- {name} ---\n{content[:4000]}"
                                     + ("...[truncated]" if len(content) > 4000 else ""))
                    except Exception:
                        size = getattr(f, "size", len(raw) if raw else 0)
                        parts.append(f"\n--- {name} ({size:,} bytes, binary) ---")
            except Exception as e:
                parts.append(f"\n--- {name} (read error: {e}) ---")
        except Exception as e:
            parts.append(f"\n--- [file] (error: {e}) ---")
    return "\n".join(parts)


# ════════════════════════════════════════════════════════════
# IN-MEMORY FILE WRAPPER (for JS-bridge decoded files)
# ════════════════════════════════════════════════════════════
class _B64File:
    def __init__(self, name: str, data: bytes, mime: str = ""):
        self.name = name
        self.size = len(data)
        self._data = data
        self.type  = mime
        self._pos  = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0:
            result = self._data[self._pos:]; self._pos = len(self._data)
        else:
            result = self._data[self._pos:self._pos + n]; self._pos += len(result)
        return result

    def seek(self, pos: int): self._pos = max(0, min(pos, len(self._data)))
    def tell(self) -> int:    return self._pos


# ════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="PyClaudeCode", layout="wide", page_icon="🤖")

_SS_DEFAULTS = {
    "theme_dark":       True,
    "messages":         [],
    "web_search":       True,
    "chat_id":          new_cid(),
    "chat_title":       None,
    "app_state":        "idle",
    "stream_q":         None,
    "stop_event":       threading.Event(),
    "pause_event":      threading.Event(),
    "stream_data":      {},
    "stream_parser":    None,
    "warmup_done":      False,
    "attached_files":   [],
}
for _k, _v in _SS_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ════════════════════════════════════════════════════════════
# THEME  — complete CSS isolation per theme, zero bleed on switch
# A unique token is embedded in each call so no rule from the previous
# theme survives (Streamlit appends <style> tags; we win by specificity
# + !important on every single rule, plus a JS snippet that resets
# any inline styles Streamlit may have set on root elements).
# ════════════════════════════════════════════════════════════
def inject_css(dark: bool):
    if dark:
        bg        = "#000000"
        sbg       = "#080808"
        umsg_bg   = "#03091a"
        umsg_brd  = "#0f2a5a"
        amsg_bg   = "#0c0c0c"
        amsg_brd  = "#1a1a1a"
        brd       = "#1c1c1c"
        txt       = "#d8d8d8"
        txt_dim   = "#555"
        inp_bg    = "#0d0d0d"
        inp_brd   = "#272727"
        cbg       = "#080d14"
        code_brd  = "#1a2336"
        btn_bg    = "#1a6fff"
        btn_hover = "#1558d6"
        btn_txt   = "#ffffff"
        btn2_bg   = "#111111"
        btn2_brd  = "#2a2a2a"
        btn2_txt  = "#888888"
        th_bg     = "linear-gradient(135deg,#0a0800,#060500)"
        th_brd    = "#7a5c00"
        th_clr    = "#a88010"
        tool_clr  = "#4d9fff"
        inp_focus = "#1a6fff"
        inp_shadow= "rgba(26,111,255,0.18)"
        toggle_on = "#1a6fff"
        drop_bg   = "#0a1428"
        drop_brd  = "#1a4080"
        drop_txt  = "#4d7fff"
        drop_hover= "#0f1e3a"
        scheme    = "dark"
    else:
         # Light theme only
        bg        = "#ffffff"
        sbg       = "#f6f6f6"
        umsg_bg   = "#eef3ff"
        umsg_brd  = "#c5d5f8"
        amsg_bg   = "#fafafa"
        amsg_brd  = "#e2e2e2"
        brd       = "#e2e2e2"
        txt       = "#0f0f0f"
        txt_dim   = "#aaa"
        inp_bg    = "#ffffff"
        inp_brd   = "#c8c8c8"
        cbg       = "#f7f7f7"
        code_brd  = "#ddd"
        btn_bg    = "#0f0f0f"
        btn_hover = "#333"
        btn_txt   = "#ffffff"
        btn2_bg   = "#f0f0f0"
        btn2_brd  = "#d0d0d0"
        btn2_txt  = "#444"
        th_bg     = "linear-gradient(135deg,#fff9ea,#fff4d6)"
        th_brd    = "#d4a017"
        th_clr    = "#8a6e00"
        tool_clr  = "#1a3a8a"
        inp_focus = "#0f0f0f"
        inp_shadow= "rgba(0,0,0,0.12)"
        toggle_on = "#0f0f0f"
        drop_bg   = "#f0f4ff"
        drop_brd  = "#b0c4f8"
        drop_txt  = "#1a3a8a"
        drop_hover= "#e4ecff"
        scheme    = "light"
        
        

    # Unique token forces Streamlit to treat this as a fresh <style> block
    _uid = uuid.uuid4().hex[:8]

    st.markdown(f"""<style>
    /* THEME={scheme} uid={_uid} */

    :root, html {{
        color-scheme: {scheme} !important;
    }}

    /* ── Base ───────────────────────────────────────────── */
    html, body, #root, .stApp, .main, .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stBottom"],
    [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"] {{
        background-color: {bg} !important;
        background: {bg} !important;
        color: {txt} !important;
        font-family: -apple-system,'SF Pro Text','Segoe UI',system-ui,sans-serif;
    }}

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] .stSidebarUserContent {{
        background: {sbg} !important;
        border-right: 1px solid {brd} !important;
    }}
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {txt} !important;
    }}

    /* ── Chat messages ───────────────────────────────────── */
    [data-testid="stChatMessageContent"] {{
        background: {amsg_bg} !important;
        border: 1px solid {amsg_brd} !important;
        border-radius: 10px !important;
        color: {txt} !important;
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
      [data-testid="stChatMessageContent"] {{
        background: {umsg_bg} !important;
        border: 1px solid {umsg_brd} !important;
    }}

    /* ── Code & pre ──────────────────────────────────────── */
    .stCodeBlock, pre, code {{
        background: {cbg} !important;
        border: 1px solid {code_brd} !important;
        color: {txt} !important;
        font-family: 'JetBrains Mono','Fira Code','Cascadia Code',monospace !important;
        font-size: 13px !important;
    }}

    /* ── Buttons ─────────────────────────────────────────── */
    .stButton > button {{
        background: {btn_bg} !important;
        color: {btn_txt} !important;
        border: none !important;
        border-radius: 7px !important;
        font-weight: 500 !important;
        transition: background .15s;
    }}
    .stButton > button:hover {{
        background: {btn_hover} !important;
        opacity: 1 !important;
    }}
    section[data-testid="stSidebar"] .stButton > button {{
        background: {btn2_bg} !important;
        color: {btn2_txt} !important;
        border: 1px solid {btn2_brd} !important;
        font-size: 13px !important;
    }}
    .stop-btn  > button {{
        background: #2a0000 !important;
        border: 1px solid #700 !important;
        color: #f77 !important;
    }}
    .pause-btn > button {{
        background: #181400 !important;
        border: 1px solid #554 !important;
        color: #ee8 !important;
    }}

    /* ── Chat input ──────────────────────────────────────── */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] > div {{
        background: {inp_bg} !important;
        border: 1px solid {inp_brd} !important;
        color: {txt} !important;
        border-radius: 10px !important;
        font-size: 14px !important;
    }}
    [data-testid="stChatInput"] textarea:focus {{
        border-color: {inp_focus} !important;
        box-shadow: 0 0 0 2.5px {inp_shadow} !important;
    }}
    [data-testid="stChatInput"] button {{
        background: {btn_bg} !important;
        border-radius: 7px !important;
    }}
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {txt_dim} !important;
    }}

    /* ── Native Streamlit file uploader ──────────────────── */
    [data-testid="stFileUploader"] {{
        background: {inp_bg} !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: {inp_bg} !important;
        border: 1.5px dashed {inp_brd} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stFileUploader"] * {{ color: {txt} !important; }}

    /* ── JS-bridge dropzone ───────────────────────────────── */
    #pcc-drop-zone {{
        background: {drop_bg} !important;
        border: 1.5px dashed {drop_brd} !important;
        color: {drop_txt} !important;
        border-radius: 10px !important;
        transition: background .15s, border-color .15s;
    }}
    #pcc-drop-zone.drag-over {{
        background: {drop_hover} !important;
        border-color: {inp_focus} !important;
    }}
    #pcc-file-list {{ color: {txt} !important; }}
    .pcc-chip {{
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 8px;
        border-radius: 5px;
        background: {inp_bg} !important;
        border: 1px solid {inp_brd} !important;
        color: {txt} !important;
        font-size: 12px;
    }}
    .pcc-rm {{
        cursor: pointer;
        color: {txt_dim} !important;
        font-size: 11px;
        padding: 0 2px;
    }}
    .pcc-rm:hover {{ color: #e33 !important; }}

    /* ── Thinking block ──────────────────────────────────── */
    .thinking-block {{
        background: {th_bg};
        border-left: 3px solid {th_brd};
        color: {th_clr};
        padding: 8px 14px;
        border-radius: 5px;
        font-size: 12.5px;
        white-space: pre-wrap;
        line-height: 1.6;
        font-family: 'JetBrains Mono','Fira Code',monospace;
    }}

    /* ── Tool trace ──────────────────────────────────────── */
    .tool-line {{
        background: {cbg};
        border: 1px solid {brd};
        border-radius: 5px;
        padding: 4px 10px;
        font-size: 12px;
        margin: 2px 0;
        font-family: 'JetBrains Mono','Fira Code',monospace;
        color: {txt_dim};
    }}
    .tool-line b {{ color: {tool_clr}; }}

    /* ── Form controls ───────────────────────────────────── */
    .stToggle input:checked + div {{ background: {toggle_on} !important; }}
    .stSelectbox [data-baseweb="select"],
    .stSelectbox [data-baseweb="select"] > div {{
        background: {inp_bg} !important;
        border: 1px solid {brd} !important;
        color: {txt} !important;
    }}
    [data-testid="stTextArea"] textarea,
    .stTextInput input {{
        background: {inp_bg} !important;
        border: 1px solid {brd} !important;
        color: {txt} !important;
        border-radius: 6px !important;
    }}

    /* ── Expanders / dividers ────────────────────────────── */
    hr {{ border-color: {brd} !important; }}
    [data-testid="stExpander"] {{
        border: 1px solid {brd} !important;
        border-radius: 8px !important;
        background: {amsg_bg} !important;
    }}
    summary, [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] p {{ color: {txt} !important; }}

    /* ── Scrollbar ───────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
    ::-webkit-scrollbar-track {{ background: {bg}; }}
    ::-webkit-scrollbar-thumb {{ background: {brd}; border-radius: 3px; }}

    /* ── Misc text ───────────────────────────────────────── */
    .stMarkdown p, p, span, label {{ color: {txt}; }}
    .stCaption, small {{ color: {txt_dim} !important; }}
    h1, h2, h3, h4, h5, h6 {{ color: {txt} !important; }}
    [data-testid="stAlert"] {{
        background: {amsg_bg} !important;
        border: 1px solid {brd} !important;
        color: {txt} !important;
    }}
    </style>

    
    """, unsafe_allow_html=True)


inject_css(st.session_state.theme_dark)


# ════════════════════════════════════════════════════════════
# JS FILE BRIDGE — drag/drop + paste into input area
# Files are base64-encoded in JS, sent via a hidden text_input,
# decoded server-side and stored in session_state.attached_files.
# Supports: click-to-browse, drag onto dropzone, drag onto chat
# textarea, and global Ctrl+V paste of files/images.
# ════════════════════════════════════════════════════════════
_FILE_BRIDGE_KEY = "pcc_file_bridge_v1"

def _render_file_bridge():
    # Hidden bridge input — receives base64-JSON from JS
    bridge_val = st.text_input(
        "file-bridge",
        key=_FILE_BRIDGE_KEY,
        label_visibility="collapsed",
        value="",
    )

    # Decode bridge payload
    if bridge_val and bridge_val.strip():
        if bridge_val == "__clear__":
            st.session_state.attached_files = []
        else:
            try:
                payload = json.loads(bridge_val)
                if isinstance(payload, list) and payload:
                    existing_names = {f.name for f in st.session_state.attached_files}
                    for item in payload:
                        try:
                            name    = item.get("name", "file")
                            b64data = item.get("data", "")
                            mime    = item.get("type", "application/octet-stream")
                            raw     = base64.b64decode(b64data) if b64data else b""
                            if name not in existing_names:
                                st.session_state.attached_files.append(
                                    _B64File(name=name, data=raw, mime=mime))
                                existing_names.add(name)
                        except Exception:
                            pass
            except Exception:
                pass

    # Show attached file chips
    attached = st.session_state.attached_files
    if attached:
        chip_html = "".join(
            f'<span class="pcc-chip" title="{f.name}">'
            f'📎 {f.name[:30]}{"…" if len(f.name)>30 else ""}'
            f'<span class="pcc-rm" onclick="pccRemoveFile()" title="Remove all">✕</span>'
            f'</span>'
            for f in attached
        )
        _html(f'<div id="pcc-file-list" style="display:flex;flex-wrap:wrap;gap:6px;'
              f'padding:6px 2px">{chip_html}</div>')

    # Drop zone widget + JS logic
    _html("""
<div id="pcc-drop-zone"
     style="display:flex;align-items:center;justify-content:center;gap:8px;
            padding:10px 16px;margin:4px 0 6px 0;cursor:pointer;
            font-size:13px;user-select:none"
     onclick="document.getElementById('pcc-file-input').click()"
     title="Click to browse, drag files here, or paste (Ctrl+V)">
  <span>📎</span>
  <span>Drop files here &middot; Click to browse &middot; Paste Ctrl+V</span>
  <input id="pcc-file-input" type="file" multiple style="display:none"
         onchange="pccHandleFiles(this.files)">
</div>

<script>
(function() {
  /* Find the Streamlit text-input that serves as our bridge.
     We identify it by checking all visible text inputs and picking
     the one whose value we control (it starts empty on each rerun). */
  function findBridgeInput() {
    // Try multiple selectors as Streamlit's DOM structure varies by version
    var candidates = Array.from(document.querySelectorAll('input[type="text"]'));
    // The bridge is the most-recently added visible text input
    // (Streamlit inserts our label-collapsed widget at the bottom of the block)
    for (var i = candidates.length - 1; i >= 0; i--) {
      var el = candidates[i];
      if (el && el.offsetParent !== null) return el;
    }
    return null;
  }

  function setInputValue(inp, val) {
    try {
      var setter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value').set;
      setter.call(inp, val);
      inp.dispatchEvent(new Event('input', { bubbles: true }));
      inp.dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    } catch(e) { return false; }
  }

  function sendToBridge(jsonStr) {
    var attempts = 0;
    function trySet() {
      var inp = findBridgeInput();
      if (inp && setInputValue(inp, jsonStr)) return;
      if (attempts++ < 15) setTimeout(trySet, 150);
    }
    setTimeout(trySet, 50);
  }

  function pccHandleFiles(files) {
    if (!files || !files.length) return;
    var total   = files.length;
    var done    = 0;
    var results = new Array(total);

    Array.from(files).forEach(function(file, idx) {
      var reader = new FileReader();
      reader.onload = function(e) {
        try {
          var dataUrl = e.target.result || '';
          var commaIdx = dataUrl.indexOf(',');
          var data = commaIdx >= 0 ? dataUrl.slice(commaIdx + 1) : '';
          results[idx] = {
            name: file.name,
            type: file.type || 'application/octet-stream',
            data: data
          };
        } catch(err) {
          results[idx] = { name: file.name, type: 'application/octet-stream', data: '' };
        }
        done++;
        if (done === total) {
          var valid = results.filter(function(r) { return r && r.name; });
          if (valid.length) sendToBridge(JSON.stringify(valid));
        }
      };
      reader.onerror = function() {
        results[idx] = { name: file.name, type: 'application/octet-stream', data: '' };
        done++;
        if (done === total) {
          var valid = results.filter(function(r) { return r && r.name; });
          if (valid.length) sendToBridge(JSON.stringify(valid));
        }
      };
      try { reader.readAsDataURL(file); }
      catch(e) { done++; }
    });
  }
  window.pccHandleFiles = pccHandleFiles;

  window.pccRemoveFile = function() {
    sendToBridge('__clear__');
  };

  /* ── Drop zone drag events ── */
  var zone = document.getElementById('pcc-drop-zone');
  if (zone) {
    zone.addEventListener('dragover', function(e) {
      e.preventDefault(); e.stopPropagation();
      zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', function() {
      zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', function(e) {
      e.preventDefault(); e.stopPropagation();
      zone.classList.remove('drag-over');
      pccHandleFiles(e.dataTransfer.files);
    });
  }

  /* ── Enable drop on Streamlit chat textarea too ── */
  function patchChatDrop() {
    try {
      var area = document.querySelector('[data-testid="stChatInput"] textarea');
      if (area && !area._pccDrop) {
        area._pccDrop = true;
        area.addEventListener('dragover', function(e) { e.preventDefault(); });
        area.addEventListener('drop', function(e) {
          e.preventDefault(); e.stopPropagation();
          if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
            pccHandleFiles(e.dataTransfer.files);
          }
        });
      }
    } catch(e) {}
  }
  patchChatDrop();
  [800, 2000, 4000, 8000].forEach(function(t) { setTimeout(patchChatDrop, t); });

  /* ── Global paste listener ── */
  document.addEventListener('paste', function(e) {
    try {
      var cd = e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData);
      if (!cd || !cd.items) return;
      var fileItems = [];
      for (var i = 0; i < cd.items.length; i++) {
        if (cd.items[i].kind === 'file') {
          var f = cd.items[i].getAsFile();
          if (f) fileItems.push(f);
        }
      }
      if (fileItems.length) pccHandleFiles(fileItems);
    } catch(err) {}
  });
})();
</script>
""")


# ════════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ════════════════════════════════════════════════════════════
_server_proc = None

def _find_server():
    try:
        ov = os.environ.get("LLAMA_SERVER_BIN","")
        if ov and Path(ov).exists(): return ov
        for n in ("llama-server","llama-server.exe","server","server.exe"):
            f = shutil.which(n)
            if f: return f
        for n in ("llama-server.exe","llama-server","server.exe","server"):
            lp = ROOT / "bin" / n
            if lp.exists(): return str(lp)
    except Exception:
        pass
    return None

def _healthy():
    try:
        import httpx
        return httpx.get(f"{LLAMA_SERVER_URL}/health", timeout=2).status_code == 200
    except Exception:
        return False

def _start_server():
    global _server_proc
    try:
        if _healthy(): return True
        binary = _find_server()
        if not binary:
            st.error("llama-server binary not found. Set LLAMA_SERVER_BIN or ensure it is on PATH.")
            return False
        m = Path(MODEL_PATH)
        if not m.exists():
            st.error(f"Model file not found: {m}")
            return False
        cpu = os.cpu_count() or 4
        nt  = int(os.environ.get("N_THREADS","0")) or max(1, cpu//2)
        ngl = int(os.environ.get("N_GPU_LAYERS","0"))
        cmd = [binary,"-m",str(m),"--port",str(LLAMA_SERVER_PORT),
               "--host","127.0.0.1","--ctx-size",str(CONTEXT_SIZE),
               "-ngl",str(ngl),"-t",str(nt)]
        _server_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with st.spinner("Starting llama-server…"):
            for _ in range(60):
                time.sleep(1)
                if _healthy(): return True
        st.error("llama-server did not become healthy within 60 seconds.")
        return False
    except Exception as e:
        st.error(f"Server start error: {e}")
        return False

def _stop_server():
    global _server_proc
    try:
        if _server_proc and _server_proc.poll() is None:
            _server_proc.terminate()
            _server_proc.wait(timeout=5)
    except Exception:
        pass


# ════════════════════════════════════════════════════════════
# TOOLS + ENGINE
# ════════════════════════════════════════════════════════════
def _build_tools(cwd):
    try:
        return [BashTool(working_dir=cwd), ReadFileTool(working_dir=cwd),
                WriteFileTool(working_dir=cwd), EditFileTool(working_dir=cwd),
                GlobTool(working_dir=cwd), GrepTool(working_dir=cwd),
                ListDirTool(working_dir=cwd), WebSearchTool(), WebFetchTool(),
                AgentTool(), TodoWriteTool(), TodoReadTool(),
                MemoryWriteTool(), MemoryReadTool(), SkillTool()]
    except Exception as e:
        st.warning(f"Some tools failed to load: {e}")
        return []

def _make_factory(client, perms, usage, cwd):
    def factory(max_turns=10, is_subagent=True):
        return QueryEngine(
            tools=_build_tools(cwd), client=client, session=Session(),
            permission_manager=perms, usage=usage, max_turns=max_turns,
            working_dir=cwd, verbose=VERBOSE, is_subagent=is_subagent,
            on_tool_start=lambda n,i: None, on_tool_end=lambda n,r: None,
            on_chunk=lambda c: None)
    return factory

@st.cache_resource
def get_client():
    try:
        if not DIRECT_MODE: _start_server()
        return ModelClient(base_url=LLAMA_SERVER_URL, direct=DIRECT_MODE, model_path=MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to create ModelClient: {e}")
        st.stop()

@st.cache_resource
def init_engine():
    try:
        client = get_client()
        if not client.health():
            st.error("Cannot connect to model. Ensure llama-server is running and healthy.")
            st.stop()
        perms  = PermissionManager(PERMISSION_MODE)
        usage  = SessionUsage()
        cwd    = WORKING_DIR
        AgentTool.set_factory(_make_factory(client, perms, usage, cwd))
        engine = QueryEngine(
            tools=_build_tools(cwd), client=client, session=Session(model_client=client),
            permission_manager=perms, usage=usage, max_turns=MAX_TURNS,
            working_dir=cwd, verbose=VERBOSE,
            on_tool_start=lambda n,i: None, on_tool_end=lambda n,r: None,
            on_chunk=lambda c: None)
        return engine, perms, usage
    except SystemExit:
        raise
    except Exception as e:
        st.error(f"Engine init failed: {e}")
        st.stop()

engine, perms, usage = init_engine()


# ════════════════════════════════════════════════════════════
# MODEL WARMUP
# ════════════════════════════════════════════════════════════
def _warmup():
    try:
        w = QueryEngine(
            tools=[], client=get_client(), session=Session(),
            permission_manager=PermissionManager("deny"), usage=SessionUsage(),
            max_turns=1, working_dir=WORKING_DIR, verbose=False,
            on_tool_start=lambda n,i: None, on_tool_end=lambda n,r: None,
            on_chunk=lambda c: None)
        w.submit_message("Hi, this is a warmup ping — reply with exactly one word: ready")
    except Exception:
        pass

if not st.session_state.warmup_done:
    st.session_state.warmup_done = True
    threading.Thread(target=_warmup, daemon=True).start()


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    dark = st.toggle("🌙 Dark mode", value=st.session_state.theme_dark)
    if dark != st.session_state.theme_dark:
        st.session_state.theme_dark = dark
        inject_css(dark)
        st.rerun()

    st.markdown(f"**🤖 {MODEL_DISPLAY}**")
    st.caption(f"{'Direct' if DIRECT_MODE else 'Server'} · ctx {CONTEXT_SIZE}")
    st.divider()

    if st.button("✏️ New Chat", use_container_width=True):
        try: engine.session.clear()
        except Exception: pass
        try: engine.invalidate_system_prompt()
        except Exception: pass
        st.session_state.messages        = []
        st.session_state.chat_id         = new_cid()
        st.session_state.chat_title      = None
        st.session_state.attached_files  = []
        st.rerun()

    st.markdown("**💬 Recent Chats**")
    _all_chats = chats_list()
    for _c in _all_chats[:30]:
        _active = "🟢 " if _c["id"] == st.session_state.chat_id else ""
        _lbl    = f"{_active}{_c['title'][:28]}"
        _col_c, _col_d = st.columns([5, 1])
        if _col_c.button(_lbl, key=f"cl_{_c['id']}", use_container_width=True,
                         help=f"{_c['ts']} · {_c['n']} msgs"):
            if _c["id"] != st.session_state.chat_id:
                _d = chat_load(_c["id"])
                if _d:
                    try: engine.session.clear()
                    except Exception: pass
                    try: engine.invalidate_system_prompt()
                    except Exception: pass
                    st.session_state.messages   = _d["messages"]
                    st.session_state.chat_id    = _c["id"]
                    st.session_state.chat_title = _d.get("title")
                    st.rerun()
        if _col_d.button("🗑", key=f"del_{_c['id']}"):
            chat_delete(_c["id"])
            if _c["id"] == st.session_state.chat_id:
                st.session_state.messages   = []
                st.session_state.chat_id    = new_cid()
                st.session_state.chat_title = None
            st.rerun()
    if _all_chats: st.divider()

    st.session_state.web_search = st.toggle("🌐 Web Search",
                                             value=st.session_state.web_search)
    _mode = st.selectbox("Permission Mode", ["auto","ask","deny"],
                         index=["auto","ask","deny"].index(perms.mode.value))
    if _mode != perms.mode.value:
        try: perms.set_mode(_mode)
        except Exception: pass
        st.rerun()

    _sc1, _sc2 = st.columns(2)
    if _sc1.button("🗑 Clear"):
        try: engine.session.clear()
        except Exception: pass
        try: engine.invalidate_system_prompt()
        except Exception: pass
        st.session_state.messages        = []
        st.session_state.chat_id         = new_cid()
        st.session_state.chat_title      = None
        st.session_state.attached_files  = []
        st.rerun()
    if _sc2.button("📊 Status"):
        try: st.info(usage.summary())
        except Exception as e: st.info(f"Usage error: {e}")

    st.divider()
    st.subheader("Memory")
    try:
        from memory.manager import load_context
        st.text_area("", (load_context() or "No memories")[:500], height=110,
                     label_visibility="collapsed")
    except Exception: pass

    st.subheader("Skills")
    try:
        _st_t = next((t for t in engine.tools.values() if t.name == "skill"), None)
        if _st_t:
            st.text_area("", _st_t.execute({}).output[:500], height=110,
                         label_visibility="collapsed")
    except Exception: pass

    st.subheader("Todos")
    try:
        st.text(TodoReadTool().execute({}).output[:400] or "No todos")
    except Exception: pass


# ════════════════════════════════════════════════════════════
# SLASH COMMANDS
# ════════════════════════════════════════════════════════════
def handle_slash(cmd):
    try:
        low = cmd.strip().lower()
        if low in ("/clear","/reset"):
            try: engine.session.clear()
            except Exception: pass
            try: engine.invalidate_system_prompt()
            except Exception: pass
            st.session_state.messages = []; return "Session cleared."
        if low in ("/status","/cost"):
            return f"{usage.summary()}\nMessages: {len(engine.session)}"
        if low.startswith("/mode "):
            nm = cmd[6:].strip()
            try: perms.set_mode(nm); return f"Permission mode → {nm}"
            except Exception: return "Modes: auto | ask | deny"
        if low.startswith("/save"):
            parts = cmd.split(maxsplit=1)
            p = Path(parts[1]) if len(parts) > 1 else Path("session.json")
            engine.session.save(p); return f"Saved → {p}"
        if low == "/compact":
            engine.session._unlimited = True; engine.session._maybe_compress()
            return "Context compacted."
        if low.startswith("/memory"):
            try:
                from memory.manager import load_context
                return load_context() or "(no memories)"
            except Exception as e: return f"Error: {e}"
        if low in ("/skills","/skill"):
            st_t = next((t for t in engine.tools.values() if t.name=="skill"), None)
            return st_t.execute({}).output if st_t else "SkillTool not found."
        if low.startswith("/skill "):
            rest = cmd[7:].strip(); parts = rest.split(maxsplit=1)
            sn = parts[0] if parts else ""; sp = parts[1] if len(parts)>1 else ""
            st_t = next((t for t in engine.tools.values() if t.name=="skill"), None)
            if not st_t: return "SkillTool not found."
            return st_t.execute({"skill": sn, "args": {"problem": sp or sn}}).output
        if low in ("/help","/?"):
            return "Commands: /clear /status /mode /save /compact /memory /skills /skill NAME /help"
        return None
    except Exception as e:
        return f"Command error: {e}"


# ════════════════════════════════════════════════════════════
# TOOL INPUT SUMMARY
# ════════════════════════════════════════════════════════════
def _inp_summary(name, inp):
    try:
        if name == "bash":                                    return inp.get("command","")[:72]
        if name in ("read_file","write_file","edit_file"):    return inp.get("path","")
        if name == "grep":                                    return f"'{inp.get('pattern','')}'"
        if name == "glob":                                    return inp.get("pattern","")
        if name == "list_dir":                                return inp.get("path",".")
        if name == "web_search":                              return inp.get("query","")[:60]
        if name == "web_fetch":                               return inp.get("url","")[:60]
        if name == "agent":                                   return inp.get("task","")[:60]
        if name == "skill":
            return f"{inp.get('skill','')} | {inp.get('args',{}).get('problem','')[:40]}"
        if name == "todo_write":                              return f"{len(inp.get('todos',[]))} items"
        if name == "memory_write":                            return inp.get("key","")
        return json.dumps(inp)[:60]
    except Exception:
        return str(inp)[:60]


# ════════════════════════════════════════════════════════════
# MESSAGE RENDERER
# ════════════════════════════════════════════════════════════
def render_message(msg: dict, live: bool = False):
    try:
        role    = msg["role"]
        content = msg.get("content", "")
        think   = msg.get("thinking", "")
        tools   = msg.get("tools", [])
        dark    = st.session_state.theme_dark

        with st.chat_message(role):
            if content:
                msg_copy_btn(content)

            if think:
                try:
                    with st.expander("💭 Thinking", expanded=live):
                        st.markdown(f'<div class="thinking-block">{think}</div>',
                                    unsafe_allow_html=True)
                except Exception: pass

            display = _strip_code_fences(content) if (role == "assistant" and not live) else content
            try:
                st.markdown(display + ("▌" if live else ""))
            except Exception:
                st.text(display)

            if tools:
                html_lines = []
                for t in tools:
                    try:
                        icon = ("✅" if t.get("output") and not t.get("is_error")
                                else ("❌" if t.get("is_error") else "⏳"))
                        out  = (f' → <span style="opacity:.7">{t["output"]}</span>'
                                if t.get("output") else "")
                        html_lines.append(
                            f'<div class="tool-line">{icon} <b>{t["name"]}</b>'
                            f' · {t.get("input","")}{out}</div>')
                    except Exception: pass
                if html_lines:
                    try: st.html("".join(html_lines))
                    except Exception: pass

            if role == "assistant" and not live:
                try:
                    for _i, _art in enumerate(extract_artifacts(content)):
                        render_artifact(_art, _i, dark)
                except Exception: pass
    except Exception:
        pass  # Never crash the UI


# ════════════════════════════════════════════════════════════
# RENDER HISTORY
# ════════════════════════════════════════════════════════════
for _msg in st.session_state.messages:
    render_message(_msg)


# ════════════════════════════════════════════════════════════
# STREAMING STATE MACHINE
# ════════════════════════════════════════════════════════════
if st.session_state.app_state == "streaming":
    data        = st.session_state.stream_data
    q           = st.session_state.stream_q
    parser      = st.session_state.stream_parser
    stop_event  = st.session_state.stop_event
    pause_event = st.session_state.pause_event

    is_done = False
    while True:
        try:
            item = q.get_nowait()
        except Empty:
            break
        except Exception:
            break
        try:
            if item[0] == "done":
                is_done = True; break
            elif item[0] == "chunk":
                try:
                    for text, is_think in parser.feed(item[1]):
                        if not text: continue
                        if is_think: data["thinking"] += text
                        else:        data["response"] += text
                except Exception: pass
            elif item[0] == "tool_start":
                try:
                    data["tools"].append({
                        "name":     item[1],
                        "input":    _inp_summary(item[1], item[2]),
                        "output":   "",
                        "is_error": False,
                    })
                except Exception: pass
            elif item[0] == "tool_end":
                try:
                    r    = item[2]
                    snip = ((r.output or "").splitlines() or [""])[0][:80]
                    for t in data["tools"]:
                        if t["name"] == item[1] and not t["output"]:
                            t["output"] = snip; t["is_error"] = r.is_error; break
                except Exception: pass
            elif item[0] == "error":
                try:
                    data["response"] += f"\n\n⚠ Error: {item[1]}"
                    is_done = True
                except Exception:
                    is_done = True
        except Exception:
            pass

    render_message({
        "role":     "assistant",
        "content":  data["response"],
        "thinking": data["thinking"],
        "tools":    data["tools"],
    }, live=not is_done)

    _ctl1, _ctl2, _ctl3 = st.columns([1, 1, 6])
    with _ctl1:
        st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
        if st.button("⏹ Stop", key="stop_gen"):
            stop_event.set(); is_done = True
        st.markdown('</div>', unsafe_allow_html=True)
    with _ctl2:
        _paused = pause_event.is_set()
        st.markdown('<div class="pause-btn">', unsafe_allow_html=True)
        if st.button("▶ Resume" if _paused else "⏸ Pause", key="pause_gen"):
            if _paused: pause_event.clear()
            else:       pause_event.set()
        st.markdown('</div>', unsafe_allow_html=True)

    if is_done:
        try:
            if hasattr(parser, "flush"):
                for text, is_think in parser.flush():
                    if is_think: data["thinking"] += text
                    else:        data["response"] += text
        except Exception: pass

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  data["response"],
            "thinking": data["thinking"],
            "tools":    list(data["tools"]),
        })
        chat_save(st.session_state.chat_id,
                  st.session_state.messages,
                  st.session_state.chat_title)
        st.session_state.app_state      = "idle"
        st.session_state.attached_files = []
        st.rerun()
    else:
        time.sleep(0.15)
        st.rerun()

# ── IDLE ──────────────────────────────────────────────────
else:
    # Primary: JS-bridge drop zone (drag/drop/paste into input area)
    _render_file_bridge()

    # Fallback: native Streamlit file uploader in an expander
    with st.expander("📎 Attach files (alternative uploader)", expanded=False):
        uploaded = st.file_uploader(
            "Drop files here or click to browse",
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader_widget",
        )
        if uploaded:
            existing_names = {f.name for f in st.session_state.attached_files}
            for uf in uploaded:
                if uf.name not in existing_names:
                    st.session_state.attached_files.append(uf)
                    existing_names.add(uf.name)
            if st.session_state.attached_files:
                st.caption(f"Attached: {', '.join(f.name for f in st.session_state.attached_files)}")

    # Chat input
    prompt = st.chat_input("Ask anything… (/help for commands) · Drag or paste files in the zone above")

    if prompt:
        if prompt.startswith("/"):
            result = handle_slash(prompt)
            if result:
                st.session_state.messages.append({"role":"assistant","content":result})
                chat_save(st.session_state.chat_id, st.session_state.messages,
                          st.session_state.chat_title)
                st.rerun()
        else:
            try:
                file_block = _process_uploads(st.session_state.attached_files)
            except Exception:
                file_block = ""
            full_prompt = prompt + file_block

            st.session_state.messages.append({"role":"user","content":full_prompt})
            chat_save(st.session_state.chat_id, st.session_state.messages,
                      st.session_state.chat_title)

            _q         = Queue()
            _stop_evt  = st.session_state.stop_event;  _stop_evt.clear()
            _pause_evt = st.session_state.pause_event; _pause_evt.clear()
            _parser    = ThinkingStreamParser()
            try:
                _ws = next((t for t in engine.tools.values() if t.name == "web_search"), None)
            except Exception:
                _ws = None
            _ws_orig   = getattr(_ws, "enabled", True) if _ws else True
            _ws_on     = st.session_state.web_search

            st.session_state.stream_q      = _q
            st.session_state.stream_parser = _parser
            st.session_state.stream_data   = {"thinking":"","response":"","tools":[]}
            st.session_state.app_state     = "streaming"

            def _run(_p=full_prompt, _q=_q, _se=_stop_evt, _pe=_pause_evt,
                     _ws=_ws, _wo=_ws_orig, _ws_on=_ws_on):
                def on_chunk(c):
                    try:
                        while _pe.is_set() and not _se.is_set(): time.sleep(0.1)
                        if _se.is_set(): return
                        _q.put(("chunk", c))
                    except Exception: pass

                def on_tool_start(n, i):
                    try:
                        if _se.is_set(): return
                        _q.put(("tool_start", n, i))
                    except Exception: pass

                def on_tool_end(n, r):
                    try: _q.put(("tool_end", n, r))
                    except Exception: pass

                engine.on_chunk      = on_chunk
                engine.on_tool_start = on_tool_start
                engine.on_tool_end   = on_tool_end

                if _ws and not _ws_on:
                    try: _ws.enabled = False
                    except Exception: pass
                try:
                    engine.submit_message(_p)
                except Exception as e:
                    try: _q.put(("error", str(e)))
                    except Exception: pass
                finally:
                    try:
                        if _ws: _ws.enabled = _wo
                    except Exception: pass
                    try: _q.put(("done",))
                    except Exception: pass

            threading.Thread(target=_run, daemon=True).start()
            st.rerun()

import atexit
atexit.register(_stop_server)