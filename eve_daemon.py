# -*- coding: utf-8 -*-
"""
eve_daemon.py  —  Drop this in your Mind_EVE folder alongside main.py.

WHAT THIS DOES
==============
Runs silently in the background 24/7. When you message EVE via WhatsApp
or email, she wakes up, thinks, and replies — no terminal needed.

    You text/email EVE → daemon receives → EVE engine processes → reply sent back

HOW TO RUN
==========
  python eve_daemon.py                     # foreground (shows log)
  pythonw eve_daemon.py                    # background, no window (Windows)
  python install_daemon.py                 # register as Windows startup task

CHANNELS
========
  WhatsApp : uses the existing wa_bridge.js (auto-started if not running)
  Email    : IMAP IDLE + SMTP (add EMAIL_* keys to your .env)

EMAIL SETUP (.env additions)
=============================
  EMAIL_IMAP_HOST=imap.gmail.com       # or imap-mail.outlook.com
  EMAIL_IMAP_PORT=993
  EMAIL_SMTP_HOST=smtp.gmail.com       # or smtp-mail.outlook.com
  EMAIL_SMTP_PORT=587
  EMAIL_ADDRESS=eve@gmail.com          # EVE's inbox
  EMAIL_PASSWORD=xxxx-xxxx-xxxx-xxxx   # Gmail app-password (not main password)
  EMAIL_OWNER=you@gmail.com            # only reply to THIS address
  EMAIL_POLL_S=30                      # how often to check (seconds), default 30
"""
from __future__ import annotations

import logging
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import urllib.request as _urllib
import json as _json
from pathlib import Path
from typing import Optional

# ── logging ────────────────────────────────────────────────────────────────────
_LOG_FILE = Path(__file__).parent / "logs" / "daemon.log"
_LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler(_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("eve_daemon")

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── load .env BEFORE importing settings ────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
except ImportError:
    pass

# ── constants ──────────────────────────────────────────────────────────────────
WA_PORT      = 5005
WA_URL       = f"http://127.0.0.1:{WA_PORT}"
POLL_TIMEOUT = 28          # seconds — slightly under wa_bridge.js 30s timeout
OWNER_WA     = os.environ.get("OWNER_WA_NUMBER", "")  # e.g. "447447148024"
EMAIL_POLL_S = int(os.environ.get("EMAIL_POLL_S", "30"))


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE  — initialise once, reuse for all channels
# ══════════════════════════════════════════════════════════════════════════════

def _build_engine():
    """
    Build a full EVE QueryEngine (same as main.py does) but headless —
    no spinner, no REPL, no warmup printout.
    Uses the cloud API from .env so no local llama-server needed.
    """
    log.info("Initialising EVE engine…")
    try:
        # Import the same pieces main.py uses
        from config.settings import (
            API_URL, API_KEY, CLOUD_MODEL, MAX_TOKENS, CONTEXT_SIZE,
            TEMPERATURE, TOP_K, TOP_P, MIN_P, REPEAT_PENALTY, MAX_TURNS,
            PERMISSION_MODE, WORKING_DIR, VERBOSE,
        )
    except Exception:
        # Fallback: read from env directly
        API_URL   = os.environ.get("API_URL", "")
        API_KEY   = os.environ.get("API_KEY", "")
        CLOUD_MODEL    = os.environ.get("CLOUD_MODEL", "gpt-4o-mini")
        MAX_TOKENS= int(os.environ.get("MAX_TOKENS", "1024"))
        CONTEXT_SIZE = int(os.environ.get("CONTEXT_SIZE", "8192"))
        TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
        TOP_K = int(os.environ.get("TOP_K", "40"))
        TOP_P = float(os.environ.get("TOP_P", "0.95"))
        MIN_P = 0.0
        REPEAT_PENALTY = 1.0
        MAX_TURNS = int(os.environ.get("MAX_TURNS", "50"))
        PERMISSION_MODE = "auto"
        WORKING_DIR = str(ROOT)
        VERBOSE = False

    cwd = Path(WORKING_DIR) if WORKING_DIR else ROOT

    # ── LLM client ────────────────────────────────────────────────────────────
    # Try cloud client first (OpenAI-compatible, used for Gemini etc.)
    try:
        from agent.llm_client import LLMClient  # adjust import path if different
        client = LLMClient(
            api_url=API_URL, api_key=API_KEY, model=CLOUD_MODEL,
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
        )
        log.info("LLM client: cloud (%s)", CLOUD_MODEL)
    except Exception:
        try:
            # Fallback: try whatever client main.py uses
            from agent.query_engine import _make_client  # type: ignore
            client = _make_client()
            log.info("LLM client: default")
        except Exception as e:
            log.error("Could not build LLM client: %s", e)
            raise

    # ── tools (minimal set suitable for conversational use) ──────────────────
    try:
        from tools.bash_tool      import BashTool
        from tools.file_tools     import ReadFileTool, WriteFileTool, EditFileTool
        from tools.search_tools   import GlobTool, GrepTool, ListDirTool
        from tools.web_tools      import WebSearchTool, WebFetchTool
        from tools.memory_tools   import MemoryWriteTool, MemoryReadTool, MemorySearchTool
        from tools.journal_tool   import JournalTool
        from tools.task_tools     import (TaskCreateTool, TaskGetTool,
                                          TaskListTool, TaskUpdateTool)
        from tools.todo_tools     import TodoWriteTool, TodoReadTool
        from tools.whatsapp_tool  import WhatsAppSendTool
        from tools.user_tools     import AskUserTool, SleepTool

        tools = [
            BashTool(working_dir=str(cwd)),
            ReadFileTool(working_dir=str(cwd)),
            WriteFileTool(working_dir=str(cwd)),
            WebSearchTool(), WebFetchTool(),
            MemoryWriteTool(), MemoryReadTool(),
            MemorySearchTool(working_dir=str(cwd)),
            JournalTool(),
            TaskCreateTool(), TaskGetTool(),
            TaskListTool(), TaskUpdateTool(),
            TodoWriteTool(), TodoReadTool(),
            WhatsAppSendTool(),
            AskUserTool(),
        ]
    except Exception as e:
        log.warning("Could not load full tool set (%s) — using minimal tools", e)
        tools = []

    # ── engine ────────────────────────────────────────────────────────────────
    from agent.query_engine import QueryEngine
    from agent.session      import Session
    from utils.permissions  import PermissionManager

    perms = PermissionManager(PERMISSION_MODE)

    engine = QueryEngine(
        tools=tools,
        client=client,
        session=Session.resume_or_create(model_client=client),
        permission_manager=perms,
        max_turns=MAX_TURNS,
        working_dir=str(cwd),
        verbose=VERBOSE,
    )
    log.info("EVE engine ready ✓")
    return engine


# ══════════════════════════════════════════════════════════════════════════════
#  wa_bridge.js  —  start/keep-alive
# ══════════════════════════════════════════════════════════════════════════════

_wa_proc: Optional[subprocess.Popen] = None


def _wa_running() -> bool:
    """Check if wa_bridge.js HTTP API is responding."""
    try:
        with _urllib.urlopen(f"{WA_URL}/status", timeout=3) as r:
            return _json.loads(r.read()).get("ready", False)
    except Exception:
        return False


def _start_wa_bridge():
    """Start wa_bridge.js as a background subprocess."""
    global _wa_proc
    bridge = ROOT / "wa_bridge.js"
    if not bridge.exists():
        log.error("wa_bridge.js not found — WhatsApp disabled")
        return

    # Kill stale process if port is already taken but not answering
    try:
        import socket
        s = socket.socket()
        s.settimeout(1)
        if s.connect_ex(("127.0.0.1", WA_PORT)) == 0:
            s.close()
            log.info("wa_bridge.js already running on port %d", WA_PORT)
            return
        s.close()
    except Exception:
        pass

    node_cmd = "node.exe" if sys.platform == "win32" else "node"
    try:
        log.info("Starting wa_bridge.js…")
        log_path = ROOT / "logs" / "wa_bridge.log"
        with open(log_path, "a") as lf:
            _wa_proc = subprocess.Popen(
                [node_cmd, str(bridge)],
                cwd=str(ROOT),
                stdout=lf,
                stderr=lf,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if sys.platform == "win32" else 0
                ),
            )
        # Wait up to 12s for it to be ready
        for _ in range(24):
            time.sleep(0.5)
            if _wa_running():
                log.info("wa_bridge.js ready ✓")
                return
        log.warning("wa_bridge.js started but not yet responding")
    except FileNotFoundError:
        log.error("Node.js not found — install from https://nodejs.org")
    except Exception as e:
        log.error("Could not start wa_bridge.js: %s", e)


def _ensure_wa_bridge():
    """Watchdog: restart bridge if it died."""
    global _wa_proc
    if _wa_proc and _wa_proc.poll() is not None:
        log.warning("wa_bridge.js crashed (rc=%d) — restarting", _wa_proc.returncode)
        _wa_proc = None
    if not _wa_proc and not _wa_running():
        _start_wa_bridge()


# ══════════════════════════════════════════════════════════════════════════════
#  WhatsApp  listener
# ══════════════════════════════════════════════════════════════════════════════

def _wa_send(to: str, text: str):
    """Send a WhatsApp message via wa_bridge.js /send."""
    try:
        payload = _json.dumps({"to": to, "text": text}).encode()
        req = _urllib.Request(
            f"{WA_URL}/send",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib.urlopen(req, timeout=15) as r:
            result = _json.loads(r.read())
            if not result.get("ok"):
                log.warning("wa /send returned: %s", result)
    except Exception as e:
        log.error("WhatsApp send failed: %s", e)


def _wa_recv_once() -> Optional[dict]:
    """
    Long-poll /recv — blocks up to POLL_TIMEOUT seconds.
    Returns {"from": "...", "text": "..."} or None on timeout/error.
    """
    try:
        req = _urllib.Request(
            f"{WA_URL}/recv",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _urllib.urlopen(req, timeout=POLL_TIMEOUT + 5) as r:
            data = _json.loads(r.read())
            if data.get("text"):
                return data
    except Exception:
        pass
    return None


def _wa_listener(engine, inbox: queue.Queue):
    """Background thread: long-poll WhatsApp, push to inbox."""
    log.info("WhatsApp listener started")
    while True:
        try:
            _ensure_wa_bridge()
            if not _wa_running():
                time.sleep(5)
                continue
            msg = _wa_recv_once()
            if msg:
                sender = msg.get("from", "")
                text   = msg.get("text", "").strip()
                if text:
                    # If OWNER_WA is set, only accept messages from that number
                    clean_sender = sender.replace("+", "").replace(" ", "")
                    clean_owner  = OWNER_WA.replace("+", "").replace(" ", "")
                    if clean_owner and clean_sender != clean_owner:
                        log.info("WhatsApp: ignoring message from %s (not owner)", sender)
                        continue
                    log.info("WhatsApp ← %s: %s", sender, text[:80])
                    inbox.put({"channel": "whatsapp", "from": sender, "text": text})
        except Exception as e:
            log.error("WhatsApp listener error: %s", e)
            time.sleep(3)


# ══════════════════════════════════════════════════════════════════════════════
#  Email  listener
# ══════════════════════════════════════════════════════════════════════════════

def _email_configured() -> bool:
    return bool(
        os.environ.get("EMAIL_ADDRESS") and
        os.environ.get("EMAIL_PASSWORD") and
        os.environ.get("EMAIL_IMAP_HOST")
    )


def _email_send(to: str, subject: str, body: str, reply_to_msg_id: str = ""):
    """Send an email reply via SMTP."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    addr     = os.environ.get("EMAIL_ADDRESS", "")
    password = os.environ.get("EMAIL_PASSWORD", "")
    host     = os.environ.get("EMAIL_SMTP_HOST", "smtp.gmail.com")
    port     = int(os.environ.get("EMAIL_SMTP_PORT", "587"))

    msg = MIMEMultipart()
    msg["From"]    = addr
    msg["To"]      = to
    msg["Subject"] = subject if subject.startswith("Re:") else f"Re: {subject}"
    if reply_to_msg_id:
        msg["In-Reply-To"] = reply_to_msg_id
        msg["References"]  = reply_to_msg_id
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(addr, password)
            smtp.send_message(msg)
        log.info("Email sent to %s: %s", to, subject[:60])
    except Exception as e:
        log.error("Email send failed: %s", e)


def _email_listener(inbox: queue.Queue):
    """
    Background thread: polls IMAP inbox every EMAIL_POLL_S seconds.
    Pushes unseen messages from the owner to the inbox queue.
    """
    import imaplib
    import email as _email_lib
    from email.header import decode_header

    if not _email_configured():
        log.info("Email bridge disabled (EMAIL_* not configured in .env)")
        return

    addr     = os.environ.get("EMAIL_ADDRESS", "")
    password = os.environ.get("EMAIL_PASSWORD", "")
    host     = os.environ.get("EMAIL_IMAP_HOST", "imap.gmail.com")
    port     = int(os.environ.get("EMAIL_IMAP_PORT", "993"))
    owner    = os.environ.get("EMAIL_OWNER", "")

    def _decode_header_str(h) -> str:
        parts = decode_header(h or "")
        out = []
        for part, enc in parts:
            if isinstance(part, bytes):
                out.append(part.decode(enc or "utf-8", errors="replace"))
            else:
                out.append(str(part))
        return " ".join(out)

    def _get_body(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                cd = str(part.get("Content-Disposition", ""))
                if ct == "text/plain" and "attachment" not in cd:
                    try:
                        return part.get_payload(decode=True).decode(
                            part.get_content_charset() or "utf-8", errors="replace"
                        )
                    except Exception:
                        pass
        else:
            try:
                return msg.get_payload(decode=True).decode(
                    msg.get_content_charset() or "utf-8", errors="replace"
                )
            except Exception:
                pass
        return ""

    log.info("Email listener started (polling every %ds)", EMAIL_POLL_S)
    seen_uids: set = set()

    while True:
        try:
            with imaplib.IMAP4_SSL(host, port) as imap:
                imap.login(addr, password)
                imap.select("INBOX")

                # Search for UNSEEN messages from the owner
                search_criteria = "(UNSEEN)"
                if owner:
                    search_criteria = f'(UNSEEN FROM "{owner}")'

                _, data = imap.search(None, search_criteria)
                uids = data[0].split() if data[0] else []

                for uid in uids:
                    uid_str = uid.decode()
                    if uid_str in seen_uids:
                        continue

                    _, msg_data = imap.fetch(uid, "(RFC822)")
                    raw = msg_data[0][1] if msg_data and msg_data[0] else None
                    if not raw:
                        continue

                    msg = _email_lib.message_from_bytes(raw)
                    sender  = msg.get("From", "")
                    subject = _decode_header_str(msg.get("Subject", "(no subject)"))
                    msg_id  = msg.get("Message-ID", "")
                    body    = _get_body(msg).strip()

                    if not body:
                        seen_uids.add(uid_str)
                        continue

                    # Mark as read
                    imap.store(uid, "+FLAGS", "\\Seen")
                    seen_uids.add(uid_str)

                    log.info("Email ← %s | %s", sender[:50], subject[:60])
                    inbox.put({
                        "channel":    "email",
                        "from":       sender,
                        "subject":    subject,
                        "msg_id":     msg_id,
                        "text":       f"[Email subject: {subject}]\n\n{body}",
                        "reply_addr": sender,
                    })

        except imaplib.IMAP4.abort as e:
            log.warning("IMAP connection aborted: %s — reconnecting", e)
        except Exception as e:
            log.error("Email listener error: %s", e)

        time.sleep(EMAIL_POLL_S)


# ══════════════════════════════════════════════════════════════════════════════
#  Message worker  — routes inbox → EVE engine → reply
# ══════════════════════════════════════════════════════════════════════════════

def _worker(engine, inbox: queue.Queue):
    """
    Single-threaded worker so EVE processes one message at a time
    (preserves conversation context / prevents race conditions).
    """
    log.info("Message worker ready")
    while True:
        try:
            item = inbox.get(timeout=2)
        except queue.Empty:
            continue

        channel = item.get("channel")
        sender  = item.get("from", "unknown")
        text    = item.get("text", "")

        if not text:
            continue

        # Build a context prefix so EVE knows which channel it's on
        if channel == "whatsapp":
            context = f"[Message from {sender} via WhatsApp]\n{text}"
        elif channel == "email":
            context = text  # already has subject header
        else:
            context = text

        log.info("Processing message from %s (%s): %s", sender, channel, text[:80])

        try:
            reply = engine.submit_message(context)
        except Exception as e:
            log.error("Engine error: %s\n%s", e, traceback.format_exc())
            reply = f"⚠️ Sorry, I ran into an error: {e}"

        if not reply or not reply.strip():
            reply = "✅ Done."

        # Route reply back to the right channel
        if channel == "whatsapp":
            wa_number = sender.replace("+", "").replace(" ", "")
            _wa_send(wa_number, reply)
            log.info("WhatsApp → %s: %s", sender, reply[:80])

        elif channel == "email":
            _email_send(
                to=item.get("reply_addr", sender),
                subject=item.get("subject", "EVE reply"),
                body=reply,
                reply_to_msg_id=item.get("msg_id", ""),
            )
            log.info("Email → %s: %s", sender, reply[:80])


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("EVE Daemon starting")
    log.info("=" * 60)

    # ── Start WhatsApp bridge ──────────────────────────────────────────────
    _start_wa_bridge()

    # ── Build engine ──────────────────────────────────────────────────────
    try:
        engine = _build_engine()
    except Exception as e:
        log.critical("Engine init failed: %s\n%s", e, traceback.format_exc())
        log.critical("Daemon cannot start without a working LLM client.")
        log.critical("Check API_URL and API_KEY in your .env file.")
        sys.exit(1)

    # ── Shared inbox ──────────────────────────────────────────────────────
    inbox: queue.Queue = queue.Queue()

    # ── Start listeners ───────────────────────────────────────────────────
    threading.Thread(
        target=_wa_listener, args=(engine, inbox),
        daemon=True, name="wa-listener"
    ).start()

    threading.Thread(
        target=_email_listener, args=(inbox,),
        daemon=True, name="email-listener"
    ).start()

    # ── Worker (main thread becomes the worker) ────────────────────────────
    log.info("Daemon running. Listening on WhatsApp + Email.")
    log.info("Send a WhatsApp message or email to start chatting with EVE.")

    try:
        _worker(engine, inbox)
    except KeyboardInterrupt:
        log.info("Daemon stopped by user.")


if __name__ == "__main__":
    main()