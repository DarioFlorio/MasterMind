# -*- coding: utf-8 -*-
"""
whatsapp_bridge.py  —  Drop this file into your Mind_EVE folder.

HOW IT WORKS
============
1. Starts a tiny Flask server (port 5005) as a background thread.
2. Twilio sends your WhatsApp messages to that server.
3. The bridge feeds them into EVE's engine exactly like you typed them.
4. EVE's reply is sent back to your WhatsApp via Twilio.
5. If you type "let's chat on whatsapp" (or similar) in the terminal,
   EVE's replies also start mirroring to WhatsApp.

SETUP (one-time, ~10 minutes)
==============================
1. Free account at https://www.twilio.com
2. Go to Messaging > Try it out > Send a WhatsApp message
3. Follow the sandbox instructions (send a join code from your phone)
4. Add to your .env:
       TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
       TWILIO_WHATSAPP_FROM=whatsapp:+14155238886   ← the sandbox number
       WHATSAPP_TO=whatsapp:+YOUR_NUMBER            ← your personal number
5. Download cloudflared (free, no account):
       https://github.com/cloudflare/cloudflared/releases/latest
   Drop cloudflared.exe next to main.py.
6. Run main.py once. It will print a Cloudflare URL like:
       [whatsapp] Tunnel: https://xxxx.trycloudflare.com
7. Paste that URL + /whatsapp into Twilio sandbox webhook field:
       https://xxxx.trycloudflare.com/whatsapp
   (Webhook method: HTTP POST)

After that: just run main.py. Text EVE's WhatsApp number any time.
The Cloudflare URL changes each restart — re-paste it in Twilio once.
For a permanent URL, use a free Cloudflare account and a named tunnel.
"""
from __future__ import annotations
import logging
import os
import queue
import re
import subprocess
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.query_engine import QueryEngine

log = logging.getLogger("whatsapp_bridge")

# ── state ──────────────────────────────────────────────────────────────────────
_engine: "QueryEngine | None" = None
_active   = False          # True  → mirror all terminal replies to WA
_wa_queue: queue.Queue = queue.Queue()   # inbound WA messages
_to_number = ""            # e.g. whatsapp:+393XXXXXXXXX
_from_number = ""          # Twilio sandbox number

# phrases that activate whatsapp mode from the terminal
_ACTIVATE_PHRASES = re.compile(
    r"(let'?s?\s+chat\s+on\s+whatsapp"
    r"|switch\s+to\s+whatsapp"
    r"|reply\s+(on|via|through)\s+whatsapp"
    r"|whatsapp\s+mode"
    r"|talk\s+on\s+whatsapp)",
    re.IGNORECASE,
)


def init(engine: "QueryEngine") -> None:
    """Call this once from main.py after the engine is built."""
    global _engine
    _engine = engine
    _load_env()
    if not _from_number:
        log.warning("[whatsapp] TWILIO_WHATSAPP_FROM not set — bridge disabled.")
        return
    _start_flask()
    _start_tunnel()
    _start_inbound_worker()
    log.info("[whatsapp] Bridge ready. Text %s to reach EVE.", _from_number)
    print(f"\033[36m[whatsapp] Bridge ready — text {_from_number} on WhatsApp\033[0m")


def notify_user_message(text: str) -> None:
    """Call this with every message the USER types in the terminal."""
    global _active
    if _ACTIVATE_PHRASES.search(text):
        _active = True
        _send_whatsapp("✅ WhatsApp mode on. I'll reply here from now on.")
        print("\033[36m[whatsapp] Mode activated — replies will mirror to WhatsApp.\033[0m")


def notify_eve_reply(reply: str) -> None:
    """Call this with every reply EVE produces."""
    if _active and reply and reply.strip():
        _send_whatsapp(reply)


# ── Twilio sender ──────────────────────────────────────────────────────────────
def _send_whatsapp(text: str) -> None:
    if not _to_number or not _from_number:
        return
    try:
        from twilio.rest import Client
        sid   = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        if not sid or not token:
            return
        client = Client(sid, token)
        # WhatsApp messages max 1600 chars — chunk if needed
        chunks = [text[i:i+1500] for i in range(0, len(text), 1500)]
        for chunk in chunks:
            client.messages.create(body=chunk, from_=_from_number, to=_to_number)
    except Exception as e:
        log.error("[whatsapp] Send failed: %s", e)


# ── Flask webhook (receives messages from Twilio) ─────────────────────────────
def _start_flask() -> None:
    def run():
        try:
            from flask import Flask, request, Response
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable, "-m", "pip", "install", "flask",
                            "--quiet", "--break-system-packages"], check=False)
            from flask import Flask, request, Response

        app = Flask(__name__)
        log_flask = logging.getLogger("werkzeug")
        log_flask.setLevel(logging.ERROR)   # silence request logs

        @app.route("/whatsapp", methods=["POST"])
        def whatsapp_webhook():
            global _active, _to_number
            sender = request.form.get("From", "")
            body   = request.form.get("Body", "").strip()
            if not body:
                return Response("", 204)

            # Remember who to reply to
            _to_number = sender

            # Activate WhatsApp mode
            _active = True

            log.info("[whatsapp] ← %s: %s", sender, body[:80])
            _wa_queue.put(body)
            return Response("", 204)   # Twilio doesn't need a TwiML reply here

        @app.route("/health")
        def health():
            return {"status": "ok", "whatsapp_active": _active}

        app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False)

    t = threading.Thread(target=run, daemon=True, name="whatsapp-flask")
    t.start()
    time.sleep(1.0)   # let Flask bind


# ── Cloudflare tunnel ─────────────────────────────────────────────────────────
def _start_tunnel() -> None:
    # Look for cloudflared next to main.py or on PATH
    candidates = [
        "cloudflared",
        "cloudflared.exe",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared"),
    ]
    exe = next((c for c in candidates if _which(c)), None)
    if not exe:
        print("\033[33m[whatsapp] cloudflared not found — skipping tunnel.\033[0m")
        print("\033[33m          Download from: https://github.com/cloudflare/cloudflared/releases\033[0m")
        print("\033[33m          Drop cloudflared.exe next to main.py, then restart.\033[0m")
        return

    def run():
        try:
            proc = subprocess.Popen(
                [exe, "tunnel", "--url", "http://localhost:5005"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True,
            )
            url_pattern = re.compile(r"https://[a-z0-9\-]+\.trycloudflare\.com")
            for line in proc.stdout:
                m = url_pattern.search(line)
                if m:
                    tunnel_url = m.group(0)
                    webhook = f"{tunnel_url}/whatsapp"
                    print(f"\033[36m[whatsapp] Tunnel: {tunnel_url}\033[0m")
                    print(f"\033[36m[whatsapp] ➜ Paste into Twilio sandbox webhook: {webhook}\033[0m")
                    break
            proc.wait()
        except Exception as e:
            log.error("[whatsapp] Tunnel error: %s", e)

    t = threading.Thread(target=run, daemon=True, name="cloudflared")
    t.start()


def _which(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None or os.path.isfile(cmd)


# ── Inbound worker (WA message → EVE engine) ─────────────────────────────────
def _start_inbound_worker() -> None:
    def run():
        while True:
            try:
                msg = _wa_queue.get(timeout=1)
            except queue.Empty:
                continue
            if _engine is None:
                continue
            try:
                log.info("[whatsapp] → EVE: %s", msg[:80])
                reply = _engine.submit_message(msg)
                if reply and reply.strip():
                    _send_whatsapp(reply)
            except Exception as e:
                log.error("[whatsapp] Engine error: %s", e)
                _send_whatsapp(f"⚠️ Error: {e}")

    t = threading.Thread(target=run, daemon=True, name="whatsapp-worker")
    t.start()


# ── .env loader ───────────────────────────────────────────────────────────────
def _load_env() -> None:
    global _from_number, _to_number
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass
    _from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "")
    _to_number   = os.environ.get("WHATSAPP_TO", "")
