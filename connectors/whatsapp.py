"""
connectors/whatsapp.py — WhatsApp bridge via Baileys (Node.js).

Manages the wa_bridge.js subprocess and the EVE ↔ WhatsApp message loop.
Self-registers into the global ConnectorRegistry on import.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request as _urllib
from pathlib import Path
from typing import Dict, Optional

from connectors import ConnectorBase, registry

ROOT     = Path(__file__).parent.parent
_WA_PORT = int(os.environ.get("WA_BRIDGE_PORT", 5005))
_SESSION = ROOT / "wa_session"


class WhatsAppConnector(ConnectorBase):
    id    = "whatsapp"
    label = "WhatsApp"
    icon  = "📱"

    def __init__(self):
        super().__init__()
        self._proc:      Optional[subprocess.Popen] = None
        self._owner_jid: str  = os.environ.get("OWNER_WA_NUMBER", "")
        self._inbound:   queue.Queue = queue.Queue()
        self._started:   bool = False
        self._connected: bool = False
        self._threads_up: bool = False

    # ── ConnectorBase interface ───────────────────────────────────────────────

    def configure(self, cfg: Dict) -> None:
        owner = cfg.get("owner", "").strip()
        if owner:
            self._owner_jid = owner
            try:
                from tools.whatsapp_tool import WhatsAppSendTool
                WhatsAppSendTool.set_owner(owner)
            except Exception:
                pass

    def start(self) -> None:
        if self._started:
            return
        if not shutil.which("node") and not shutil.which("node.exe"):
            print("[WA] Node.js not found — WhatsApp disabled", flush=True)
            return
        self._ensure_npm_deps()
        self._start_bridge_proc()
        self._sync_owner()
        if not self._threads_up:
            self._start_threads()
            self._threads_up = True
        self._started = True

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass
        self._started   = False
        self._connected = False

    def on_eve_reply(self, reply: str) -> None:
        if self._enabled and reply and reply.strip():
            self._send(reply)

    @property
    def connected(self) -> bool:
        return self._connected

    def status(self) -> Dict:
        return {
            "id":        self.id,
            "label":     self.label,
            "icon":      self.icon,
            "enabled":   self._enabled,
            "connected": self._connected,
            "started":   self._started,
            "owner":     self._owner_jid,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _ensure_npm_deps(self):
        baileys = ROOT / "node_modules" / "@whiskeysockets" / "baileys"
        if baileys.exists():
            return
        pkg = ROOT / "package.json"
        if not pkg.exists():
            pkg.write_text(json.dumps({
                "name": "eve-wa-bridge", "version": "1.0.0", "private": True,
                "dependencies": {
                    "@whiskeysockets/baileys": "^6.7.0",
                    "@hapi/boom": "^10.0.1",
                    "qrcode-terminal": "^0.12.0",
                    "pino": "^8.0.0",
                },
            }, indent=2))
        npm = shutil.which("npm") or shutil.which("npm.cmd")
        if npm:
            kw = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
            subprocess.run([npm, "install", "--prefer-offline"],
                           cwd=str(ROOT), capture_output=True, **kw)

    def _kill_stale(self):
        try:
            if sys.platform == "win32":
                out = subprocess.check_output(["netstat", "-ano"], text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                for line in out.splitlines():
                    if f":{_WA_PORT}" in line and "LISTENING" in line:
                        pid = line.split()[-1]
                        if pid.isdigit():
                            subprocess.run(["taskkill", "/F", "/PID", pid],
                                capture_output=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                try:
                    out = subprocess.check_output(
                        ["lsof", "-ti", f"tcp:{_WA_PORT}"], text=True).strip()
                    for pid in out.splitlines():
                        if pid.isdigit():
                            subprocess.run(["kill", "-9", pid], capture_output=True)
                except FileNotFoundError:
                    subprocess.run(["fuser", "-k", f"{_WA_PORT}/tcp"], capture_output=True)
        except Exception:
            pass

    def _start_bridge_proc(self):
        bridge_js = ROOT / "wa_bridge.js"
        if not bridge_js.exists():
            print("[WA] wa_bridge.js not found", flush=True)
            return
        node = shutil.which("node") or shutil.which("node.exe")
        if not node:
            return
        self._kill_stale()
        time.sleep(0.4)
        env = {**os.environ}
        kw  = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        self._proc = subprocess.Popen(
            [node, str(bridge_js)], cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env, **kw,
        )

        def _pipe():
            for line in self._proc.stdout:
                if line.strip():
                    print(f"[WA] {line.rstrip()}", flush=True)
        threading.Thread(target=_pipe, daemon=True, name="wa-pipe").start()
        time.sleep(1.5)

    def _sync_owner(self):
        for _ in range(30):
            try:
                with _urllib.urlopen(
                    f"http://127.0.0.1:{_WA_PORT}/status", timeout=2
                ) as r:
                    data = json.loads(r.read())
                    self._connected = data.get("ready", False)
                    owner = data.get("owner", "")
                    if owner:
                        self._owner_jid = owner
                        try:
                            from tools.whatsapp_tool import WhatsAppSendTool
                            WhatsAppSendTool.set_owner(owner)
                        except Exception:
                            pass
                        return
            except Exception:
                pass
            time.sleep(0.5)

    def _start_threads(self):
        # Poll inbound messages
        def _poll():
            while True:
                try:
                    req = _urllib.Request(
                        f"http://127.0.0.1:{_WA_PORT}/recv",
                        data=b"{}", headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with _urllib.urlopen(req, timeout=35) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read())
                            if data.get("text"):
                                self._inbound.put(data)
                except Exception:
                    time.sleep(2)
        threading.Thread(target=_poll, daemon=True, name="wa-recv").start()

        # Process inbound → engine → reply
        def _work():
            while True:
                try:
                    msg = self._inbound.get(timeout=1)
                except queue.Empty:
                    continue
                if not self._engine or not self._enabled:
                    continue
                self._owner_jid = msg.get("from", self._owner_jid)
                try:
                    from tools.whatsapp_tool import WhatsAppSendTool
                    WhatsAppSendTool.set_owner(self._owner_jid)
                except Exception:
                    pass
                try:
                    reply = self._engine.submit_message(msg["text"])
                    if reply and reply.strip():
                        self._send(reply)
                except Exception as e:
                    self._send(f"⚠️ Error: {e}")
        threading.Thread(target=_work, daemon=True, name="wa-worker").start()

        # Status poller
        def _status():
            while True:
                try:
                    with _urllib.urlopen(
                        f"http://127.0.0.1:{_WA_PORT}/status", timeout=3
                    ) as r:
                        data = json.loads(r.read())
                        self._connected = data.get("ready", False)
                        owner = data.get("owner", "")
                        if owner and not self._owner_jid:
                            self._owner_jid = owner
                            try:
                                from tools.whatsapp_tool import WhatsAppSendTool
                                WhatsAppSendTool.set_owner(owner)
                            except Exception:
                                pass
                except Exception:
                    self._connected = False
                time.sleep(5)
        threading.Thread(target=_status, daemon=True, name="wa-status").start()

    def _send(self, text: str):
        if not self._owner_jid:
            return
        try:
            body = json.dumps({"to": self._owner_jid, "text": text}).encode()
            req  = _urllib.Request(
                f"http://127.0.0.1:{_WA_PORT}/send",
                data=body, headers={"Content-Type": "application/json"},
                method="POST",
            )
            _urllib.urlopen(req, timeout=10)
        except Exception:
            pass


# Self-register
_wa_connector = WhatsAppConnector()
registry.register(_wa_connector)
