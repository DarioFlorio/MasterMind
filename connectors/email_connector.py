"""
connectors/email_connector.py — IMAP / SMTP email bridge for EVE.

Polls a mailbox every 60 s for unseen messages and routes them to the
QueryEngine exactly as if they were typed in the chat.
Self-registers into the global ConnectorRegistry on import.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

from connectors import ConnectorBase, registry


class EmailConnector(ConnectorBase):
    id    = "email"
    label = "Email"
    icon  = "✉️"

    def __init__(self):
        super().__init__()
        self._email_addr: str = ""
        self._password:   str = ""
        self._imap_host:  str = ""
        self._smtp_host:  str = ""
        self._stop_evt:   threading.Event = threading.Event()
        self._thread:     Optional[threading.Thread] = None
        self._connected:  bool = False

    # ── ConnectorBase ─────────────────────────────────────────────────────────

    def configure(self, cfg: Dict) -> None:
        self._email_addr = cfg.get("email", self._email_addr)
        self._password   = cfg.get("password", self._password)
        self._imap_host  = cfg.get("imap_host", self._imap_host)
        self._smtp_host  = cfg.get("smtp_host", self._smtp_host)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True, name="email-poll")
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        self._connected = False

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
            "email":     self._email_addr,
            "imap_host": self._imap_host,
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _poll(self):
        while not self._stop_evt.wait(timeout=60):
            if not self._enabled or not self._engine:
                continue
            if not self._imap_host or not self._email_addr or not self._password:
                continue
            try:
                import imaplib
                import email as _email_mod

                M = imaplib.IMAP4_SSL(self._imap_host)
                M.login(self._email_addr, self._password)
                M.select("INBOX")
                self._connected = True

                _, msgs = M.search(None, "UNSEEN")
                for num in (msgs[0].split() or [])[:5]:
                    _, data = M.fetch(num, "(RFC822)")
                    msg  = _email_mod.message_from_bytes(data[0][1])
                    subj = msg.get("Subject", "(no subject)")
                    frm  = msg.get("From", "unknown")
                    body = self._extract_body(msg)
                    prompt = (
                        f"[Email from {frm}]\n"
                        f"Subject: {subj}\n\n"
                        f"{body[:2000]}"
                    )
                    try:
                        self._engine.submit_message(prompt)
                    except Exception:
                        pass
                    M.store(num, "+FLAGS", "\\Seen")

                M.close()
                M.logout()
            except Exception as e:
                self._connected = False
                print(f"[email] Poll error: {e}", flush=True)

    @staticmethod
    def _extract_body(msg) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        return part.get_payload(decode=True).decode("utf-8", errors="replace")
                    except Exception:
                        pass
        try:
            return msg.get_payload(decode=True).decode("utf-8", errors="replace")
        except Exception:
            return ""


# Self-register
_email_connector = EmailConnector()
registry.register(_email_connector)
