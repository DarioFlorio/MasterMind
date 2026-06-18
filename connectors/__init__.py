"""
connectors/ — Self-contained bridge modules for EVE.

Each connector lives in its own file, inherits ConnectorBase, and
self-registers via ConnectorRegistry.  webui.py imports only the
registry; it never needs to know the implementation details.

Folder layout
─────────────
connectors/
  __init__.py         ← registry + base (this file)
  whatsapp.py         ← WhatsApp / Baileys bridge
  email_connector.py  ← IMAP / SMTP bridge
  (future: slack.py, telegram.py, sms.py …)
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional


# ── Base ──────────────────────────────────────────────────────────────────────

class ConnectorBase(ABC):
    """Abstract bridge between EVE and an external messaging channel."""

    #: Machine-readable id used in the registry and the UI toggle.
    id: str = ""
    #: Human label shown in the settings panel.
    label: str = ""
    #: Font-awesome-style emoji icon for the settings card.
    icon: str = "🔌"

    def __init__(self):
        self._engine: Optional[object]   = None   # QueryEngine, set by init()
        self._enabled: bool              = False
        self._lock                       = threading.Lock()

    # ── Life-cycle ────────────────────────────────────────────────────────────

    def init(self, engine) -> None:
        """Attach to the QueryEngine.  Called once on first chat request."""
        self._engine = engine
        if self._enabled:
            self.start()

    @abstractmethod
    def start(self) -> None:
        """Open the connection / start background threads."""

    @abstractmethod
    def stop(self) -> None:
        """Close the connection / stop threads."""

    # ── Control ───────────────────────────────────────────────────────────────

    def enable(self, cfg: Optional[Dict] = None) -> None:
        with self._lock:
            if cfg:
                self.configure(cfg)
            self._enabled = True
        if self._engine:
            self.start()

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
        self.stop()

    def configure(self, cfg: Dict) -> None:
        """Override to accept settings dict from the UI."""

    # ── Outbound ─────────────────────────────────────────────────────────────

    def on_eve_reply(self, reply: str) -> None:
        """Called by webui.py after EVE generates a response."""

    # ── Status ───────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    @abstractmethod
    def connected(self) -> bool:
        """True when the external channel is reachable."""

    @abstractmethod
    def status(self) -> Dict:
        """Return a JSON-serialisable status dict for the UI."""


# ── Registry ──────────────────────────────────────────────────────────────────

class ConnectorRegistry:
    """Singleton that owns all connector instances."""

    _instance: Optional["ConnectorRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connectors: Dict[str, ConnectorBase] = {}
        return cls._instance

    def register(self, connector: ConnectorBase) -> None:
        self._connectors[connector.id] = connector

    def get(self, cid: str) -> Optional[ConnectorBase]:
        return self._connectors.get(cid)

    def all(self) -> Dict[str, ConnectorBase]:
        return dict(self._connectors)

    def init_all(self, engine) -> None:
        for c in self._connectors.values():
            try:
                c.init(engine)
            except Exception as e:
                print(f"[connectors] {c.id} init failed: {e}", flush=True)

    def on_eve_reply(self, reply: str) -> None:
        for c in self._connectors.values():
            if c.enabled:
                try:
                    c.on_eve_reply(reply)
                except Exception:
                    pass

    def status_all(self) -> Dict:
        return {cid: c.status() for cid, c in self._connectors.items()}


# Module-level singleton
registry = ConnectorRegistry()
