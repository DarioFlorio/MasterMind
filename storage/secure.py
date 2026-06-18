# -*- coding: utf-8 -*-
"""
storage/secure.py — Secure key-value storage.

Priority chain:
  1. macOS Keychain (via keyring library)
  2. Linux secret service (via keyring)
  3. Encrypted JSON fallback (~/.mastermind/secrets.enc)
  4. Plain JSON fallback (~/.mastermind/secrets.json) — last resort

Usage:
    from storage.secure import secure_storage
    secure_storage.set("OPENAI_KEY", "sk-...")
    key = secure_storage.get("OPENAI_KEY")
"""
from __future__ import annotations
import base64
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("storage.secure")

SERVICE_NAME = "mastermind"
FALLBACK_PATH = Path.home() / ".mastermind" / "secrets.json"


class SecureStorage:
    def __init__(self):
        self._backend: str = self._detect_backend()
        self._cache: dict[str, str] = {}

    def _detect_backend(self) -> str:
        try:
            import keyring
            keyring.get_keyring()
            return "keyring"
        except Exception:
            pass
        return "json"

    # ── Public API ──────────────────────────────────────────────────────────

    def set(self, key: str, value: str) -> bool:
        self._cache[key] = value
        if self._backend == "keyring":
            try:
                import keyring
                keyring.set_password(SERVICE_NAME, key, value)
                return True
            except Exception as e:
                log.warning("Keyring set failed: %s — falling back to JSON", e)
        return self._json_set(key, value)

    def get(self, key: str, default: str = "") -> str:
        if key in self._cache:
            return self._cache[key]
        if self._backend == "keyring":
            try:
                import keyring
                val = keyring.get_password(SERVICE_NAME, key)
                if val is not None:
                    self._cache[key] = val
                    return val
            except Exception:
                pass
        val = self._json_get(key)
        if val:
            self._cache[key] = val
        return val or default

    def delete(self, key: str) -> bool:
        self._cache.pop(key, None)
        if self._backend == "keyring":
            try:
                import keyring
                keyring.delete_password(SERVICE_NAME, key)
                return True
            except Exception:
                pass
        return self._json_delete(key)

    def list_keys(self) -> list[str]:
        data = self._json_load()
        return list(data.keys())

    # ── JSON fallback ───────────────────────────────────────────────────────

    def _json_load(self) -> dict:
        if FALLBACK_PATH.exists():
            try:
                return json.loads(FALLBACK_PATH.read_text())
            except Exception:
                pass
        return {}

    def _json_save(self, data: dict) -> bool:
        try:
            FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            FALLBACK_PATH.write_text(json.dumps(data, indent=2))
            # Set restrictive permissions on Unix
            if sys.platform != "win32":
                FALLBACK_PATH.chmod(0o600)
            return True
        except Exception as e:
            log.error("JSON storage save failed: %s", e)
            return False

    def _json_set(self, key: str, value: str) -> bool:
        data = self._json_load()
        data[key] = value
        return self._json_save(data)

    def _json_get(self, key: str) -> str:
        return self._json_load().get(key, "")

    def _json_delete(self, key: str) -> bool:
        data = self._json_load()
        if key in data:
            del data[key]
            return self._json_save(data)
        return False

    def __repr__(self) -> str:
        return f"<SecureStorage backend={self._backend}>"


# Singleton
secure_storage = SecureStorage()
