# -*- coding: utf-8 -*-
"""plugins/marketplace.py — Plugin marketplace browser and installer."""
from __future__ import annotations
import json
import logging
from dataclasses import dataclass

log = logging.getLogger("plugins.marketplace")

DEFAULT_REGISTRY = "https://raw.githubusercontent.com/mastermind-ai/plugins/main/registry.json"


@dataclass
class MarketplaceEntry:
    name: str
    version: str
    description: str
    author: str
    download_url: str
    tags: list[str]


class Marketplace:
    def __init__(self, registry_url: str = DEFAULT_REGISTRY):
        self.registry_url = registry_url
        self._cache: list[MarketplaceEntry] = []

    def fetch(self) -> list[MarketplaceEntry]:
        try:
            import httpx
            r = httpx.get(self.registry_url, timeout=10)
            r.raise_for_status()
            data = r.json()
            self._cache = [
                MarketplaceEntry(
                    name=p["name"], version=p.get("version", "0.0.1"),
                    description=p.get("description", ""),
                    author=p.get("author", ""),
                    download_url=p.get("download_url", ""),
                    tags=p.get("tags", []),
                )
                for p in data.get("plugins", [])
            ]
        except Exception as e:
            log.warning("Marketplace fetch failed: %s", e)
        return self._cache

    def search(self, query: str) -> list[MarketplaceEntry]:
        if not self._cache:
            self.fetch()
        q = query.lower()
        return [
            p for p in self._cache
            if q in p.name.lower() or q in p.description.lower()
            or any(q in t for t in p.tags)
        ]

    def install(self, name: str) -> bool:
        if not self._cache:
            self.fetch()
        entry = next((p for p in self._cache if p.name == name), None)
        if not entry:
            log.error("Plugin %r not found in marketplace", name)
            return False

        try:
            import httpx
            import tempfile
            from pathlib import Path
            from plugins.manager import plugin_manager

            r = httpx.get(entry.download_url, timeout=30)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
                f.write(r.content)
                tmp = Path(f.name)
            return plugin_manager.install_from_zip(tmp)
        except Exception as e:
            log.error("Marketplace install %s failed: %s", name, e)
            return False
