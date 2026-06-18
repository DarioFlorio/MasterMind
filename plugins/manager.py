# -*- coding: utf-8 -*-
"""
plugins/manager.py — Plugin system MasterMind built-in.

Plugins are Python packages installed under ~/.mastermind/plugins/ or
./plugins/ and follow a simple manifest format.

A plugin is a directory (or .zip) containing:
    plugin.json   — manifest
    main.py       — entry point with register(engine) function

Example plugin.json:
    {
      "name": "git-smart",
      "version": "1.0.0",
      "description": "Smart git operations",
      "author": "you",
      "tools": ["git_smart_commit", "git_smart_pr"],
      "hooks": ["tool_use:pre"]
    }
"""
from __future__ import annotations
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("plugins.manager")

PLUGIN_DIRS = [
    Path.home() / ".mastermind" / "plugins",
    Path.cwd() / "plugins",
]


@dataclass
class PluginManifest:
    name: str
    version: str
    description: str
    author: str = ""
    tools: list[str] = field(default_factory=list)
    hooks: list[str] = field(default_factory=list)
    mcp_servers: list[dict] = field(default_factory=list)
    path: Path | None = None

    @classmethod
    def from_json(cls, data: dict, path: Path | None = None) -> "PluginManifest":
        return cls(
            name=data["name"],
            version=data.get("version", "0.0.1"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tools=data.get("tools", []),
            hooks=data.get("hooks", []),
            mcp_servers=data.get("mcp_servers", []),
            path=path,
        )


@dataclass
class LoadedPlugin:
    manifest: PluginManifest
    module: Any
    active: bool = True


class PluginManager:
    def __init__(self):
        self._plugins: dict[str, LoadedPlugin] = {}
        self._engine = None

    def set_engine(self, engine) -> None:
        """Bind to the QueryEngine so plugins can add tools/hooks."""
        self._engine = engine

    def discover(self) -> list[PluginManifest]:
        """Scan plugin directories for available plugins."""
        manifests = []
        for base in PLUGIN_DIRS:
            if not base.exists():
                continue
            for entry in base.iterdir():
                manifest_path = entry / "plugin.json" if entry.is_dir() else None
                if manifest_path and manifest_path.exists():
                    try:
                        data = json.loads(manifest_path.read_text())
                        manifests.append(PluginManifest.from_json(data, entry))
                    except Exception as e:
                        log.warning("Bad plugin manifest at %s: %s", entry, e)
        return manifests

    def load(self, plugin_path: Path) -> bool:
        """Load a plugin from a directory path."""
        manifest_file = plugin_path / "plugin.json"
        main_file = plugin_path / "main.py"

        if not manifest_file.exists() or not main_file.exists():
            log.error("Plugin at %s missing plugin.json or main.py", plugin_path)
            return False

        try:
            manifest = PluginManifest.from_json(
                json.loads(manifest_file.read_text()), plugin_path
            )
        except Exception as e:
            log.error("Failed to parse plugin manifest at %s: %s", plugin_path, e)
            return False

        if manifest.name in self._plugins:
            log.info("Plugin %s already loaded", manifest.name)
            return True

        # Load the Python module
        spec = importlib.util.spec_from_file_location(
            f"mastermind_plugin_{manifest.name}", main_file
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[f"mastermind_plugin_{manifest.name}"] = module
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            log.error("Failed to exec plugin %s: %s", manifest.name, e)
            return False

        # Register the plugin
        plugin = LoadedPlugin(manifest=manifest, module=module)
        self._plugins[manifest.name] = plugin

        # Call register() if present
        if hasattr(module, "register") and self._engine:
            try:
                module.register(self._engine)
            except Exception as e:
                log.error("Plugin %s register() failed: %s", manifest.name, e)

        # Register any MCP servers declared by the plugin
        if manifest.mcp_servers:
            try:
                from mcp.registry import mcp_registry
                for srv in manifest.mcp_servers:
                    mcp_registry.add(
                        srv["name"], srv["url"],
                        srv.get("transport", "http"),
                        srv.get("api_key", "")
                    )
            except Exception as e:
                log.warning("Plugin %s MCP registration failed: %s", manifest.name, e)

        log.info("Plugin loaded: %s v%s", manifest.name, manifest.version)
        return True

    def load_all(self) -> int:
        """Auto-discover and load all plugins. Returns count loaded."""
        manifests = self.discover()
        loaded = 0
        for m in manifests:
            if m.path and self.load(m.path):
                loaded += 1
        return loaded

    def unload(self, name: str) -> bool:
        plugin = self._plugins.get(name)
        if not plugin:
            return False
        if hasattr(plugin.module, "unregister") and self._engine:
            try:
                plugin.module.unregister(self._engine)
            except Exception:
                pass
        mod_key = f"mastermind_plugin_{name}"
        sys.modules.pop(mod_key, None)
        del self._plugins[name]
        log.info("Plugin unloaded: %s", name)
        return True

    def list_plugins(self) -> list[dict]:
        return [
            {
                "name": p.manifest.name,
                "version": p.manifest.version,
                "description": p.manifest.description,
                "author": p.manifest.author,
                "tools": p.manifest.tools,
                "active": p.active,
            }
            for p in self._plugins.values()
        ]

    def install_from_zip(self, zip_path: Path) -> bool:
        """Install a plugin from a zip file."""
        import zipfile
        import tempfile
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            tmp_path = Path(tmp)
            dirs = [d for d in tmp_path.iterdir() if d.is_dir()]
            plugin_dir = dirs[0] if dirs else tmp_path

            # Read manifest to get name
            manifest_file = plugin_dir / "plugin.json"
            if not manifest_file.exists():
                log.error("No plugin.json found in zip")
                return False
            data = json.loads(manifest_file.read_text())
            name = data["name"]

            install_base = Path.home() / ".mastermind" / "plugins"
            install_base.mkdir(parents=True, exist_ok=True)
            dest = install_base / name
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(str(plugin_dir), str(dest))

        return self.load(dest)

    def __repr__(self) -> str:
        return f"<PluginManager {len(self._plugins)} plugins>"


# Singleton
plugin_manager = PluginManager()
