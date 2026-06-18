"""
Local network discovery via mDNS/Bonjour.
MasterMind local network discovery.

Advertises the MasterMind gateway on the local network so:
  - Mobile apps / companion tools can discover it automatically
  - Multiple instances can find each other
  - No manual IP/port configuration needed

Uses the 'zeroconf' Python library (pip install zeroconf).
Falls back gracefully if zeroconf is not installed.

Service type: _mastermind-gw._tcp.local.
"""
from __future__ import annotations
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

_SERVICE_TYPE = "_mastermind-gw._tcp.local."
_ZEROCONF_AVAILABLE = False

try:
    from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf, IPVersion
    _ZEROCONF_AVAILABLE = True
except ImportError:
    pass


@dataclass
class DiscoveredGateway:
    name: str
    host: str
    port: int
    properties: dict[str, str] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    @property
    def url(self) -> str:
        tls = self.properties.get("tls") == "1"
        scheme = "wss" if tls else "ws"
        return f"{scheme}://{self.host}:{self.port}"


class GatewayAdvertiser:
    """
    Advertises this MasterMind instance on the local network via mDNS.
    
    Usage:
        advertiser = GatewayAdvertiser(port=18234, name="MyMasterMind")
        advertiser.start()
        # ... later ...
        advertiser.stop()
    """

    def __init__(
        self,
        port: int,
        name: str | None = None,
        tls: bool = False,
        tls_fingerprint: str | None = None,
        extra_props: dict | None = None,
        minimal: bool = False,
    ):
        self._port = port
        self._name = name or f"MasterMind-{socket.gethostname()}"
        self._tls = tls
        self._tls_fingerprint = tls_fingerprint
        self._extra_props = extra_props or {}
        self._minimal = minimal
        self._zc = None
        self._info = None
        self._running = False

    def start(self) -> bool:
        """Start advertising. Returns True if successful."""
        if not _ZEROCONF_AVAILABLE:
            print("[discovery] zeroconf not installed — mDNS advertising disabled")
            print("[discovery] Install with: pip install zeroconf")
            return False

        try:
            self._zc = Zeroconf(ip_version=IPVersion.All)
            props = self._build_props()
            host = self._get_local_ip()
            svc_name = f"{self._name}.{_SERVICE_TYPE}"

            self._info = ServiceInfo(
                _SERVICE_TYPE,
                svc_name,
                addresses=[socket.inet_aton(host)],
                port=self._port,
                properties={k: v.encode() for k, v in props.items()},
                server=f"{socket.gethostname()}.local.",
            )
            self._zc.register_service(self._info)
            self._running = True
            print(f"[discovery] Advertising {self._name} on port {self._port} ({host})")
            return True
        except Exception as e:
            print(f"[discovery] Failed to start mDNS advertising: {e}")
            return False

    def stop(self) -> None:
        if self._zc and self._info:
            try:
                self._zc.unregister_service(self._info)
                self._zc.close()
            except Exception:
                pass
        self._running = False
        self._zc = None
        self._info = None

    def _build_props(self) -> dict[str, str]:
        props = {
            "role": "gateway",
            "displayName": self._name,
            "port": str(self._port),
            "transport": "gateway",
        }
        if self._tls:
            props["tls"] = "1"
        if self._tls_fingerprint and not self._minimal:
            props["tlsSha256"] = self._tls_fingerprint
        if not self._minimal:
            props["host"] = f"{socket.gethostname()}.local"
        props.update(self._extra_props)
        return props

    @staticmethod
    def _get_local_ip() -> str:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


class GatewayDiscovery:
    """
    Discover MasterMind instances on the local network.
    
    Usage:
        discovery = GatewayDiscovery()
        discovery.start()
        
        gateways = discovery.get_gateways()
        for gw in gateways:
            print(f"Found: {gw.name} at {gw.url}")
        
        discovery.stop()
    """

    def __init__(
        self,
        on_add: Optional[callable] = None,
        on_remove: Optional[callable] = None,
        timeout: float = 5.0,
    ):
        self._on_add = on_add
        self._on_remove = on_remove
        self._timeout = timeout
        self._lock = threading.Lock()
        self._gateways: dict[str, DiscoveredGateway] = {}
        self._zc = None
        self._browser = None

    def start(self) -> bool:
        if not _ZEROCONF_AVAILABLE:
            print("[discovery] zeroconf not installed — mDNS discovery disabled")
            return False
        try:
            self._zc = Zeroconf(ip_version=IPVersion.All)
            listener = _DiscoveryListener(self._on_service_added, self._on_service_removed)
            self._browser = ServiceBrowser(self._zc, _SERVICE_TYPE, listener)
            return True
        except Exception as e:
            print(f"[discovery] Failed to start discovery: {e}")
            return False

    def stop(self) -> None:
        if self._browser:
            try:
                self._browser.cancel()
            except Exception:
                pass
        if self._zc:
            try:
                self._zc.close()
            except Exception:
                pass
        self._zc = None
        self._browser = None

    def get_gateways(self) -> list[DiscoveredGateway]:
        with self._lock:
            return list(self._gateways.values())

    def discover_once(self, timeout: float | None = None) -> list[DiscoveredGateway]:
        """Discover gateways, wait for timeout, return results, and stop."""
        self.start()
        time.sleep(timeout or self._timeout)
        gateways = self.get_gateways()
        self.stop()
        return gateways

    def _on_service_added(self, info: "ServiceInfo") -> None:
        try:
            host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "127.0.0.1"
            props = {k.decode(): v.decode() for k, v in (info.properties or {}).items()}
            name = info.name.replace(f".{_SERVICE_TYPE}", "")
            gw = DiscoveredGateway(name=name, host=host, port=info.port, properties=props)
            with self._lock:
                self._gateways[info.name] = gw
            if self._on_add:
                self._on_add(gw)
        except Exception:
            pass

    def _on_service_removed(self, name: str) -> None:
        gw = None
        with self._lock:
            gw = self._gateways.pop(name, None)
        if gw and self._on_remove:
            self._on_remove(gw)


if _ZEROCONF_AVAILABLE:
    from zeroconf import ServiceListener

    class _DiscoveryListener(ServiceListener):
        def __init__(self, on_add, on_remove):
            self._on_add = on_add
            self._on_remove = on_remove

        def add_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info and self._on_add:
                self._on_add(info)

        def update_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info and self._on_add:
                self._on_add(info)

        def remove_service(self, zc: "Zeroconf", type_: str, name: str) -> None:
            if self._on_remove:
                self._on_remove(name)
else:
    class _DiscoveryListener:  # type: ignore
        def __init__(self, *a, **kw): pass


def scan_local_network(timeout: float = 5.0) -> list[DiscoveredGateway]:
    """One-shot scan for MasterMind gateways on the local network."""
    disc = GatewayDiscovery(timeout=timeout)
    return disc.discover_once(timeout)
