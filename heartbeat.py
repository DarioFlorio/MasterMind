"""heartbeat.py — Background periodic task runner."""
from __future__ import annotations
import threading, time
from typing import Callable


class Heartbeat:
    def __init__(self):
        self._tasks:  list[tuple[int, Callable, float]] = []
        self._thread: threading.Thread | None = None
        self._stop    = threading.Event()

    def register(self, every: int, task: Callable) -> None:
        self._tasks.append((every, task, time.time()))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        while not self._stop.wait(timeout=10):
            now = time.time()
            for i, (interval, task, last) in enumerate(self._tasks):
                if now - last >= interval:
                    try:
                        task()
                    except Exception:
                        pass
                    self._tasks[i] = (interval, task, now)
