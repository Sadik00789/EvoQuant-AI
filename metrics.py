"""
Lightweight, dependency-free metrics registry (Phase 0 observability).

Thread-safe counters and gauges that surface operational truth the previous
build lacked: bars processed, debate failures, orders submitted/skipped,
reconciliation runs, and dead-letter routing.
"""

from __future__ import annotations

import threading
import time
from typing import Dict

_lock = threading.Lock()
_counters: Dict[str, float] = {}
_gauges: Dict[str, float] = {}
_started = time.time()


def increment(name: str, value: float = 1.0) -> None:
    with _lock:
        _counters[name] = _counters.get(name, 0.0) + float(value)


def set_gauge(name: str, value: float) -> None:
    with _lock:
        _gauges[name] = float(value)


def snapshot() -> Dict[str, object]:
    with _lock:
        return {
            "uptime_seconds": round(time.time() - _started, 1),
            "counters": dict(_counters),
            "gauges": dict(_gauges),
        }


def summary_line() -> str:
    snap = snapshot()
    counters = snap.get("counters", {})
    if not counters:
        return "metrics: (no events yet)"
    parts = ", ".join(f"{k}={int(v)}" for k, v in sorted(counters.items()))
    return f"metrics: {parts}"
