"""Lightweight metrics helpers for the runtime."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class RuntimeMetrics:
    routing_ms: float = 0.0
    execution_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0
    errors: List[str] = field(default_factory=list)


class Timer:
    def __init__(self) -> None:
        self._start = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


class TraceWriter:
    def __init__(self, path: str) -> None:
        self._path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._f = open(path, "a", encoding="utf-8")

    def event(self, payload: dict) -> None:
        self._f.write(json.dumps(payload, default=_json_default) + "\n")
        self._f.flush()

    def close(self) -> None:
        self._f.close()
