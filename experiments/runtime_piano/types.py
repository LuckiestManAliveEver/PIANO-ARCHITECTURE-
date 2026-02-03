"""Core types for the Piano runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class Capability:
    id: str
    name: str
    handler: Callable["CapabilityContext", Any]
    cost_hint_ms: float = 0.0
    tags: set[str] = field(default_factory=set)
    provides: set[str] = field(default_factory=set)
    requires: set[str] = field(default_factory=set)


@dataclass
class CapabilityContext:
    task_id: str
    step_id: int
    input: dict
    memory: dict
    trace: list[dict]


@dataclass
class RouteDecision:
    primary: list[str]
    warm: list[str]
    confidence: float
    scores: list[tuple[str, float]]


@dataclass
class StepResult:
    task_id: str
    step_id: int
    ok: bool
    outputs: dict
    used: list[str]
    used_count: int
    noop: int
    required_count: int
    missing_required_count: int
    satisfied_from_state: int
    satisfied_from_new_outputs: int
    fallback_used: bool
    latency_ms: float
    error: str | None = None


@dataclass
class TaskResult:
    task_id: str
    ok: bool
    steps: list[StepResult]
    latency_ms: float
    errors: list[str]
