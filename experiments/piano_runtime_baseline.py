#!/usr/bin/env python3
"""Piano runtime baseline wrapper."""

from __future__ import annotations

import asyncio
import random
import time
from typing import List

from experiments.runtime_piano.examples.demo_agent import build_plan, build_registry
from experiments.runtime_piano.executor import MatrixExecutor
from experiments.runtime_piano.router import ConductorRouter
from experiments.runtime_piano.runtime import PianoRuntime
from experiments.runtime_piano.validator import OGIVValidator


def _expand_plan(steps: int) -> List[dict]:
    base = build_plan()
    if steps <= len(base):
        return [dict(base[i]) for i in range(steps)]
    plan: List[dict] = []
    for i in range(steps):
        plan.append(dict(base[i % len(base)]))
    return plan


def run_piano_runtime_task_once(
    steps: int,
    seed: int,
    warmup: int = 0,
    write_outputs: bool = False,
) -> dict:
    random.seed(seed)
    registry = build_registry()
    router = ConductorRouter(registry)
    executor = MatrixExecutor(registry)
    validator = OGIVValidator()
    runtime = PianoRuntime(registry, router, executor, validator)

    plan = _expand_plan(steps)
    t0 = time.perf_counter()
    result = asyncio.run(runtime.run_task(plan, task_id=f"runtime-{seed}"))
    t1 = time.perf_counter()

    step_latencies = [s.latency_ms for s in result.steps[warmup:]] if result.steps else []
    step_total_avg = sum(step_latencies) / len(step_latencies) if step_latencies else 0.0

    return {
        "latency_ms": (t1 - t0) * 1000.0,
        "step_total_ms_avg": step_total_avg,
    }


if __name__ == "__main__":
    out = run_piano_runtime_task_once(steps=5, seed=42)
    print(out)
