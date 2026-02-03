#!/usr/bin/env python3
"""Asyncio benchmark runner for CPU-bound tasks.

Run:
  python3 -m experiments.benchmarks.harness --system dense --tasks 200 --concurrency 1,5,10 --steps 30 --warmup 0 --seed 42
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import time
from typing import Iterable, List, Tuple

from experiments.dense_baseline import run_dense_task_once
from experiments.pi_baseline import run_pi_task_once
from experiments.piano_runtime_baseline import run_piano_runtime_task_once
from experiments.poc import run_piano_task_once


RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "results", "throughput_concurrency.csv"
)


def parse_concurrency(value: str) -> List[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("concurrency must be a positive integer or comma-separated list")
    concurrencies = []
    for p in parts:
        try:
            n = int(p)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid concurrency value: {p}") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError("concurrency values must be positive")
        concurrencies.append(n)
    return concurrencies


def percentile(values: Iterable[float], p: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if p <= 0:
        return vals[0]
    if p >= 100:
        return vals[-1]
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def write_rows(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "concurrency",
                "tasks",
                "total_time_s",
                "throughput_tps",
                "avg_latency_ms",
                "p95_latency_ms",
                "p99_latency_ms",
                "failures",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def _task_fn(system: str, steps: int, warmup: int, seed: int) -> dict:
    if system == "piano":
        return run_piano_task_once(steps=steps, seed=seed, warmup=warmup, write_outputs=False)
    if system == "piano_runtime":
        return run_piano_runtime_task_once(steps=steps, seed=seed, warmup=warmup, write_outputs=False)
    if system == "pi":
        return run_pi_task_once(steps=steps, seed=seed, warmup=warmup, write_outputs=False)
    if system == "dense":
        return run_dense_task_once(steps=steps, seed=seed, warmup=warmup, write_outputs=False)
    raise ValueError(f"Unknown system: {system}")


async def _run_one(
    sem: asyncio.Semaphore,
    system: str,
    steps: int,
    warmup: int,
    seed: int,
) -> Tuple[float, bool]:
    async with sem:
        ok = True
        try:
            result = await asyncio.to_thread(_task_fn, system, steps, warmup, seed)
            latency_ms = float(result.get("latency_ms", 0.0))
        except Exception:
            ok = False
            latency_ms = 0.0
        return latency_ms, ok


async def run_benchmark(
    system: str,
    tasks: int,
    concurrency: int,
    steps: int,
    warmup: int,
    seed: int,
) -> Tuple[List[float], int, float]:
    sem = asyncio.Semaphore(concurrency)

    async def task_wrapper(task_index: int) -> Tuple[float, bool]:
        task_seed = seed + task_index
        return await _run_one(sem, system, steps, warmup, task_seed)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(task_wrapper(i) for i in range(tasks)))
    t1 = time.perf_counter()

    latencies = [r[0] for r in results if r[1]]
    failures = sum(1 for _, ok in results if not ok)
    total_time_s = t1 - t0
    return latencies, failures, total_time_s


def main() -> None:
    p = argparse.ArgumentParser(description="Async benchmark harness")
    p.add_argument("--system", type=str, default="dense", choices=["piano", "piano_runtime", "dense", "pi"])
    p.add_argument("--tasks", type=int, default=200)
    p.add_argument("--concurrency", type=parse_concurrency, default=[20])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.tasks <= 0 or any(c <= 0 for c in args.concurrency):
        raise SystemExit("tasks and concurrency must be positive")
    if args.steps <= 0:
        raise SystemExit("steps must be positive")
    if args.warmup < 0:
        raise SystemExit("warmup must be >= 0")

    rows = []
    for conc in args.concurrency:
        latencies, failures, total_time_s = asyncio.run(
            run_benchmark(args.system, args.tasks, conc, args.steps, args.warmup, args.seed)
        )
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95 = percentile(latencies, 95)
        p99 = percentile(latencies, 99)
        throughput = (args.tasks - failures) / total_time_s if total_time_s > 0 else 0.0

        rows.append(
            {
                "system": args.system,
                "concurrency": conc,
                "tasks": args.tasks,
                "total_time_s": f"{total_time_s:.6f}",
                "throughput_tps": f"{throughput:.6f}",
                "avg_latency_ms": f"{avg_latency:.6f}",
                "p95_latency_ms": f"{p95:.6f}",
                "p99_latency_ms": f"{p99:.6f}",
                "failures": failures,
            }
        )

    write_rows(RESULTS_PATH, rows)
    print(f"Wrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

# How to run:
#   python3 -m experiments.benchmarks.harness --system dense --tasks 200 --concurrency 1,5,10 --steps 30 --warmup 0 --seed 42
