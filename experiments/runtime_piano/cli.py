#!/usr/bin/env python3
"""CLI entrypoint for the Piano runtime."""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, List

from experiments.runtime_piano.examples.demo_agent import build_plan, build_registry
from experiments.runtime_piano.examples.demo_agent import main as demo_main
from experiments.runtime_piano.executor import MatrixExecutor
from experiments.runtime_piano.router import ConductorRouter
from experiments.runtime_piano.runtime import PianoRuntime
from experiments.runtime_piano.validator import OGIVValidator


def _percentile(values: List[float], p: float) -> float:
    vals = sorted(values)
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
    return vals[f] * (c - k) + vals[c] * (k - f)


def _run_bench(runs: int, seed: int, out_path: str) -> None:
    rng = random.Random(seed)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary_path = os.path.splitext(out_path)[0] + "_summary.csv"

    rows: List[Dict[str, object]] = []
    plan = build_plan()

    for run_id in range(runs):
        run_seed = seed + run_id
        rng.seed(run_seed)
        registry = build_registry()
        router = ConductorRouter(registry)
        executor = MatrixExecutor(registry)
        validator = OGIVValidator()
        runtime = PianoRuntime(registry, router, executor, validator)

        state: Dict[str, object] = {}
        for step_id, step in enumerate(plan):
            effective_input = dict(state)
            effective_input.update(step.get("input", {}) or {})
            step_payload = dict(step)
            step_payload["input"] = effective_input

            r0 = time.perf_counter()
            decision = router.route(step_payload)
            r1 = time.perf_counter()
            routing_ms = (r1 - r0) * 1000.0

            t0 = time.perf_counter()
            result = asyncio_run(
                executor.execute_step(
                    task_id=f"bench-{run_id}",
                    step_id=step_id,
                    step_input=step_payload,
                    decision=decision,
                    validator=validator,
                )
            )
            t1 = time.perf_counter()
            total_ms = (t1 - t0) * 1000.0
            exec_ms = result.latency_ms

            combined = result.outputs.get("combined", {}) if isinstance(result.outputs, dict) else {}
            cleaned = runtime._clean_output(combined if isinstance(combined, dict) else {})
            for k, v in cleaned.items():
                state[k] = v

            rows.append(
                {
                    "run_id": run_id,
                    "step_id": step_id,
                    "ok": int(result.ok),
                    "fallback": int(result.fallback_used),
                    "routing_ms": f"{routing_ms:.6f}",
                    "exec_ms": f"{exec_ms:.6f}",
                    "total_ms": f"{total_ms:.6f}",
                    "used_count": result.used_count,
                    "noop": result.noop,
                    "state_keys": len(state),
                }
            )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "step_id",
                "ok",
                "fallback",
                "routing_ms",
                "exec_ms",
                "total_ms",
                "used_count",
                "noop",
                "state_keys",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics: Dict[str, List[float]] = {
        "routing_ms": [],
        "exec_ms": [],
        "total_ms": [],
        "used_count": [],
        "noop": [],
        "state_keys": [],
    }
    for row in rows:
        metrics["routing_ms"].append(float(row["routing_ms"]))
        metrics["exec_ms"].append(float(row["exec_ms"]))
        metrics["total_ms"].append(float(row["total_ms"]))
        metrics["used_count"].append(float(row["used_count"]))
        metrics["noop"].append(float(row["noop"]))
        metrics["state_keys"].append(float(row["state_keys"]))

    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "avg", "p95"])
        for metric, values in metrics.items():
            avg = sum(values) / len(values) if values else 0.0
            p95 = _percentile(values, 95)
            writer.writerow([metric, f"{avg:.6f}", f"{p95:.6f}"])

    print(f"Wrote benchmark rows to {out_path}")
    print(f"Wrote benchmark summary to {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Piano runtime CLI")
    p.add_argument("--demo", action="store_true", help="Run demo agent")
    p.add_argument("--bench", action="store_true", help="Run benchmark mode")
    p.add_argument("--runs", type=int, default=100, help="Benchmark runs")
    p.add_argument("--seed", type=int, default=42, help="Benchmark seed")
    p.add_argument("--out", type=str, default="experiments/results/runtime_bench.csv", help="CSV output path")
    p.add_argument(
        "--trace-path",
        type=str,
        default="experiments/results/piano_trace.jsonl",
        help="Trace JSONL output path",
    )
    args = p.parse_args()

    if args.demo:
        demo_main(trace_path=args.trace_path)
        return
    if args.bench:
        _run_bench(args.runs, args.seed, args.out)
        return

    print("No action specified. Use --demo to run the demo agent.")


def asyncio_run(coro):
    try:
        import asyncio

        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


if __name__ == "__main__":
    main()
