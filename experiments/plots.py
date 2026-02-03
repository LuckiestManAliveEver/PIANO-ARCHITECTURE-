#!/usr/bin/env python3
"""Plot CSV outputs from poc.py.

Generates PNGs in experiments/results/.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: str, skip_warmup: bool = False) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if skip_warmup and int(float(row.get("is_warmup", 0))) == 1:
                continue
            for k, v in row.items():
                data.setdefault(k, []).append(float(v))
    return data


def rolling_mean(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values
    out = []
    s = 0.0
    q = []
    for v in values:
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_series(x, y, title, ylabel, out_path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_compare(x, y1, y2, label1, label2, title, ylabel, out_path) -> None:
    plt.figure(figsize=(8, 4))
    plt.plot(x, y1, linewidth=1.5, label=label1)
    plt.plot(x, y2, linewidth=1.5, label=label2)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def summarize_runtime_bench(data: Dict[str, List[float]]) -> Tuple[List[int], List[float], List[float], float, float]:
    step_ids = sorted({int(s) for s in data.get("step_id", [])})
    used_by_step = []
    noop_by_step = []
    for step_id in step_ids:
        indices = [i for i, s in enumerate(data.get("step_id", [])) if int(s) == step_id]
        if not indices:
            used_by_step.append(0.0)
            noop_by_step.append(0.0)
            continue
        used_vals = [data.get("used_count", [0.0])[i] for i in indices]
        noop_vals = [data.get("noop", [0.0])[i] for i in indices]
        used_by_step.append(float(np.mean(used_vals)) if used_vals else 0.0)
        noop_by_step.append(float(np.mean(noop_vals)) if noop_vals else 0.0)
    avg_used_count = float(np.mean(data.get("used_count", [0.0]))) if data.get("used_count") else 0.0
    noop_rate = float(np.mean(data.get("noop", [0.0]))) if data.get("noop") else 0.0
    return step_ids, used_by_step, noop_by_step, avg_used_count, noop_rate


def summarize_metrics(data: Dict[str, List[float]]) -> Tuple[float, float, float, float, float, float, float, float]:
    step_total = np.asarray(data["step_total_ms"], dtype=float)
    routing = np.asarray(data["routing_ms"], dtype=float)
    kv = np.asarray(data["kv_proxy"], dtype=float)

    avg_step_total = float(np.mean(step_total))
    p95_step_total = float(np.percentile(step_total, 95))
    max_step_total = float(np.max(step_total))

    avg_routing = float(np.mean(routing))
    p95_routing = float(np.percentile(routing, 95))
    max_routing = float(np.max(routing))

    avg_kv = float(np.mean(kv))
    max_kv = float(np.max(kv))

    return (
        avg_step_total,
        p95_step_total,
        max_step_total,
        avg_routing,
        p95_routing,
        max_routing,
        avg_kv,
        max_kv,
    )


def write_summary_csv(
    out_path: str,
    piano: Dict[str, List[float]],
    dense: Dict[str, List[float]],
) -> None:
    header = [
        "system",
        "avg_step_total_ms",
        "p95_step_total_ms",
        "max_step_total_ms",
        "avg_routing_ms",
        "p95_routing_ms",
        "max_routing_ms",
        "avg_kv_proxy",
        "max_kv_proxy",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name, data in (("piano", piano), ("dense", dense)):
            (
                avg_step_total,
                p95_step_total,
                max_step_total,
                avg_routing,
                p95_routing,
                max_routing,
                avg_kv,
                max_kv,
            ) = summarize_metrics(data)
            writer.writerow(
                [
                    name,
                    avg_step_total,
                    p95_step_total,
                    max_step_total,
                    avg_routing,
                    p95_routing,
                    max_routing,
                    avg_kv,
                    max_kv,
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Piano Architecture simulation results")
    p.add_argument(
        "--piano-steps",
        type=str,
        default="experiments/results/piano_steps.csv",
        help="Piano steps CSV",
    )
    p.add_argument(
        "--dense-steps",
        type=str,
        default="experiments/results/dense_steps.csv",
        help="Dense steps CSV",
    )
    p.add_argument(
        "--piano-summary",
        type=str,
        default="experiments/results/piano_summary.csv",
        help="Piano summary CSV",
    )
    p.add_argument(
        "--dense-summary",
        type=str,
        default="experiments/results/dense_summary.csv",
        help="Dense summary CSV",
    )
    p.add_argument("--out", type=str, default="experiments/results")
    p.add_argument("--window", type=int, default=10, help="Rolling window for fallback frequency")
    args = p.parse_args()

    data = read_csv(args.piano_steps, skip_warmup=True)
    dense = read_csv(args.dense_steps, skip_warmup=True)
    moe_steps_path = os.path.join(args.out, "moe_steps.csv")
    moe = read_csv(moe_steps_path, skip_warmup=True) if os.path.exists(moe_steps_path) else None
    steps = data.get("step", list(range(1, len(next(iter(data.values()))) + 1)))

    ensure_dir(args.out)

    plot_series(
        steps,
        data["matrix_count"],
        "Matrix Count vs Time",
        "Matrix Count",
        os.path.join(args.out, "matrix_count_vs_time.png"),
    )

    plot_series(
        steps,
        data["routing_ms"],
        "Routing vs Time",
        "Routing (ms)",
        os.path.join(args.out, "routing_vs_time.png"),
    )

    plot_series(
        steps,
        data["kv_proxy"],
        "KV-Cache Proxy Size vs Time",
        "KV Proxy Size",
        os.path.join(args.out, "kv_proxy_vs_time.png"),
    )

    fallback_roll = rolling_mean(data["fallback"], args.window)
    plot_series(
        steps,
        fallback_roll,
        f"Fallback Frequency (Rolling Window = {args.window})",
        "Fallback Rate",
        os.path.join(args.out, "fallback_frequency.png"),
    )

    plot_compare(
        steps,
        data["routing_ms"],
        dense["routing_ms"],
        "Piano",
        "Dense",
        "Routing Comparison",
        "Routing (ms)",
        os.path.join(args.out, "routing_compare.png"),
    )

    plot_compare(
        steps,
        data["step_total_ms"],
        dense["step_total_ms"],
        "Piano",
        "Dense",
        "Step Total Comparison",
        "Step Total (ms)",
        os.path.join(args.out, "step_total_compare.png"),
    )

    # Step total with routing overlay (secondary y-axis)
    fig, ax = plt.subplots(figsize=(8, 4))
    l1, = ax.plot(steps, data["step_total_ms"], linewidth=1.5, label="Piano (step_total)")
    l2, = ax.plot(steps, dense["step_total_ms"], linewidth=1.5, label="Dense (step_total)")
    l4 = None
    if moe is not None:
        l4, = ax.plot(steps, moe["step_total_ms"], linewidth=1.5, label="MoE (step_total)")
    ax.set_title("Step Total Comparison (with Routing Overlay)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Step Total (ms)")

    piano_routing = data["routing_ms"]
    ax2 = ax.twinx()
    l3, = ax2.plot(
        steps,
        piano_routing,
        linestyle="--",
        alpha=0.7,
        label="Piano Routing",
    )
    l5 = None
    if moe is not None:
        l5, = ax2.plot(
            steps,
            moe["routing_ms"],
            linestyle=":",
            alpha=0.7,
            label="MoE Routing",
        )
    ax2.set_ylabel("Routing (ms)")

    lines = [l1, l2]
    if l4 is not None:
        lines.append(l4)
    lines.append(l3)
    if l5 is not None:
        lines.append(l5)
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "step_total_compare_with_routing.png"), dpi=200)
    plt.close(fig)

    plot_compare(
        steps,
        data["kv_proxy"],
        dense["kv_proxy"],
        "Piano",
        "Dense",
        "KV-Cache Proxy Comparison",
        "KV Proxy Size",
        os.path.join(args.out, "kv_proxy_compare.png"),
    )

    def read_summary(path: str) -> Dict[str, Tuple[float, float, float]]:
        out: Dict[str, Tuple[float, float, float]] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                out[row["metric"]] = (
                    float(row["avg"]),
                    float(row["p95"]),
                    float(row["max"]),
                )
        return out

    piano_summary = read_summary(args.piano_summary)
    dense_summary = read_summary(args.dense_summary)
    moe_summary_path = os.path.join(args.out, "moe_summary.csv")
    moe_summary = read_summary(moe_summary_path) if os.path.exists(moe_summary_path) else None

    compare_path = os.path.join(args.out, "summary_compare.csv")
    with open(compare_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "avg_step_total_ms",
                "p95_step_total_ms",
                "max_step_total_ms",
                "avg_routing_ms",
                "p95_routing_ms",
            ]
        )
        for name, summary in (("piano", piano_summary), ("dense", dense_summary)):
            step_stats = summary.get("step_total_ms", (0.0, 0.0, 0.0))
            routing_stats = summary.get("routing_ms", (0.0, 0.0, 0.0))
            writer.writerow(
                [
                    name,
                    f"{step_stats[0]:.6f}",
                    f"{step_stats[1]:.6f}",
                    f"{step_stats[2]:.6f}",
                    f"{routing_stats[0]:.6f}",
                    f"{routing_stats[1]:.6f}",
                ]
            )
        if moe_summary is not None:
            step_stats = moe_summary.get("step_total_ms", (0.0, 0.0, 0.0))
            routing_stats = moe_summary.get("routing_ms", (0.0, 0.0, 0.0))
            writer.writerow(
                [
                    "moe",
                    f"{step_stats[0]:.6f}",
                    f"{step_stats[1]:.6f}",
                    f"{step_stats[2]:.6f}",
                    f"{routing_stats[0]:.6f}",
                    f"{routing_stats[1]:.6f}",
                ]
            )

    # Throughput/latency vs concurrency
    throughput_path = os.path.join(args.out, "throughput_concurrency.csv")
    if not os.path.exists(throughput_path):
        print(f"Missing throughput data at {throughput_path}; skipping throughput plots.")
    else:
        by_system = {}
        with open(throughput_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = row.get("system", "").strip()
                if system not in ("piano", "dense", "pi", "piano_runtime"):
                    continue
                by_system.setdefault(system, []).append(
                    (
                        int(float(row.get("concurrency", 0))),
                        float(row.get("throughput_tps", 0.0)),
                        float(row.get("p95_latency_ms", 0.0)),
                    )
                )

        def _plot_metric(metric_index: int, title: str, ylabel: str, out_name: str) -> None:
            plt.figure(figsize=(8, 4))
            for system in ("piano", "dense", "pi", "piano_runtime"):
                rows = by_system.get(system, [])
                if not rows:
                    continue
                rows.sort(key=lambda x: x[0])
                xs = [r[0] for r in rows]
                ys = [r[metric_index] for r in rows]
                plt.plot(xs, ys, linewidth=1.5, label=system.capitalize())
            plt.title(title)
            plt.xlabel("Concurrency")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, out_name), dpi=200)
            plt.close()

        _plot_metric(
            1,
            "Throughput vs Concurrency",
            "Throughput (tasks/s)",
            "throughput_vs_concurrency.png",
        )
        _plot_metric(
            2,
            "P95 Latency vs Concurrency",
            "P95 Latency (ms)",
            "p95_latency_vs_concurrency.png",
        )

    # Scale experts plots (if available)
    scale_path = os.path.join(args.out, "scale_experts.csv")
    if os.path.exists(scale_path):
        scale_by_system = {}
        rss_present = False
        with open(scale_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = row.get("system", "").strip()
                if system not in ("piano", "dense", "pi"):
                    continue
                expert_count = int(float(row.get("expert_count", 0)))
                p95_step = float(row.get("p95_step_total_ms", 0.0))
                p95_routing = float(row.get("p95_routing_ms", 0.0))
                rss_raw = row.get("rss_mb", "")
                rss_val = None
                if rss_raw not in ("", None):
                    try:
                        rss_val = float(rss_raw)
                        rss_present = True
                    except ValueError:
                        rss_val = None
                scale_by_system.setdefault(system, []).append(
                    (expert_count, p95_step, p95_routing, rss_val)
                )

        def _plot_scale(metric_index: int, title: str, ylabel: str, out_name: str) -> None:
            plt.figure(figsize=(8, 4))
            for system in ("piano", "dense", "pi"):
                rows = scale_by_system.get(system, [])
                if not rows:
                    continue
                rows.sort(key=lambda x: x[0])
                xs = [r[0] for r in rows]
                ys = [r[metric_index] for r in rows]
                plt.plot(xs, ys, linewidth=1.5, label=system.capitalize())
            plt.title(title)
            plt.xlabel("Expert Count")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, out_name), dpi=200)
            plt.close()

        _plot_scale(
            1,
            "P95 Step Total vs Expert Count",
            "P95 Step Total (ms)",
            "scale_step_total_p95_vs_experts.png",
        )
        _plot_scale(
            2,
            "P95 Routing vs Expert Count",
            "P95 Routing (ms)",
            "scale_routing_p95_vs_experts.png",
        )
        if rss_present:
            _plot_scale(
                3,
                "RSS vs Expert Count",
                "RSS (MB)",
                "scale_rss_vs_experts.png",
            )

    # Overload/backpressure plots (if available)
    overload_path = os.path.join(args.out, "overload_backpressure.csv")
    if os.path.exists(overload_path):
        overload_by_system = {}

        def _get_float(row: dict, key: str, default: float = 0.0) -> float:
            try:
                value = row.get(key, "")
                if value in ("", None):
                    return default
                return float(value)
            except (ValueError, TypeError):
                return default

        with open(overload_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                system = str(row.get("system", "")).strip()
                if not system:
                    continue
                arrival = _get_float(row, "arrival_tps", None)
                if arrival is None:
                    continue
                overload_by_system.setdefault(system, []).append(
                    (
                        arrival,
                        _get_float(row, "throughput_tps"),
                        _get_float(row, "p99_latency_ms"),
                        _get_float(row, "dropped"),
                        _get_float(row, "generated"),
                        _get_float(row, "avg_queue_depth"),
                        _get_float(row, "max_queue_depth"),
                    )
                )

        def _plot_overload(metric_index: int, title: str, ylabel: str, out_name: str) -> None:
            plt.figure(figsize=(8, 4))
            for system, rows in overload_by_system.items():
                if not rows:
                    continue
                rows.sort(key=lambda x: x[0])
                xs = [r[0] for r in rows]
                ys = [r[metric_index] for r in rows]
                plt.plot(xs, ys, linewidth=1.5, label=system.capitalize())
            plt.title(title)
            plt.xlabel("Arrival TPS")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(args.out, out_name), dpi=200)
            plt.close()

        _plot_overload(
            1,
            "Throughput vs Arrival",
            "Throughput (tasks/s)",
            "overload_throughput_vs_arrival.png",
        )
        _plot_overload(
            2,
            "P99 Latency vs Arrival",
            "P99 Latency (ms)",
            "overload_p99_latency_vs_arrival.png",
        )

        plt.figure(figsize=(8, 4))
        for system, rows in overload_by_system.items():
            if not rows:
                continue
            rows.sort(key=lambda x: x[0])
            xs = [r[0] for r in rows]
            ys = [(r[3] / r[4]) if r[4] else 0.0 for r in rows]
            plt.plot(xs, ys, linewidth=1.5, label=system.capitalize())
        plt.title("Drop Rate vs Arrival")
        plt.xlabel("Arrival TPS")
        plt.ylabel("Drop Rate (dropped/generated)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "overload_drop_rate_vs_arrival.png"), dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        for system, rows in overload_by_system.items():
            if not rows:
                continue
            rows.sort(key=lambda x: x[0])
            xs = [r[0] for r in rows]
            avg_vals = [r[5] for r in rows]
            plt.plot(xs, avg_vals, linewidth=1.5, label=f"{system.capitalize()} Avg")
            max_vals = [r[6] for r in rows]
            if any(v > 0 for v in max_vals):
                plt.plot(xs, max_vals, linewidth=1.0, linestyle="--", label=f"{system.capitalize()} Max")
        plt.title("Queue Depth vs Arrival")
        plt.xlabel("Arrival TPS")
        plt.ylabel("Queue Depth")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "overload_queue_depth_vs_arrival.png"), dpi=200)
        plt.close()

    # Runtime bench plots (if available)
    runtime_bench_path = os.path.join(args.out, "runtime_bench.csv")
    if os.path.exists(runtime_bench_path):
        runtime_data = read_csv(runtime_bench_path, skip_warmup=False)
        (
            step_ids,
            used_by_step,
            noop_by_step,
            avg_used_count,
            noop_rate,
        ) = summarize_runtime_bench(runtime_data)

        plot_series(
            step_ids,
            used_by_step,
            "Runtime Bench Used Count vs Step",
            "Used Count",
            os.path.join(args.out, "runtime_bench_used_count_vs_step.png"),
        )

        noop_roll = rolling_mean(noop_by_step, args.window)
        plot_series(
            step_ids,
            noop_roll,
            f"Runtime Bench Noop Rate (Rolling Window = {args.window})",
            "Noop Rate",
            os.path.join(args.out, "runtime_bench_noop_rate_vs_step.png"),
        )

        bench_summary_path = os.path.join(args.out, "runtime_bench_summary.csv")
        with open(bench_summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            writer.writerow(["avg_used_count", f"{avg_used_count:.6f}"])
            writer.writerow(["noop_rate", f"{noop_rate:.6f}"])

    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()

# How to run:
#   python3 -m experiments.plots
