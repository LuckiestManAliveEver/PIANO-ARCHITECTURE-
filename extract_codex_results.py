#!/usr/bin/env python3
"""Extract Codex replication metrics from local artifacts."""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional, Tuple


ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}


def _parse_token_sweep(path: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return (None, None)

    # Extract percentages from the sweep table rows
    percents = []
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*%", text):
        try:
            percents.append(float(match.group(1)))
        except ValueError:
            continue
    if not percents:
        return (None, None)
    return (min(percents), max(percents))


def _parse_routing_accuracy(path: str) -> Optional[float]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        return None

    match = re.search(r"Routing Accuracy \(Exact Match\):\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _get_float(d: dict, key: str) -> Optional[float]:
    val = d.get(key)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> int:
    baseline = _load_json(os.path.join(ROOT, "codex_baseline.json"))
    piano = _load_json(os.path.join(ROOT, "codex_piano.json"))

    baseline_kv = _get_float(baseline, "kv_cache_mb")
    piano_kv = _get_float(piano, "kv_cache_mb")
    baseline_ttft = _get_float(baseline, "ttft_ms")
    piano_ttft = _get_float(piano, "ttft_ms")

    token_min, token_max = _parse_token_sweep(os.path.join(ROOT, "codex_token_sweep_stdout.txt"))
    routing_acc = _parse_routing_accuracy(os.path.join(ROOT, "codex_routing_stdout.txt"))

    kv_reduction_pct = None
    if baseline_kv is not None and piano_kv is not None and baseline_kv != 0:
        kv_reduction_pct = (1.0 - (piano_kv / baseline_kv)) * 100.0

    result = {
        "baseline_kv_mb": baseline_kv,
        "piano_kv_mb": piano_kv,
        "kv_reduction_pct": kv_reduction_pct,
        "baseline_ttft_ms": baseline_ttft,
        "piano_ttft_ms": piano_ttft,
        "token_savings_min_pct": token_min,
        "token_savings_max_pct": token_max,
        "routing_accuracy_pct": routing_acc,
    }

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
