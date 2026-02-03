"""Deterministic conductor router."""

from __future__ import annotations

import math
from typing import Dict, List

from experiments.runtime_piano.registry import CapabilityRegistry
from experiments.runtime_piano.types import RouteDecision


class ConductorRouter:
    def __init__(self, registry: CapabilityRegistry, primary_k: int = 3, warm_m: int = 2) -> None:
        self._registry = registry
        self._primary_k = primary_k
        self._warm_m = warm_m

    def route(self, step_input: dict) -> RouteDecision:
        intent = str(step_input.get("intent", "")).lower()
        tools = step_input.get("tools", []) or []
        tags = set(step_input.get("tags", []) or [])
        needed = set(step_input.get("required", []) or [])
        needed.update(step_input.get("needs_any", []) or [])
        available_input_keys = set((step_input.get("input", {}) or {}).keys())
        missing_needed = needed - available_input_keys

        scores: List[tuple[str, float]] = []
        for cap in self._registry.list():
            if cap.requires and not cap.requires.issubset(available_input_keys):
                continue
            provides = cap.provides or set()
            new_keys = provides - available_input_keys
            tools_explicit = any(
                isinstance(tool, str)
                and tool.lower() in (cap.id.lower(), cap.name.lower())
                for tool in tools
            )
            if needed and not missing_needed:
                if provides and provides.issubset(available_input_keys) and not tools_explicit:
                    continue
                if not new_keys and not tools_explicit:
                    continue
            score = 0.0
            if cap.id.lower() in intent or cap.name.lower() in intent:
                score += 1.0
            for t in tags:
                if t in cap.tags:
                    score += 1.0
            # Penalize higher cost hints
            score -= cap.cost_hint_ms * 0.001

            if needed and provides:
                if missing_needed:
                    if not (provides & missing_needed):
                        score -= 5.0
                else:
                    if not (provides & needed):
                        score -= 5.0
                        if not tools_explicit and score <= 0 and not tags and cap.id.lower() not in intent and cap.name.lower() not in intent:
                            continue

            # Boost if tool names overlap (if any)
            for tool in tools:
                if isinstance(tool, str) and tool.lower() in (cap.id.lower(), cap.name.lower()):
                    score += 0.5

            if score < -3.0:
                continue

            scores.append((cap.id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        primary = [cid for cid, _ in scores[: self._primary_k]]
        warm = [cid for cid, _ in scores[self._primary_k : self._primary_k + self._warm_m]]

        total = sum(max(s, 0.0) for _, s in scores) or 1.0
        top_score = max(scores[0][1], 0.0) if scores else 0.0
        confidence = top_score / total

        return RouteDecision(
            primary=primary,
            warm=warm,
            confidence=confidence,
            scores=scores,
        )
