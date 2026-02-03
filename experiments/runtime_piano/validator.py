"""Outcome-grounded intent validation (OGIV) and fallback selection."""

from __future__ import annotations

from typing import List, Tuple

from experiments.runtime_piano.types import RouteDecision


class OGIVValidator:
    def validate(
        self, step_input: dict, outputs: dict, decision: RouteDecision
    ) -> tuple[bool, str, dict]:
        combined = outputs.get("combined", outputs) if isinstance(outputs, dict) else {}
        if not isinstance(combined, dict):
            combined = {}
        effective_input = step_input.get("input", {}) or {}
        required = set(step_input.get("required", []) or [])
        missing = required - set(effective_input.keys()) - set(combined.keys())
        satisfied_from_new_outputs = len(required & set(combined.keys()))
        required_count = len(required)
        missing_required_count = len(missing)
        satisfied_from_state = required_count - missing_required_count - satisfied_from_new_outputs
        stats = {
            "required_count": required_count,
            "missing_required_count": missing_required_count,
            "satisfied_from_state": satisfied_from_state,
            "satisfied_from_new_outputs": satisfied_from_new_outputs,
        }
        effective = dict(effective_input)
        effective.update(combined)
        if missing:
            return False, f"missing_required:{sorted(missing)[0]}", stats

        if effective.get("disclaimer") is True and step_input.get("disclaimer_allowed", True) is False:
            return False, "output_disclaimer", stats
        if "error" in effective and effective.get("error"):
            if step_input.get("strict_errors", False):
                return False, "output_error", stats
            return True, "ok_with_warnings", stats

        must_any = step_input.get("must_include_any", []) or []
        if must_any:
            text = effective.get("text", "")
            found = False
            for token in must_any:
                if token in effective:
                    found = True
                    break
                if isinstance(text, str) and token in text:
                    found = True
                    break
            if not found:
                return False, "must_include_any_not_satisfied", stats

        return True, "ok", stats

    def choose_fallback(self, decision: RouteDecision) -> list[str]:
        used = set(decision.primary)
        fallback = list(decision.warm)
        for cid, _score in decision.scores:
            if cid in used or cid in fallback:
                continue
            fallback.append(cid)
        return fallback
