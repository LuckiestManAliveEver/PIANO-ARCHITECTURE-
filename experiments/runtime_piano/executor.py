"""Concurrent capability executor with validation and fallback."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

from experiments.runtime_piano.registry import CapabilityRegistry
from experiments.runtime_piano.types import CapabilityContext, RouteDecision, StepResult
from experiments.runtime_piano.validator import OGIVValidator


class MatrixExecutor:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    @staticmethod
    def _is_empty(value: Any) -> bool:
        if isinstance(value, str) and value.strip() == "":
            return True
        return value is None or value == "" or value == [] or value == {}

    @classmethod
    def _safe_merge(cls, base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        for key, val in new.items():
            if key == "disclaimer":
                base["disclaimer"] = bool(base.get("disclaimer", False) or bool(val))
                continue
            if key not in base:
                base[key] = val
                continue
            if cls._is_empty(base[key]) and not cls._is_empty(val):
                base[key] = val
                continue
            if cls._is_empty(val):
                continue
            # keep base value if both are non-empty
        return base

    @staticmethod
    def _clean_output(d: Dict[str, Any]) -> Dict[str, Any]:
        cleaned: Dict[str, Any] = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if v == "" or v == [] or v == {}:
                continue
            cleaned[k] = v
        return cleaned

    async def execute_step(
        self,
        task_id: str,
        step_id: int,
        step_input: dict,
        decision: RouteDecision,
        validator: OGIVValidator,
        timeout_ms: int = 30000,
    ) -> StepResult:
        t0 = time.perf_counter()
        used: List[str] = []
        fallback_used = False
        per_tool: Dict[str, Any] = {}
        errors: List[str] = []

        def _counts() -> tuple[int, int]:
            used_count = len(used)
            return used_count, 1 if used_count == 0 else 0

        if not decision.primary and not decision.warm:
            outputs = {"combined": {}}
            ok, reason, stats = validator.validate(step_input, outputs, decision)
            t1 = time.perf_counter()
            used_count, noop = _counts()
            return StepResult(
                task_id=task_id,
                step_id=step_id,
                ok=ok,
                outputs=outputs,
                used=[],
                used_count=used_count,
                noop=noop,
                required_count=stats["required_count"],
                missing_required_count=stats["missing_required_count"],
                satisfied_from_state=stats["satisfied_from_state"],
                satisfied_from_new_outputs=stats["satisfied_from_new_outputs"],
                fallback_used=False,
                latency_ms=(t1 - t0) * 1000.0,
                error=None if ok else reason,
            )

        async def _run_capability(cid: str) -> tuple[str, Any]:
            cap = self._registry.get(cid)
            ctx = CapabilityContext(
                task_id=task_id,
                step_id=step_id,
                input=step_input.get("input", {}),
                memory={},
                trace=[],
            )
            return cid, await asyncio.to_thread(cap.handler, ctx)

        def _merge_output(combined_out: Dict[str, Any], out: Any) -> None:
            if isinstance(out, dict):
                self._safe_merge(combined_out, out)
            elif isinstance(out, str):
                prev = combined_out.get("text", "")
                combined_out["text"] = (prev + "\n" + out).strip() if prev else out
            else:
                combined_out.setdefault("values", []).append(out)

        def build_combined(tool_outputs: Dict[str, Any]) -> Dict[str, Any]:
            combined_out: Dict[str, Any] = {}
            for out in tool_outputs.values():
                _merge_output(combined_out, out)
            return combined_out

        async def _execute_batch(batch: List[str]) -> None:
            tasks = []
            for cid in batch:
                tasks.append(asyncio.wait_for(_run_capability(cid), timeout=timeout_ms / 1000.0))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for cid, res in zip(batch, results):
                if isinstance(res, Exception):
                    error_text = str(res)
                    errors.append(error_text)
                    per_tool[cid] = {"error": error_text, "disclaimer": True}
                    used.append(cid)
                    continue
                _cid, out = res
                per_tool[_cid] = out
                used.append(_cid)

        try:
            await _execute_batch(decision.primary)
            combined = build_combined(per_tool)
            ok, reason, stats = validator.validate(step_input, {"combined": combined}, decision)
            if ok:
                combined = self._clean_output(combined)
                t1 = time.perf_counter()
                used_count, noop = _counts()
                return StepResult(
                    task_id=task_id,
                    step_id=step_id,
                    ok=True,
                    outputs={"combined": combined, "by_tool": per_tool},
                    used=used,
                    used_count=used_count,
                    noop=noop,
                    required_count=stats["required_count"],
                    missing_required_count=stats["missing_required_count"],
                    satisfied_from_state=stats["satisfied_from_state"],
                    satisfied_from_new_outputs=stats["satisfied_from_new_outputs"],
                    fallback_used=fallback_used,
                    latency_ms=(t1 - t0) * 1000.0,
                )

            fallback_used = True
            fallback = validator.choose_fallback(decision)
            for cid in fallback:
                if cid in per_tool:
                    continue
                await _execute_batch([cid])
                combined = build_combined(per_tool)
                ok, reason, stats = validator.validate(step_input, {"combined": combined}, decision)
                if ok:
                    combined = self._clean_output(combined)
                    t1 = time.perf_counter()
                    used_count, noop = _counts()
                    return StepResult(
                        task_id=task_id,
                        step_id=step_id,
                        ok=True,
                        outputs={"combined": combined, "by_tool": per_tool},
                        used=used,
                        used_count=used_count,
                        noop=noop,
                        required_count=stats["required_count"],
                        missing_required_count=stats["missing_required_count"],
                        satisfied_from_state=stats["satisfied_from_state"],
                        satisfied_from_new_outputs=stats["satisfied_from_new_outputs"],
                        fallback_used=fallback_used,
                        latency_ms=(t1 - t0) * 1000.0,
                    )

            t1 = time.perf_counter()
            combined = self._clean_output(combined)
            used_count, noop = _counts()
            return StepResult(
                task_id=task_id,
                step_id=step_id,
                ok=False,
                outputs={"combined": combined, "by_tool": per_tool},
                used=used,
                used_count=used_count,
                noop=noop,
                required_count=stats["required_count"],
                missing_required_count=stats["missing_required_count"],
                satisfied_from_state=stats["satisfied_from_state"],
                satisfied_from_new_outputs=stats["satisfied_from_new_outputs"],
                fallback_used=fallback_used,
                latency_ms=(t1 - t0) * 1000.0,
                error=reason,
            )
        except Exception as exc:
            t1 = time.perf_counter()
            combined = self._clean_output(build_combined(per_tool))
            used_count, noop = _counts()
            return StepResult(
                task_id=task_id,
                step_id=step_id,
                ok=False,
                outputs={"combined": combined, "by_tool": per_tool},
                used=used,
                used_count=used_count,
                noop=noop,
                required_count=0,
                missing_required_count=0,
                satisfied_from_state=0,
                satisfied_from_new_outputs=0,
                fallback_used=fallback_used,
                latency_ms=(t1 - t0) * 1000.0,
                error=str(exc),
            )
