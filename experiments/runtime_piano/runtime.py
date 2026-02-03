"""Minimal Piano runtime orchestrator."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from experiments.runtime_piano.executor import MatrixExecutor
from experiments.runtime_piano.metrics import TraceWriter
from experiments.runtime_piano.router import ConductorRouter
from experiments.runtime_piano.types import RouteDecision, StepResult, TaskResult
from experiments.runtime_piano.validator import OGIVValidator


class PianoRuntime:
    def __init__(
        self,
        registry,
        router: ConductorRouter,
        executor: MatrixExecutor,
        validator: OGIVValidator,
        queue_maxsize: int = 0,
    ) -> None:
        self._registry = registry
        self._router = router
        self._executor = executor
        self._validator = validator
        self._queue: Optional[asyncio.Queue] = None
        if queue_maxsize > 0:
            self._queue = asyncio.Queue(maxsize=queue_maxsize)

    async def run_task(
        self,
        plan: List[dict],
        task_id: str | None = None,
        max_inflight_steps: int = 1,
        trace_path: str | None = None,
    ) -> TaskResult:
        t0 = time.perf_counter()
        tid = task_id or str(uuid.uuid4())
        memory: Dict[str, Any] = {}
        state: Dict[str, Any] = {}
        steps: List[StepResult] = []
        errors: List[str] = []
        trace = TraceWriter(trace_path) if trace_path else None
        if trace:
            trace.event({"event": "task_start", "task_id": tid, "ts": time.time()})

        for idx, step in enumerate(plan):
            effective_input = dict(state)
            effective_input.update(step.get("input", {}) or {})
            step_payload = dict(step)
            step_payload["input"] = effective_input
            decision = self._router.route(step_payload)
            if trace:
                trace.event(
                    {
                        "event": "step_route",
                        "task_id": tid,
                        "step_id": idx,
                        "primary": decision.primary,
                        "warm": decision.warm,
                        "confidence": decision.confidence,
                        "scores": decision.scores[:5],
                    }
                )
            result = await self._executor.execute_step(
                task_id=tid,
                step_id=idx,
                step_input=step_payload,
                decision=decision,
                validator=self._validator,
            )
            steps.append(result)
            if not result.ok and result.error:
                errors.append(result.error)
            combined = result.outputs
            if isinstance(result.outputs, dict):
                combined = result.outputs.get("combined", result.outputs) or {}
            cleaned = self._clean_output(combined if isinstance(combined, dict) else {})
            memory[f"step_{idx}"] = cleaned
            for k, v in cleaned.items():
                state[k] = v
            if trace:
                trace.event(
                    {
                        "event": "step_result",
                        "task_id": tid,
                        "step_id": idx,
                        "ok": result.ok,
                        "used": result.used,
                        "used_count": result.used_count,
                        "noop": result.noop,
                        "required_count": result.required_count,
                        "missing_required_count": result.missing_required_count,
                        "satisfied_from_state": result.satisfied_from_state,
                        "satisfied_from_new_outputs": result.satisfied_from_new_outputs,
                        "fallback_used": result.fallback_used,
                        "latency_ms": result.latency_ms,
                        "error": result.error,
                        "combined_keys_count": len(cleaned),
                    }
                )

        t1 = time.perf_counter()
        ok = all(s.ok for s in steps)
        if trace:
            trace.event(
                {
                    "event": "task_end",
                    "task_id": tid,
                    "ok": ok,
                    "latency_ms": (t1 - t0) * 1000.0,
                    "steps_ok_count": sum(1 for s in steps if s.ok),
                    "steps_fail_count": sum(1 for s in steps if not s.ok),
                }
            )
            trace.close()
        return TaskResult(
            task_id=tid,
            ok=ok,
            steps=steps,
            latency_ms=(t1 - t0) * 1000.0,
            errors=errors,
        )

    @staticmethod
    def _clean_output(d: dict) -> dict:
        cleaned = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if v == "" or v == [] or v == {}:
                continue
            cleaned[k] = v
        return cleaned

    async def submit_task(self, plan: List[dict], task_id: str | None = None) -> TaskResult:
        if self._queue is None:
            return await self.run_task(plan, task_id=task_id)
        try:
            self._queue.put_nowait((plan, task_id))
        except asyncio.QueueFull:
            return TaskResult(
                task_id=task_id or "dropped",
                ok=False,
                steps=[],
                latency_ms=0.0,
                errors=["dropped"],
            )
        return TaskResult(
            task_id=task_id or "queued",
            ok=True,
            steps=[],
            latency_ms=0.0,
            errors=[],
        )

    async def start_workers(self, n: int) -> None:
        if self._queue is None:
            return

        async def _worker() -> None:
            while True:
                item = await self._queue.get()
                if item is None:
                    self._queue.task_done()
                    return
                plan, task_id = item
                await self.run_task(plan, task_id=task_id)
                self._queue.task_done()

        for _ in range(n):
            asyncio.create_task(_worker())
