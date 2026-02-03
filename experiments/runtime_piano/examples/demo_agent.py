#!/usr/bin/env python3
"""Demo agent using the Piano runtime."""

from __future__ import annotations

from experiments.runtime_piano.executor import MatrixExecutor
from experiments.runtime_piano.registry import CapabilityRegistry
from experiments.runtime_piano.router import ConductorRouter
from experiments.runtime_piano.runtime import PianoRuntime
from experiments.runtime_piano.types import Capability, CapabilityContext, TaskResult
from experiments.runtime_piano.validator import OGIVValidator


def search_catalog(ctx: CapabilityContext):
    return {"items": ["keyboard", "sustain pedal", "bench"]}


def price_quote(ctx: CapabilityContext):
    return {"quote": "$199"}


def inventory_check(ctx: CapabilityContext):
    if "items" not in ctx.input or not ctx.input.get("items"):
        return {"disclaimer": True, "error": "missing_items"}
    return {"in_stock": True, "warehouse": "A1"}


def order_create(ctx: CapabilityContext):
    return {"order_id": "ORD-12345", "status": "created"}


def delivery_eta(ctx: CapabilityContext):
    if "order_id" not in ctx.input:
        return {"disclaimer": True, "error": "missing_order_id"}
    return {"text": "Estimated delivery in 3-5 business days."}


def refund_policy(ctx: CapabilityContext):
    return {"policy": "30-day returns with receipt."}


def build_registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    registry.register(
        Capability(
            id="search_catalog",
            name="Search Catalog",
            handler=search_catalog,
            tags={"search", "catalog"},
            provides={"items"},
            requires=set(),
        )
    )
    registry.register(
        Capability(
            id="price_quote",
            name="Price Quote",
            handler=price_quote,
            tags={"pricing"},
            provides={"quote"},
            requires={"items"},
        )
    )
    registry.register(
        Capability(
            id="inventory_check",
            name="Inventory Check",
            handler=inventory_check,
            tags={"inventory"},
            provides={"in_stock", "warehouse"},
            requires={"items"},
        )
    )
    registry.register(
        Capability(
            id="order_create",
            name="Order Create",
            handler=order_create,
            tags={"order"},
            provides={"order_id", "status"},
            requires={"items"},
        )
    )
    registry.register(
        Capability(
            id="delivery_eta",
            name="Delivery ETA",
            handler=delivery_eta,
            tags={"delivery"},
            provides={"text"},
            requires={"order_id"},
        )
    )
    registry.register(
        Capability(
            id="refund_policy",
            name="Refund Policy",
            handler=refund_policy,
            tags={"policy"},
            provides={"policy"},
            requires=set(),
        )
    )
    return registry


def build_plan() -> list[dict]:
    return [
        {"intent": "catalog_search", "input": {"query": "piano accessories"}, "required": ["items"], "tags": ["search"]},
        {"intent": "quote_and_stock", "input": {}, "required": ["quote", "in_stock", "warehouse"], "tags": ["pricing", "inventory"]},
        {"intent": "confirm_stock", "input": {}, "required": ["in_stock"], "tags": ["inventory"]},
        {"intent": "create_order", "input": {"qty": 1}, "required": ["order_id", "status"], "tags": ["order"]},
        {"intent": "delivery_eta", "input": {}, "required": ["text"], "tags": ["delivery"]},
    ]


def print_trace(result: TaskResult) -> None:
    print(f"Task {result.task_id} ok={result.ok} latency_ms={result.latency_ms:.2f}")
    for step in result.steps:
        print(
            f"  step {step.step_id} ok={step.ok} used={step.used} fallback={step.fallback_used} latency_ms={step.latency_ms:.2f}"
        )
        if step.error:
            print(f"    error: {step.error}")
        combined = step.outputs.get("combined", {}) if isinstance(step.outputs, dict) else {}
        if combined:
            print(f"    combined: {combined}")


def main(trace_path: str | None = None) -> None:
    registry = build_registry()
    router = ConductorRouter(registry)
    executor = MatrixExecutor(registry)
    validator = OGIVValidator()
    runtime = PianoRuntime(registry, router, executor, validator)

    plan = build_plan()
    result = asyncio_run(runtime.run_task(plan, task_id="demo-ecom", trace_path=trace_path))
    print_trace(result)


def asyncio_run(coro):
    try:
        import asyncio

        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)


if __name__ == "__main__":
    main()
