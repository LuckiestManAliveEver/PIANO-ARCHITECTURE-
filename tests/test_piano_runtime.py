import asyncio
import unittest

from experiments.runtime_piano.examples.demo_agent import build_plan, build_registry
from experiments.runtime_piano.executor import MatrixExecutor
from experiments.runtime_piano.registry import CapabilityRegistry
from experiments.runtime_piano.router import ConductorRouter
from experiments.runtime_piano.runtime import PianoRuntime
from experiments.runtime_piano.types import Capability, RouteDecision
from experiments.runtime_piano.validator import OGIVValidator


class PianoRuntimeTests(unittest.TestCase):
    def test_state_propagation_items_available(self) -> None:
        registry = build_registry()
        runtime = PianoRuntime(
            registry,
            ConductorRouter(registry),
            MatrixExecutor(registry),
            OGIVValidator(),
        )
        plan = build_plan()
        result = asyncio.run(runtime.run_task(plan, task_id="test-state"))
        self.assertTrue(result.ok)
        self.assertTrue(result.steps[2].ok)
        combined = result.steps[2].outputs.get("combined", {})
        self.assertIn("in_stock", combined)

    def test_requirements_gating_delivery_eta(self) -> None:
        registry = build_registry()
        router = ConductorRouter(registry)
        decision = router.route({"intent": "delivery_eta", "input": {}})
        scored_ids = [cid for cid, _ in decision.scores]
        self.assertNotIn("delivery_eta", scored_ids)
        self.assertNotIn("delivery_eta", decision.primary)
        self.assertNotIn("delivery_eta", decision.warm)

    def test_safe_merge_disclaimer_does_not_erase(self) -> None:
        registry = CapabilityRegistry()

        def in_stock(_ctx):
            return {"in_stock": True}

        def disclaimer(_ctx):
            return {"disclaimer": True}

        registry.register(Capability(id="stock", name="Stock", handler=in_stock))
        registry.register(Capability(id="disc", name="Disclaimer", handler=disclaimer))

        executor = MatrixExecutor(registry)
        validator = OGIVValidator()
        decision = RouteDecision(primary=["stock", "disc"], warm=[], confidence=1.0, scores=[])
        result = asyncio.run(
            executor.execute_step(
                task_id="test-merge",
                step_id=0,
                step_input={"input": {}},
                decision=decision,
                validator=validator,
            )
        )
        combined = result.outputs.get("combined", {})
        self.assertIn("in_stock", combined)
        self.assertTrue(combined["in_stock"])

    def test_fallback_additive_adds_required(self) -> None:
        registry = CapabilityRegistry()

        def primary(_ctx):
            return {"disclaimer": True}

        def fallback(_ctx):
            return {"in_stock": True}

        registry.register(Capability(id="primary", name="Primary", handler=primary))
        registry.register(Capability(id="fallback", name="Fallback", handler=fallback))

        executor = MatrixExecutor(registry)
        validator = OGIVValidator()
        decision = RouteDecision(
            primary=["primary"],
            warm=["fallback"],
            confidence=1.0,
            scores=[("primary", 1.0), ("fallback", 0.5)],
        )
        result = asyncio.run(
            executor.execute_step(
                task_id="test-fallback",
                step_id=0,
                step_input={"input": {}, "required": ["in_stock"]},
                decision=decision,
                validator=validator,
            )
        )
        combined = result.outputs.get("combined", {})
        self.assertTrue(result.ok)
        self.assertTrue(result.fallback_used)
        self.assertIn("in_stock", combined)
        self.assertTrue(combined["in_stock"])


if __name__ == "__main__":
    unittest.main()
