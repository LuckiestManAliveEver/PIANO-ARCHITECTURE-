"""Capability registry."""

from __future__ import annotations

from typing import Dict, Optional

from experiments.runtime_piano.types import Capability


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: Dict[str, Capability] = {}

    def register(self, capability: Capability) -> None:
        self._capabilities[capability.id] = capability

    def get(self, cid: str) -> Capability:
        if cid not in self._capabilities:
            raise KeyError(f"Unknown capability id: {cid}")
        return self._capabilities[cid]

    def list(self) -> list[Capability]:
        return list(self._capabilities.values())

    def has(self, cid: str) -> bool:
        return cid in self._capabilities


_DEFAULT_REGISTRY: Optional[CapabilityRegistry] = None


def _get_default_registry() -> CapabilityRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = CapabilityRegistry()
    return _DEFAULT_REGISTRY


def capability(id: str, name: str, registry: Optional[CapabilityRegistry] = None):
    def _decorator(fn):
        reg = registry or _get_default_registry()
        reg.register(Capability(id=id, name=name, handler=fn))
        return fn

    return _decorator
