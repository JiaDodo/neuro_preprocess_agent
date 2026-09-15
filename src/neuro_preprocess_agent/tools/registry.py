from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from neuro_preprocess_agent.state import PipelineState

WorkerHandler = Callable[[PipelineState], dict[str, Any]]


class WorkerPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    description: str
    output_key: str
    risk: str
    handler: WorkerHandler


class WorkerRegistry:
    """Small, explicit registry for deterministic pipeline workers."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerSpec] = {}

    def register(self, spec: WorkerSpec) -> None:
        if spec.name in self._workers:
            raise ValueError(f"Worker already registered: {spec.name}")
        self._workers[spec.name] = spec

    def get(self, name: str) -> WorkerSpec:
        try:
            return self._workers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown worker: {name}") from exc

    def catalog(self) -> list[dict[str, str]]:
        return [
            {
                "name": item.name,
                "description": item.description,
                "risk": item.risk,
                "output_key": item.output_key,
            }
            for item in self._workers.values()
        ]

    def execute(self, name: str, state: PipelineState) -> dict[str, Any]:
        spec = self.get(name)
        policy = state.get("config", {}).get("permissions", {}).get(name, "allow")
        if policy == "deny":
            raise WorkerPolicyError(f"Worker '{name}' is denied by permissions.{name}")
        if policy == "ask" and not state.get("plan_approved"):
            raise WorkerPolicyError(f"Worker '{name}' requires an approved execution plan")
        return spec.handler(state)
