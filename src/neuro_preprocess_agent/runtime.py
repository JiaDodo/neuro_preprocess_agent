from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langgraph.types import Command

from neuro_preprocess_agent.graph import build_local_graph


def interrupt_message(result: dict[str, Any]) -> str | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None
    value = getattr(interrupts[0], "value", interrupts[0])
    if isinstance(value, dict):
        return str(value.get("message") or json.dumps(value, ensure_ascii=False, indent=2))
    return str(value)


class AgentRuntime:
    """One API used by CLI today and a future web/API entry point."""

    def __init__(self, compiled_graph=None) -> None:
        self.graph = compiled_graph or build_local_graph()

    @staticmethod
    def graph_config(thread_id: str, max_concurrency: int | None = None) -> dict[str, Any]:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if max_concurrency is not None:
            config["max_concurrency"] = max_concurrency
        return config

    def start(self, request: str, config_path: str | Path, thread_id: str) -> dict[str, Any]:
        graph_config = self.graph_config(thread_id)
        existing = self.graph.get_state(graph_config)
        if existing.values:
            raise ValueError(
                f"thread_id={thread_id} already exists. Use a new thread_id for a new task, "
                "or resume the existing task."
            )
        return self.graph.invoke(
            {"request": request, "config_path": str(config_path), "errors": [], "events": []},
            config=graph_config,
        )

    def resume(self, response: str | bool | dict[str, Any], thread_id: str) -> dict[str, Any]:
        graph_config = self.graph_config(thread_id)
        snapshot = self.graph.get_state(graph_config)
        if not snapshot.values:
            raise ValueError(f"thread_id={thread_id} does not exist and cannot be resumed.")
        if not snapshot.next and not getattr(snapshot, "interrupts", ()):
            raise ValueError(f"thread_id={thread_id} has already completed and cannot be resumed.")
        state = dict(snapshot.values)
        max_workers = state.get("config", {}).get("preprocess", {}).get("max_workers")
        return self.graph.invoke(Command(resume=response), config=self.graph_config(thread_id, max_workers))

    def state(self, thread_id: str) -> dict[str, Any]:
        snapshot = self.graph.get_state(self.graph_config(thread_id))
        return dict(snapshot.values) if snapshot.values else {}
