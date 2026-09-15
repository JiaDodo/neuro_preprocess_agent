from __future__ import annotations

from datetime import datetime
from typing import Any

from langgraph.config import get_stream_writer


def make_event(node: str, message: str, level: str = "info", **extra: Any) -> dict[str, Any]:
    event = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "node": node,
        "level": level,
        "message": message,
    }
    event.update(extra)
    return event


def emit(event: dict[str, Any]) -> None:
    try:
        get_stream_writer()(event)
    except RuntimeError:
        pass
