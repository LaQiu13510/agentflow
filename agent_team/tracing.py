"""Persistent execution traces for AgentFlow."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import TRACE_LOG_PATH
from agent_team.safety import get_safety_controller


class AgentTraceStore:
    """Append-only JSONL trace store."""

    def __init__(self, path: Path = TRACE_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.safety = get_safety_controller()

    def append(self, state: dict[str, Any]) -> dict:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": state.get("session_id", ""),
            "task": self.safety.redact(state.get("task", "")),
            "route": state.get("route", ""),
            "route_reason": state.get("route_reason", ""),
            "used_tools": state.get("used_tools", []),
            "observations": self._redact_observations(state.get("observations", [])),
            "final_answer": self.safety.redact(state.get("final_answer", "")),
            "latency_ms": state.get("latency_ms", 0),
            "estimated_units": self.safety.estimate_units(state.get("final_answer", "")),
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def recent(self, limit: int = 10) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-limit:]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def _redact_observations(self, observations: list[dict]) -> list[dict]:
        rows = []
        for item in observations:
            rows.append(
                {
                    **item,
                    "content": self.safety.redact(item.get("content", "")),
                }
            )
        return rows


_trace_store: AgentTraceStore | None = None


def get_trace_store() -> AgentTraceStore:
    global _trace_store
    if _trace_store is None:
        _trace_store = AgentTraceStore()
    return _trace_store
