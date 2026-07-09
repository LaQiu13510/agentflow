"""AgentFlow 运行时状态缓存。

保存短期会话状态、任务状态、限流计数、工具调用计数和预算用量。
优先使用 Redis；未配置 Redis 或连接失败时降级到进程内 TTL 缓存。
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from threading import RLock
from typing import Any
from uuid import uuid4

from config import (
    AGENTFLOW_DAILY_BUDGET_UNITS,
    AGENTFLOW_RATE_LIMIT_PER_MINUTE,
    AGENTFLOW_RUNTIME_BACKEND,
    AGENTFLOW_SESSION_TTL_SECONDS,
    REDIS_URL,
)


_current_session_id: ContextVar[str] = ContextVar("agentflow_session_id", default="")
_current_task_id: ContextVar[str] = ContextVar("agentflow_task_id", default="")


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def current_day() -> str:
    return datetime.now().strftime("%Y%m%d")


class RuntimeStateStore:
    """进程内运行时状态缓存。"""

    backend = "memory"

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.rows: dict[str, tuple[float, dict[str, Any]]] = {}
        self.ints: dict[str, tuple[float, int]] = {}
        self.recent_task_ids: list[str] = []
        self.lock = RLock()

    def touch_session(
        self,
        session_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key = f"session:{session_id}"
        record = self._get_json(key) or {
            "session_id": session_id,
            "created_at": now_iso(),
            "run_count": 0,
        }
        record.update(metadata or {})
        record["last_seen"] = now_iso()
        self._set_json(key, record, AGENTFLOW_SESSION_TTL_SECONDS)
        return record

    def create_task(self, session_id: str, task: str) -> dict[str, Any]:
        task_id = uuid4().hex[:16]
        record = {
            "task_id": task_id,
            "session_id": session_id,
            "task": task[:500],
            "status": "queued",
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "tool_calls": 0,
        }
        self._set_json(f"task:{task_id}", record, AGENTFLOW_SESSION_TTL_SECONDS)
        self._push_recent_task(task_id)

        session = self._get_json(f"session:{session_id}") or {
            "session_id": session_id,
            "created_at": now_iso(),
            "run_count": 0,
        }
        session["run_count"] = int(session.get("run_count", 0)) + 1
        session["active_task_id"] = task_id
        session["last_task"] = task[:300]
        self.touch_session(session_id, session)
        return record

    def update_task(
        self,
        task_id: str,
        status: str,
        **fields: Any,
    ) -> dict[str, Any]:
        if not task_id:
            return {}
        key = f"task:{task_id}"
        record = self._get_json(key) or {"task_id": task_id, "created_at": now_iso()}
        record.update(fields)
        record["status"] = status
        record["updated_at"] = now_iso()
        self._set_json(key, record, AGENTFLOW_SESSION_TTL_SECONDS)
        return record

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._get_json(f"task:{task_id}")

    def recent_tasks(self, limit: int = 10) -> list[dict[str, Any]]:
        records = []
        for task_id in self._recent_task_ids(limit):
            record = self.get_task(task_id)
            if record:
                records.append(record)
        return records

    def check_rate_limit(self, session_id: str) -> dict[str, Any]:
        limit = AGENTFLOW_RATE_LIMIT_PER_MINUTE
        if limit <= 0:
            return {"allowed": True, "limit": 0, "remaining": None, "count": 0}

        window = int(time.time() // 60)
        key = f"rate:{session_id}:{window}"
        count = self._incr_int(key, ttl_seconds=120)
        return {
            "allowed": count <= limit,
            "limit": limit,
            "remaining": max(0, limit - count),
            "count": count,
            "window_seconds": 60,
        }

    def add_budget_units(
        self,
        session_id: str,
        units: int,
        category: str = "llm",
    ) -> dict[str, Any]:
        limit = AGENTFLOW_DAILY_BUDGET_UNITS
        key = f"budget:{session_id}:{current_day()}"
        current = self._get_int(key)
        units = max(0, int(units))
        next_total = current + units
        if limit > 0 and next_total > limit:
            return {
                "allowed": False,
                "category": category,
                "used": current,
                "requested": units,
                "limit": limit,
                "remaining": max(0, limit - current),
            }
        self._set_int(key, next_total, ttl_seconds=172800)
        return {
            "allowed": True,
            "category": category,
            "used": next_total,
            "requested": units,
            "limit": limit,
            "remaining": None if limit <= 0 else max(0, limit - next_total),
        }

    def record_tool_call(
        self,
        session_id: str,
        task_id: str,
        worker: str,
        tool_name: str,
    ) -> dict[str, Any]:
        if not session_id:
            return {}
        key = f"tools:{session_id}:{current_day()}"
        stats = self._get_json(key) or {
            "session_id": session_id,
            "date": current_day(),
            "total": 0,
            "by_tool": {},
            "by_worker": {},
        }
        stats["total"] = int(stats.get("total", 0)) + 1
        stats["by_tool"][tool_name] = int(stats["by_tool"].get(tool_name, 0)) + 1
        stats["by_worker"][worker] = int(stats["by_worker"].get(worker, 0)) + 1
        stats["updated_at"] = now_iso()
        self._set_json(key, stats, ttl_seconds=172800)

        if task_id:
            task = self.get_task(task_id) or {}
            task["tool_calls"] = int(task.get("tool_calls", 0)) + 1
            task["last_tool"] = tool_name
            status = task.get("status", "running")
            fields = {
                key: value
                for key, value in task.items()
                if key not in {"task_id", "status"}
            }
            self.update_task(task_id, status, **fields)
        return stats

    def get_tool_stats(self, session_id: str) -> dict[str, Any]:
        return self._get_json(f"tools:{session_id}:{current_day()}") or {
            "session_id": session_id,
            "date": current_day(),
            "total": 0,
            "by_tool": {},
            "by_worker": {},
        }

    def stats(self) -> dict[str, Any]:
        self._purge_expired()
        return {
            "backend": self.backend,
            "active_sessions": self._count_prefix("session:"),
            "recent_tasks": len(self.recent_tasks(limit=20)),
            "session_ttl_seconds": AGENTFLOW_SESSION_TTL_SECONDS,
            "rate_limit_per_minute": AGENTFLOW_RATE_LIMIT_PER_MINUTE,
            "daily_budget_units": AGENTFLOW_DAILY_BUDGET_UNITS,
            "fallback_reason": self.reason,
        }

    def _get_json(self, key: str) -> dict[str, Any] | None:
        with self.lock:
            self._purge_expired()
            row = self.rows.get(key)
            return json.loads(json.dumps(row[1])) if row else None

    def _set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        with self.lock:
            self.rows[key] = (time.time() + ttl_seconds, json.loads(json.dumps(value)))

    def _get_int(self, key: str) -> int:
        with self.lock:
            self._purge_expired()
            row = self.ints.get(key)
            return int(row[1]) if row else 0

    def _set_int(self, key: str, value: int, ttl_seconds: int) -> None:
        with self.lock:
            self.ints[key] = (time.time() + ttl_seconds, int(value))

    def _incr_int(self, key: str, ttl_seconds: int) -> int:
        value = self._get_int(key) + 1
        self._set_int(key, value, ttl_seconds)
        return value

    def _push_recent_task(self, task_id: str) -> None:
        with self.lock:
            self.recent_task_ids.insert(0, task_id)
            self.recent_task_ids = self.recent_task_ids[:100]

    def _recent_task_ids(self, limit: int) -> list[str]:
        with self.lock:
            return self.recent_task_ids[:limit]

    def _count_prefix(self, prefix: str) -> int:
        with self.lock:
            self._purge_expired()
            return len([key for key in self.rows if key.startswith(prefix)])

    def _purge_expired(self) -> None:
        now = time.time()
        for key, (expires_at, _) in list(self.rows.items()):
            if expires_at < now:
                self.rows.pop(key, None)
        for key, (expires_at, _) in list(self.ints.items()):
            if expires_at < now:
                self.ints.pop(key, None)


class RedisRuntimeStateStore(RuntimeStateStore):
    """Redis 运行时状态缓存。"""

    backend = "redis"

    def __init__(self, redis_url: str = REDIS_URL, prefix: str = "agentflow:"):
        if not redis_url:
            raise RuntimeError("REDIS_URL 未配置")
        try:
            import redis
        except Exception as exc:
            raise RuntimeError(f"redis package unavailable: {exc}") from exc

        self.reason = ""
        self.prefix = prefix
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=0.5,
            socket_timeout=0.8,
        )
        self.client.ping()

    def _get_json(self, key: str) -> dict[str, Any] | None:
        payload = self.client.get(self._key(key))
        return json.loads(payload) if payload else None

    def _set_json(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        self.client.setex(self._key(key), ttl_seconds, json.dumps(value, ensure_ascii=False))

    def _get_int(self, key: str) -> int:
        value = self.client.get(self._key(key))
        return int(value) if value else 0

    def _set_int(self, key: str, value: int, ttl_seconds: int) -> None:
        self.client.setex(self._key(key), ttl_seconds, int(value))

    def _incr_int(self, key: str, ttl_seconds: int) -> int:
        redis_key = self._key(key)
        value = int(self.client.incr(redis_key))
        if value == 1:
            self.client.expire(redis_key, ttl_seconds)
        return value

    def _push_recent_task(self, task_id: str) -> None:
        key = self._key("tasks:recent")
        self.client.lpush(key, task_id)
        self.client.ltrim(key, 0, 99)
        self.client.expire(key, AGENTFLOW_SESSION_TTL_SECONDS)

    def _recent_task_ids(self, limit: int) -> list[str]:
        return self.client.lrange(self._key("tasks:recent"), 0, max(0, limit - 1))

    def _count_prefix(self, prefix: str) -> int:
        return len(list(self.client.scan_iter(self._key(prefix) + "*", count=200)))

    def _purge_expired(self) -> None:
        return None

    def _key(self, key: str) -> str:
        return f"{self.prefix}{key}"


@contextmanager
def runtime_context(session_id: str, task_id: str = ""):
    session_token = _current_session_id.set(session_id or "")
    task_token = _current_task_id.set(task_id or "")
    try:
        yield
    finally:
        _current_session_id.reset(session_token)
        _current_task_id.reset(task_token)


def current_runtime_context() -> tuple[str, str]:
    return _current_session_id.get(), _current_task_id.get()


_runtime_state: RuntimeStateStore | RedisRuntimeStateStore | None = None


def get_runtime_state() -> RuntimeStateStore | RedisRuntimeStateStore:
    global _runtime_state
    if _runtime_state is not None:
        return _runtime_state

    if AGENTFLOW_RUNTIME_BACKEND == "memory" or (
        AGENTFLOW_RUNTIME_BACKEND == "auto" and not REDIS_URL
    ):
        _runtime_state = RuntimeStateStore()
        return _runtime_state

    try:
        _runtime_state = RedisRuntimeStateStore()
    except Exception as exc:
        _runtime_state = RuntimeStateStore(str(exc)[:200])
    return _runtime_state
