"""长期记忆模块。

长期记忆保存可复用的任务经验摘要，并在后续任务中按相关性检索出来。
优先使用 PostgreSQL 持久化；数据库不可用时降级为进程内实现，保证本地 Demo 和离线测试可运行。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from agent_team.safety import get_safety_controller
from config import (
    AGENTFLOW_LONG_TERM_MEMORY_TABLE,
    DB_URL,
    LONG_TERM_MEMORY_LIMIT,
)


STOP_TERMS = {
    "的",
    "了",
    "和",
    "与",
    "或",
    "在",
    "是",
    "为",
    "对",
    "将",
    "并",
    "中",
    "后",
    "前",
    "这",
    "那",
    "一个",
    "用户",
    "任务",
    "摘要",
    "输出",
}


class Base(DeclarativeBase):
    pass


class LongTermMemoryRecord(Base):
    __tablename__ = AGENTFLOW_LONG_TERM_MEMORY_TABLE

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    route = Column(String(32), default="", index=True)
    skill_name = Column(String(64), default="")
    task = Column(Text, nullable=False)
    summary = Column(Text, nullable=False)
    keywords = Column(Text, default="")
    content = Column(Text, nullable=False)
    importance = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


@dataclass
class LongTermMemoryItem:
    session_id: str
    route: str
    skill_name: str
    task: str
    summary: str
    keywords: str
    content: str
    importance: float
    created_at: datetime
    id: int | None = None
    score: float = 0.0


class LongTermMemory:
    """用 PostgreSQL 保存 AgentFlow 的长期任务记忆。"""

    def __init__(self, database_url: str = DB_URL):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.session_factory = sessionmaker(bind=self.engine)
        self.safety = get_safety_controller()

    def init_tables(self):
        Base.metadata.create_all(self.engine)

    def add_memory(
        self,
        session_id: str,
        task: str,
        answer: str,
        route: str = "",
        skill_name: str = "",
    ) -> LongTermMemoryItem | None:
        item = self._build_item(session_id, task, answer, route, skill_name)
        if item is None:
            return None

        self.init_tables()
        with self.session_factory() as session:
            row = LongTermMemoryRecord(
                session_id=item.session_id,
                route=item.route,
                skill_name=item.skill_name,
                task=item.task,
                summary=item.summary,
                keywords=item.keywords,
                content=item.content,
                importance=item.importance,
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return self._to_item(row)

    def search(
        self,
        query: str,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> list[LongTermMemoryItem]:
        self.init_tables()
        with self.session_factory() as session:
            rows_query = session.query(LongTermMemoryRecord)
            if session_id:
                rows_query = rows_query.filter_by(session_id=session_id)
            rows = (
                rows_query.order_by(LongTermMemoryRecord.created_at.desc())
                .limit(200)
                .all()
            )
            items = [self._to_item(row) for row in rows]
        return self._rank(query, items, limit)

    def recent(
        self,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> list[LongTermMemoryItem]:
        self.init_tables()
        with self.session_factory() as session:
            rows_query = session.query(LongTermMemoryRecord)
            if session_id:
                rows_query = rows_query.filter_by(session_id=session_id)
            rows = (
                rows_query.order_by(LongTermMemoryRecord.created_at.desc())
                .limit(limit)
                .all()
            )
            return [self._to_item(row) for row in rows]

    def format_context(
        self,
        query: str,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> str:
        items = self.search(query, session_id=session_id, limit=limit)
        if not items:
            return "暂无长期记忆。"

        lines = []
        for index, item in enumerate(items, start=1):
            stamp = item.created_at.strftime("%Y-%m-%d")
            route = f" / {item.route}" if item.route else ""
            skill = f" / {item.skill_name}" if item.skill_name else ""
            keywords = f"；关键词：{item.keywords}" if item.keywords else ""
            lines.append(
                f"{index}. [{stamp}{route}{skill}] {item.summary[:260]}{keywords}"
            )
        return "\n".join(lines)

    def stats(self) -> dict:
        self.init_tables()
        with self.session_factory() as session:
            total = session.query(LongTermMemoryRecord).count()
            sessions = {
                row[0]
                for row in session.query(LongTermMemoryRecord.session_id).distinct().all()
            }
        return {
            "backend": "postgresql",
            "total_memories": total,
            "sessions": len(sessions),
        }

    def _to_item(self, row: LongTermMemoryRecord) -> LongTermMemoryItem:
        return LongTermMemoryItem(
            id=row.id,
            session_id=row.session_id,
            route=row.route or "",
            skill_name=row.skill_name or "",
            task=row.task,
            summary=row.summary,
            keywords=row.keywords or "",
            content=row.content,
            importance=float(row.importance or 0.5),
            created_at=row.created_at or datetime.utcnow(),
        )

    def _build_item(
        self,
        session_id: str,
        task: str,
        answer: str,
        route: str,
        skill_name: str,
    ) -> LongTermMemoryItem | None:
        task = self.safety.redact(_compact_text(task))[:1200]
        answer_body = self.safety.redact(_strip_runtime_footer(answer))
        if not task and not answer_body:
            return None

        summary = _summarize_answer(task, answer_body)
        keywords = ", ".join(_extract_keywords(f"{task}\n{summary}"))
        content = f"任务: {task}\n结论: {summary}"[:3000]
        return LongTermMemoryItem(
            session_id=session_id or "default",
            route=route or "",
            skill_name=skill_name or "",
            task=task,
            summary=summary,
            keywords=keywords,
            content=content,
            importance=_estimate_importance(task, summary, route),
            created_at=datetime.utcnow(),
        )

    def _rank(
        self,
        query: str,
        items: list[LongTermMemoryItem],
        limit: int,
    ) -> list[LongTermMemoryItem]:
        query_terms = _tokenize(query)
        ranked = []
        for item in items:
            item_terms = _tokenize(
                f"{item.task}\n{item.summary}\n{item.keywords}\n{item.route}\n{item.skill_name}"
            )
            overlap = query_terms & item_terms
            score = len(overlap) * 2.0
            if query_terms and item_terms:
                score += len(overlap) / max(len(query_terms), 1)
            score += item.importance * 0.25
            item.score = round(score, 4)
            if score > 0:
                ranked.append(item)

        if not ranked:
            ranked = items[:limit]
        ranked.sort(key=lambda row: (row.score, row.created_at), reverse=True)
        return ranked[: max(1, limit)]


class InMemoryLongTermMemory:
    """PostgreSQL 不可用时的长期记忆降级实现。"""

    def __init__(self, reason: str = ""):
        self.reason = reason
        self.rows: list[LongTermMemoryItem] = []
        self.safety = get_safety_controller()

    def init_tables(self):
        return None

    def add_memory(
        self,
        session_id: str,
        task: str,
        answer: str,
        route: str = "",
        skill_name: str = "",
    ) -> LongTermMemoryItem | None:
        item = LongTermMemory._build_item(self, session_id, task, answer, route, skill_name)
        if item is None:
            return None
        item.id = len(self.rows) + 1
        self.rows.append(item)
        return item

    def search(
        self,
        query: str,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> list[LongTermMemoryItem]:
        rows = [row for row in self.rows if not session_id or row.session_id == session_id]
        return LongTermMemory._rank(self, query, list(reversed(rows[-200:])), limit)

    def recent(
        self,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> list[LongTermMemoryItem]:
        rows = [row for row in self.rows if not session_id or row.session_id == session_id]
        return rows[-limit:]

    def format_context(
        self,
        query: str,
        session_id: str = "",
        limit: int = LONG_TERM_MEMORY_LIMIT,
    ) -> str:
        return LongTermMemory.format_context(self, query, session_id=session_id, limit=limit)

    def stats(self) -> dict:
        sessions = {row.session_id for row in self.rows}
        return {
            "backend": "memory",
            "total_memories": len(self.rows),
            "sessions": len(sessions),
            "fallback_reason": self.reason,
        }


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_runtime_footer(answer: str) -> str:
    text = str(answer or "")
    return text.split("\n\n---\nSupervisor 路由:", 1)[0].strip()


def _summarize_answer(task: str, answer: str) -> str:
    answer = _compact_text(answer)
    if not answer:
        return _compact_text(task)[:600]
    sentences = re.split(r"(?<=[。！？.!?])\s+", answer)
    selected = " ".join(sentence for sentence in sentences[:3] if sentence).strip()
    return (selected or answer)[:700]


def _estimate_importance(task: str, summary: str, route: str) -> float:
    text = f"{task}\n{summary}".lower()
    score = 0.45
    if route in {"engineer", "researcher"}:
        score += 0.1
    for keyword in [
        "架构",
        "方案",
        "测试",
        "评测",
        "接口",
        "数据库",
        "部署",
        "风险",
        "architecture",
        "test",
    ]:
        if keyword in text:
            score += 0.04
    return min(score, 1.0)


def _extract_keywords(text: str, limit: int = 10) -> list[str]:
    terms = list(_tokenize(text))
    terms.sort(key=lambda term: (-len(term), term))
    return terms[:limit]


def _tokenize(text: str) -> set[str]:
    normalized = str(text or "").lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
    cjk_bigrams = {
        "".join(cjk_chars[index : index + 2])
        for index in range(max(0, len(cjk_chars) - 1))
    }
    return (words | set(cjk_chars) | cjk_bigrams) - STOP_TERMS


_long_term_memory_instance: LongTermMemory | InMemoryLongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory | InMemoryLongTermMemory:
    global _long_term_memory_instance
    if _long_term_memory_instance is None:
        try:
            _long_term_memory_instance = LongTermMemory()
            _long_term_memory_instance.init_tables()
        except Exception as exc:
            _long_term_memory_instance = InMemoryLongTermMemory(str(exc)[:200])
    return _long_term_memory_instance
