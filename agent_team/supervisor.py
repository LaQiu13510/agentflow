"""LangGraph Supervisor Agent。"""

import json
import re
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agent_team.long_term_memory import get_long_term_memory
from agent_team.memory import get_memory
from agent_team.skills import Skill, get_skill_registry
from agent_team.tracing import get_trace_store
from agent_team.workers.engineer import EngineerWorker
from agent_team.workers.general import GeneralWorker
from agent_team.workers.researcher import ResearcherWorker
from agent_team.workers.writer import WriterWorker
from config import DEFAULT_SESSION_ID
from models.llm import get_llm
from tools.mcp_base import ToolRegistry
from tools.registry import build_tool_registry


class AgentFlowState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    task: str
    route: str
    route_reason: str
    short_term_context: str
    long_term_context: str
    memory_context: str
    worker_output: str
    final_answer: str
    observations: list[dict]
    used_tools: list[str]
    latency_ms: float
    skill_name: str
    trace_record: dict
    _start_time: float


ROUTES = {"researcher", "engineer", "writer", "general"}


class SupervisorAgent:
    """负责根据任务意图选择 Worker。"""

    def __init__(self):
        self.llm = get_llm(temperature=0.0, max_tokens=512)

    def route(self, task: str) -> tuple[str, str]:
        skill = self._match_skill(task)
        if skill:
            return skill.route, f"skill:{skill.name}"

        prompt = f"""你是 Supervisor Agent，请把用户任务路由到一个 worker。

可选 worker:
- researcher: 检索资料、查知识库、总结证据
- engineer: 设计架构、实现方案、调试、测试
- writer: 项目文档、技术报告、发布说明、内容润色
- general: 其他泛化问题

只输出 JSON: {{"route": "...", "reason": "..."}}

用户任务: {task}
"""
        try:
            response = self.llm.chat(
                [
                    SystemMessage(content="你只输出合法 JSON。"),
                    HumanMessage(content=prompt),
                ],
                temperature=0.0,
                max_tokens=256,
            )
            data = self._parse_json(response)
            route = data.get("route", "general")
            if route not in ROUTES:
                route = "general"
            return route, data.get("reason", "llm")
        except Exception as exc:
            return "general", f"fallback: {exc}"

    def _keyword_route(self, task: str) -> str | None:
        skill = get_skill_registry().match(task)
        return skill.route if skill else None

    def _match_skill(self, task: str) -> Skill | None:
        return get_skill_registry().match(task)

    def _parse_json(self, text: str) -> dict:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        return json.loads(match.group(0))


def _latest_user_message(state: AgentFlowState) -> str:
    messages = state.get("messages", [])
    if not messages:
        return ""
    return str(messages[-1].content)


def build_agent_team(
    tools: ToolRegistry | None = None,
):
    """构建 AgentFlow LangGraph。"""
    registry = tools or build_tool_registry()
    supervisor = SupervisorAgent()
    memory = get_memory()
    long_term_memory = get_long_term_memory()
    workers = {
        "researcher": ResearcherWorker(registry),
        "engineer": EngineerWorker(registry),
        "writer": WriterWorker(registry),
        "general": GeneralWorker(registry),
    }

    def load_memory_node(state: AgentFlowState) -> AgentFlowState:
        session_id = state.get("session_id") or DEFAULT_SESSION_ID
        state["session_id"] = session_id
        state["task"] = _latest_user_message(state)
        short_term_context = memory.recent_context(session_id)
        long_term_context = long_term_memory.format_context(
            state.get("task", ""),
            session_id=session_id,
        )
        state["short_term_context"] = short_term_context
        state["long_term_context"] = long_term_context
        state["memory_context"] = (
            f"短期记忆:\n{short_term_context}\n\n"
            f"长期记忆:\n{long_term_context}"
        )
        state["_start_time"] = time.time()
        return state

    def supervisor_node(state: AgentFlowState) -> AgentFlowState:
        route, reason = supervisor.route(state.get("task", ""))
        state["route"] = route
        state["route_reason"] = reason
        if reason.startswith("skill:"):
            state["skill_name"] = reason.split(":", 1)[1]
        return state

    def worker_node(worker_name: str):
        def _run(state: AgentFlowState) -> AgentFlowState:
            result = workers[worker_name].run(
                task=state.get("task", ""),
                memory_context=state.get("memory_context", ""),
            )
            state["worker_output"] = result.content
            state["observations"] = result.observations
            state["used_tools"] = result.used_tools
            return state

        return _run

    def finalize_node(state: AgentFlowState) -> AgentFlowState:
        route = state.get("route", "general")
        tools_used = ", ".join(state.get("used_tools", [])) or "无"
        state["final_answer"] = (
            f"{state.get('worker_output', '')}\n\n"
            f"---\n"
            f"Supervisor 路由: `{route}` ({state.get('route_reason', '')})\n"
            f"调用工具: {tools_used}"
        )
        start_time = state.get("_start_time", time.time())
        state["latency_ms"] = round((time.time() - start_time) * 1000, 1)
        return state

    def save_memory_node(state: AgentFlowState) -> AgentFlowState:
        session_id = state.get("session_id") or DEFAULT_SESSION_ID
        task = state.get("task", "")
        answer = state.get("final_answer", "")
        route = state.get("route", "general")
        if task:
            memory.add_message(session_id, "user", task, route=route)
        if answer:
            memory.add_message(session_id, "assistant", answer, route=route)
            long_term_memory.add_memory(
                session_id=session_id,
                task=task,
                answer=answer,
                route=route,
                skill_name=state.get("skill_name", ""),
            )
        state["trace_record"] = get_trace_store().append(state)
        return state

    def route_condition(state: AgentFlowState) -> Literal["researcher", "engineer", "writer", "general"]:
        route = state.get("route", "general")
        return route if route in ROUTES else "general"

    graph = StateGraph(AgentFlowState)
    graph.add_node("load_memory", load_memory_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("researcher", worker_node("researcher"))
    graph.add_node("engineer", worker_node("engineer"))
    graph.add_node("writer", worker_node("writer"))
    graph.add_node("general", worker_node("general"))
    graph.add_node("finalize", finalize_node)
    graph.add_node("save_memory", save_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_condition,
        {
            "researcher": "researcher",
            "engineer": "engineer",
            "writer": "writer",
            "general": "general",
        },
    )
    for name in ROUTES:
        graph.add_edge(name, "finalize")
    graph.add_edge("finalize", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()


