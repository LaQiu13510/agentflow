"""AgentFlow Streamlit 应用入口。"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
from langchain_core.messages import HumanMessage

from agent_team.memory import get_memory
from agent_team.skills import get_skill_registry
from agent_team.supervisor import build_agent_team
from agent_team.tracing import get_trace_store
from config import DEFAULT_SESSION_ID
from models.embedding import get_embedding_model
from models.llm import get_llm
from tools.registry import build_tool_registry


st.set_page_config(
    page_title="AgentFlow",
    page_icon="AF",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
    max-width: 1280px;
}
.main-title {
    font-size: 2rem;
    font-weight: 720;
    margin: 0;
}
.subtle {
    color: #667085;
    font-size: 0.92rem;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.75rem;
}
.metric-box {
    border: 1px solid #e4e7ec;
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    background: #ffffff;
}
.metric-label {
    color: #667085;
    font-size: 0.76rem;
}
.metric-value {
    color: #101828;
    font-size: 1.05rem;
    font-weight: 650;
    margin-top: 0.15rem;
    overflow-wrap: anywhere;
}
.tool-chip {
    display: inline-block;
    border: 1px solid #d0d5dd;
    border-radius: 999px;
    padding: 0.18rem 0.55rem;
    margin: 0.12rem 0.15rem 0.12rem 0;
    font-size: 0.78rem;
    color: #344054;
    background: #f9fafb;
}
</style>
"""


EXAMPLES = [
    "检索 SmartKB 中关于混合检索和 RRF 的内容，并输出技术摘要。",
    "设计一个 MCP Server，用来给 AgentFlow 查询 PostgreSQL 里的对话记忆。",
    "把 AgentFlow 的架构整理成一段项目 README 摘要。",
    "比较 SmartKB 和 AgentFlow 的架构边界，并说明如何组合使用。",
]


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = DEFAULT_SESSION_ID
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_state" not in st.session_state:
        st.session_state.last_state = None


@st.cache_resource(show_spinner=False)
def get_registry():
    return build_tool_registry()


@st.cache_resource(show_spinner=False)
def get_graph():
    return build_agent_team(get_registry())


def safe_check(label: str, fn) -> tuple[str, bool, str]:
    try:
        ok, detail = fn()
        return label, bool(ok), str(detail)
    except Exception as exc:
        return label, False, str(exc)[:180]


def run_health_checks() -> list[tuple[str, bool, str]]:
    registry = get_registry()
    checks = [
        safe_check("DeepSeek", lambda: get_llm().test_connection()),
        safe_check("Embedding", lambda: get_embedding_model().test_connection()),
        safe_check(
            "PostgreSQL",
            lambda: _tool_check(registry, "postgres", "health"),
        ),
        safe_check(
            "Milvus",
            lambda: _tool_check(registry, "milvus", "health"),
        ),
    ]
    return checks


def _tool_check(registry, server: str, tool: str) -> tuple[bool, str]:
    result = registry.call(server, tool)
    return result.success, result.content


def render_sidebar() -> None:
    with st.sidebar:
        st.header("AgentFlow")
        st.caption("Supervisor + Worker Agents + MCP Tools")

        st.session_state.session_id = st.text_input(
            "Session ID",
            value=st.session_state.session_id,
            help="同一个 Session 会复用 PostgreSQL 短期记忆。",
        )

        if st.button("连接检查", use_container_width=True):
            st.session_state.health_checks = run_health_checks()

        checks = st.session_state.get("health_checks", [])
        for label, ok, detail in checks:
            with st.status(
                f"{label}: {'正常' if ok else '异常'}",
                state="complete" if ok else "error",
            ):
                st.write(detail)

        st.divider()
        st.subheader("MCP 工具")
        for item in get_registry().list_tools():
            st.caption(f"{item['server']}.{item['name']} - {item['description']}")

        st.divider()
        st.subheader("Skills")
        for item in get_skill_registry().list_skills():
            tools = ", ".join(item["suggested_tools"]) or "none"
            st.caption(f"{item['name']} -> {item['route']} · tools: {tools}")
            st.caption(item["worker_detail"])

        st.divider()
        if st.button("清空本页对话", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_state = None
            st.rerun()


def render_header() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown('<p class="main-title">AgentFlow Multi-Agent 协作平台</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtle">用 LangGraph 编排 Supervisor，由 researcher、engineer、writer、general 四类 Worker 调用 MCP 风格工具完成任务。</p>',
        unsafe_allow_html=True,
    )


def render_metrics(last_state: dict[str, Any] | None) -> None:
    memory_stats = {"total_messages": 0, "sessions": 0}
    try:
        memory_stats = get_memory().stats()
    except Exception:
        pass

    route = (last_state or {}).get("route", "-")
    latency = (last_state or {}).get("latency_ms", "-")
    tools = (last_state or {}).get("used_tools", [])
    backend = memory_stats.get("backend", "-")

    st.markdown(
        f"""
<div class="metric-row">
  <div class="metric-box"><div class="metric-label">最近路由</div><div class="metric-value">{route}</div></div>
  <div class="metric-box"><div class="metric-label">延迟</div><div class="metric-value">{latency} ms</div></div>
  <div class="metric-box"><div class="metric-label">工具调用</div><div class="metric-value">{len(tools)}</div></div>
  <div class="metric-box"><div class="metric-label">记忆</div><div class="metric-value">{memory_stats['total_messages']} / {backend}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_examples() -> None:
    cols = st.columns(2)
    for idx, example in enumerate(EXAMPLES):
        with cols[idx % 2]:
            if st.button(example, key=f"example-{idx}", use_container_width=True):
                st.session_state.pending_prompt = example


def run_agent(task: str) -> dict[str, Any]:
    graph = get_graph()
    start = time.time()
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "session_id": st.session_state.session_id,
        }
    )
    state.setdefault("latency_ms", round((time.time() - start) * 1000, 1))
    return state


def render_trace(state: dict[str, Any] | None) -> None:
    if not state:
        st.info("运行一次任务后，这里会展示 Supervisor 路由、工具调用和观察结果。")
        return

    st.subheader("执行轨迹")
    col1, col2, col3 = st.columns([1, 1, 2])
    col1.metric("路由", state.get("route", "-"))
    col2.metric("耗时", f"{state.get('latency_ms', '-')} ms")
    col3.write(f"路由原因: {state.get('route_reason', '-')}")

    tools = state.get("used_tools", [])
    if tools:
        st.markdown(
            "".join(f'<span class="tool-chip">{tool}</span>' for tool in tools),
            unsafe_allow_html=True,
        )

    for item in state.get("observations", []):
        with st.expander(item.get("tool", "tool observation")):
            st.write(item.get("content", ""))
            metadata = item.get("metadata") or {}
            image_path = metadata.get("image_path")
            image_url = metadata.get("image_url")
            if image_path:
                st.image(image_path)
            elif image_url:
                st.markdown(f"[打开生成图片]({image_url})")

    st.subheader("最近 Trace")
    for row in reversed(get_trace_store().recent(limit=5)):
        label = f"{row.get('timestamp', '')} · {row.get('route', '-')}"
        with st.expander(label):
            st.write(row.get("task", ""))
            st.caption(f"skill={row.get('route_reason', '-')} · tools={', '.join(row.get('used_tools', []))}")
            st.write(row.get("final_answer", "")[:800])


def render_chat() -> None:
    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    pending = st.session_state.pop("pending_prompt", None)
    prompt = pending or st.chat_input("输入一个任务，让 Supervisor 自动分配给合适的 Worker")
    if not prompt:
        return

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("AgentFlow 正在路由、调用工具并生成回答..."):
            try:
                state = run_agent(prompt)
                answer = state.get("final_answer") or state.get("worker_output", "")
                st.session_state.last_state = state
            except Exception as exc:
                answer = f"运行失败: {exc}"
                st.session_state.last_state = {"route": "error", "observations": []}
            st.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()


def main() -> None:
    _init_state()
    render_sidebar()
    render_header()

    render_metrics(st.session_state.last_state)
    st.divider()

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("协作对话")
        render_examples()
        render_chat()
    with right:
        render_trace(st.session_state.last_state)


if __name__ == "__main__":
    main()

