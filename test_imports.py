"""AgentFlow 基础导入与结构检查。

默认不访问外部 API；加上 --live 时检查 DeepSeek、Embedding、PostgreSQL、Milvus。
"""

from __future__ import annotations

import sys
import py_compile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def step(name: str, fn):
    print(f"[TEST] {name} ...", end=" ")
    try:
        result = fn()
        print("[OK]")
        if result:
            print(f"       {result}")
    except Exception as exc:
        print("[FAIL]")
        print(f"       {type(exc).__name__}: {exc}")
        raise


def check_config():
    import config

    assert config.ROOT_DIR.exists()
    assert config.DATA_DIR.exists()
    assert config.DOCS_DIR.exists()
    assert config.DEFAULT_SESSION_ID
    assert config.AGENTFLOW_SESSION_TTL_SECONDS > 0
    return f"root={config.ROOT_DIR}"


def check_core_imports():
    import agent_team.context  # noqa: F401
    import agent_team.long_term_memory  # noqa: F401
    import models.embedding  # noqa: F401
    import models.llm  # noqa: F401
    warnings.filterwarnings("ignore")
    import agent_team.memory  # noqa: F401
    import agent_team.runtime_state  # noqa: F401
    import agent_team.skills  # noqa: F401
    import agent_team.supervisor  # noqa: F401
    import tools.mcp_base  # noqa: F401
    py_compile.compile(str(Path(__file__).parent / "app.py"), doraise=True)
    return "核心模块导入成功，app.py 语法检查通过"


def check_registry():
    from tools.registry import build_tool_registry

    registry = build_tool_registry()
    tools = registry.list_tools()
    assert len(tools) >= 8
    names = {f"{item['server']}.{item['name']}" for item in tools}
    required = {
        "postgres.list_smartkb_documents",
        "milvus.search_smartkb",
        "project.project_summary",
        "image.generate_image",
    }
    assert required.issubset(names)
    return f"tools={len(tools)}"


def check_keyword_router():
    from agent_team.supervisor import SupervisorAgent

    router = SupervisorAgent._keyword_route
    assert router(None, "帮我写一段项目技术报告") == "writer"
    assert router(None, "设计 MCP server 的代码结构") == "engineer"
    assert router(None, "检索知识库里的 RAG 文档") == "researcher"
    assert router(None, "生成一张 AgentFlow 架构图片") == "general"
    return "keyword routes ok"


def check_memory_fallback():
    from agent_team.long_term_memory import InMemoryLongTermMemory
    from agent_team.memory import InMemoryAgentMemory

    memory = InMemoryAgentMemory("test")
    memory.add_message("s1", "user", "你好", route="general")
    memory.add_message("s1", "assistant", "你好，我是 AgentFlow", route="general")
    context = memory.recent_context("s1")
    stats = memory.stats()
    assert "AgentFlow" in context
    assert stats["total_messages"] == 2

    long_term = InMemoryLongTermMemory("test")
    long_term.add_memory(
        "s1",
        "设计 AgentFlow 长期记忆模块",
        "长期记忆用于沉淀任务经验，并在后续任务中按相关性检索。",
        route="engineer",
    )
    long_context = long_term.format_context("长期记忆怎么设计", "s1")
    long_stats = long_term.stats()
    assert "长期记忆" in long_context
    assert long_stats["total_memories"] == 1
    return f"short={stats['backend']} long={long_stats['backend']}"


def check_context_and_skills():
    from agent_team.context import ContextManager, ContextPacket
    from agent_team.skills import get_skill_registry

    manager = ContextManager(memory_budget=20, evidence_budget=30)
    prompt = manager.build_worker_prompt(
        ContextPacket(
            task="生成一张架构图片",
            memory_context="很长的历史记忆" * 20,
            tool_observations=[{"tool": "image.generate_image", "content": "图片已生成"}],
        ),
        style="简洁输出",
    )
    assert "用户任务" in prompt
    assert "图片已生成" in prompt

    registry = get_skill_registry()
    assert registry.match("生成一张 AgentFlow 图片").name == "image_generation"
    assert registry.match("检索 RAG 知识库").route == "researcher"
    assert registry.match("设计 MCP server 测试方案").route == "engineer"
    return f"skills={len(registry.list_skills())}"


def check_runtime_state():
    from agent_team.runtime_state import RuntimeStateStore, runtime_context, current_runtime_context

    store = RuntimeStateStore("test")
    session = store.touch_session("s1", {"status": "active"})
    assert session["status"] == "active"

    task = store.create_task("s1", "设计一个任务队列")
    assert task["status"] == "queued"
    updated = store.update_task(task["task_id"], "running", route="engineer")
    assert updated["status"] == "running"
    assert store.recent_tasks(limit=1)[0]["task_id"] == task["task_id"]

    rate = store.check_rate_limit("s1")
    assert rate["allowed"]
    budget = store.add_budget_units("s1", 12, category="prompt")
    assert budget["allowed"]
    tools = store.record_tool_call("s1", task["task_id"], "engineer", "project.summary")
    assert tools["total"] == 1
    assert store.get_task(task["task_id"])["tool_calls"] == 1

    with runtime_context("s1", task["task_id"]):
        assert current_runtime_context() == ("s1", task["task_id"])
    return f"runtime={store.stats()['backend']}"


def live_checks():
    from models.embedding import get_embedding_model
    from models.llm import get_llm
    from tools.registry import build_tool_registry

    llm_ok, llm_msg = get_llm(max_tokens=64).test_connection()
    print(f"[LIVE] DeepSeek: {llm_ok} {llm_msg}")

    emb_ok, emb_msg = get_embedding_model().test_connection()
    print(f"[LIVE] Embedding: {emb_ok} {emb_msg}")

    registry = build_tool_registry()
    for server in ["postgres", "milvus"]:
        result = registry.call(server, "health")
        print(f"[LIVE] {server}: {result.success} {result.content}")


def main():
    step("配置加载", check_config)
    step("核心模块导入", check_core_imports)
    step("MCP 工具注册", check_registry)
    step("Supervisor 关键词路由", check_keyword_router)
    step("内存降级记忆", check_memory_fallback)
    step("上下文与 Skills", check_context_and_skills)
    step("运行时状态缓存", check_runtime_state)

    if "--live" in sys.argv:
        step("外部服务连通", live_checks)

    print("\nAgentFlow import checks passed.")


if __name__ == "__main__":
    main()

