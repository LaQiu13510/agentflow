"""AgentFlow 端到端流程测试。

默认使用 Fake LLM 与 Fake MCP Server，稳定验证 LangGraph 编排流程。
加上 --live 可以运行真实外部服务健康检查。
"""

from __future__ import annotations

import sys
import warnings

from langchain_core.messages import HumanMessage

from agent_team.long_term_memory import InMemoryLongTermMemory
from agent_team.memory import InMemoryAgentMemory
from tools.mcp_base import LocalMCPServer, ToolRegistry, ToolResult, ToolSpec

warnings.filterwarnings("ignore")


class FakeLLM:
    def chat(self, messages, temperature=None, max_tokens=None) -> str:
        task = str(messages[-1].content)[:120]
        return (
            "【离线测试回答】AgentFlow 已完成任务处理。\n"
            f"- 任务摘要: {task}\n"
            "- 输出包含工具证据、Worker 分工和可执行建议。"
        )

    def test_connection(self):
        return True, "fake llm ok"


class FakePostgresServer(LocalMCPServer):
    server_name = "postgres"

    def register_tools(self):
        self.add_tool(ToolSpec("health", "fake postgres health"), self.health)
        self.add_tool(ToolSpec("list_smartkb_documents", "fake documents"), self.list_docs)

    def health(self):
        return ToolResult(True, "Fake PostgreSQL 正常")

    def list_docs(self, limit: int = 8):
        return ToolResult(
            True,
            "- test_rag_guide_zh.md (md, 8 chunks, indexed)\n"
            "- test_hybrid_search_en.md (md, 4 chunks, indexed)",
        )


class FakeMilvusServer(LocalMCPServer):
    server_name = "milvus"

    def register_tools(self):
        self.add_tool(ToolSpec("health", "fake milvus health"), self.health)
        self.add_tool(ToolSpec("search_smartkb", "fake smartkb search"), self.search)

    def health(self):
        return ToolResult(True, "Fake Milvus 正常")

    def search(self, query: str, top_k: int = 4):
        return ToolResult(
            True,
            f"[1] test_rag_guide_zh.md score=0.91\n与 {query} 相关的 SmartKB 片段。",
            {"rows": [{"file_name": "test_rag_guide_zh.md", "score": 0.91}]},
        )


class FakeProjectServer(LocalMCPServer):
    server_name = "project"

    def register_tools(self):
        self.add_tool(ToolSpec("engineering_requirements", "fake engineering requirements"), self.requirements)
        self.add_tool(ToolSpec("project_summary", "fake project summary"), self.summary)
        self.add_tool(ToolSpec("quality_checklist", "fake quality checklist"), self.checklist)

    def requirements(self):
        return ToolResult(True, "Engineering requirements: Python, LangGraph, RAG, MCP, and storage integration.")

    def summary(self, project: str):
        return ToolResult(True, f"{project}  summary: background, architecture, tools, and metrics.")

    def checklist(self, stack: str):
        return ToolResult(True, "Checklist: MCP boundaries, agent evaluation, and deployment readiness.")


class FakeImageServer(LocalMCPServer):
    server_name = "image"

    def register_tools(self):
        self.add_tool(ToolSpec("health", "fake image health"), self.health)
        self.add_tool(ToolSpec("generate_image", "fake image generation"), self.generate_image)

    def health(self):
        return ToolResult(True, "Fake image API 正常")

    def generate_image(self, prompt: str, size: str = "1024x1024"):
        return ToolResult(
            True,
            "图片已生成: data/generated_images/fake.png",
            {"image_path": "data/generated_images/fake.png", "size": size},
        )


def build_fake_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            FakePostgresServer(),
            FakeMilvusServer(),
            FakeProjectServer(),
            FakeImageServer(),
        ]
    )


def patch_runtime(
    memory: InMemoryAgentMemory,
    long_term_memory: InMemoryLongTermMemory,
):
    import agent_team.supervisor as supervisor_module
    import agent_team.workers.base as worker_base

    fake_llm = FakeLLM()
    supervisor_module.get_llm = lambda *args, **kwargs: fake_llm
    worker_base.get_llm = lambda *args, **kwargs: fake_llm
    supervisor_module.get_memory = lambda: memory
    supervisor_module.get_long_term_memory = lambda: long_term_memory


def run_case(graph, task: str, expected_route: str):
    state = graph.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "session_id": "offline-e2e",
        }
    )
    assert state["route"] == expected_route, state
    assert state.get("final_answer"), state
    assert state.get("used_tools"), state
    assert state.get("latency_ms", 0) >= 0, state
    print(
        f"[OK] {expected_route:<10} tools={len(state['used_tools'])} "
        f"latency={state['latency_ms']}ms"
    )
    return state


def run_offline_e2e():
    from agent_team.supervisor import build_agent_team

    memory = InMemoryAgentMemory("offline-e2e")
    long_term_memory = InMemoryLongTermMemory("offline-e2e")
    patch_runtime(memory, long_term_memory)
    graph = build_agent_team(build_fake_registry())

    cases = [
        ("请检索知识库中关于混合检索和 RRF 的资料", "researcher"),
        ("帮我设计一个 MCP server 的代码结构和测试方案", "engineer"),
        ("写一段 AgentFlow README 摘要", "writer"),
        ("生成一张 AgentFlow 架构图片", "general"),
    ]
    last_state = None
    for task, expected_route in cases:
        last_state = run_case(graph, task, expected_route)

    stats = memory.stats()
    assert stats["total_messages"] == len(cases) * 2
    print(f"[OK] memory backend={stats['backend']} messages={stats['total_messages']}")

    long_stats = long_term_memory.stats()
    assert long_stats["total_memories"] == len(cases)
    long_context = long_term_memory.format_context("MCP server 测试方案", "offline-e2e")
    assert "MCP" in long_context
    print(
        f"[OK] long-term memory backend={long_stats['backend']} "
        f"memories={long_stats['total_memories']}"
    )
    print(f"[OK] last answer preview={last_state['final_answer'][:80]}")


def run_live_health():
    from test_imports import live_checks

    live_checks()


def main():
    run_offline_e2e()
    if "--live" in sys.argv:
        run_live_health()
    print("\nAgentFlow e2e checks passed.")


if __name__ == "__main__":
    main()


