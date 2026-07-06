"""工程师 Worker。"""

from agent_team.workers.base import BaseWorker, WorkerResult


class EngineerWorker(BaseWorker):
    name = "engineer"
    role_prompt = "你是工程师 Agent，负责把需求拆成架构、模块、接口和测试。"

    def run(self, task: str, memory_context: str = "") -> WorkerResult:
        self._reset_tool_budget()
        observations = []
        used_tools = []

        requirements = self._call_tool("project", "engineering_requirements")
        observations.append({"tool": "project.engineering_requirements", "content": requirements.content})
        used_tools.append("project.engineering_requirements")

        kb = self._call_tool("milvus", "search_smartkb", query=task, top_k=3)
        observations.append({"tool": "milvus.search_smartkb", "content": kb.content})
        used_tools.append("milvus.search_smartkb")

        evidence = "\n\n".join(f"[{item['tool']}]\n{item['content']}" for item in observations)
        answer = self._compose(
            task,
            evidence,
            memory_context,
            "输出工程方案: 目标、模块划分、核心流程、测试方案、风险和可观测性。",
        )
        return WorkerResult(answer, observations, used_tools)


