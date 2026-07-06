"""写手 Worker。"""

from agent_team.workers.base import BaseWorker, WorkerResult


class WriterWorker(BaseWorker):
    name = "writer"
    role_prompt = "你是文档 Agent，负责把技术内容整理成项目说明、技术报告和发布说明。"

    def run(self, task: str, memory_context: str = "") -> WorkerResult:
        self._reset_tool_budget()
        observations = []
        used_tools = []

        summary = self._call_tool("project", "project_summary", project=task)
        observations.append({"tool": "project.project_summary", "content": summary.content})
        used_tools.append("project.project_summary")

        checklist = self._call_tool(
            "project",
            "quality_checklist",
            stack="Python, LangChain, LangGraph, MCP, Agent, RAG, PostgreSQL, Milvus",
        )
        observations.append({"tool": "project.quality_checklist", "content": checklist.content})
        used_tools.append("project.quality_checklist")

        evidence = "\n\n".join(f"[{item['tool']}]\n{item['content']}" for item in observations)
        answer = self._compose(
            task,
            evidence,
            memory_context,
            "输出可直接用于项目文档的中文表达，结构清楚，避免夸大。",
        )
        return WorkerResult(answer, observations, used_tools)


