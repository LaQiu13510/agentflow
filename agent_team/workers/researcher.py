"""研究员 Worker。"""

from agent_team.workers.base import BaseWorker, WorkerResult


class ResearcherWorker(BaseWorker):
    name = "researcher"
    role_prompt = "你是研究员 Agent，负责检索资料、提炼证据和指出来源。"

    def run(self, task: str, memory_context: str = "") -> WorkerResult:
        self._reset_tool_budget()
        observations = []
        used_tools = []

        docs = self._call_tool("postgres", "list_smartkb_documents", limit=8)
        observations.append({"tool": "postgres.list_smartkb_documents", "content": docs.content})
        used_tools.append("postgres.list_smartkb_documents")

        kb = self._call_tool("milvus", "search_smartkb", query=task, top_k=4)
        observations.append({"tool": "milvus.search_smartkb", "content": kb.content})
        used_tools.append("milvus.search_smartkb")

        evidence = "\n\n".join(f"[{item['tool']}]\n{item['content']}" for item in observations)
        answer = self._compose(
            task,
            evidence,
            memory_context,
            "用中文给出研究结论。先列关键发现，再列证据来源，最后给下一步建议。",
        )
        return WorkerResult(answer, observations, used_tools)
