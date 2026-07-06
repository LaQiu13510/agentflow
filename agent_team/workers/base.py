"""Worker Agent 基类。"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from agent_team.context import ContextPacket, get_context_manager
from agent_team.safety import get_safety_controller
from config import MAX_TOOL_CALLS_PER_WORKER
from models.llm import get_llm
from tools.mcp_base import ToolRegistry, ToolResult


@dataclass
class WorkerResult:
    content: str
    observations: list[dict]
    used_tools: list[str]


class BaseWorker:
    name = "base"
    role_prompt = "你是一个通用 AI 助手。"

    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self.llm = get_llm()
        self.context_manager = get_context_manager()
        self.safety = get_safety_controller()
        self._tool_calls = 0

    def run(self, task: str, memory_context: str = "") -> WorkerResult:
        raise NotImplementedError

    def _compose(
        self,
        task: str,
        evidence: str,
        memory_context: str,
        style: str,
    ) -> str:
        prompt = self.context_manager.build_worker_prompt(
            ContextPacket(
                task=task,
                memory_context=memory_context,
                tool_observations=[{"tool": "worker.evidence", "content": evidence}],
            ),
            style=style,
        )
        prompt = self.safety.redact(prompt)
        return self.llm.chat(
            [
                SystemMessage(content=self.role_prompt),
                HumanMessage(content=prompt),
            ],
            temperature=0.2,
        )

    def _reset_tool_budget(self):
        self._tool_calls = 0

    def _call_tool(self, server_name: str, tool_name: str, **kwargs) -> ToolResult:
        self._tool_calls += 1
        if self._tool_calls > MAX_TOOL_CALLS_PER_WORKER:
            return ToolResult(
                False,
                f"工具调用超过上限: {MAX_TOOL_CALLS_PER_WORKER}",
                {"max_tool_calls": MAX_TOOL_CALLS_PER_WORKER},
            )
        result = self.tools.call(server_name, tool_name, **kwargs)
        result.content = self.safety.redact(result.content)
        return result
