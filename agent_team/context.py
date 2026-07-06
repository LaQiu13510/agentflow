"""Context management for AgentFlow workers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextPacket:
    """Compact context passed from Supervisor to Workers."""

    task: str
    memory_context: str = ""
    tool_observations: list[dict] = field(default_factory=list)


class ContextManager:
    """Build and trim worker context within simple character budgets."""

    def __init__(
        self,
        memory_budget: int = 1200,
        evidence_budget: int = 2400,
    ):
        self.memory_budget = memory_budget
        self.evidence_budget = evidence_budget

    def trim_memory(self, memory_context: str) -> str:
        """Keep the most recent memory text within budget."""
        return self._trim_from_end(memory_context or "暂无历史记忆。", self.memory_budget)

    def format_observations(self, observations: list[dict]) -> str:
        """Render tool observations for prompts."""
        if not observations:
            return "暂无工具观察。"
        parts = []
        for item in observations:
            tool = item.get("tool", "tool")
            content = str(item.get("content", ""))
            parts.append(f"[{tool}]\n{content}")
        return self._trim_from_end("\n\n".join(parts), self.evidence_budget)

    def build_worker_prompt(
        self,
        packet: ContextPacket,
        style: str,
    ) -> str:
        """Build the final prompt body used by worker LLM calls."""
        memory = self.trim_memory(packet.memory_context)
        evidence = self.format_observations(packet.tool_observations)
        return f"""用户任务:
{packet.task}

历史记忆:
{memory}

工具观察:
{evidence}

请按以下风格输出:
{style}
"""

    def _trim_from_end(self, text: str, budget: int) -> str:
        text = text.strip()
        if len(text) <= budget:
            return text
        return "..." + text[-budget:]


_context_manager: ContextManager | None = None


def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
