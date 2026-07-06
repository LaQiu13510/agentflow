"""通用 Worker。"""

from agent_team.workers.base import BaseWorker, WorkerResult


class GeneralWorker(BaseWorker):
    name = "general"
    role_prompt = "你是协调型 Agent，负责综合回答用户问题。"

    def run(self, task: str, memory_context: str = "") -> WorkerResult:
        self._reset_tool_budget()
        observations = []
        used_tools = []

        if self._is_image_task(task):
            result = self._call_tool("image", "generate_image", prompt=task, size="1024x1024")
            observations.append({"tool": "image.generate_image", "content": result.content, "metadata": result.metadata})
            used_tools.append("image.generate_image")
            evidence = result.content
            answer = self._compose(
                task,
                evidence,
                memory_context,
                "简洁说明图片生成结果。若工具返回本地路径或 URL，请原样列出，方便用户打开或下载。",
            )
            return WorkerResult(answer, observations, used_tools)

        req = self._call_tool("project", "engineering_requirements")
        observations.append({"tool": "project.engineering_requirements", "content": req.content})
        used_tools.append("project.engineering_requirements")

        answer = self._compose(
            task,
            req.content,
            memory_context,
            "给出简洁可执行的回答。必要时说明可以继续交给 researcher、engineer 或 writer。",
        )
        return WorkerResult(answer, observations, used_tools)

    def _is_image_task(self, task: str) -> bool:
        text = task.lower()
        return any(
            keyword in text
            for keyword in ["生图", "图片", "图像", "架构图", "流程图", "示意图", "image", "generate image", "gpt-image"]
        )



