"""Reusable skill registry for AgentFlow routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Skill:
    """A reusable task capability exposed to the Supervisor."""

    name: str
    route: str
    description: str
    triggers: tuple[str, ...]
    suggested_tools: tuple[str, ...] = field(default_factory=tuple)
    input_schema: dict = field(default_factory=dict)
    output_format: str = ""
    fallback_route: str = "general"
    worker_detail: str = ""


class SkillRegistry:
    """Keyword-based skill matcher used before LLM routing fallback."""

    def __init__(self, skills: list[Skill]):
        self.skills = skills

    def match(self, task: str) -> Skill | None:
        text = task.lower()
        best: tuple[int, Skill] | None = None
        for skill in self.skills:
            score = sum(1 for trigger in skill.triggers if trigger.lower() in text)
            if score and (best is None or score > best[0]):
                best = (score, skill)
        return best[1] if best else None

    def list_skills(self) -> list[dict]:
        return [
            {
                "name": skill.name,
                "route": skill.route,
                "description": skill.description,
                "triggers": list(skill.triggers),
                "suggested_tools": list(skill.suggested_tools),
                "input_schema": skill.input_schema,
                "output_format": skill.output_format,
                "fallback_route": skill.fallback_route,
                "worker_detail": skill.worker_detail,
            }
            for skill in self.skills
        ]


def build_default_skill_registry() -> SkillRegistry:
    return SkillRegistry(
        [
            Skill(
                name="image_generation",
                route="general",
                description="Generate images, architecture diagrams, flow diagrams, and visual sketches from text prompts.",
                triggers=("生图", "图片", "图像", "架构图", "流程图", "示意图", "image", "generate image", "gpt-image"),
                suggested_tools=("image.generate_image",),
                input_schema={"prompt": "required string", "size": "optional image size, default 1024x1024"},
                output_format="Return the generated image URL or local image path with a concise status note.",
                fallback_route="general",
                worker_detail=(
                    "General worker handles image generation tasks by calling the image MCP server, "
                    "then reports the generated URL/path without pretending to inspect the image."
                ),
            ),
            Skill(
                name="knowledge_research",
                route="researcher",
                description="Retrieve and summarize evidence from SmartKB, Milvus, and PostgreSQL metadata.",
                triggers=("检索", "知识库", "资料", "rag", "milvus", "postgresql"),
                suggested_tools=("postgres.list_smartkb_documents", "milvus.search_smartkb"),
                input_schema={"task": "research question or evidence request"},
                output_format="Key findings, evidence sources, and recommended next steps.",
                fallback_route="general",
                worker_detail=(
                    "Researcher worker lists available SmartKB documents, searches vector knowledge, "
                    "summarizes retrieved evidence, and calls out source files."
                ),
            ),
            Skill(
                name="engineering_design",
                route="engineer",
                description="Plan architecture, interfaces, implementation, debugging, deployment, and tests.",
                triggers=("架构", "实现", "代码", "接口", "测试", "bug", "部署", "mcp server"),
                suggested_tools=("project.engineering_requirements", "milvus.search_smartkb"),
                input_schema={"task": "engineering problem, feature request, bug, or test plan"},
                output_format="Goal, modules, flow, tests, risks, and observability.",
                fallback_route="general",
                worker_detail=(
                    "Engineer worker turns requirements into modules, APIs, implementation steps, "
                    "test strategy, operational risks, and observability checkpoints."
                ),
            ),
            Skill(
                name="technical_writing",
                route="writer",
                description="Produce project documentation, reports, summaries, release notes, and polished technical text.",
                triggers=("文档", "报告", "摘要", "润色", "说明", "readme", "release", "pitch"),
                suggested_tools=("project.project_summary", "project.quality_checklist"),
                input_schema={"task": "documentation or writing request"},
                output_format="Clear Chinese technical writing with concise structure and no exaggeration.",
                fallback_route="general",
                worker_detail=(
                    "Writer worker converts technical facts into readable README text, reports, summaries, "
                    "release notes, and interview-friendly project descriptions."
                ),
            ),
        ]
    )


_skill_registry: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = build_default_skill_registry()
    return _skill_registry
