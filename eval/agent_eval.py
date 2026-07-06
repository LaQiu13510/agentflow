"""Offline agent routing and skill evaluation for AgentFlow."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_team.skills import get_skill_registry
from agent_team.supervisor import SupervisorAgent


CASES = [
    {"task": "检索知识库中关于 RAG 和 RRF 的资料", "expected_route": "researcher", "expected_skill": "knowledge_research"},
    {"task": "设计一个 MCP server 的测试方案", "expected_route": "engineer", "expected_skill": "engineering_design"},
    {"task": "写一段 AgentFlow README 摘要", "expected_route": "writer", "expected_skill": "technical_writing"},
    {"task": "生成一张图片，展示多 Agent 协作流程", "expected_route": "general", "expected_skill": "image_generation"},
    {"task": "生成一个架构图", "expected_route": "general", "expected_skill": "image_generation"},
]


def run_eval() -> dict:
    registry = get_skill_registry()
    details = []
    for case in CASES:
        skill = registry.match(case["task"])
        actual_route = SupervisorAgent._keyword_route(None, case["task"])
        actual_skill = skill.name if skill else ""
        details.append(
            {
                "task": case["task"],
                "expected_route": case["expected_route"],
                "actual_route": actual_route,
                "expected_skill": case["expected_skill"],
                "actual_skill": actual_skill,
                "passed": actual_route == case["expected_route"] and actual_skill == case["expected_skill"],
            }
        )

    passed = sum(1 for item in details if item["passed"])
    total = len(details)
    skills = registry.list_skills()
    schema_ready = sum(
        1
        for skill in skills
        if skill.get("input_schema") and skill.get("output_format") and skill.get("worker_detail")
    )
    return {
        "project": "agentflow-multi-agent",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "total": total,
        "passed": passed,
        "route_accuracy": round(passed / total, 4) if total else 0,
        "skill_schema_coverage": round(schema_ready / len(skills), 4) if skills else 0,
        "skill_count": len(skills),
        "details": details,
    }


def main():
    report = run_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
