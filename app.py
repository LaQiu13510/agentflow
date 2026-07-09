"""AgentFlow FastAPI application."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from agent_team.long_term_memory import get_long_term_memory
from agent_team.memory import get_memory
from agent_team.runtime_state import get_runtime_state
from agent_team.safety import get_safety_controller
from agent_team.skills import get_skill_registry
from agent_team.supervisor import build_agent_team
from agent_team.tracing import get_trace_store
from config import DEFAULT_SESSION_ID
from models.embedding import get_embedding_model
from models.llm import get_llm
from tools.registry import build_tool_registry


app = FastAPI(
    title="AgentFlow",
    description="Supervisor + Worker Agents + MCP-style tools",
    version="1.0.0",
)


EXAMPLES = [
    "检索 SmartKB 中关于混合检索和 RRF 的内容，并输出技术摘要。",
    "设计一个 MCP Server，用来查询 PostgreSQL 里的对话记忆。",
    "写一段 AgentFlow README 摘要。",
    "生成一个多 Agent 协作架构图。",
]


class RunRequest(BaseModel):
    task: str = Field(..., min_length=1, max_length=4000)
    session_id: str = DEFAULT_SESSION_ID


@lru_cache(maxsize=1)
def get_registry_cached():
    return build_tool_registry()


@lru_cache(maxsize=1)
def get_graph_cached():
    return build_agent_team(get_registry_cached())


def memory_stats() -> dict[str, Any]:
    try:
        short_term = get_memory().stats()
    except Exception as exc:
        short_term = {
            "backend": "unavailable",
            "total_messages": 0,
            "sessions": 0,
            "error": str(exc)[:180],
        }
    try:
        long_term = get_long_term_memory().stats()
    except Exception as exc:
        long_term = {
            "backend": "unavailable",
            "total_memories": 0,
            "sessions": 0,
            "error": str(exc)[:180],
        }
    return {
        "backend": f"短期:{short_term.get('backend', '-')} 长期:{long_term.get('backend', '-')}",
        "total_messages": short_term.get("total_messages", 0),
        "long_term_memories": long_term.get("total_memories", 0),
        "sessions": max(short_term.get("sessions", 0), long_term.get("sessions", 0)),
        "short_term": short_term,
        "long_term": long_term,
    }


def runtime_stats() -> dict[str, Any]:
    try:
        return get_runtime_state().stats()
    except Exception as exc:
        return {"backend": "unavailable", "error": str(exc)[:180]}


def service_checks() -> list[dict[str, Any]]:
    checks = []
    checks.append(_safe_check("DeepSeek", lambda: get_llm(max_tokens=64).test_connection()))
    checks.append(_safe_check("Embedding", lambda: get_embedding_model().test_connection()))
    checks.append(_safe_tool_check("PostgreSQL", "postgres", "health"))
    checks.append(_safe_tool_check("Milvus", "milvus", "health"))
    checks.append(_safe_tool_check("Image", "image", "health"))
    checks.append(_safe_check("Runtime", lambda: (True, runtime_stats())))
    return checks


def _safe_check(label: str, fn) -> dict[str, Any]:
    try:
        ok, detail = fn()
        return {"label": label, "ok": bool(ok), "detail": str(detail)}
    except Exception as exc:
        return {"label": label, "ok": False, "detail": str(exc)[:180]}


def _safe_tool_check(label: str, server: str, tool: str) -> dict[str, Any]:
    result = get_registry_cached().call(server, tool)
    return {"label": label, "ok": result.success, "detail": result.content}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "examples": EXAMPLES,
        "tools": get_registry_cached().list_tools(),
        "skills": get_skill_registry().list_skills(),
        "memory": memory_stats(),
        "runtime": runtime_stats(),
        "traces": get_trace_store().recent(limit=8),
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"checks": service_checks(), "memory": memory_stats(), "runtime": runtime_stats()}


@app.get("/api/memory")
def memory() -> dict[str, Any]:
    return {"memory": memory_stats()}


@app.get("/api/runtime")
def runtime() -> dict[str, Any]:
    store = get_runtime_state()
    return {"runtime": runtime_stats(), "tasks": store.recent_tasks(limit=12)}


@app.get("/api/tasks/{task_id}")
def task_status(task_id: str) -> dict[str, Any]:
    task = get_runtime_state().get_task(task_id)
    return {"ok": bool(task), "task": task or {}}


@app.get("/api/traces")
def traces(limit: int = 8) -> dict[str, Any]:
    return {"traces": get_trace_store().recent(limit=limit)}


@app.post("/api/run")
def run_agent(request: RunRequest) -> dict[str, Any]:
    session_id = request.session_id or DEFAULT_SESSION_ID
    runtime_store = get_runtime_state()
    safety = get_safety_controller()
    runtime_store.touch_session(
        session_id,
        {"status": "checking", "last_task": request.task[:300]},
    )

    rate_limit = runtime_store.check_rate_limit(session_id)
    if not rate_limit.get("allowed", True):
        runtime_store.touch_session(session_id, {"status": "rate_limited"})
        return blocked_run_response(
            request,
            session_id,
            "请求过快，请稍后再试。",
            {"rate_limit": rate_limit, "state": runtime_stats()},
        )

    prompt_budget = runtime_store.add_budget_units(
        session_id,
        safety.estimate_units(request.task),
        category="prompt",
    )
    if not prompt_budget.get("allowed", True):
        runtime_store.touch_session(session_id, {"status": "budget_blocked"})
        return blocked_run_response(
            request,
            session_id,
            "今日预算已达到上限，请明天再试或调高预算配置。",
            {"rate_limit": rate_limit, "budget": prompt_budget, "state": runtime_stats()},
        )

    task_record = runtime_store.create_task(session_id, request.task)
    task_id = task_record["task_id"]
    runtime_store.update_task(task_id, "running")

    graph = get_graph_cached()
    try:
        state = graph.invoke(
            {
                "messages": [HumanMessage(content=request.task)],
                "session_id": session_id,
                "task_id": task_id,
            }
        )
    except Exception as exc:
        runtime_store.update_task(task_id, "failed", error=str(exc)[:300])
        runtime_store.touch_session(session_id, {"status": "failed", "last_task_id": task_id})
        raise

    final_answer = state.get("final_answer") or state.get("worker_output", "")
    answer_budget = runtime_store.add_budget_units(
        session_id,
        safety.estimate_units(final_answer),
        category="answer",
    )
    task_record = runtime_store.update_task(
        task_id,
        "done",
        route=state.get("route", "-"),
        route_reason=state.get("route_reason", "-"),
        latency_ms=state.get("latency_ms", 0),
        used_tools=state.get("used_tools", []),
        budget=answer_budget,
    )
    runtime_store.touch_session(
        session_id,
        {"status": "idle", "last_task_id": task_id, "last_route": state.get("route", "-")},
    )

    return {
        "ok": True,
        "session_id": state.get("session_id", request.session_id),
        "task_id": task_id,
        "task": state.get("task", request.task),
        "route": state.get("route", "-"),
        "route_reason": state.get("route_reason", "-"),
        "skill_name": state.get("skill_name", ""),
        "used_tools": state.get("used_tools", []),
        "observations": state.get("observations", []),
        "final_answer": final_answer,
        "latency_ms": state.get("latency_ms", 0),
        "memory": memory_stats(),
        "trace": state.get("trace_record", {}),
        "runtime": {
            "task": task_record,
            "rate_limit": rate_limit,
            "budget": answer_budget,
            "tool_stats": runtime_store.get_tool_stats(session_id),
            "state": runtime_stats(),
        },
    }


def blocked_run_response(
    request: RunRequest,
    session_id: str,
    message: str,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ok": False,
        "session_id": session_id,
        "task": request.task,
        "route": "-",
        "route_reason": "runtime_guard",
        "skill_name": "",
        "used_tools": [],
        "observations": [],
        "final_answer": message,
        "latency_ms": 0,
        "memory": memory_stats(),
        "trace": {},
        "runtime": runtime,
    }


@app.get("/api/run/stream")
def run_agent_stream(task: str, session_id: str = DEFAULT_SESSION_ID) -> StreamingResponse:
    def generate():
        yield sse("stage", {"message": "Supervisor 正在分析任务"})
        try:
            result = run_agent(RunRequest(task=task, session_id=session_id))
            yield sse("stage", {"message": f"已路由到 {result.get('route', '-')}"})
            answer = result.get("final_answer", "")
            for chunk in chunk_text(answer):
                yield sse("delta", {"text": chunk})
            yield sse("final", result)
        except Exception as exc:
            yield sse("app_error", {"message": str(exc)[:500]})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = 28):
    for index in range(0, len(text), size):
        yield text[index:index + size]


INDEX_HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AgentFlow</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --muted: #667085;
      --text: #101828;
      --accent: #2563eb;
      --ok: #087443;
      --warn: #b54708;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    button, textarea, input { font: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(246, 247, 249, 0.96);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.04);
    }
    .top-inner {
      max-width: 1440px;
      margin: 0 auto;
      padding: 14px 20px;
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 3fr;
      gap: 16px;
      align-items: center;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      line-height: 1.2;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 3px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 9px 10px;
      min-width: 0;
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
    }
    .metric-value {
      font-weight: 650;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px 20px 32px;
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(360px, 0.9fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .main-panel { padding: 16px; }
    .workspace-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 14px;
    }
    .workspace-title h2 {
      font-size: 18px;
      margin: 0;
      letter-spacing: 0;
    }
    .workspace-title p {
      margin: 4px 0 0;
      color: var(--muted);
      font-size: 13px;
    }
    .kbd {
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 4px 7px;
      color: var(--muted);
      background: #f9fafb;
      font-size: 12px;
      white-space: nowrap;
    }
    .run-state {
      border: 1px solid #bfdbfe;
      background: #eff6ff;
      color: #1d4ed8;
      border-radius: 8px;
      padding: 9px 11px;
      margin-bottom: 12px;
      font-size: 13px;
    }
    .run-state.busy {
      border-color: #fed7aa;
      background: #fff7ed;
      color: #c2410c;
    }
    .run-state.done {
      border-color: #bbf7d0;
      background: #f0fdf4;
      color: #047857;
    }
    .inspector {
      position: sticky;
      top: 96px;
      max-height: calc(100vh - 116px);
      overflow: auto;
      padding: 16px;
    }
    .section-title {
      font-weight: 700;
      margin: 0 0 10px;
    }
    .session-row {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    .session-row label {
      color: var(--muted);
      font-size: 13px;
    }
    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: #fff;
      color: var(--text);
      outline: none;
    }
    input:focus, textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
    }
    textarea {
      min-height: 120px;
      resize: vertical;
    }
    .examples {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .btn {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 9px 11px;
      cursor: pointer;
      text-align: left;
    }
    .btn:hover { border-color: var(--accent); }
    .primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      text-align: center;
      font-weight: 650;
    }
    .ghost { color: var(--muted); }
    .chat {
      display: flex;
      flex-direction: column;
      gap: 10px;
      margin-top: 14px;
    }
    .msg {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .msg.user {
      background: #f8fbff;
      border-color: #c7d7fe;
    }
    .msg.assistant {
      background: #fff;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 8px 0 12px;
    }
    .chip {
      display: inline-flex;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: #344054;
      background: #f9fafb;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .trace-block {
      border-top: 1px solid var(--line);
      padding-top: 12px;
      margin-top: 12px;
    }
    .stage-list {
      display: grid;
      gap: 6px;
      margin: 8px 0 12px;
    }
    .stage {
      display: grid;
      grid-template-columns: 18px minmax(0, 1fr);
      gap: 8px;
      align-items: center;
      color: #344054;
      font-size: 13px;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--line);
      justify-self: center;
    }
    .stage.done .dot { background: var(--ok); }
    .kv {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 8px;
      font-size: 14px;
      margin: 6px 0;
    }
    .k { color: var(--muted); }
    .v { overflow-wrap: anywhere; }
    details {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      margin: 8px 0;
      background: #fff;
    }
    summary { cursor: pointer; font-weight: 650; }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      color: #344054;
    }
    .status-ok { color: var(--ok); }
    .status-bad { color: var(--bad); }
    .muted { color: var(--muted); }
    .loading {
      opacity: 0.65;
      pointer-events: none;
    }
    @media (max-width: 980px) {
      .top-inner, .shell {
        grid-template-columns: 1fr;
      }
      .metrics {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .inspector {
        position: static;
        max-height: none;
      }
      .examples {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="top-inner">
      <div>
        <h1>AgentFlow</h1>
        <div class="subtitle">Supervisor · Workers · Skills · MCP Tools</div>
      </div>
      <div class="metrics">
        <div class="metric"><div class="metric-label">最近路由</div><div class="metric-value" id="metric-route">-</div></div>
        <div class="metric"><div class="metric-label">延迟</div><div class="metric-value" id="metric-latency">-</div></div>
        <div class="metric"><div class="metric-label">工具调用</div><div class="metric-value" id="metric-tools">0</div></div>
        <div class="metric"><div class="metric-label">记忆</div><div class="metric-value" id="metric-memory">-</div></div>
      </div>
    </div>
  </header>

  <main class="shell">
    <section class="panel main-panel">
      <div class="workspace-title">
        <div>
          <h2>任务控制台</h2>
          <p>输入任务后由 Supervisor 选择 Skill、Worker 和工具链。</p>
        </div>
        <div class="kbd">Ctrl / ⌘ + Enter</div>
      </div>
      <div class="run-state" id="run-state">Ready · 等待任务输入</div>
      <div class="session-row">
        <label for="session">Session ID</label>
        <input id="session" />
        <button class="btn" id="health-btn">检查连接</button>
      </div>
      <div class="section-title">协作任务</div>
      <textarea id="task" placeholder="输入一个任务，让 Supervisor 自动分配给合适的 Worker"></textarea>
      <div class="examples" id="examples"></div>
      <button class="btn primary" id="run-btn">运行 AgentFlow</button>
      <div class="chat" id="chat"></div>
    </section>

    <aside class="panel inspector">
      <div class="section-title">执行轨迹</div>
      <div id="trace-empty" class="msg assistant muted">运行一次任务后，这里会展示 Supervisor 路由、工具调用和观察结果。</div>
      <div id="trace"></div>
      <div class="trace-block">
        <div class="section-title">Skills</div>
        <div id="skills"></div>
      </div>
      <div class="trace-block">
        <div class="section-title">MCP Tools</div>
        <div id="tools"></div>
      </div>
      <div class="trace-block">
        <div class="section-title">最近 Trace</div>
        <div id="recent-traces"></div>
      </div>
      <div class="trace-block">
        <div class="section-title">服务状态</div>
        <div id="health"></div>
      </div>
    </aside>
  </main>

  <script>
    const sessionInput = document.getElementById("session");
    const taskInput = document.getElementById("task");
    const runBtn = document.getElementById("run-btn");
    const healthBtn = document.getElementById("health-btn");
    const chat = document.getElementById("chat");
    const sessionId = localStorage.getItem("agentflow_session") || "agentflow-demo";
    sessionInput.value = sessionId;

    sessionInput.addEventListener("input", () => {
      localStorage.setItem("agentflow_session", sessionInput.value || "agentflow-demo");
    });

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function setMetric(id, value) {
      document.getElementById(id).textContent = value;
    }

    function addMessage(role, content) {
      const div = document.createElement("div");
      div.className = `msg ${role}`;
      div.textContent = content;
      chat.prepend(div);
      return div;
    }

    async function loadMeta() {
      const response = await fetch("/api/meta");
      const data = await response.json();
      renderExamples(data.examples || []);
      renderSkills(data.skills || []);
      renderTools(data.tools || []);
      renderTraces(data.traces || []);
      updateMemoryMetric(data.memory || {});
    }

    function renderExamples(examples) {
      const box = document.getElementById("examples");
      box.innerHTML = "";
      examples.forEach(example => {
        const btn = document.createElement("button");
        btn.className = "btn";
        btn.textContent = example;
        btn.onclick = () => { taskInput.value = example; };
        box.appendChild(btn);
      });
    }

    function renderSkills(skills) {
      document.getElementById("skills").innerHTML = skills.map(item => `
        <details>
          <summary>${escapeHtml(item.name)} -> ${escapeHtml(item.route)}</summary>
          <div class="kv"><div class="k">说明</div><div class="v">${escapeHtml(item.description)}</div></div>
          <div class="kv"><div class="k">工具</div><div class="v">${escapeHtml((item.suggested_tools || []).join(", ") || "none")}</div></div>
          <div class="kv"><div class="k">职责</div><div class="v">${escapeHtml(item.worker_detail || "")}</div></div>
        </details>
      `).join("");
    }

    function renderTools(tools) {
      document.getElementById("tools").innerHTML = tools.map(item =>
        `<span class="chip">${escapeHtml(item.server)}.${escapeHtml(item.name)}</span>`
      ).join("");
    }

    function renderTrace(result) {
      document.getElementById("trace-empty").style.display = "none";
      setMetric("metric-route", result.route || "-");
      setMetric("metric-latency", `${result.latency_ms || 0} ms`);
      setMetric("metric-tools", (result.used_tools || []).length);
      updateMemoryMetric(result.memory || {});
      const budget = (result.runtime || {}).budget || {};
      const budgetText = budget.limit
        ? `${budget.used ?? 0}/${budget.limit}`
        : `${budget.used ?? 0}`;

      const tools = (result.used_tools || []).map(tool =>
        `<span class="chip">${escapeHtml(tool)}</span>`
      ).join("");
      const observations = (result.observations || []).map(item => {
        const metadata = item.metadata || {};
        const image = metadata.image_url
          ? `<p><a href="${escapeHtml(metadata.image_url)}" target="_blank">打开生成图片</a></p>`
          : metadata.image_path
            ? `<p class="muted">图片路径: ${escapeHtml(metadata.image_path)}</p>`
            : "";
        return `
          <details>
            <summary>${escapeHtml(item.tool || "tool observation")}</summary>
            <pre>${escapeHtml(item.content || "")}</pre>
            ${image}
          </details>
        `;
      }).join("");

      document.getElementById("trace").innerHTML = `
        <div class="stage-list">
          <div class="stage done"><span class="dot"></span><span>Load memory</span></div>
          <div class="stage done"><span class="dot"></span><span>Skill routing · ${escapeHtml(result.route_reason || "-")}</span></div>
          <div class="stage done"><span class="dot"></span><span>Worker · ${escapeHtml(result.route || "-")}</span></div>
          <div class="stage done"><span class="dot"></span><span>Tool calls · ${(result.used_tools || []).length}</span></div>
          <div class="stage done"><span class="dot"></span><span>Trace persisted</span></div>
        </div>
        <div class="kv"><div class="k">Task ID</div><div class="v">${escapeHtml(result.task_id || "-")}</div></div>
        <div class="kv"><div class="k">路由</div><div class="v">${escapeHtml(result.route || "-")}</div></div>
        <div class="kv"><div class="k">Skill</div><div class="v">${escapeHtml(result.route_reason || "-")}</div></div>
        <div class="kv"><div class="k">预算</div><div class="v">${escapeHtml(budgetText)}</div></div>
        <div class="kv"><div class="k">工具</div><div class="v"><div class="chips">${tools || "<span class='muted'>无</span>"}</div></div></div>
        ${observations}
      `;
    }

    function updateMemoryMetric(memory) {
      const total = memory.total_messages ?? 0;
      const longTerm = memory.long_term_memories ?? 0;
      const backend = memory.backend || "-";
      setMetric("metric-memory", `短 ${total} · 长 ${longTerm} / ${backend}`);
    }

    function renderTraces(traces) {
      document.getElementById("recent-traces").innerHTML = traces.slice().reverse().map(row => `
        <details>
          <summary>${escapeHtml(row.timestamp || "")} · ${escapeHtml(row.route || "-")}</summary>
          <div class="kv"><div class="k">任务</div><div class="v">${escapeHtml(row.task || "")}</div></div>
          <div class="kv"><div class="k">Skill</div><div class="v">${escapeHtml(row.route_reason || "-")}</div></div>
          <pre>${escapeHtml((row.final_answer || "").slice(0, 800))}</pre>
        </details>
      `).join("") || "<div class='muted'>暂无 Trace 记录。</div>";
    }

    async function loadHealth() {
      const response = await fetch("/api/health");
      const data = await response.json();
      document.getElementById("health").innerHTML = (data.checks || []).map(item => `
        <div class="kv">
          <div class="k">${escapeHtml(item.label)}</div>
          <div class="v ${item.ok ? "status-ok" : "status-bad"}">${item.ok ? "正常" : "异常"} · ${escapeHtml(item.detail)}</div>
        </div>
      `).join("");
      updateMemoryMetric(data.memory || {});
    }

    async function runAgent() {
      const task = taskInput.value.trim();
      if (!task) return;
      document.body.classList.add("loading");
      const runState = document.getElementById("run-state");
      runState.className = "run-state busy";
      runState.textContent = "Running · Supervisor 正在路由并调用工具";
      addMessage("user", task);
      const assistant = addMessage("assistant", "");
      const params = new URLSearchParams({
        task,
        session_id: sessionInput.value || "agentflow-demo"
      });
      const events = new EventSource(`/api/run/stream?${params.toString()}`);
      let streamDone = false;
      events.addEventListener("stage", event => {
        const data = JSON.parse(event.data);
        runState.textContent = `Running · ${data.message || "处理中"}`;
      });
      events.addEventListener("delta", event => {
        const data = JSON.parse(event.data);
        assistant.textContent += data.text || "";
      });
      events.addEventListener("final", async event => {
        streamDone = true;
        const result = JSON.parse(event.data);
        renderTrace(result);
        runState.className = result.ok === false ? "run-state" : "run-state done";
        runState.textContent = result.ok === false
          ? `Blocked · ${result.route_reason || "runtime_guard"}`
          : `Done · ${result.route || "-"} · ${result.latency_ms || 0} ms`;
        const traces = await fetch("/api/traces").then(r => r.json());
        renderTraces(traces.traces || []);
        events.close();
        document.body.classList.remove("loading");
      });
      events.addEventListener("app_error", event => {
        let message = "流式连接失败";
        try { message = JSON.parse(event.data).message || message; } catch {}
        assistant.textContent = `运行失败: ${message}`;
        runState.className = "run-state";
        runState.textContent = `Failed · ${message}`;
        events.close();
        document.body.classList.remove("loading");
      });
      events.onerror = () => {
        if (streamDone) return;
        assistant.textContent = "运行失败: 流式连接失败";
        runState.className = "run-state";
        runState.textContent = "Failed · 流式连接失败";
        events.close();
        document.body.classList.remove("loading");
      };
    }

    runBtn.addEventListener("click", runAgent);
    healthBtn.addEventListener("click", loadHealth);
    taskInput.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
        runAgent();
      }
    });

    loadMeta();
  </script>
</body>
</html>
"""
