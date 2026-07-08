# AgentFlow 多 Agent 协作平台

AgentFlow 是一个本地 Multi-Agent 协作平台，用于构建可观察、可扩展的 LLM 应用。项目基于 LangGraph 实现 Supervisor 路由，将任务分配给不同 Worker，并通过 MCP 风格工具层连接 PostgreSQL、Milvus、项目工具和图片生成能力，同时提供 FastAPI Web 界面。

## 项目背景

企业中的 AI 助手往往不只需要回答单个问题，还需要完成多步骤任务，例如检索资料、分析需求、设计方案、生成文档、调用数据库或外部工具。单一 Agent 在处理这类复杂任务时，容易出现职责不清、工具调用不可控、过程不可追踪等问题。

AgentFlow 面向这个问题构建一个可观察、可扩展的 Multi-Agent 协作平台：Supervisor 负责理解任务并选择合适的 Skill 和 Worker，不同 Worker 通过 MCP 风格工具层连接知识库、数据库、项目工具和图片生成能力，让复杂任务的执行过程可分工、可观察、可复盘。

## 功能特性

- 使用 LangGraph 构建 Supervisor 工作流。
- 支持四类 Worker：
  - `researcher`：检索知识、总结证据、指出来源。
  - `engineer`：设计架构、接口、实现步骤、测试方案和风险控制。
  - `writer`：生成 README、技术报告、摘要和发布说明。
  - `general`：处理通用任务和图片生成任务。
- 使用 SkillRegistry 管理技能触发词、输入 schema、输出格式、fallback route 和建议工具。
- 使用 ContextManager 管理历史记忆和工具观察，避免 prompt 无限膨胀。
- 提供 MCP 风格工具协议：`list_tools` 和 `call_tool`。
- 支持 PostgreSQL 短期记忆，数据库不可用时自动降级到内存记忆。
- 支持长期记忆，将任务经验沉淀为可检索摘要，并在后续任务中注入相关上下文。
- 支持调用 SmartKB 的 Milvus 知识库。
- 支持 Right Code 兼容的 `gpt-image-2` 图片生成。
- 支持执行轨迹持久化，记录路由、技能、工具调用、观察结果、最终回答、延迟和估算用量。
- 支持敏感信息脱敏和 Worker 工具调用次数限制。
- 提供 FastAPI Web 界面，展示聊天、指标、技能、工具、Trace 和服务状态。
- 支持 SSE（Server-Sent Events）流式输出。

## 系统架构

```text
User task
  -> memory loader
       -> short-term dialogue memory
       -> long-term task memory retrieval
  -> Supervisor + SkillRegistry
  -> researcher | engineer | writer | general
  -> ContextManager
  -> MCP-style tools
       -> PostgreSQL metadata and memory
       -> Milvus SmartKB retrieval
       -> project helper tools
       -> image generation
  -> final answer
  -> memory + trace storage
```

## 目录结构

```text
agentflow-multi-agent/
├── app.py
├── config.py
├── test_imports.py
├── test_e2e.py
├── agent_team/
│   ├── context.py
│   ├── long_term_memory.py
│   ├── memory.py
│   ├── safety.py
│   ├── skills.py
│   ├── supervisor.py
│   ├── tracing.py
│   └── workers/
├── docs/
├── eval/
├── models/
└── tools/
```

## 安装

```bash
git clone <your-repository-url>
cd agentflow-multi-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 配置

复制环境变量模板，并填写自己的模型、数据库、向量库和图片生成配置。

```bash
cp .env.example .env
```

常用配置示例：

```env
SMARTKB_PROJECT_DIR=../smartkb-rag
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DB_URL=your_postgresql_connection_string
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
COLLECTION_NAME=my_rag_collection
AGENTFLOW_MEMORY_TABLE=agentflow_memories
AGENTFLOW_LONG_TERM_MEMORY_TABLE=agentflow_long_term_memories
AGENTFLOW_LONG_TERM_MEMORY_LIMIT=4
IMAGE_API_KEY=your_image_api_key
IMAGE_API_BASE=https://www.right.codes/draw/v1
IMAGE_MODEL=gpt-image-2
```

请不要提交真实密钥。

## 运行

启动 FastAPI 应用：

```bash
uvicorn app:app --host 127.0.0.1 --port 8502
```

打开浏览器访问：

```text
http://127.0.0.1:8502
```

示例任务：

```text
检索 SmartKB 中关于混合检索和 RRF 的内容
设计一个 MCP server 的测试方案
写一段 AgentFlow README 摘要
生成一个多 Agent 协作架构图
```

## 测试

默认测试为离线测试，不依赖外部服务。

```bash
python test_imports.py
python test_e2e.py
python eval/agent_eval.py
```

配置 `.env` 后可以运行 live 检查：

```bash
python test_imports.py --live
python test_e2e.py --live
```

## 文档

- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `docs/MCP_TOOLS.md`
- `docs/PROJECT_REPORT.md`
