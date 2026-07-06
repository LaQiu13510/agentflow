# AgentFlow

AgentFlow is a local multi-agent orchestration platform for LLM applications. It uses LangGraph to route tasks through a Supervisor, delegates work to specialized workers, and exposes PostgreSQL, Milvus, project utilities, and image generation through MCP-style tools.

## Features

- Route tasks with a LangGraph Supervisor workflow.
- Use four worker roles:
  - `researcher`: retrieves knowledge, summarizes evidence, and cites sources.
  - `engineer`: plans architecture, interfaces, implementation steps, tests, and risks.
  - `writer`: produces documentation, reports, summaries, and release-style text.
  - `general`: handles lightweight coordination and image generation tasks.
- Register reusable skills with triggers, input schemas, output formats, fallback routes, and suggested tools.
- Manage worker context with memory trimming and tool-observation formatting.
- Expose local MCP-style tools with `list_tools` and `call_tool`.
- Store short-term memory in PostgreSQL, with an in-memory fallback for offline demos and tests.
- Query the SmartKB Milvus collection from the companion RAG project.
- Generate images through a Right Code-compatible `gpt-image-2` image API.
- Persist execution traces with routes, skills, tool calls, observations, final answers, latency, and estimated usage.
- Redact common secrets from prompts and traces, and limit per-worker tool calls.

## Architecture

```text
User task
  -> memory loader
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

## Project Structure

```text
agentflow-multi-agent/
├── app.py
├── config.py
├── test_imports.py
├── test_e2e.py
├── agent_team/
│   ├── context.py
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

## Installation

```bash
git clone <your-repository-url>
cd agentflow-multi-agent
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and fill in your own credentials.

```bash
cp .env.example .env
```

Common configuration values:

```env
SMARTKB_PROJECT_DIR=../smartkb-rag
DEEPSEEK_API_KEY=your_deepseek_key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
DB_URL=your_postgresql_connection_string
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
COLLECTION_NAME=my_rag_collection
IMAGE_API_KEY=your_image_api_key
IMAGE_API_BASE=https://www.right.codes/draw/v1
IMAGE_MODEL=gpt-image-2
```

Do not commit real credentials.

## Usage

Run the Streamlit app:

```bash
streamlit run app.py --server.port 8502
```

Open the local URL shown by Streamlit and enter a task. The UI shows route decisions, skill selection, tool calls, observations, latency, memory status, and recent traces.

Example tasks:

```text
检索 SmartKB 中关于混合检索和 RRF 的内容
设计一个 MCP server 的测试方案
写一段 AgentFlow README 摘要
生成一个多 Agent 协作架构图
```

## Tests

The default tests are offline and do not require external services.

```bash
python test_imports.py
python test_e2e.py
python eval/agent_eval.py
```

Live service checks can be run after `.env` is configured:

```bash
python test_imports.py --live
python test_e2e.py --live
```

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `docs/MCP_TOOLS.md`
- `docs/PROJECT_REPORT.md`
