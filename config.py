"""
AgentFlow 全局配置
==================
AgentFlow 默认复用同级 SmartKB 项目的 .env，避免重复维护 API Key 和数据库地址。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).parent
DEFAULT_SMARTKB_DIR = ROOT_DIR.parent / "smartkb-rag"
SMARTKB_PROJECT_DIR = Path(os.getenv("SMARTKB_PROJECT_DIR", str(DEFAULT_SMARTKB_DIR)))
DATA_DIR = ROOT_DIR / "data"
DOCS_DIR = ROOT_DIR / "docs"

DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# 先加载 SmartKB 配置，再允许 AgentFlow 自己的 .env 覆盖。
load_dotenv(SMARTKB_PROJECT_DIR / ".env", override=False)
load_dotenv(ROOT_DIR / ".env", override=True)


# LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# Embedding
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# Storage
DB_URL = os.getenv("DB_URL", "")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
SMARTKB_COLLECTION = os.getenv("COLLECTION_NAME", "my_rag_collection")

# AgentFlow
AGENTFLOW_MEMORY_TABLE = os.getenv(
    "AGENTFLOW_MEMORY_TABLE",
    "agentflow_memories",
)
AGENTFLOW_LONG_TERM_MEMORY_TABLE = os.getenv(
    "AGENTFLOW_LONG_TERM_MEMORY_TABLE",
    "agentflow_long_term_memories",
)
DEFAULT_SESSION_ID = "agentflow-demo"
TOP_K_KNOWLEDGE = int(os.getenv("AGENTFLOW_TOP_K", "4"))
LONG_TERM_MEMORY_LIMIT = int(os.getenv("AGENTFLOW_LONG_TERM_MEMORY_LIMIT", "4"))
MAX_TOOL_CALLS_PER_WORKER = int(os.getenv("MAX_TOOL_CALLS_PER_WORKER", "4"))
TRACE_LOG_PATH = DATA_DIR / os.getenv("AGENTFLOW_TRACE_FILE", "agentflow_traces.jsonl")

# Image generation
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_API_BASE = os.getenv("IMAGE_API_BASE", "https://www.right.codes/draw/v1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")
IMAGE_TIMEOUT_SECONDS = float(os.getenv("IMAGE_TIMEOUT_SECONDS", "120"))
IMAGE_RETRY_COUNT = int(os.getenv("IMAGE_RETRY_COUNT", "1"))
IMAGE_OUTPUT_DIR = DATA_DIR / "generated_images"
IMAGE_OUTPUT_DIR.mkdir(exist_ok=True)


