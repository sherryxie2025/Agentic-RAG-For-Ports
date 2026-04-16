# src/online_pipeline/__init__.py
"""
Agentic RAG (DAG-based) online pipeline for port decision support.

Two graph builders, both compiled via LangGraph:

- `build_langgraph_workflow(...)`        — single-turn DAG (default for the
                                             205-sample golden evaluation).
- `build_langgraph_workflow_with_memory(...)` — same DAG plus an upfront
                                             follow-up resolution node and
                                             memory-context injection. Use
                                             this for multi-turn sessions /
                                             API endpoints.

Memory:
- `MemoryManager` is the single facade. It owns a 3-layer per-session
  short-term buffer (raw turns / LLM summaries / distilled key facts) and
  a DuckDB long-term store with BGE vector retrieval.

The pre-agent ReAct/Plan-Execute architecture (`agent_*.py`) was retired
and lives under `legacy/react_agent/`.
"""

from .langgraph_workflow import (
    build_langgraph_workflow,
    build_langgraph_workflow_presynthesis,
    build_langgraph_workflow_with_memory,
)
from .conversation_memory import (
    BGEEmbedder,
    MemoryManager,
    ShortTermMemory,
    LongTermMemory,
    extract_entities,
    extract_key_facts,
)

__all__ = [
    "build_langgraph_workflow",
    "build_langgraph_workflow_presynthesis",
    "build_langgraph_workflow_with_memory",
    "MemoryManager",
    "ShortTermMemory",
    "LongTermMemory",
    "BGEEmbedder",
    "extract_entities",
    "extract_key_facts",
]
