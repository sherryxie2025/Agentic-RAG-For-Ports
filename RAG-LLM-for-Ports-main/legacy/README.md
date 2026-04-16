# Legacy archive

This directory holds historical code and reports that are **no longer part
of the active agentic-RAG (DAG) pipeline**. Everything here is preserved
for reference (and to keep git history) but is not imported by current
production / evaluation code.

The active system is the LangGraph DAG built by
`src/online_pipeline/langgraph_workflow.py`
(driven by `evaluation/run_rag_evaluation.py`).

## Subdirectories

### `react_agent/` — Plan-and-Execute (ReAct-style) agent
Earlier exploratory architecture B. Was a true LLM-driven Plan→Act→Observe
loop on top of the same retrievers. Dropped in favour of the deterministic
DAG because:
- The DAG is faster (~3× lower p50 latency on the 205-sample golden set).
- The agent pipeline's re-plan loop didn't add measurable answer-quality.
- Maintenance cost (planner prompt + tool descriptors + observation
  judge) was high.

Contains:
- `agent_graph.py` / `agent_state.py` / `agent_tools.py` / `agent_prompts.py`
- `agent_memory.py` — short-term + long-term memory (SQLite). **Memory
  for the agentic-RAG DAG was rewritten from scratch in
  `src/online_pipeline/conversation_memory.py`** rather than reused.
- `session_manager.py` — multi-turn session driver
- `demo_agent.py`

### `old_offline/` — superseded offline-pipeline scripts
v1 chunker + embedding builder + Neo4j builder; replaced by their `_v2`
versions which are still active in `src/offline_pipeline/`.

- `chunk_documents.py` (v1, RecursiveCharacterTextSplitter)
- `semantic_chunker.py` (v1.5)
- `build_embeddings.py` (v1, all-MiniLM-L6-v2)
- `build_neo4j_graph.py` (v1, fully hardcoded)
- `build_vector_db.py` (v1 Chroma populator)
- `run_offline_pipeline.py` (v1 entry point chaining the above)

### `old_reports/` — superseded system reports
Earlier writeups; canonical doc is now `FINAL_SYSTEM_REPORT_V2_CN.md`
in the project root.

- `FINAL_SYSTEM_REPORT_CN.md` (v1)
- `SYSTEM_REPORT.md` (English baseline)

### `root_scraps/` — assorted root-level cruft
- `RAG-LLM-for-Ports-main_原文件.zip` — snapshot of the very first
  upstream codebase
- `env_example - 副本.txt` — duplicate of `env_example.txt`

## Restoring something
```
git log --diff-filter=R --follow -- legacy/<path>
git mv legacy/<path> <original-location>
```
Backup tag for the cleanup commit: `backup-pre-cleanup-2026-04-15`.
