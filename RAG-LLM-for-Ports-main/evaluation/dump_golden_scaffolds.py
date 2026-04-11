"""
Dump golden-dataset scaffolds for Opus-driven query generation.

Runs Phase 1 of the new golden_dataset_v3_rag.json build: stratified
sampling + raw context dump. No LLM calls. Produces per-category task
files that downstream Opus subagents will read and populate with queries.

Output files (under evaluation/scaffolds/):
    vector_tasks.json      — 50 vector chunks to generate queries for
    sql_tasks.json         — 30 SQL result scenarios
    rules_tasks.json       — 20 grounded rules
    graph_tasks.json       — 15 Neo4j edges
    multi_tasks.json       — 49 multi-source anchors

Guardrails (25) and metadata filters (16) are handwritten in
build_golden_v3_rag.py's build_guardrail_samples() and
build_metadata_filter_samples() — no scaffold needed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("dump_scaffolds")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

# Reuse stratification + sampler helpers from build_golden_v3_rag.py
from build_golden_v3_rag import (
    load_children,
    stratify_children_by_metadata,
    sample_sql_queries,
    sample_rule_queries,
    sample_graph_edges,
    TARGETS,
)

SCAFFOLD_DIR = PROJECT_ROOT / "evaluation" / "scaffolds"


def dump_vector_tasks() -> int:
    chunks = load_children()
    sampled = stratify_children_by_metadata(chunks, TARGETS["vector_only"])
    tasks = []
    for i, c in enumerate(sampled):
        tasks.append({
            "task_id": f"VEC_{i+1:03d}",
            "chunk_id": c["chunk_id"],
            "parent_id": c.get("parent_id"),
            "source_file": c.get("source_file", ""),
            "page": c.get("page"),
            "section_number": c.get("section_number", ""),
            "section_title": c.get("section_title", ""),
            "doc_type": c.get("doc_type", "document"),
            "category": c.get("category", "unknown"),
            "publish_year": c.get("publish_year"),
            "is_table": bool(c.get("is_table", False)),
            "word_count": c.get("word_count", 0),
            "chunk_text": c.get("text", ""),
        })
    (SCAFFOLD_DIR / "vector_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(tasks)


def dump_sql_tasks() -> int:
    scenarios = sample_sql_queries(TARGETS["sql_only"])
    tasks = []
    for i, sc in enumerate(scenarios):
        tasks.append({
            "task_id": f"SQL_{i+1:03d}",
            "sql": sc["sql"],
            "result": sc["result"],
            "tables": sc["tables"],
            "column": sc.get("column"),
            "aggregation": sc.get("aggregation"),
            "year": sc.get("year"),
            "phrase": sc.get("phrase"),
        })
    (SCAFFOLD_DIR / "sql_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(tasks)


def dump_rule_tasks() -> int:
    rules = sample_rule_queries(TARGETS["rules_only"])
    tasks = []
    for i, r in enumerate(rules):
        tasks.append({
            "task_id": f"RUL_{i+1:03d}",
            "variable": r.get("variable", ""),
            "sql_variable": r.get("sql_variable", ""),
            "operator": r.get("operator", ""),
            "value": r.get("value"),
            "action": r.get("action", ""),
            "rule_text": r.get("rule_text", ""),
            "source_file": r.get("source_file", ""),
            "page": r.get("page"),
        })
    (SCAFFOLD_DIR / "rules_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(tasks)


def dump_graph_tasks() -> int:
    edges = sample_graph_edges(TARGETS["graph_only"])
    tasks = []
    for i, e in enumerate(edges):
        tasks.append({
            "task_id": f"GRA_{i+1:03d}",
            "source": e.get("source", ""),
            "target": e.get("target", ""),
            "rule_text": (e.get("rule_text") or "")[:300],
            "operator": e.get("op", ""),
            "threshold": e.get("threshold"),
            "source_file": e.get("source_file", ""),
        })
    (SCAFFOLD_DIR / "graph_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(tasks)


def dump_multi_tasks() -> int:
    """Multi-source anchors: pick chunks + indicate which sources are required."""
    import random
    chunks = load_children()
    rng = random.Random(42 + 1)

    combos = [
        (["vector", "sql"], 5),
        (["vector", "rules"], 5),
        (["vector", "graph"], 4),
        (["sql", "rules"], 7),
        (["sql", "graph"], 4),
        (["rules", "graph"], 5),
        (["vector", "sql", "rules"], 5),
        (["vector", "sql", "graph"], 4),
        (["vector", "rules", "graph"], 3),
        (["sql", "rules", "graph"], 3),
        (["vector", "sql", "rules", "graph"], 4),
    ]

    tasks = []
    counter = 0
    for sources, count in combos:
        for _ in range(count):
            counter += 1
            anchor = rng.choice(chunks)
            tasks.append({
                "task_id": f"MULTI_{counter:03d}",
                "expected_sources": sources,
                "anchor_chunk_id": anchor.get("chunk_id"),
                "anchor_parent_id": anchor.get("parent_id"),
                "anchor_source_file": anchor.get("source_file", ""),
                "anchor_section_title": anchor.get("section_title", ""),
                "anchor_doc_type": anchor.get("doc_type", "document"),
                "anchor_text": anchor.get("text", "")[:800],
            })
    (SCAFFOLD_DIR / "multi_tasks.json").write_text(
        json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(tasks)


def main() -> None:
    SCAFFOLD_DIR.mkdir(parents=True, exist_ok=True)
    print("Dumping golden dataset scaffolds (no LLM calls)...")
    n_vec = dump_vector_tasks()
    print(f"  vector_tasks.json: {n_vec}")
    n_sql = dump_sql_tasks()
    print(f"  sql_tasks.json:    {n_sql}")
    n_rule = dump_rule_tasks()
    print(f"  rules_tasks.json:  {n_rule}")
    n_graph = dump_graph_tasks()
    print(f"  graph_tasks.json:  {n_graph}")
    n_multi = dump_multi_tasks()
    print(f"  multi_tasks.json:  {n_multi}")
    print(f"\nTotal scaffolds: {n_vec + n_sql + n_rule + n_graph + n_multi}")
    print(f"Output dir: {SCAFFOLD_DIR}")


if __name__ == "__main__":
    main()
