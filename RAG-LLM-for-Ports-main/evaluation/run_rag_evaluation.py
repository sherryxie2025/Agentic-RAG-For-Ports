"""
Final RAG Evaluation Runner — Agentic RAG DAG + v2 data pipeline + golden v3.

Runs the LangGraph DAG workflow (NOT the Plan-Execute agent) against the
unbiased Opus-generated golden_dataset_v3_rag.json. This is the canonical
runner for the FINAL version of the system.

Usage (from project root):
    cd RAG-LLM-for-Ports-main
    python evaluation/run_rag_evaluation.py [--limit N] [--skip-llm-judge]

Output: evaluation/agent/reports/rag_v2_n205_final.json  (by default)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "evaluation"
AGENT_EVAL_DIR = EVAL_ROOT / "agent"   # reuse metric modules
REPORTS_DIR = AGENT_EVAL_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(AGENT_EVAL_DIR))   # for eval_* modules
os.chdir(str(PROJECT_ROOT))

# Metric modules (reuse from agent/)
from eval_routing import evaluate_routing, print_routing_report
from eval_retrieval import evaluate_retrieval_all, print_retrieval_report
from eval_answer_quality import evaluate_answers, print_answer_report
from eval_guardrails import evaluate_guardrails, print_guardrail_report
from eval_latency import evaluate_latency, print_latency_report


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_v3() -> List[Dict[str, Any]]:
    """Load the new Opus-generated golden dataset v3."""
    path = EVAL_ROOT / "golden_dataset_v3_rag.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run evaluation/merge_golden_v3.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("samples", [])


# ---------------------------------------------------------------------------
# DAG runner
# ---------------------------------------------------------------------------

def _process_sample(
    workflow,
    sample: Dict[str, Any],
    idx: int,
    total: int,
    print_lock: threading.Lock,
) -> Dict[str, Any]:
    """Run one sample through the DAG and extract eval-shaped result."""
    query = sample.get("query", "")
    sample_id = sample.get("id", f"Q{idx}")
    t0 = time.time()
    try:
        state = workflow.invoke({
            "user_query": query,
            "reasoning_trace": [],
            "warnings": [],
        })
        total_time = time.time() - t0

        # --- Extract for eval modules ---
        final = state.get("final_answer", {}) or {}
        router_decision = state.get("router_decision", {}) or {}

        sql_results_list = state.get("sql_results", []) or []
        tables_used = []
        for sr in sql_results_list:
            if isinstance(sr, dict):
                p = sr.get("plan", {})
                if isinstance(p, dict):
                    tables_used.extend(p.get("target_tables", []) or [])

        rule_results = state.get("rule_results", {}) or {}
        rule_variables = []
        for rm in rule_results.get("matched_rules", []) or []:
            var = rm.get("variable") or rm.get("sql_variable")
            if var:
                rule_variables.append(var)

        graph_results = state.get("graph_results", {}) or {}
        graph_entities = graph_results.get("query_entities", []) or []
        graph_rels = []
        for p in graph_results.get("reasoning_paths", []) or []:
            graph_rels.extend(p.get("path_edges", []) or [])

        retrieved_docs = state.get("retrieved_docs", []) or []
        retrieved_chunk_ids = [
            d.get("chunk_id", "") for d in retrieved_docs if isinstance(d, dict)
        ]
        retrieved_sources_list = [
            d.get("source_file", "") for d in retrieved_docs if isinstance(d, dict)
        ]
        pre_rerank_docs = state.get("pre_rerank_docs", []) or []
        pre_rerank_ids = [
            d.get("chunk_id", "") for d in pre_rerank_docs if isinstance(d, dict)
        ]
        pre_rerank_sources_list = [
            d.get("source_file", "") for d in pre_rerank_docs if isinstance(d, dict)
        ]

        result = {
            "id": sample_id,
            # Routing
            "needs_vector": state.get("needs_vector", router_decision.get("needs_vector", False)),
            "needs_sql": state.get("needs_sql", router_decision.get("needs_sql", False)),
            "needs_rules": state.get("needs_rules", router_decision.get("needs_rules", False)),
            "needs_graph": state.get("needs_graph_reasoning", router_decision.get("needs_graph_reasoning", False)),
            "question_type": state.get("question_type"),
            "answer_mode": state.get("answer_mode"),
            # Retrieval
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "retrieved_sources": retrieved_sources_list,
            "pre_rerank_chunk_ids": pre_rerank_ids,
            "pre_rerank_sources": pre_rerank_sources_list,
            "tables_used": tables_used,
            "execution_ok": any(
                r.get("execution_ok", False) for r in sql_results_list if isinstance(r, dict)
            ),
            "row_count": sum(
                r.get("row_count", 0) for r in sql_results_list if isinstance(r, dict)
            ),
            "rule_variables": rule_variables,
            "entities": graph_entities,
            "relationships": graph_rels,
            "path_count": len(graph_results.get("reasoning_paths", []) or []),
            # Answer quality
            "answer_text": final.get("answer", "") if isinstance(final, dict) else str(final),
            "sources_used": final.get("sources_used", []) if isinstance(final, dict) else [],
            "evidence_bundle": state.get("evidence_bundle", {}),
            "final_answer": final,
            # Latency
            "total_time": total_time,
            "stage_timings": {},
        }
        with print_lock:
            print(f"  [{idx+1}/{total}] {sample_id}: {total_time:.1f}s", flush=True)
        return result
    except Exception as e:
        with print_lock:
            print(f"  [{idx+1}/{total}] {sample_id}: FAILED — {e}", flush=True)
        return {
            "id": sample_id,
            "error": str(e),
            "total_time": time.time() - t0,
        }


def run_dag_on_samples(
    samples: List[Dict[str, Any]],
    limit: Optional[int] = None,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    Run the LangGraph DAG workflow on each golden sample.
    If workers > 1, runs samples in parallel via a thread pool.
    All shared resources (OpenAI SDK, Chroma, Neo4j, DuckDB per-call conn,
    transformer inference) are thread-safe.
    """
    from online_pipeline.langgraph_workflow import build_langgraph_workflow
    from online_pipeline.pipeline_logger import setup_pipeline_logging

    setup_pipeline_logging(level="WARNING")

    workflow = build_langgraph_workflow(
        project_root=PROJECT_ROOT,
        chroma_collection_name="port_documents_v2",  # v2 BGE + Small-to-Big
        use_llm_sql_planner=True,
    )

    if limit is not None:
        samples = samples[:limit]
    total = len(samples)
    print_lock = threading.Lock()

    if workers <= 1:
        # Serial path
        return [
            _process_sample(workflow, sample, i, total, print_lock)
            for i, sample in enumerate(samples)
        ]

    # Parallel path: ThreadPoolExecutor, preserve original order
    results: List[Optional[Dict[str, Any]]] = [None] * total
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_idx = {
            pool.submit(_process_sample, workflow, sample, i, total, print_lock): i
            for i, sample in enumerate(samples)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
    return [r for r in results if r is not None]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm-judge", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="Thread pool size for parallel sample processing.")
    parser.add_argument("--output", default="rag_v2_n205_final.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  FINAL RAG EVALUATION — Agentic RAG DAG + v2 data + golden v3")
    print("=" * 70)

    samples = load_golden_v3()
    print(f"\nDataset: golden_dataset_v3_rag.json")
    print(f"Samples: {len(samples)}")
    if args.limit:
        print(f"Limit:   {args.limit}")

    print(f"\n--- Running DAG workflow on samples (workers={args.workers}) ---\n", flush=True)
    results = run_dag_on_samples(samples, limit=args.limit, workers=args.workers)

    print("\n--- Computing metrics ---\n")
    routing = evaluate_routing(results, samples)
    retrieval = evaluate_retrieval_all(results, samples)
    answers = evaluate_answers(
        results, samples,
        use_llm_judge=not args.skip_llm_judge,
        max_llm_samples=20,
    )
    guardrails = evaluate_guardrails(results, samples)
    latency = evaluate_latency(results)

    print_routing_report(routing)
    print_retrieval_report(retrieval)
    print_answer_report(answers)
    print_guardrail_report(guardrails)
    print_latency_report(latency)

    # Save report
    report = {
        "timestamp": datetime.now().isoformat(),
        "architecture": "Agentic RAG LangGraph DAG",
        "data_pipeline": "v2 (Small-to-Big + BGE + auto-taxonomy + rule-driven graph)",
        "golden_dataset": "golden_dataset_v3_rag.json (Opus 4.1 generated, 205 samples)",
        "dataset_size": len(samples),
        "evaluated_samples": len(results),
        "single_turn": {
            "routing": routing.to_dict(),
            "retrieval": retrieval.to_dict(),
            "answer_quality": answers.to_dict(),
            "guardrails": guardrails.to_dict(),
            "latency": latency.to_dict(),
        },
    }

    out_path = REPORTS_DIR / args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n>> Wrote {out_path}")


if __name__ == "__main__":
    main()
