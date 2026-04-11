"""
Unified Agent Evaluation Driver.

Runs the full evaluation suite:
1. Single-turn golden dataset -> routing/retrieval/answer_quality/guardrails/latency
2. Multi-turn conversations -> multi-turn specific metrics
3. Aggregates everything into a single JSON report

Usage:
    cd RAG-LLM-for-Ports-main
    python evaluation/run_full_evaluation.py [--skip-llm-judge] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVAL_DIR))
os.chdir(str(PROJECT_ROOT))

# Evaluation modules
from eval_routing import evaluate_routing, print_routing_report
from eval_retrieval import evaluate_retrieval_all, print_retrieval_report
from eval_answer_quality import evaluate_answers, print_answer_report
from eval_multi_turn import (
    aggregate_multi_turn,
    coherence_judge,
    evaluate_multi_turn_conversation,
    print_multi_turn_report,
)
from eval_guardrails import evaluate_guardrails, print_guardrail_report
from eval_latency import evaluate_latency, print_latency_report


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_golden_dataset() -> Dict[str, Any]:
    """Load base golden dataset + v3 extras (multi-turn + guardrails)."""
    base_path = EVAL_DIR / "golden_dataset.json"
    extras_path = EVAL_DIR / "golden_dataset_v3_extras.json"

    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    extras = {"sections": {}}
    if extras_path.exists():
        with open(extras_path, "r", encoding="utf-8") as f:
            extras = json.load(f)

    sections = extras.get("sections", {})

    # Flatten all single-turn samples
    single_turn = list(base)
    single_turn.extend(sections.get("single_turn_gap_fill", []))
    single_turn.extend(sections.get("guardrails", []))

    # Multi-turn conversations kept separate
    multi_turn = sections.get("multi_turn_conversations", [])

    return {
        "single_turn": single_turn,
        "multi_turn": multi_turn,
    }


# ---------------------------------------------------------------------------
# Agent runner (builds agent, runs queries, collects data)
# ---------------------------------------------------------------------------

def run_agent_on_samples(
    samples: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run the agent on each golden sample and collect evaluation data.

    Returns a list of result dicts with the shape expected by eval modules.
    """
    from online_pipeline.agent_graph import build_agent_graph
    from online_pipeline.pipeline_logger import setup_pipeline_logging

    setup_pipeline_logging(level="WARNING")

    agent = build_agent_graph(
        project_root=PROJECT_ROOT,
        chroma_collection_name="port_documents_v2",  # BGE + Small-to-Big
        use_llm_sql_planner=True,
        enable_react_observations=True,
    )

    results = []
    for i, sample in enumerate(samples):
        if limit and i >= limit:
            break

        query = sample.get("query", "")
        sample_id = sample.get("id", f"Q{i}")

        t0 = time.time()
        try:
            state = agent.invoke({
                "user_query": query,
                "reasoning_trace": [],
                "warnings": [],
                "tool_results": [],
                "observations": [],
            })
            total_time = time.time() - t0

            # Extract for evaluation
            final = state.get("final_answer", {}) or {}
            plan = state.get("plan", [])
            tool_results = state.get("tool_results", [])

            # Tables used in SQL tool calls
            sql_results_list = state.get("sql_results", [])
            tables_used = []
            for sr in sql_results_list:
                if isinstance(sr, dict):
                    p = sr.get("plan", {})
                    if isinstance(p, dict):
                        tables_used.extend(p.get("target_tables", []) or [])

            # Rule variables matched
            rule_results = state.get("rule_results", {}) or {}
            rule_variables = []
            for rm in rule_results.get("matched_rules", []) or []:
                var = rm.get("variable") or rm.get("sql_variable")
                if var:
                    rule_variables.append(var)

            # Graph entities/relationships
            graph_results = state.get("graph_results", {}) or {}
            graph_entities = graph_results.get("query_entities", [])
            graph_rels = []
            for p in graph_results.get("reasoning_paths", []) or []:
                graph_rels.extend(p.get("path_edges", []) or [])

            # Chunk ids + source files retrieved (for IR metrics)
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

            # Derive routing decision from plan tool names
            # (new Plan-Execute agent doesn't set needs_* directly)
            plan_tools = set(
                (s.get("tool_name") or "").lower() for s in plan
            )
            tool_to_cap = {
                "document_search": "vector",
                "sql_query": "sql",
                "rule_lookup": "rules",
                "graph_reason": "graph",
            }
            activated_caps = set()
            for tn in plan_tools:
                cap = tool_to_cap.get(tn)
                if cap:
                    activated_caps.add(cap)

            result = {
                "id": sample_id,
                # For routing eval (derived from plan)
                "needs_vector": "vector" in activated_caps,
                "needs_sql": "sql" in activated_caps,
                "needs_rules": "rules" in activated_caps,
                "needs_graph": "graph" in activated_caps,
                "question_type": state.get("question_type"),
                "answer_mode": state.get("answer_mode"),
                # For retrieval eval
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
                # For answer quality eval
                "answer_text": final.get("answer", "") if isinstance(final, dict) else str(final),
                "sources_used": final.get("sources_used", []) if isinstance(final, dict) else [],
                "evidence_bundle": state.get("evidence_bundle", {}),
                "final_answer": final,
                # For latency eval
                "total_time": total_time,
                "iteration": state.get("iteration", 1),
                "observations": state.get("observations", []),
                "stage_timings": state.get("stage_timings", {}),
            }
            results.append(result)
            print(f"  [{i+1}/{len(samples)}] {sample_id}: {total_time:.1f}s")
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {sample_id}: FAILED — {e}")
            results.append({
                "id": sample_id,
                "error": str(e),
                "total_time": time.time() - t0,
            })

    return results


def run_multi_turn_conversations(
    conversations: List[Dict[str, Any]],
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run multi-turn conversations through the agent with session management.
    Returns runs in the format expected by eval_multi_turn.
    """
    from online_pipeline.agent_graph import build_agent_graph
    from online_pipeline.agent_memory import MemoryManager
    from online_pipeline.session_manager import SessionManager
    from online_pipeline.pipeline_logger import setup_pipeline_logging

    setup_pipeline_logging(level="WARNING")

    agent = build_agent_graph(
        project_root=PROJECT_ROOT,
        chroma_collection_name="port_documents_v2",  # BGE + Small-to-Big
        use_llm_sql_planner=True,
        enable_react_observations=True,
    )
    memory_mgr = MemoryManager(PROJECT_ROOT)
    session_mgr = SessionManager(memory_mgr)

    runs = []
    for i, conv in enumerate(conversations):
        if limit and i >= limit:
            break

        conv_id = conv.get("conversation_id", f"MT{i}")
        print(f"\n  Running conversation {conv_id} ({len(conv.get('turns', []))} turns)")

        session_id, short_term = session_mgr.get_or_create()
        run_turns = []

        for turn_spec in conv.get("turns", []):
            raw_query = turn_spec.get("query", "")
            try:
                resolved = session_mgr.resolve_query(session_id, raw_query)
                state_extras = session_mgr.build_agent_state_extras(session_id, resolved)

                state = agent.invoke({
                    "user_query": resolved,
                    "original_query": raw_query,
                    "reasoning_trace": [],
                    "warnings": [],
                    "tool_results": [],
                    "observations": [],
                    **state_extras,
                })

                final = state.get("final_answer", {}) or {}
                answer_text = final.get("answer", "") if isinstance(final, dict) else str(final)

                session_mgr.record_turn(session_id, "user", raw_query)
                session_mgr.record_turn(session_id, "assistant", answer_text[:300])

                tracked = list(short_term.active_entities.keys())

                run_turns.append({
                    "turn_id": turn_spec.get("turn_id"),
                    "raw_query": raw_query,
                    "resolved_query": resolved,
                    "answer": answer_text,
                    "tracked_entities": tracked,
                    "tool_results_summary": [
                        tr.get("tool_name") for tr in state.get("tool_results", [])
                    ],
                })
                print(f"    Turn {turn_spec.get('turn_id')}: {raw_query[:50]}")
            except Exception as e:
                print(f"    Turn failed: {e}")
                run_turns.append({
                    "turn_id": turn_spec.get("turn_id"),
                    "raw_query": raw_query,
                    "error": str(e),
                })

        runs.append({
            "conversation_id": conv_id,
            "session_id": session_id,
            "turns": run_turns,
            "spec": conv,
        })

        # Clean up session
        try:
            session_mgr.end_session(session_id)
        except Exception:
            pass

    memory_mgr.close()
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm-judge", action="store_true",
                        help="Skip LLM-as-judge scoring (faster, cheaper)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples (single-turn) for smoke test")
    parser.add_argument("--mt-limit", type=int, default=None,
                        help="Limit number of multi-turn conversations")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-multi", action="store_true")
    parser.add_argument("--output", default="evaluation_report_v3.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  FULL AGENT EVALUATION")
    print("=" * 70)

    dataset = load_golden_dataset()
    print(f"\n  Single-turn samples: {len(dataset['single_turn'])}")
    print(f"  Multi-turn conversations: {len(dataset['multi_turn'])}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "single_turn_count": len(dataset["single_turn"]),
            "multi_turn_count": len(dataset["multi_turn"]),
        },
    }

    # --- Single-turn evaluation ---
    if not args.skip_single:
        print("\n--- Running agent on single-turn samples ---")
        single_results = run_agent_on_samples(dataset["single_turn"], limit=args.limit)

        print("\n--- Computing single-turn metrics ---")
        routing_metrics = evaluate_routing(single_results, dataset["single_turn"])
        retrieval_metrics = evaluate_retrieval_all(single_results, dataset["single_turn"])
        answer_metrics = evaluate_answers(
            single_results, dataset["single_turn"],
            use_llm_judge=not args.skip_llm_judge,
            max_llm_samples=20,
        )
        guardrail_metrics = evaluate_guardrails(single_results, dataset["single_turn"])
        latency_metrics = evaluate_latency(single_results)

        print_routing_report(routing_metrics)
        print_retrieval_report(retrieval_metrics)
        print_answer_report(answer_metrics)
        print_guardrail_report(guardrail_metrics)
        print_latency_report(latency_metrics)

        report["single_turn"] = {
            "routing": routing_metrics.to_dict(),
            "retrieval": retrieval_metrics.to_dict(),
            "answer_quality": answer_metrics.to_dict(),
            "guardrails": guardrail_metrics.to_dict(),
            "latency": latency_metrics.to_dict(),
        }

    # --- Multi-turn evaluation ---
    if not args.skip_multi and dataset["multi_turn"]:
        print("\n--- Running agent on multi-turn conversations ---")
        mt_runs = run_multi_turn_conversations(
            dataset["multi_turn"], limit=args.mt_limit
        )

        print("\n--- Computing multi-turn metrics ---")
        per_conv = []
        judge_results = []
        for run in mt_runs:
            spec = run["spec"]
            conv_metrics = evaluate_multi_turn_conversation(spec, run)
            per_conv.append(conv_metrics)

            if not args.skip_llm_judge:
                judge = coherence_judge([
                    {"turn_id": t.get("turn_id"),
                     "query": t.get("raw_query"),
                     "answer": t.get("answer", "")}
                    for t in run["turns"]
                ])
                judge_results.append(judge)

        mt_metrics = aggregate_multi_turn(per_conv, judge_results if judge_results else None)
        print_multi_turn_report(mt_metrics)

        report["multi_turn"] = {
            "aggregate": mt_metrics.to_dict(),
            "per_conversation": per_conv,
        }

    # --- Save report ---
    output_path = EVAL_DIR / args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n--- Evaluation report saved to {output_path} ---")
    print("Done.")


if __name__ == "__main__":
    main()
