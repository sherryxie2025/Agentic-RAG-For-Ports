# src/online_pipeline/demo_agent.py
"""
Demo script for the Plan-and-Execute Agent pipeline.

Run:
    python -m src.online_pipeline.demo_agent

Compares the agent output with the original DAG pipeline side-by-side.
"""

from __future__ import annotations

import json
from pathlib import Path


def print_section(title: str) -> None:
    print("\n" + "=" * 100)
    print(f"  {title}")
    print("=" * 100)


def print_plan(plan: list) -> None:
    for step in plan:
        status = step.get("status", "?")
        icon = {"completed": "+", "failed": "x", "pending": ".", "running": "~"}.get(status, "?")
        print(f"  [{icon}] Step {step.get('step_id', '?')}: "
              f"{step.get('tool_name', '?')} — {step.get('purpose', '')}")
        if step.get("result_summary"):
            print(f"      => {step['result_summary']}")


def main() -> None:
    from .pipeline_logger import setup_pipeline_logging
    setup_pipeline_logging(level="INFO")

    project_root = Path(__file__).resolve().parents[2]

    # Build Agent graph
    from .agent_graph import build_agent_graph
    agent = build_agent_graph(
        project_root=project_root,
        chroma_collection_name=None,
        use_llm_sql_planner=True,
        sql_model_name=None,
    )

    demo_queries = [
        "Why might berth delays be related to weather conditions and crane slowdown?",
        "Based on wave height and berth productivity, should operations be paused?",
        "Under what wind conditions should vessel entry be restricted?",
    ]

    print_section("Plan-and-Execute Agent Demo")
    print(f"Project root: {project_root}")
    print(f"Queries: {len(demo_queries)}")

    for idx, query in enumerate(demo_queries, start=1):
        print_section(f"Query {idx}: {query}")

        state = agent.invoke({
            "user_query": query,
            "reasoning_trace": [],
            "warnings": [],
            "tool_results": [],
        })

        # 1. Execution Plan
        print("\n[1] EXECUTION PLAN")
        plan = state.get("plan", [])
        print_plan(plan)

        # 2. Iterations
        print(f"\n[2] ITERATIONS: {state.get('iteration', 0)}")
        print(f"    Evidence sufficient: {state.get('evidence_sufficient', '?')}")
        gaps = state.get("evidence_gaps", [])
        if gaps:
            print(f"    Evidence gaps: {gaps}")

        # 3. Tool Results Summary
        print("\n[3] TOOL RESULTS")
        tool_results = state.get("tool_results", [])
        for tr in tool_results:
            status = "OK" if tr.get("success") else "FAIL"
            print(f"  [{status}] {tr.get('tool_name', '?')} "
                  f"({tr.get('execution_time_s', 0):.2f}s) "
                  f"query={tr.get('input_query', '')[:60]}")

        # 4. Evidence Bundle Summary
        print("\n[4] EVIDENCE BUNDLE")
        bundle = state.get("evidence_bundle", {})
        docs = bundle.get("documents", [])
        sql = bundle.get("sql_results", [])
        rules = bundle.get("rules", {})
        graph = bundle.get("graph", {})
        print(f"    Documents: {len(docs)}")
        print(f"    SQL results: {len(sql)}")
        matched_rules = rules.get("matched_rules", []) if isinstance(rules, dict) else []
        print(f"    Rules: {len(matched_rules)}")
        graph_paths = graph.get("reasoning_paths", []) if isinstance(graph, dict) else []
        print(f"    Graph paths: {len(graph_paths)}")

        # 5. Final Answer
        print("\n[5] FINAL ANSWER")
        final = state.get("final_answer", {})
        if isinstance(final, dict):
            answer_text = final.get("answer", "(no answer)")
            print(f"    {answer_text[:500]}")
            print(f"\n    Confidence: {final.get('confidence')}")
            print(f"    Sources: {final.get('sources_used', [])}")
            print(f"    Grounding: {final.get('grounding_status', '?')}")
            print(f"    LLM fallback: {final.get('llm_answer_used', '?')}")
        else:
            print(f"    {final}")

        # 6. Reasoning Trace
        print("\n[6] REASONING TRACE")
        for step in state.get("reasoning_trace", []):
            print(f"    - {step}")

        print()


if __name__ == "__main__":
    main()
