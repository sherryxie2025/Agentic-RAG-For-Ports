# src/online_pipeline/demo_langgraph_pipeline.py

from __future__ import annotations

from pathlib import Path

from .langgraph_workflow import build_langgraph_workflow


def print_section(title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)


def main() -> None:
    from .pipeline_logger import setup_pipeline_logging
    setup_pipeline_logging(level="INFO")

    project_root = Path(__file__).resolve().parents[2]

    app = build_langgraph_workflow(
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

    print_section("LangGraph + Neo4j Graph Reasoner Demo (Product Answer + LLM Fallback)")

    for idx, query in enumerate(demo_queries, start=1):
        print_section(f"Demo Query {idx}")
        print("USER QUERY:")
        print(query)

        state = app.invoke({
            "user_query": query,
            "reasoning_trace": [],
            "warnings": [],
        })

        print("\n[1] ROUTER DECISION")
        print(state.get("router_decision"))

        print("\n[2] PLANNER OUTPUT")
        print("source_plan:", state.get("source_plan"))
        print("answer_mode:", state.get("answer_mode"))
        print("execution_strategy:", state.get("execution_strategy"))
        print("sub_queries:")
        for sq in state.get("sub_queries", []):
            print(" -", sq)

        print("\n[3] SQL RESULTS")
        sql_results = state.get("sql_results", [])
        if sql_results:
            sql_result = sql_results[0]
            print("execution_ok:", sql_result.get("execution_ok"))
            print("plan:", sql_result.get("plan"))
            print("rows:", sql_result.get("rows")[:3])
        else:
            print("- No SQL query executed.")

        print("\n[4] GRAPH RESULTS")
        graph_results = state.get("graph_results", {})
        print("query_entities:", graph_results.get("query_entities"))
        print("expanded_nodes:", graph_results.get("expanded_nodes"))
        print("reasoning_paths:")
        for rp in graph_results.get("reasoning_paths", []):
            print(" -", rp)

        print("\n[5] FINAL ANSWER")
        final_answer = state.get("final_answer", {})
        print("answer:", final_answer.get("answer"))
        print("confidence:", final_answer.get("confidence"))
        print("sources_used:", final_answer.get("sources_used"))
        print("reasoning_summary:", final_answer.get("reasoning_summary"))
        print("caveats:", final_answer.get("caveats"))
        print("grounding_status:", final_answer.get("grounding_status"))
        print("llm_answer_used:", final_answer.get("llm_answer_used"))
        print("knowledge_fallback_used:", final_answer.get("knowledge_fallback_used"))
        print("knowledge_fallback_notes:", final_answer.get("knowledge_fallback_notes"))

        print("\n[6] REASONING TRACE")
        for step in state.get("reasoning_trace", []):
            print("-", step)


if __name__ == "__main__":
    main()