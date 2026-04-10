# src/online_pipeline/demo_sql_v2.py

from __future__ import annotations

from pathlib import Path

from .build_sql_db import DuckDBBuilder
from .sql_agent_v2 import SQLAgentV2


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    print("=" * 120)
    print("Building DuckDB database...")
    builder = DuckDBBuilder(project_root)
    db_path = builder.build(overwrite=True)
    print(f"Built DB at: {db_path}")

    agent = SQLAgentV2(
        db_path=db_path,
        use_llm_sql=True,
        model_name=None,
    )

    benchmark_queries = [
        "What was the average crane productivity in 2015?",
        "What was the average wave height in 2015?",
        "What was the average turn time in 2015?",
        "What are the top 5 highest arrival delay cases?",
        "Why might berth delays be related to weather conditions and crane slowdown?",
        "Under what wind conditions should vessel entry be restricted?",
    ]

    print("\n" + "=" * 120)
    print("Phase 2 Benchmark Queries")
    print("=" * 120)

    for idx, q in enumerate(benchmark_queries, start=1):
        print("\n" + "-" * 120)
        print(f"[Query {idx}] {q}")

        result = agent.run(q)
        plan = result["plan"]

        print("Generation mode:", plan.get("generation_mode"))
        print("Used tables:", plan.get("target_tables"))
        print("Aggregation:", plan.get("aggregation"))
        print("Generated SQL:")
        print(plan.get("generated_sql"))

        print("Execution OK:", result["execution_ok"])
        print("Row count:", result["row_count"])
        print("Preview rows:")
        for row in result["rows"][:5]:
            print("  ", row)
        print("Error:", result["error"])


if __name__ == "__main__":
    main()