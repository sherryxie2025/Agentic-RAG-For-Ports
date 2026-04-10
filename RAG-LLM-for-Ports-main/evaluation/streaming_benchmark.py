"""
Streaming vs Blocking Latency Benchmark
========================================
Compares time-to-first-token (TTFT) in streaming mode vs total wait time
in blocking mode for the answer synthesis stage.

Produces a comparison table suitable for resume/interview presentation.

Usage:
    cd RAG-LLM-for-Ports-main
    uv run python evaluation/streaming_benchmark.py
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

from online_pipeline.langgraph_workflow import LangGraphWorkflowBuilder
from online_pipeline.pipeline_logger import setup_pipeline_logging


# Representative queries covering different intent types
BENCHMARK_QUERIES = [
    {"id": "DOC", "query": "What are the key environmental sustainability initiatives described in port annual reports?"},
    {"id": "SQL", "query": "What was the average crane productivity in moves per hour in 2015?"},
    {"id": "RULE", "query": "What are the wind speed restrictions for vessel entry?"},
    {"id": "HYB", "query": "Based on the current wind speed data and port rules, should vessel entry be restricted?"},
    {"id": "CAUSAL", "query": "Why might berth delays be related to weather conditions and crane slowdown?"},
]


def run_benchmark():
    setup_pipeline_logging(level="INFO")
    print("Loading pipeline...")
    builder = LangGraphWorkflowBuilder(
        project_root=PROJECT_ROOT,
        use_llm_sql_planner=True,
    )
    app = builder.build()
    synthesizer = builder.factory.answer_synthesizer

    results = []

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        query = item["query"]
        print(f"\n{'='*80}")
        print(f"[{qid}] {query[:70]}...")

        # --- Run full pipeline to get state (shared for both modes) ---
        t_pipeline_start = time.time()
        state = app.invoke({
            "user_query": query,
            "reasoning_trace": [],
            "warnings": [],
        })
        pipeline_time = time.time() - t_pipeline_start

        # --- Blocking mode: measure synthesize time ---
        t_block_start = time.time()
        block_answer = synthesizer.synthesize(state)
        block_total = time.time() - t_block_start
        block_text = block_answer.get("answer", "")

        # --- Streaming mode: measure TTFT and total ---
        t_stream_start = time.time()
        ttft = None
        stream_chunks = []
        for chunk in synthesizer.synthesize_stream(state):
            if ttft is None:
                ttft = time.time() - t_stream_start
            stream_chunks.append(chunk)
        stream_total = time.time() - t_stream_start
        stream_text = "".join(stream_chunks)

        if ttft is None:
            ttft = stream_total  # fallback if no chunks

        speedup = block_total / ttft if ttft > 0 else 0

        result = {
            "id": qid,
            "query": query[:60],
            "pipeline_s": round(pipeline_time, 1),
            "block_total_s": round(block_total, 1),
            "stream_ttft_s": round(ttft, 2),
            "stream_total_s": round(stream_total, 1),
            "ttft_speedup": round(speedup, 1),
            "block_len": len(block_text),
            "stream_len": len(stream_text),
        }
        results.append(result)

        print(f"  Pipeline:        {pipeline_time:.1f}s")
        print(f"  Blocking total:  {block_total:.1f}s")
        print(f"  Streaming TTFT:  {ttft:.2f}s")
        print(f"  Streaming total: {stream_total:.1f}s")
        print(f"  TTFT speedup:    {speedup:.1f}x")

    # --- Summary table ---
    print("\n" + "=" * 80)
    print("STREAMING vs BLOCKING BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'ID':8s} | {'Block (s)':>10s} | {'TTFT (s)':>9s} | {'Stream (s)':>10s} | {'Speedup':>8s} | {'Pipeline (s)':>12s}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")

    ttft_list = []
    block_list = []
    for r in results:
        print(f"{r['id']:8s} | {r['block_total_s']:10.1f} | {r['stream_ttft_s']:9.2f} | {r['stream_total_s']:10.1f} | {r['ttft_speedup']:7.1f}x | {r['pipeline_s']:12.1f}")
        ttft_list.append(r["stream_ttft_s"])
        block_list.append(r["block_total_s"])

    avg_block = sum(block_list) / len(block_list)
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_speedup = avg_block / avg_ttft if avg_ttft > 0 else 0

    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")
    print(f"{'AVG':8s} | {avg_block:10.1f} | {avg_ttft:9.2f} | {'':10s} | {avg_speedup:7.1f}x |")

    print(f"\nKey insight: Streaming reduces perceived latency from {avg_block:.1f}s to {avg_ttft:.2f}s ({avg_speedup:.0f}x improvement)")
    print(f"Users see the first token in ~{avg_ttft:.1f}s instead of waiting {avg_block:.0f}s for the full response.")

    # Save results
    out_path = PROJECT_ROOT / "evaluation" / "streaming_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "queries": results,
            "summary": {
                "avg_blocking_total_s": round(avg_block, 2),
                "avg_streaming_ttft_s": round(avg_ttft, 2),
                "avg_ttft_speedup": round(avg_speedup, 1),
            },
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    run_benchmark()
