"""
Re-run samples that ran under a degraded API (DashScope Arrearage /
Access denied errors) and merge them back into the existing eval report,
then re-compute ALL metrics and write a fresh JSON + MD report.

Workflow:
  1. Load an existing report (must have per_sample_results)
  2. Identify contaminated sample ids:
     - llm_error contains 'Arrearage' / 'Access denied' / 'BadRequest'
       / 'code: 400'
     - OR any upstream failure the user passes via --extra-ids
  3. Look them up in golden_dataset_v3_rag.json
  4. Run the DAG on just those samples via run_rag_evaluation.run_dag_on_samples
  5. Replace the contaminated rows in per_sample_results
  6. Re-compute routing / retrieval / answer / guardrail / latency
     metrics on the merged list
  7. Write merged JSON and auto-render MD

Usage:
  python evaluation/rerun_contaminated.py \
      --input-report evaluation/agent/reports/rag_v2_n205_improved.json \
      --output-report evaluation/agent/reports/rag_v2_n205_improved_merged.json \
      --workers 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
REPORTS_DIR = EVAL_DIR / "agent" / "reports"

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EVAL_DIR / "agent"))

# Import runner functions by file path since evaluation/ is not a package
_runner_spec = importlib.util.spec_from_file_location(
    "run_rag_evaluation", EVAL_DIR / "run_rag_evaluation.py"
)
_runner = importlib.util.module_from_spec(_runner_spec)
_runner_spec.loader.exec_module(_runner)

_md_spec = importlib.util.spec_from_file_location(
    "render_eval_markdown", EVAL_DIR / "render_eval_markdown.py"
)
_md = importlib.util.module_from_spec(_md_spec)
_md_spec.loader.exec_module(_md)

from eval_routing import evaluate_routing, print_routing_report
from eval_retrieval import evaluate_retrieval_all, print_retrieval_report
from eval_answer_quality import evaluate_answers, print_answer_report
from eval_guardrails import evaluate_guardrails, print_guardrail_report
from eval_latency import evaluate_latency, print_latency_report


CONTAMINATION_MARKERS = (
    "Arrearage",
    "Access denied",
    "BadRequest",
    "code: 400",
    "overdue-payment",
)


def is_contaminated(sample_row: Dict[str, Any]) -> bool:
    """A sample is contaminated if its synth LLM error matches any Arrearage marker."""
    err = (sample_row.get("llm_error") or "") or ""
    if any(m in err for m in CONTAMINATION_MARKERS):
        return True
    # Also check if the synth LLM was skipped due to a 400-flavoured error
    # stored in the final_answer.
    final = sample_row.get("final_answer") or {}
    llm_error_final = (final.get("llm_error") or "") if isinstance(final, dict) else ""
    if any(m in llm_error_final for m in CONTAMINATION_MARKERS):
        return True
    return False


def load_golden() -> List[Dict[str, Any]]:
    g_path = EVAL_DIR / "golden_dataset_v3_rag.json"
    data = json.loads(g_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data.get("samples", [])
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-report", required=True,
                        help="Existing JSON report with per_sample_results to salvage")
    parser.add_argument("--output-report", required=True,
                        help="Path to write the merged report")
    parser.add_argument("--workers", type=int, default=3,
                        help="Thread pool size for the rerun")
    parser.add_argument("--extra-ids", default="",
                        help="Comma-separated extra sample ids to force re-run")
    args = parser.parse_args()

    in_path = Path(args.input_report)
    if not in_path.is_absolute():
        in_path = PROJECT_ROOT / in_path
    out_path = Path(args.output_report)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path

    print(f"Loading report: {in_path}")
    report = json.loads(in_path.read_text(encoding="utf-8"))
    per_sample: List[Dict[str, Any]] = report.get("per_sample_results") or []
    if not per_sample:
        print("ERROR: input report has no per_sample_results")
        sys.exit(1)
    print(f"  per_sample_results: {len(per_sample)}")

    # Identify contaminated rows
    contaminated_ids: Set[str] = {
        r.get("id") for r in per_sample if is_contaminated(r)
    }
    contaminated_ids.discard(None)
    contaminated_ids.discard("")
    if args.extra_ids:
        extras = {x.strip() for x in args.extra_ids.split(",") if x.strip()}
        contaminated_ids.update(extras)
        print(f"  extra ids to rerun: {extras}")
    print(f"  contaminated ids to rerun: {len(contaminated_ids)}")

    if not contaminated_ids:
        print("Nothing to rerun, exiting.")
        return

    # Load golden, filter to contaminated samples
    golden = load_golden()
    print(f"Golden samples: {len(golden)}")
    golden_by_id = {g["id"]: g for g in golden}
    rerun_samples = [golden_by_id[i] for i in contaminated_ids if i in golden_by_id]
    missing = contaminated_ids - {g["id"] for g in golden}
    if missing:
        print(f"  WARNING: {len(missing)} contaminated ids not in golden: {sorted(missing)[:10]}")
    print(f"  rerun batch size: {len(rerun_samples)}")

    # Run DAG on only those samples
    print("\n--- Re-running contaminated samples ---\n", flush=True)
    t0 = time.time()
    new_rows = _runner.run_dag_on_samples(
        rerun_samples, limit=None, workers=args.workers
    )
    elapsed = time.time() - t0
    print(f"\nRerun complete in {elapsed:.1f}s")

    new_by_id = {r.get("id"): r for r in new_rows}

    # Quick sanity: any new rows still contaminated?
    still_bad = [i for i, r in new_by_id.items() if is_contaminated(r)]
    if still_bad:
        print(f"  WARNING: {len(still_bad)} samples STILL contaminated after rerun: {still_bad[:5]}")

    # Merge back — replace contaminated rows in-place so ordering is preserved
    merged: List[Dict[str, Any]] = []
    replaced = 0
    for row in per_sample:
        sid = row.get("id")
        if sid in new_by_id:
            merged.append(new_by_id[sid])
            replaced += 1
        else:
            merged.append(row)
    print(f"  replaced {replaced}/{len(per_sample)} rows")

    # Re-compute all metrics on the merged list
    print("\n--- Recomputing metrics ---\n")
    routing = evaluate_routing(merged, golden)
    retrieval = evaluate_retrieval_all(merged, golden)
    answers = evaluate_answers(merged, golden, use_llm_judge=False)
    guardrails = evaluate_guardrails(merged, golden)
    latency = evaluate_latency(merged)

    print_routing_report(routing)
    print_retrieval_report(retrieval)
    print_answer_report(answers)
    print_guardrail_report(guardrails)
    print_latency_report(latency)

    merged_report = {
        "timestamp": datetime.now().isoformat(),
        "architecture": report.get("architecture", "Agentic RAG LangGraph DAG"),
        "data_pipeline": report.get("data_pipeline"),
        "golden_dataset": report.get("golden_dataset"),
        "dataset_size": len(golden),
        "evaluated_samples": len(merged),
        "workers": args.workers,
        "rerun_info": {
            "input_report": str(in_path.name),
            "contaminated_count": len(contaminated_ids),
            "rerun_batch_size": len(rerun_samples),
            "replaced_rows": replaced,
            "rerun_seconds": round(elapsed, 1),
            "still_contaminated_after_rerun": len(still_bad),
        },
        "single_turn": {
            "routing": routing.to_dict(),
            "retrieval": retrieval.to_dict(),
            "answer_quality": answers.to_dict(),
            "guardrails": guardrails.to_dict(),
            "latency": latency.to_dict(),
        },
        "per_sample_results": merged,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(merged_report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n>> Wrote {out_path}")

    md_path = out_path.with_suffix(".md")
    _md.render_report_md(merged_report, md_path, source_json=out_path)
    print(f">> Wrote {md_path}")


if __name__ == "__main__":
    main()
