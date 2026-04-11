"""
Rescore answer_quality on an existing eval report without re-running the DAG.

Loads:
  - evaluation/agent/reports/<report>.json  (has per_sample_results)
  - evaluation/golden_dataset_v3_rag.json   (has reference_answer)

Computes:
  - avg_embedding_similarity (BGE cosine)
  - avg_rougeL_f1
  - re-computes the rest of answer_quality for completeness

Writes back into the report under single_turn.answer_quality and adds a
"rescored_at" timestamp, so you can tell it was post-processed.

Usage:
  python evaluation/rescore_answer_quality.py \
      --report evaluation/agent/reports/rag_v2_n205_final.json \
      --golden evaluation/golden_dataset_v3_rag.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.agent.eval_answer_quality import (
    AnswerQualityMetrics,
    evaluate_answers,
    print_answer_report,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True,
                        help="Path to an existing eval report JSON")
    parser.add_argument("--golden", default="evaluation/golden_dataset_v3_rag.json",
                        help="Path to the golden dataset with reference_answer")
    parser.add_argument("--skip-llm-judge", action="store_true", default=True,
                        help="Skip LLM judge (default true for cheap rescore)")
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = PROJECT_ROOT / report_path
    golden_path = Path(args.golden)
    if not golden_path.is_absolute():
        golden_path = PROJECT_ROOT / golden_path

    print(f"Loading report:  {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    print(f"Loading golden:  {golden_path}")
    golden_data = json.loads(golden_path.read_text(encoding="utf-8"))

    # Golden file is either a list or {"samples": [...]}
    if isinstance(golden_data, dict):
        golden = golden_data.get("samples", [])
    else:
        golden = golden_data

    per_sample = report.get("per_sample_results") or []
    if not per_sample:
        print("ERROR: report has no per_sample_results — rescoring needs per-sample answer_text")
        sys.exit(1)

    print(f"  per_sample_results: {len(per_sample)}")
    print(f"  golden samples:     {len(golden)}")
    print()

    # Filter out error rows that have no answer_text
    valid_rows = [r for r in per_sample if r.get("answer_text") or r.get("final_answer")]
    print(f"  rows with answer_text: {len(valid_rows)}")

    print("\n--- Scoring (BGE model will load on first call) ---")
    metrics: AnswerQualityMetrics = evaluate_answers(
        results=valid_rows,
        golden=golden,
        use_llm_judge=not args.skip_llm_judge,
    )
    print_answer_report(metrics)

    # Write back
    report.setdefault("single_turn", {})
    report["single_turn"]["answer_quality"] = metrics.to_dict()
    report["answer_quality_rescored_at"] = datetime.utcnow().isoformat() + "Z"

    out_path = report_path
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n>> Updated {out_path}")


if __name__ == "__main__":
    main()
