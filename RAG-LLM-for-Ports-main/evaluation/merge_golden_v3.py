"""
Merge Opus-generated scaffold results + handwritten guardrails/metadata
into the final golden_dataset_v3_rag.json.

Inputs:
    evaluation/scaffolds/vector_results.json   (50 vector samples, opus)
    evaluation/scaffolds/sql_results.json      (30 sql samples, opus)
    evaluation/scaffolds/rules_results.json    (20 rule samples, opus)
    evaluation/scaffolds/graph_results.json    (15 graph samples, opus)
    evaluation/scaffolds/multi_results.json    (49 multi-source samples, opus)

Plus handwritten (reused from build_golden_v3_rag.py):
    25 guardrail samples (9 types)
    16 metadata filter tests

Output:
    evaluation/golden_dataset_v3_rag.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "evaluation"))

from build_golden_v3_rag import (
    build_guardrail_samples,
    build_metadata_filter_samples,
)

SCAFFOLD_DIR = PROJECT_ROOT / "evaluation" / "scaffolds"
OUTPUT_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset_v3_rag.json"


def load(path: Path) -> list:
    if not path.exists():
        print(f"[WARN] Missing: {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    vector_r = load(SCAFFOLD_DIR / "vector_results.json")
    sql_r = load(SCAFFOLD_DIR / "sql_results.json")
    rules_r = load(SCAFFOLD_DIR / "rules_results.json")
    graph_r = load(SCAFFOLD_DIR / "graph_results.json")
    multi_r = load(SCAFFOLD_DIR / "multi_results.json")

    guardrails = build_guardrail_samples()
    metadata_filters = build_metadata_filter_samples()

    all_samples = (
        vector_r + sql_r + rules_r + graph_r + multi_r
        + guardrails + metadata_filters
    )

    # Sanity checks
    ids_seen = Counter(s.get("id", "") for s in all_samples)
    duplicates = {k: v for k, v in ids_seen.items() if v > 1}
    if duplicates:
        print(f"[WARN] Duplicate IDs: {duplicates}")

    print(f"\n{'='*60}")
    print(f"  Final Golden Dataset v3 RAG — {len(all_samples)} samples")
    print(f"{'='*60}\n")

    # Category counts
    print("Category counts:")
    cats = [
        ("vector (opus)", len(vector_r)),
        ("sql (opus)", len(sql_r)),
        ("rules (opus)", len(rules_r)),
        ("graph (opus)", len(graph_r)),
        ("multi-source (opus)", len(multi_r)),
        ("guardrails (handwritten)", len(guardrails)),
        ("metadata filters (handwritten)", len(metadata_filters)),
    ]
    for name, count in cats:
        print(f"  {name:<35} {count}")
    print(f"  {'TOTAL':<35} {len(all_samples)}")

    # Routing combo distribution
    print("\nSource combinations (2^4):")
    combos = Counter(
        tuple(sorted(s.get("expected_sources", []))) for s in all_samples
    )
    for k, v in sorted(combos.items()):
        label = str(k) if k else "() [guardrail]"
        print(f"  {label:<42} {v}")

    # Answer mode distribution
    print("\nAnswer mode distribution:")
    modes = Counter(s.get("answer_mode", "?") for s in all_samples)
    for m, c in modes.most_common():
        print(f"  {m:<20} {c}")

    # Guardrail types
    guardrail_types = Counter(
        s.get("guardrail_type", "") for s in all_samples if s.get("guardrail_type")
    )
    print("\nGuardrail types:")
    for gt, c in guardrail_types.most_common():
        print(f"  {gt:<30} {c}")

    # Generation method distribution
    print("\nGeneration methods:")
    methods = Counter(s.get("generation_method", "?") for s in all_samples)
    for m, c in methods.most_common():
        print(f"  {m:<40} {c}")

    # Save
    output_obj = {
        "description": "Golden dataset v3 for Agentic RAG DAG + v2 data pipeline",
        "version": "3.0",
        "generated_at": datetime.now().isoformat(),
        "generator": "Opus 4.1 (via Claude Code subagents, chunk-first unbiased)",
        "evaluated_system": "Qwen 3.5 Flash (DashScope)",
        "total_samples": len(all_samples),
        "category_counts": dict(cats),
        "source_combos": {str(k): v for k, v in combos.items()},
        "answer_modes": dict(modes),
        "guardrail_types": dict(guardrail_types),
        "samples": all_samples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, indent=2, ensure_ascii=False)

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\n>> Wrote {OUTPUT_PATH} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
