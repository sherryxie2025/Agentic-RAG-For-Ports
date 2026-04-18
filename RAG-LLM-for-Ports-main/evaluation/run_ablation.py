"""
Weight-ablation driver for long-term memory retrieval.

Runs `run_cross_session_evaluation.py` repeatedly under N different
`scoring_weights` configurations and aggregates the deltas into a single
summary table.

Configs:
    baseline       (current defaults)
    cos_only       (vector similarity only)
    no_decay       (remove time decay)
    no_entity      (remove entity overlap)
    equal          (all weights equal)
    entity_heavy   (entity dominates)

Usage:
    python evaluation/run_ablation.py [--limit N] [--skip-check]
    # --skip-check skips the sanity check that Phase B beats Phase A first.

Output:
    evaluation/agent/reports/ablation_summary.json  — side-by-side deltas
    evaluation/agent/reports/rag_v3_cross_session_<config>.json  — per-config
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "evaluation" / "agent" / "reports"
STORAGE_SQL = PROJECT_ROOT / "storage" / "sql"

PYTHON = sys.executable


# Weight configs. `cos` is used when embedder is ON (Phase B); `word` would
# be used for Phase A but ablation runs Phase B only (we already have the
# Phase A vs B comparison as the sanity check).
CONFIGS: Dict[str, Dict[str, float]] = {
    "baseline":     {"cos": 0.55, "entity": 0.25, "decay": 0.12, "access": 0.03, "importance": 0.05},
    "cos_only":     {"cos": 1.00, "entity": 0.00, "decay": 0.00, "access": 0.00, "importance": 0.00},
    "no_decay":     {"cos": 0.60, "entity": 0.27, "decay": 0.00, "access": 0.05, "importance": 0.08},
    "no_entity":    {"cos": 0.70, "entity": 0.00, "decay": 0.15, "access": 0.05, "importance": 0.10},
    "equal":        {"cos": 0.20, "entity": 0.20, "decay": 0.20, "access": 0.20, "importance": 0.20},
    "entity_heavy": {"cos": 0.35, "entity": 0.50, "decay": 0.10, "access": 0.00, "importance": 0.05},
}


def weights_to_cli(w: Dict[str, float]) -> str:
    return ",".join(f"{k}={v}" for k, v in w.items())


def run_one_config(
    name: str,
    weights: Dict[str, float],
    limit: Optional[int],
    dataset_path: Optional[Path] = None,
) -> Path:
    db_path = STORAGE_SQL / f"memory_ablation_{name}.duckdb"
    out_path = REPORTS_DIR / f"rag_v3_cross_session_{name}.json"

    # Wipe prior DB so runs are reproducible (ablation isolates scoring,
    # not the accumulation effect of prior runs' long-term writes).
    if db_path.exists():
        db_path.unlink()

    cmd = [
        PYTHON, "-u", "evaluation/run_cross_session_evaluation.py",
        "--config-label", name,
        "--memory-db", str(db_path),
        "--output", str(out_path),
        "--weights", weights_to_cli(weights),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    if dataset_path is not None:
        cmd += ["--dataset", str(dataset_path)]

    print(f"\n{'=' * 70}")
    print(f"  [ablation] running config: {name}")
    print(f"  weights: {weights}")
    print(f"{'=' * 70}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return out_path


def summarise(per_config_paths: Dict[str, Path]) -> Dict[str, Any]:
    rows = []
    for name, path in per_config_paths.items():
        if not path.exists():
            rows.append({"config": name, "error": "report missing"})
            continue
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        agg = report.get("aggregate") or {}
        rows.append({
            "config": name,
            "weights": report.get("weights"),
            "cross_session_hit_rate": agg.get("cross_session_hit_rate"),
            "correct_session_recall_rate": agg.get("correct_session_recall_rate"),
            "cross_session_leak_rate": agg.get("cross_session_leak_rate"),
            "avg_lt_top_score_positive": agg.get("avg_lt_top_score_positive"),
            "avg_lt_top_score_negative": agg.get("avg_lt_top_score_negative"),
            "score_gap_pos_minus_neg": agg.get("score_gap_pos_minus_neg"),
        })
    return {
        "configs": list(per_config_paths.keys()),
        "rows": rows,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 100)
    print("  ABLATION SUMMARY")
    print("=" * 100)
    header = f"{'config':<14} | {'hit_rate':>10} | {'correct':>10} | {'leak':>8} | {'+score':>8} | {'-score':>8} | {'gap':>8}"
    print(header)
    print("-" * len(header))
    for r in summary["rows"]:
        if "error" in r:
            print(f"{r['config']:<14} | ERROR: {r['error']}")
            continue
        hr = r.get("cross_session_hit_rate")
        cr = r.get("correct_session_recall_rate")
        lk = r.get("cross_session_leak_rate")
        ps = r.get("avg_lt_top_score_positive") or 0
        ns = r.get("avg_lt_top_score_negative") or 0
        gp = r.get("score_gap_pos_minus_neg") or 0

        def pct(x):
            return f"{x*100:.2f}%" if x is not None else "   n/a"
        print(f"{r['config']:<14} | {pct(hr):>10} | {pct(cr):>10} | {pct(lk):>8} | "
              f"{ps:>8.4f} | {ns:>8.4f} | {gp:>+8.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit conversations per config (smoke mode)")
    parser.add_argument(
        "--configs", type=str, default=None,
        help="Comma-separated subset of configs to run (default: all 6)",
    )
    parser.add_argument(
        "--dataset", type=Path, default=None,
        help="Override the cross-session dataset path (default: v3 cross_session)",
    )
    args = parser.parse_args()

    to_run = list(CONFIGS.keys())
    if args.configs:
        to_run = [c.strip() for c in args.configs.split(",") if c.strip()]
        missing = [c for c in to_run if c not in CONFIGS]
        if missing:
            raise SystemExit(f"Unknown configs: {missing}")

    paths: Dict[str, Path] = {}
    for name in to_run:
        paths[name] = run_one_config(name, CONFIGS[name], args.limit,
                                     dataset_path=args.dataset)

    summary = summarise(paths)
    print_summary(summary)

    out = REPORTS_DIR / "ablation_summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary: {out}")


if __name__ == "__main__":
    main()
