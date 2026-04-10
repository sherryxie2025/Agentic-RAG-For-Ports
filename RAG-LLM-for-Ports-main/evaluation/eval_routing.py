"""
Routing Evaluation: Multi-label capability classification metrics.

The router is NOT a single-label classifier — it predicts 4 independent
boolean capabilities (vector, sql, rules, graph). Standard single-label
accuracy is misleading. This module computes:

- Per-capability Precision / Recall / F1
- Micro-F1 (over all 4*N predictions)
- Macro-F1 (average of 4 per-capability F1 scores)
- Exact-match rate (all 4 bools correct)
- Over-routing / Under-routing rates (cost vs coverage)
- Question-type and answer-mode classification accuracy
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CAPABILITIES = ["vector", "sql", "rules", "graph"]


@dataclass
class RoutingMetrics:
    """Container for all routing evaluation metrics."""
    per_capability: Dict[str, Dict[str, float]] = field(default_factory=dict)
    micro: Dict[str, float] = field(default_factory=dict)
    macro: Dict[str, float] = field(default_factory=dict)
    exact_match_rate: float = 0.0
    over_routing_rate: float = 0.0
    under_routing_rate: float = 0.0
    question_type_accuracy: float = 0.0
    answer_mode_accuracy: float = 0.0
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_capability": self.per_capability,
            "micro": self.micro,
            "macro": self.macro,
            "exact_match_rate": round(self.exact_match_rate, 4),
            "over_routing_rate": round(self.over_routing_rate, 4),
            "under_routing_rate": round(self.under_routing_rate, 4),
            "question_type_accuracy": round(self.question_type_accuracy, 4),
            "answer_mode_accuracy": round(self.answer_mode_accuracy, 4),
            "total_samples": self.total_samples,
        }


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Precision/Recall/F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def _expected_caps(sample: Dict[str, Any]) -> Dict[str, bool]:
    """Extract expected capability flags from a golden sample."""
    return {
        "vector": bool(sample.get("needs_vector", False)),
        "sql": bool(sample.get("needs_sql", False)),
        "rules": bool(sample.get("needs_rules", False)),
        "graph": bool(sample.get("needs_graph", False)),
    }


def _actual_caps(router_output: Dict[str, Any]) -> Dict[str, bool]:
    """
    Extract actual capability flags from router output.
    Handles both IntentRouter.route() output and AgentState.
    """
    return {
        "vector": bool(router_output.get("needs_vector", False)),
        "sql": bool(router_output.get("needs_sql", False)),
        "rules": bool(router_output.get("needs_rules", False)),
        "graph": bool(
            router_output.get("needs_graph", False)
            or router_output.get("needs_graph_reasoning", False)
        ),
    }


def evaluate_routing(
    predictions: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> RoutingMetrics:
    """
    Compute routing metrics by matching predictions to golden samples by id.

    Args:
        predictions: list of dicts with 'id' and router decision fields
            (needs_vector, needs_sql, needs_rules, needs_graph,
             question_type, answer_mode)
        golden: list of golden samples with expected values

    Returns:
        RoutingMetrics object with all computed metrics.
    """
    golden_by_id = {g["id"]: g for g in golden if "id" in g}
    pred_by_id = {p["id"]: p for p in predictions if "id" in p}

    # Per-capability confusion counts
    cap_counts = {cap: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for cap in CAPABILITIES}

    exact_match = 0
    over_count = 0
    under_count = 0
    qtype_correct = 0
    amode_correct = 0
    qtype_total = 0
    amode_total = 0
    total = 0

    for gid, g in golden_by_id.items():
        p = pred_by_id.get(gid)
        if p is None:
            continue
        total += 1

        expected = _expected_caps(g)
        actual = _actual_caps(p)

        sample_exact = True
        expected_count = sum(expected.values())
        actual_count = sum(actual.values())

        for cap in CAPABILITIES:
            e, a = expected[cap], actual[cap]
            if e and a:
                cap_counts[cap]["tp"] += 1
            elif not e and a:
                cap_counts[cap]["fp"] += 1
            elif e and not a:
                cap_counts[cap]["fn"] += 1
            else:
                cap_counts[cap]["tn"] += 1
            if e != a:
                sample_exact = False

        if sample_exact:
            exact_match += 1
        if actual_count > expected_count:
            over_count += 1
        elif actual_count < expected_count:
            under_count += 1

        # Question type / answer mode
        if "question_type" in g:
            qtype_total += 1
            if p.get("question_type") == g["question_type"]:
                qtype_correct += 1
        if "answer_mode" in g:
            amode_total += 1
            if p.get("answer_mode") == g["answer_mode"]:
                amode_correct += 1

    # Per-capability P/R/F1
    per_cap = {}
    for cap in CAPABILITIES:
        c = cap_counts[cap]
        per_cap[cap] = _prf(c["tp"], c["fp"], c["fn"])

    # Micro: sum TP/FP/FN across all capabilities
    micro_tp = sum(c["tp"] for c in cap_counts.values())
    micro_fp = sum(c["fp"] for c in cap_counts.values())
    micro_fn = sum(c["fn"] for c in cap_counts.values())
    micro = _prf(micro_tp, micro_fp, micro_fn)

    # Macro: average of per-capability F1
    macro_p = sum(per_cap[cap]["precision"] for cap in CAPABILITIES) / len(CAPABILITIES)
    macro_r = sum(per_cap[cap]["recall"] for cap in CAPABILITIES) / len(CAPABILITIES)
    macro_f1 = sum(per_cap[cap]["f1"] for cap in CAPABILITIES) / len(CAPABILITIES)
    macro = {
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "f1": round(macro_f1, 4),
    }

    return RoutingMetrics(
        per_capability=per_cap,
        micro=micro,
        macro=macro,
        exact_match_rate=exact_match / total if total else 0.0,
        over_routing_rate=over_count / total if total else 0.0,
        under_routing_rate=under_count / total if total else 0.0,
        question_type_accuracy=qtype_correct / qtype_total if qtype_total else 0.0,
        answer_mode_accuracy=amode_correct / amode_total if amode_total else 0.0,
        total_samples=total,
    )


def print_routing_report(metrics: RoutingMetrics) -> None:
    """Pretty-print routing metrics."""
    print("\n" + "=" * 70)
    print("  ROUTING EVALUATION")
    print("=" * 70)
    print(f"  Samples evaluated: {metrics.total_samples}")
    print(f"  Exact-match rate:  {metrics.exact_match_rate:.2%}")
    print(f"  Over-routing:      {metrics.over_routing_rate:.2%}  (extra sources activated)")
    print(f"  Under-routing:     {metrics.under_routing_rate:.2%}  (missing sources)")
    print(f"\n  Question-type accuracy: {metrics.question_type_accuracy:.2%}")
    print(f"  Answer-mode accuracy:   {metrics.answer_mode_accuracy:.2%}")

    print("\n  Per-Capability Metrics:")
    print(f"  {'Capability':<12} {'P':>8} {'R':>8} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    for cap in CAPABILITIES:
        m = metrics.per_capability[cap]
        print(f"  {cap:<12} {m['precision']:>8.3f} {m['recall']:>8.3f} "
              f"{m['f1']:>8.3f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")

    print(f"\n  Micro-avg: P={metrics.micro['precision']:.3f}  "
          f"R={metrics.micro['recall']:.3f}  F1={metrics.micro['f1']:.3f}")
    print(f"  Macro-avg: P={metrics.macro['precision']:.3f}  "
          f"R={metrics.macro['recall']:.3f}  F1={metrics.macro['f1']:.3f}")


if __name__ == "__main__":
    # Standalone demo with synthetic data
    golden = [
        {"id": "Q1", "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": False,
         "question_type": "hybrid_reasoning", "answer_mode": "decision_support"},
        {"id": "Q2", "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
         "question_type": "structured_data", "answer_mode": "lookup"},
    ]
    preds = [
        {"id": "Q1", "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": False,
         "question_type": "hybrid_reasoning", "answer_mode": "decision_support"},
        {"id": "Q2", "needs_vector": True, "needs_sql": True, "needs_rules": False, "needs_graph": False,
         "question_type": "structured_data", "answer_mode": "lookup"},
    ]
    metrics = evaluate_routing(preds, golden)
    print_routing_report(metrics)
    print(json.dumps(metrics.to_dict(), indent=2))
