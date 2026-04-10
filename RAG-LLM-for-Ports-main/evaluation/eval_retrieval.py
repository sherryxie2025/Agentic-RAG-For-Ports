"""
Retrieval Evaluation: Per-source retrieval + reranking metrics.

Measures retrieval quality separately for each data source:
- Vector (documents): Recall@k, Precision@k, MRR, nDCG@k
- SQL: Table coverage, execution success, row count match
- Rules: Variable match, threshold exact match
- Graph: Entity recall, relationship recall

Also computes reranking lift (post-rerank vs pre-rerank).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    hits = len(top_k & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    if not retrieved_ids[:k]:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(top_k)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank of first relevant hit."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    if not relevant_ids:
        return 1.0

    def dcg(ids: List[str]) -> float:
        return sum(
            1.0 / math.log2(i + 2) for i, rid in enumerate(ids) if rid in relevant_ids
        )

    actual_dcg = dcg(retrieved_ids[:k])
    ideal_k = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-source evaluators
# ---------------------------------------------------------------------------

@dataclass
class RetrievalMetrics:
    vector: Dict[str, float] = field(default_factory=dict)
    sql: Dict[str, float] = field(default_factory=dict)
    rules: Dict[str, float] = field(default_factory=dict)
    graph: Dict[str, float] = field(default_factory=dict)
    reranking_lift: Dict[str, float] = field(default_factory=dict)
    samples_counted: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector": self.vector,
            "sql": self.sql,
            "rules": self.rules,
            "graph": self.graph,
            "reranking_lift": self.reranking_lift,
            "samples_counted": self.samples_counted,
        }


def evaluate_vector(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate document retrieval.

    Each result dict should have:
        - id: matching the golden id
        - retrieved_chunk_ids: list of chunk_ids returned (post-rerank, top-5)
        - pre_rerank_chunk_ids: chunks before reranking (top-20)

    Metrics @5 use post-rerank output; metrics @20 use pre-rerank output
    (otherwise recall@20 is bounded by post-rerank size).
    """
    golden_by_id = {g["id"]: g for g in golden}

    r5, r20, p5, mrr_total, ndcg10 = 0.0, 0.0, 0.0, 0.0, 0.0
    count = 0

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("golden_vector"):
            continue
        relevant = set(g["golden_vector"].get("relevant_chunk_ids", []))
        if not relevant:
            continue

        post = r.get("retrieved_chunk_ids", [])
        pre = r.get("pre_rerank_chunk_ids", []) or post  # fallback

        # @5 metrics from reranked output
        r5 += recall_at_k(post, relevant, 5)
        p5 += precision_at_k(post, relevant, 5)
        mrr_total += mrr(post, relevant)
        # @10 / @20 metrics from pre-rerank (larger candidate pool)
        ndcg10 += ndcg_at_k(pre, relevant, 10)
        r20 += recall_at_k(pre, relevant, 20)
        count += 1

    if count == 0:
        return {"count": 0}

    return {
        "recall@5": round(r5 / count, 4),
        "recall@20": round(r20 / count, 4),
        "precision@5": round(p5 / count, 4),
        "mrr": round(mrr_total / count, 4),
        "ndcg@10": round(ndcg10 / count, 4),
        "count": count,
    }


def evaluate_reranking_lift(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Compute nDCG@5 lift from reranking (post-rerank - pre-rerank)."""
    golden_by_id = {g["id"]: g for g in golden}
    lifts_ndcg = []
    lifts_r5 = []
    top1_hit_before = 0
    top1_hit_after = 0
    count = 0

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("golden_vector"):
            continue
        relevant = set(g["golden_vector"].get("relevant_chunk_ids", []))
        if not relevant:
            continue

        pre = r.get("pre_rerank_chunk_ids", [])
        post = r.get("retrieved_chunk_ids", [])
        if not pre or not post:
            continue

        pre_ndcg = ndcg_at_k(pre, relevant, 5)
        post_ndcg = ndcg_at_k(post, relevant, 5)
        lifts_ndcg.append(post_ndcg - pre_ndcg)

        pre_r5 = recall_at_k(pre, relevant, 5)
        post_r5 = recall_at_k(post, relevant, 5)
        lifts_r5.append(post_r5 - pre_r5)

        if pre and pre[0] in relevant:
            top1_hit_before += 1
        if post and post[0] in relevant:
            top1_hit_after += 1
        count += 1

    if count == 0:
        return {"count": 0}

    return {
        "ndcg@5_lift": round(sum(lifts_ndcg) / count, 4),
        "recall@5_lift": round(sum(lifts_r5) / count, 4),
        "top1_hit_before": round(top1_hit_before / count, 4),
        "top1_hit_after": round(top1_hit_after / count, 4),
        "top1_lift": round((top1_hit_after - top1_hit_before) / count, 4),
        "count": count,
    }


def evaluate_sql(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Evaluate SQL retrieval.

    Each result dict should have:
        - id: matching golden id
        - tables_used: list of table names queried
        - execution_ok: bool
        - row_count: int
    """
    golden_by_id = {g["id"]: g for g in golden}
    table_f1 = 0.0
    table_f1_count = 0
    exec_ok_count = 0
    row_count_reasonable = 0
    row_count_total = 0  # denominator fix: only count samples that have expected_row_count
    count = 0

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("golden_sql"):
            continue
        gs = g["golden_sql"]
        expected_tables = set((gs.get("expected_tables") or {}).keys())

        actual_tables = set(r.get("tables_used", []))
        if expected_tables:
            tp = len(expected_tables & actual_tables)
            fp = len(actual_tables - expected_tables)
            fn = len(expected_tables - actual_tables)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
            table_f1 += f1
            table_f1_count += 1

        if r.get("execution_ok"):
            exec_ok_count += 1

        # Expected row count (only counted when specified in golden)
        expected_rc = gs.get("expected_row_count")
        if expected_rc is not None:
            row_count_total += 1
            actual_rc = r.get("row_count", -1)
            if expected_rc == 0 and actual_rc == 0:
                row_count_reasonable += 1
            elif expected_rc > 0 and actual_rc > 0:
                row_count_reasonable += 1

        count += 1

    if count == 0:
        return {"count": 0}

    result: Dict[str, float] = {
        "table_f1": round(table_f1 / max(table_f1_count, 1), 4),
        "execution_ok_rate": round(exec_ok_count / count, 4),
        "count": count,
    }
    # Only report row_count metric when at least one sample has the annotation
    if row_count_total > 0:
        result["row_count_reasonable"] = round(row_count_reasonable / row_count_total, 4)
        result["row_count_annotated"] = row_count_total
    return result


def evaluate_rules(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate rule retrieval by variable match."""
    golden_by_id = {g["id"]: g for g in golden}
    var_recall = 0.0
    var_precision = 0.0
    count = 0

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("golden_rules"):
            continue
        expected_vars = set(
            v.lower() for v in g["golden_rules"].get("expected_rule_variables", [])
        )
        if not expected_vars:
            continue

        actual_vars = set(v.lower() for v in r.get("rule_variables", []))
        if actual_vars:
            tp = len(expected_vars & actual_vars)
            var_recall += tp / len(expected_vars)
            var_precision += tp / len(actual_vars)
        count += 1

    if count == 0:
        return {"count": 0}

    return {
        "variable_recall": round(var_recall / count, 4),
        "variable_precision": round(var_precision / count, 4),
        "count": count,
    }


def evaluate_graph(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate graph reasoning by entity + relationship overlap."""
    golden_by_id = {g["id"]: g for g in golden}
    entity_recall = 0.0
    rel_recall = 0.0
    path_found_rate = 0
    count = 0

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("golden_graph"):
            continue
        expected_ents = set(
            e.lower() for e in g["golden_graph"].get("expected_entities", [])
        )
        expected_rels = set(
            rel.upper() for rel in g["golden_graph"].get("expected_relationships", [])
        )

        actual_ents = set(e.lower() for e in r.get("entities", []))
        actual_rels = set(rel.upper() for rel in r.get("relationships", []))

        if expected_ents:
            entity_recall += len(expected_ents & actual_ents) / len(expected_ents)
        if expected_rels:
            rel_recall += len(expected_rels & actual_rels) / len(expected_rels)
        if r.get("path_count", 0) > 0:
            path_found_rate += 1
        count += 1

    if count == 0:
        return {"count": 0}

    return {
        "entity_recall": round(entity_recall / count, 4),
        "relationship_recall": round(rel_recall / count, 4),
        "path_found_rate": round(path_found_rate / count, 4),
        "count": count,
    }


def evaluate_retrieval_all(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> RetrievalMetrics:
    """Compute all per-source retrieval metrics."""
    return RetrievalMetrics(
        vector=evaluate_vector(results, golden),
        sql=evaluate_sql(results, golden),
        rules=evaluate_rules(results, golden),
        graph=evaluate_graph(results, golden),
        reranking_lift=evaluate_reranking_lift(results, golden),
        samples_counted={"total": len(results)},
    )


def print_retrieval_report(metrics: RetrievalMetrics) -> None:
    """Pretty-print retrieval metrics."""
    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION")
    print("=" * 70)

    for source_name in ["vector", "sql", "rules", "graph"]:
        source_metrics = getattr(metrics, source_name)
        if not source_metrics or source_metrics.get("count", 0) == 0:
            print(f"\n  {source_name.upper()}: no samples")
            continue
        print(f"\n  {source_name.upper()} ({source_metrics['count']} samples)")
        for k, v in source_metrics.items():
            if k == "count":
                continue
            print(f"    {k:<24}: {v:.4f}" if isinstance(v, float) else f"    {k:<24}: {v}")

    if metrics.reranking_lift and metrics.reranking_lift.get("count", 0) > 0:
        print(f"\n  RERANKING LIFT ({metrics.reranking_lift['count']} samples)")
        for k, v in metrics.reranking_lift.items():
            if k == "count":
                continue
            sign = "+" if isinstance(v, float) and v >= 0 else ""
            print(f"    {k:<24}: {sign}{v:.4f}" if isinstance(v, float) else f"    {k:<24}: {v}")


if __name__ == "__main__":
    # Demo with synthetic data
    golden = [
        {
            "id": "Q1",
            "golden_vector": {"relevant_chunk_ids": ["c1", "c2", "c3"]},
        },
    ]
    preds = [
        {
            "id": "Q1",
            "retrieved_chunk_ids": ["c1", "cx", "c2", "cy", "c3"],
            "pre_rerank_chunk_ids": ["cx", "cy", "c1", "c2", "c3"],
        },
    ]
    metrics = evaluate_retrieval_all(preds, golden)
    print_retrieval_report(metrics)
