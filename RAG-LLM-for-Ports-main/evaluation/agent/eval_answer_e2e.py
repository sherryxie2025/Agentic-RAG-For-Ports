"""
End-to-end answer quality scoring for multi-turn and cross-session evaluation.

Wraps the single-turn eval_answer_quality.py functions so they can be called
per-turn in multi-turn/cross-session drivers. Handles the key difference:

  - Turns with `from_sample_id` (session 1): have reference_answer and
    expected_evidence_keywords → all 7 metrics available.
  - Free-form turns (session 2): no reference_answer → only citation_validity,
    grounding_flag, and optionally LLM judge (using memory_context as reference).

Also provides an aggregate function and a combined report printer.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_answer_quality import (
    keyword_coverage,
    citation_validity,
    numerical_accuracy,
    grounding_flag,
    embedding_similarity,
    rouge_l_f1,
)


def score_answer_e2e(
    answer_text: str,
    final_answer: Dict[str, Any],
    evidence_bundle: Optional[Dict[str, Any]],
    turn_spec: Dict[str, Any],
    base_samples: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Score a single turn's answer for end-to-end quality.

    Returns a dict with available metrics (some may be None if the turn
    lacks reference data).
    """
    result: Dict[str, Any] = {}

    # Resolve golden fields: from turn_spec directly or via base sample
    sample_id = turn_spec.get("from_sample_id") or turn_spec.get("derived_from_sample_id")
    golden = {}
    if sample_id and base_samples:
        golden = base_samples.get(sample_id, {})

    expected_kw = turn_spec.get("expected_evidence_keywords") or golden.get("expected_evidence_keywords", [])
    reference_answer = turn_spec.get("reference_answer") or golden.get("reference_answer", "")

    sources_used = (
        final_answer.get("sources_used", []) if isinstance(final_answer, dict) else []
    )

    # --- Objective metrics ---

    # 1. Keyword coverage (available when golden keywords exist)
    if expected_kw:
        result["keyword_coverage"] = round(keyword_coverage(expected_kw, answer_text), 4)
    else:
        result["keyword_coverage"] = None

    # 2. Citation validity (always available — checks [sql]/[doc]/[rule]/[graph] tags)
    if evidence_bundle:
        result["citation_validity"] = round(citation_validity(sources_used, evidence_bundle), 4)
    else:
        result["citation_validity"] = None

    # 3. Numerical accuracy (available when reference_answer has numbers)
    if reference_answer:
        num_acc = numerical_accuracy(reference_answer, answer_text)
        result["numerical_accuracy"] = round(num_acc, 4) if num_acc is not None else None
    else:
        result["numerical_accuracy"] = None

    # 4. Grounding flag (always available)
    result["grounding"] = grounding_flag(final_answer) if isinstance(final_answer, dict) else "unknown"

    # 5. Embedding similarity (available when reference_answer exists)
    if reference_answer and answer_text:
        sim = embedding_similarity(reference_answer, answer_text)
        result["embedding_similarity"] = round(sim, 4) if sim is not None else None
    else:
        result["embedding_similarity"] = None

    # 6. ROUGE-L F1 (available when reference_answer exists)
    if reference_answer and answer_text:
        rouge = rouge_l_f1(reference_answer, answer_text)
        result["rouge_l_f1"] = round(rouge, 4) if rouge is not None else None
    else:
        result["rouge_l_f1"] = None

    result["has_reference"] = bool(reference_answer)
    result["has_golden_keywords"] = bool(expected_kw)

    return result


def aggregate_answer_quality(
    per_turn_scores: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate answer quality scores across turns."""

    def _avg(key: str) -> Optional[float]:
        vals = [r[key] for r in per_turn_scores if r.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    # Split by whether they have reference or not
    with_ref = [r for r in per_turn_scores if r.get("has_reference")]
    without_ref = [r for r in per_turn_scores if not r.get("has_reference")]

    grounding_dist: Dict[str, int] = {}
    for r in per_turn_scores:
        g = r.get("grounding", "unknown")
        grounding_dist[g] = grounding_dist.get(g, 0) + 1

    return {
        "total_turns": len(per_turn_scores),
        "turns_with_reference": len(with_ref),
        "turns_without_reference": len(without_ref),
        "keyword_coverage": _avg("keyword_coverage"),
        "citation_validity": _avg("citation_validity"),
        "numerical_accuracy": _avg("numerical_accuracy"),
        "embedding_similarity": _avg("embedding_similarity"),
        "rouge_l_f1": _avg("rouge_l_f1"),
        "grounding_distribution": grounding_dist,
    }


def print_answer_quality_report(agg: Dict[str, Any], label: str = "") -> None:
    print(f"\n{'=' * 60}")
    print(f"  ANSWER QUALITY (end-to-end) {label}")
    print(f"{'=' * 60}")
    print(f"  Total turns: {agg['total_turns']} "
          f"(with ref: {agg['turns_with_reference']}, "
          f"without: {agg['turns_without_reference']})")

    def _pct(v):
        return f"{v:.2%}" if v is not None else "n/a"

    print(f"\n  Keyword coverage:      {_pct(agg['keyword_coverage'])}")
    print(f"  Citation validity:     {_pct(agg['citation_validity'])}")
    print(f"  Numerical accuracy:    {_pct(agg['numerical_accuracy'])}")
    print(f"  Embedding similarity:  {_pct(agg['embedding_similarity'])}")
    print(f"  ROUGE-L F1:            {_pct(agg['rouge_l_f1'])}")
    print(f"  Grounding dist:        {agg['grounding_distribution']}")
