"""
Guardrail Evaluation: Out-of-domain, empty evidence, conflict detection.

Tests the agent's behavior on edge cases that have historically caused
failure modes:
- Out-of-domain queries (should refuse/redirect)
- Impossible queries (future dates, nonexistent entities)
- Empty evidence (should say "no data" not fabricate)
- Ambiguous queries (should clarify or state assumptions)
- Evidence conflicts (should detect and report)
- False premises (should identify and correct)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Guardrail detection patterns
# ---------------------------------------------------------------------------

# Signal phrases that indicate the agent is refusing or indicating OOD
_OOD_PHRASES = [
    "out of domain", "out of scope", "not a port", "beyond the scope",
    "cannot help", "i don't have", "no information about",
    "not applicable", "not related to port", "outside the scope",
    "outside my scope", "falls outside", "focused on port", "focus on port",
    "port operations", "not about port", "port decision-support",
    "domain gate", "only answer questions", "do not have information",
]

# Signal phrases for "no data"
_NO_DATA_PHRASES = [
    "no data", "no records", "no matching", "not found", "no results",
    "empty", "does not exist", "no rows", "returned zero", "no entries",
]

# Signal phrases for conflict reporting
_CONFLICT_PHRASES = [
    "conflict", "contradiction", "discrepancy", "does not match",
    "inconsistent", "differs from", "however", "but the data shows",
    "while the rule says", "the actual value", "exceeds the threshold",
]

# Signal phrases for clarification
_CLARIFY_PHRASES = [
    "clarify", "could you specify", "which", "what specifically",
    "please provide", "more context", "unclear", "ambiguous",
    "assume", "assuming", "i'll interpret",
]

# False-premise detection
_FALSE_PREMISE_PHRASES = [
    "not possible", "future date", "cannot predict", "no data yet",
    "incorrect assumption", "false premise", "this is in the future",
    "data does not exist for",
]


def _contains_any(text: str, phrases: List[str]) -> bool:
    text_lower = text.lower()
    return any(p in text_lower for p in phrases)


# ---------------------------------------------------------------------------
# Per-guardrail-type evaluators
# ---------------------------------------------------------------------------

def evaluate_ood_response(answer: str, final: Dict[str, Any]) -> Dict[str, Any]:
    """Did the agent properly handle an out-of-domain query?"""
    is_refusing = _contains_any(answer, _OOD_PHRASES)

    # Also check: fallback flags
    llm_fallback = final.get("knowledge_fallback_used", False) if isinstance(final, dict) else False
    has_fake_sources = bool(final.get("sources_used")) if isinstance(final, dict) else False

    # A good OOD response refuses AND does not cite fake sources
    pass_score = 1.0 if (is_refusing and not has_fake_sources) else 0.0
    return {
        "passed": pass_score,
        "refused": is_refusing,
        "llm_fallback_used": llm_fallback,
        "falsely_cited_sources": has_fake_sources and not is_refusing,
    }


def evaluate_empty_evidence(answer: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Did the agent acknowledge empty evidence instead of fabricating?"""
    # Check: any evidence actually retrieved?
    has_docs = bool(bundle.get("documents", []))
    has_sql = any(
        (r.get("row_count", 0) or 0) > 0
        for r in bundle.get("sql_results", []) if isinstance(r, dict)
    )
    has_rules = bool(
        (bundle.get("rules", {}) or {}).get("matched_rules", [])
    )
    has_graph = bool(
        (bundle.get("graph", {}) or {}).get("reasoning_paths", [])
    )

    any_evidence = has_docs or has_sql or has_rules or has_graph
    says_no_data = _contains_any(answer, _NO_DATA_PHRASES)

    # If no evidence retrieved, answer should indicate it
    if not any_evidence:
        passed = 1.0 if says_no_data else 0.0
    else:
        # Evidence was retrieved but might still be insufficient — pass by default
        passed = 1.0

    return {
        "passed": passed,
        "any_evidence_retrieved": any_evidence,
        "acknowledged_empty": says_no_data,
    }


def evaluate_conflict_detection(
    answer: str,
    bundle: Dict[str, Any],
    expected_conflicts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Did the agent detect and report expected evidence conflicts?"""
    conflict_annotations = bundle.get("conflict_annotations", []) or []
    mentions_conflict = _contains_any(answer, _CONFLICT_PHRASES)

    # Expected conflict variables
    expected_vars = set(
        c.get("variable", "").lower() for c in expected_conflicts
    )

    detected_vars = set()
    for ca in conflict_annotations:
        var = (ca.get("rule_variable") or "").lower()
        if var:
            detected_vars.add(var)

    if not expected_vars:
        recall = 1.0
    else:
        recall = len(expected_vars & detected_vars) / len(expected_vars)

    passed = 1.0 if (mentions_conflict or len(detected_vars) > 0) else 0.0

    return {
        "passed": passed,
        "conflicts_detected_count": len(conflict_annotations),
        "conflict_recall": round(recall, 4),
        "mentioned_in_answer": mentions_conflict,
    }


def evaluate_ambiguous(answer: str) -> Dict[str, Any]:
    """Did the agent clarify an ambiguous query or state its assumptions?"""
    clarified = _contains_any(answer, _CLARIFY_PHRASES)
    return {
        "passed": 1.0 if clarified else 0.0,
        "clarified_or_assumed": clarified,
    }


def evaluate_false_premise(answer: str) -> Dict[str, Any]:
    """Did the agent identify a false premise?"""
    identified = _contains_any(answer, _FALSE_PREMISE_PHRASES)
    return {
        "passed": 1.0 if identified else 0.0,
        "identified_false_premise": identified,
    }


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

@dataclass
class GuardrailMetrics:
    pass_rates: Dict[str, float] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass_rates": {k: round(v, 4) for k, v in self.pass_rates.items()},
            "counts": self.counts,
            "sample_details": self.details[:10],
        }


def evaluate_guardrails(
    results: List[Dict[str, Any]],
    golden: List[Dict[str, Any]],
) -> GuardrailMetrics:
    """
    Evaluate all guardrail samples. Only samples with 'guardrail_type' field
    are evaluated by this module.
    """
    golden_by_id = {g["id"]: g for g in golden}

    type_pass = {}
    type_count = {}
    details = []

    for r in results:
        g = golden_by_id.get(r.get("id"))
        if not g or not g.get("guardrail_type"):
            continue

        gtype = g["guardrail_type"]
        type_count[gtype] = type_count.get(gtype, 0) + 1

        answer = r.get("answer_text", "") or ""
        bundle = r.get("evidence_bundle", {}) or {}
        final = r.get("final_answer", {}) or {}

        if gtype == "out_of_domain":
            eval_result = evaluate_ood_response(answer, final)
        elif gtype in ("empty_evidence", "impossible_query"):
            eval_result = evaluate_empty_evidence(answer, bundle)
            if gtype == "impossible_query":
                # Extra check: should identify impossibility
                eval_result["identified_impossibility"] = _contains_any(
                    answer, _FALSE_PREMISE_PHRASES + ["future", "cannot", "not possible"]
                )
        elif gtype in ("evidence_conflict", "doc_vs_sql_conflict", "doc_vs_rule_conflict"):
            expected_conflicts = g.get("expected_conflicts", [])
            eval_result = evaluate_conflict_detection(answer, bundle, expected_conflicts)
        elif gtype == "ambiguous_query":
            eval_result = evaluate_ambiguous(answer)
        elif gtype == "false_premise":
            eval_result = evaluate_false_premise(answer)
        else:
            eval_result = {"passed": 0.0, "unknown_type": gtype}

        type_pass[gtype] = type_pass.get(gtype, 0.0) + eval_result.get("passed", 0.0)
        details.append({
            "id": g["id"],
            "guardrail_type": gtype,
            "result": eval_result,
        })

    pass_rates = {
        t: type_pass[t] / type_count[t] if type_count[t] else 0.0
        for t in type_pass
    }

    return GuardrailMetrics(
        pass_rates=pass_rates,
        counts=type_count,
        details=details,
    )


def print_guardrail_report(metrics: GuardrailMetrics) -> None:
    print("\n" + "=" * 70)
    print("  GUARDRAIL EVALUATION")
    print("=" * 70)
    print(f"  {'Guardrail Type':<25} {'Pass Rate':>12} {'Samples':>8}")
    print(f"  {'-' * 48}")
    for gtype in sorted(metrics.pass_rates.keys()):
        rate = metrics.pass_rates[gtype]
        cnt = metrics.counts[gtype]
        flag = "PASS" if rate >= 0.8 else ("WARN" if rate >= 0.5 else "FAIL")
        print(f"  {gtype:<25} {rate:>12.2%} {cnt:>8}  [{flag}]")
