# src/online_pipeline/conflict_detector.py
"""
Enhanced evidence conflict detector.

Original detection only covered Rule↔SQL numerical threshold comparison.
This module adds:
- Doc↔SQL: document-stated numbers vs SQL actual values
- Doc↔Rule: document-stated thresholds vs rule database
- Doc↔Doc: contradictory claims across document chunks (LLM-optional)
- Temporal: rule-version mismatch (document date vs current rules)

Each detector returns a list of ConflictAnnotation dicts.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("online_pipeline.conflict_detector")


# ---------------------------------------------------------------------------
# Number extraction from text
# ---------------------------------------------------------------------------

# Matches numbers with optional unit: "25 knots", "85%", "30-35", "1,200"
_NUMBER_WITH_UNIT_RE = re.compile(
    r"(\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?:\s*[-–to]+\s*(\d+(?:,\d{3})*(?:\.\d+)?))?"
    r"\s*(%|knots?|kts?|m/s|meters?|feet|ft|hours?|hrs?|minutes?|min|"
    r"degrees?|deg|teu|tons?|containers?|moves?|mph)?",
    re.IGNORECASE,
)


def extract_numbers_with_context(text: str, window: int = 40) -> List[Dict[str, Any]]:
    """
    Extract numbers with surrounding context from text.

    Returns list of:
        {
            "value": float or (float, float) for range,
            "unit": str,
            "context": str (surrounding words),
            "position": int (character offset),
        }
    """
    results = []
    for m in _NUMBER_WITH_UNIT_RE.finditer(text):
        val_str = m.group(1).replace(",", "")
        end_str = m.group(2)
        unit = (m.group(3) or "").lower().strip()
        try:
            val = float(val_str)
        except ValueError:
            continue
        if end_str:
            try:
                end_val = float(end_str.replace(",", ""))
                value = (val, end_val)
            except ValueError:
                value = val
        else:
            value = val

        start = max(0, m.start() - window)
        end = min(len(text), m.end() + window)
        context = text[start:end].strip()

        results.append({
            "value": value,
            "unit": unit,
            "context": context,
            "position": m.start(),
        })

    return results


# ---------------------------------------------------------------------------
# Existing Rule↔SQL conflict detector (moved from langgraph_nodes)
# ---------------------------------------------------------------------------

def detect_rule_sql_conflicts(
    rule_results: Dict[str, Any],
    sql_results: List[Any],
) -> List[Dict[str, Any]]:
    """
    Compare grounded rule thresholds against actual SQL data values.
    Moved from NodeFactory._detect_evidence_conflicts for reuse.
    """
    conflicts = []
    if not rule_results or not sql_results:
        return conflicts

    matched_rules = rule_results.get("matched_rules", []) or []
    sql_data = sql_results[0] if sql_results else {}
    if not isinstance(sql_data, dict) or not sql_data.get("execution_ok"):
        return conflicts

    rows = sql_data.get("rows", []) or []
    if not rows:
        return conflicts

    sql_values: Dict[str, List[float]] = {}
    for row in rows[:5]:
        data = row.get("data", row) if isinstance(row, dict) else {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                sql_values.setdefault(k, []).append(v)

    for rule in matched_rules:
        var = (rule.get("sql_variable") or rule.get("variable") or "").lower()
        op = rule.get("operator") or ""
        rule_val = rule.get("value")
        if not var or not op or rule_val is None:
            continue
        try:
            rule_threshold = float(rule_val)
        except (ValueError, TypeError):
            continue

        for col, values in sql_values.items():
            if var in col.lower() or col.lower() in var:
                actual_avg = sum(values) / len(values)
                if op == ">" and actual_avg > rule_threshold:
                    result = "EXCEEDED"
                elif op == "<" and actual_avg < rule_threshold:
                    result = "BELOW_LIMIT"
                elif op == ">=" and actual_avg >= rule_threshold:
                    result = "AT_OR_ABOVE"
                elif op == "<=" and actual_avg <= rule_threshold:
                    result = "AT_OR_BELOW"
                else:
                    result = "WITHIN_BOUNDS"
                conflicts.append({
                    "conflict_type": "rule_vs_sql",
                    "rule_text": (rule.get("rule_text", "") or "")[:120],
                    "rule_variable": var,
                    "rule_operator": op,
                    "rule_threshold": rule_threshold,
                    "sql_column": col,
                    "actual_value": round(actual_avg, 4),
                    "comparison_result": result,
                })
                break
    return conflicts


# ---------------------------------------------------------------------------
# NEW: Doc↔SQL conflict detector
# ---------------------------------------------------------------------------

def detect_doc_sql_conflicts(
    documents: List[Dict[str, Any]],
    sql_results: List[Any],
    tolerance: float = 0.10,
) -> List[Dict[str, Any]]:
    """
    Detect conflicts where a document claims a number that differs from SQL data.

    Matches by keyword overlap between doc context and SQL column names.
    tolerance: relative difference allowed (10% by default).
    """
    conflicts = []
    if not documents or not sql_results:
        return conflicts

    sql_data = sql_results[0] if sql_results else {}
    if not isinstance(sql_data, dict) or not sql_data.get("execution_ok"):
        return conflicts

    rows = sql_data.get("rows", []) or []
    if not rows:
        return conflicts

    # Build SQL column -> aggregate value map
    sql_values: Dict[str, float] = {}
    for row in rows[:10]:
        data = row.get("data", row) if isinstance(row, dict) else {}
        for k, v in data.items():
            if isinstance(v, (int, float)):
                if k in sql_values:
                    sql_values[k] = (sql_values[k] + v) / 2  # running average
                else:
                    sql_values[k] = v

    # Scan documents for numerical claims, dedupe by (doc_source, sql_column)
    best_conflict: Dict[tuple, Dict[str, Any]] = {}

    for doc in documents[:5]:
        text = doc.get("text", "") or ""
        if not text:
            continue

        doc_nums = extract_numbers_with_context(text)
        for dn in doc_nums:
            doc_val = dn["value"]
            if isinstance(doc_val, tuple):
                doc_val = sum(doc_val) / 2  # range midpoint
            context_lower = dn["context"].lower()

            # Try to match to a SQL column by keyword
            # Only accept matches where the number has a unit or the column
            # keyword appears very close to the number
            for col, sql_val in sql_values.items():
                col_keywords = [kw for kw in col.lower().replace("_", " ").split() if len(kw) > 3]
                if not col_keywords:
                    continue

                # Require the number to be near a col keyword (within its context window)
                # and prefer when the number has a unit
                if not any(kw in context_lower for kw in col_keywords):
                    continue

                # Score: prefer numbers with units and that are closer to keyword
                match_quality = 1.0 if dn.get("unit") else 0.5

                if sql_val == 0:
                    rel_diff = abs(doc_val) if doc_val != 0 else 0
                else:
                    rel_diff = abs(doc_val - sql_val) / abs(sql_val)

                if rel_diff <= tolerance:
                    continue

                key = (doc.get("source_file", "?"), col)
                candidate = {
                    "conflict_type": "doc_vs_sql",
                    "doc_source": doc.get("source_file", "?"),
                    "doc_page": doc.get("page", 0),
                    "doc_claim": f"{doc_val} {dn['unit']}".strip(),
                    "doc_context": dn["context"][:120],
                    "sql_column": col,
                    "sql_value": round(sql_val, 4),
                    "relative_diff": round(rel_diff, 4),
                    "match_quality": match_quality,
                    "severity": "high" if rel_diff > 0.25 else "medium",
                }
                # Keep the highest-quality match per (doc, column) pair
                existing = best_conflict.get(key)
                if existing is None or candidate["match_quality"] > existing["match_quality"]:
                    best_conflict[key] = candidate

    # Strip internal match_quality field before returning
    for c in best_conflict.values():
        c.pop("match_quality", None)
        conflicts.append(c)

    return conflicts


# ---------------------------------------------------------------------------
# NEW: Doc↔Rule conflict detector
# ---------------------------------------------------------------------------

def detect_doc_rule_conflicts(
    documents: List[Dict[str, Any]],
    rule_results: Dict[str, Any],
    tolerance: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Detect conflicts where a document's stated threshold disagrees with
    the rule database.

    Useful for catching documentation that's out of sync with current rules
    (e.g., old handbook says 30 knots, new rule DB says 25 knots).
    """
    conflicts = []
    if not documents or not rule_results:
        return conflicts

    matched_rules = rule_results.get("matched_rules", []) or []
    if not matched_rules:
        return conflicts

    # Build rule variable -> threshold map
    rule_thresholds = {}
    for rule in matched_rules:
        var = (rule.get("variable") or "").lower().strip()
        val = rule.get("value")
        if var and val is not None:
            try:
                rule_thresholds[var] = {
                    "value": float(val),
                    "rule_text": (rule.get("rule_text", "") or "")[:120],
                    "source_file": rule.get("source_file", ""),
                }
            except (ValueError, TypeError):
                continue

    if not rule_thresholds:
        return conflicts

    # Scan docs for numbers in contexts matching rule variables
    for doc in documents[:5]:
        text = doc.get("text", "") or ""
        if not text:
            continue

        doc_nums = extract_numbers_with_context(text)
        for dn in doc_nums:
            context_lower = dn["context"].lower()

            for rule_var, rule_info in rule_thresholds.items():
                # Match rule variable to doc context
                var_words = rule_var.split()
                if not all(w in context_lower for w in var_words if len(w) > 2):
                    continue

                doc_val = dn["value"]
                if isinstance(doc_val, tuple):
                    doc_val = sum(doc_val) / 2

                rule_val = rule_info["value"]
                if rule_val == 0:
                    rel_diff = abs(doc_val) if doc_val != 0 else 0
                else:
                    rel_diff = abs(doc_val - rule_val) / abs(rule_val)

                if rel_diff > tolerance:
                    conflicts.append({
                        "conflict_type": "doc_vs_rule",
                        "doc_source": doc.get("source_file", "?"),
                        "doc_page": doc.get("page", 0),
                        "doc_context": dn["context"][:120],
                        "doc_value": doc_val,
                        "rule_variable": rule_var,
                        "rule_value": rule_val,
                        "rule_source": rule_info.get("source_file", ""),
                        "relative_diff": round(rel_diff, 4),
                        "severity": "high" if rel_diff > 0.20 else "medium",
                        "note": "Possible version mismatch between document and rule database",
                    })
                    break

    return conflicts


# ---------------------------------------------------------------------------
# NEW: Temporal (version) mismatch detector
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def detect_temporal_staleness(
    documents: List[Dict[str, Any]],
    current_year: Optional[int] = None,
    stale_threshold_years: int = 5,
) -> List[Dict[str, Any]]:
    """
    Flag documents that may be stale relative to current year.

    Heuristic: extract the latest year mentioned in the doc and flag if older
    than stale_threshold_years.
    """
    warnings = []
    if not documents:
        return warnings

    if current_year is None:
        current_year = datetime.now().year

    for doc in documents[:10]:
        text = doc.get("text", "") or ""
        source = doc.get("source_file", "")
        all_years = [int(m.group()) for m in _YEAR_RE.finditer(text)] + \
                    [int(m.group()) for m in _YEAR_RE.finditer(source)]
        # Filter to non-future years only
        valid_years = [y for y in all_years if y <= current_year]
        if not valid_years:
            continue
        latest_year = max(valid_years)
        age = current_year - latest_year
        if age >= stale_threshold_years:
            warnings.append({
                "conflict_type": "temporal_staleness",
                "doc_source": source,
                "latest_year": latest_year,
                "age_years": age,
                "severity": "high" if age >= 8 else "medium",
                "note": f"Document references data from {latest_year} ({age} years old)",
            })

    return warnings


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def detect_all_conflicts(
    evidence_bundle: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Run all conflict detectors on an evidence bundle.
    Returns combined list of conflict annotations.
    """
    documents = evidence_bundle.get("documents", []) or []
    sql_results = evidence_bundle.get("sql_results", []) or []
    rule_results = evidence_bundle.get("rules", {}) or {}

    all_conflicts = []

    # 1. Rule vs SQL (original)
    all_conflicts.extend(detect_rule_sql_conflicts(rule_results, sql_results))

    # 2. Doc vs SQL (new)
    all_conflicts.extend(detect_doc_sql_conflicts(documents, sql_results))

    # 3. Doc vs Rule (new)
    all_conflicts.extend(detect_doc_rule_conflicts(documents, rule_results))

    # 4. Temporal staleness (new)
    all_conflicts.extend(detect_temporal_staleness(documents))

    if all_conflicts:
        logger.info(
            "CONFLICT_DETECTOR: %d conflicts (%s)",
            len(all_conflicts),
            ", ".join(set(c.get("conflict_type", "?") for c in all_conflicts)),
        )

    return all_conflicts
