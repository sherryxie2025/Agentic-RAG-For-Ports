# src/online_pipeline/rule_retriever.py

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

from .source_registry import SourceRegistry
from .state_schema import PortQAState, RuleEngineResult, RuleMatch

logger = logging.getLogger("online_pipeline.rule_retriever")


class RuleRetriever:
    """
    Lightweight rule retriever over JSON rule files.

    Design goals:
    - compatible with current semi-structured rules
    - future-compatible with more normalized fields
    - simple keyword-based retrieval for MVP
    """

    def __init__(self, registry: SourceRegistry) -> None:
        self.registry = registry
        self.grounded_rules = self._safe_load_json_list(self.registry.grounded_rules_file)
        self.policy_rules = self._safe_load_json_list(self.registry.policy_rules_file)

        self.all_rules = self._prepare_rule_pool(
            grounded_rules=self.grounded_rules,
            policy_rules=self.policy_rules,
        )

    @staticmethod
    def _safe_load_json_list(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        return []

    def _prepare_rule_pool(
        self,
        grounded_rules: List[Dict[str, Any]],
        policy_rules: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []

        for idx, rule in enumerate(grounded_rules):
            normalized = self._normalize_rule_record(rule, source_type="grounded", fallback_idx=idx)
            pool.append(normalized)

        for idx, rule in enumerate(policy_rules):
            normalized = self._normalize_rule_record(rule, source_type="policy", fallback_idx=idx)
            pool.append(normalized)

        return pool

    def _normalize_rule_record(
        self,
        rule: Dict[str, Any],
        source_type: str,
        fallback_idx: int,
    ) -> Dict[str, Any]:
        """
        Normalizes both:
        1) your current rule format
        2) the future recommended structured format

        Current supported examples:
        - old format:
            rule_text, condition, action, variable, threshold, unit, source_file, page, sql_variable
        - future format:
            rule_id, threshold_raw, unit_raw, operator, value, value_min, value_max, source_unit, canonical_unit
        """
        rule_id = rule.get("rule_id") or f"{source_type}_rule_{fallback_idx:04d}"

        threshold_raw = rule.get("threshold_raw")
        if threshold_raw is None:
            threshold_raw = rule.get("threshold")

        unit_raw = rule.get("unit_raw")
        if unit_raw is None:
            unit_raw = rule.get("unit")

        normalized = {
            "rule_id": rule_id,
            "source_type": source_type,
            "rule_text": rule.get("rule_text", ""),
            "condition": rule.get("condition"),
            "action": rule.get("action"),
            "variable": rule.get("variable"),
            "threshold_raw": threshold_raw,
            "unit_raw": unit_raw,
            "sql_variable": rule.get("sql_variable"),
            "operator": rule.get("operator"),
            "value": rule.get("value"),
            "value_min": rule.get("value_min"),
            "value_max": rule.get("value_max"),
            "source_unit": rule.get("source_unit") or unit_raw,
            "canonical_unit": rule.get("canonical_unit"),
            "source_file": rule.get("source_file"),
            "page": rule.get("page"),
        }

        normalized["search_text"] = self._build_search_text(normalized)
        return normalized

    @staticmethod
    def _build_search_text(rule: Dict[str, Any]) -> str:
        parts = [
            str(rule.get("rule_text", "")),
            str(rule.get("condition", "")),
            str(rule.get("action", "")),
            str(rule.get("variable", "")),
            str(rule.get("threshold_raw", "")),
            str(rule.get("unit_raw", "")),
            str(rule.get("sql_variable", "")),
        ]
        return " ".join(parts).lower()

    @staticmethod
    def _normalize_query(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = re.sub(r"[^a-zA-Z0-9_\-\. ]+", " ", text.lower())
        tokens = [tok for tok in text.split() if tok]
        return tokens

    @staticmethod
    def _extract_query_keywords(query: str) -> List[str]:
        base_tokens = RuleRetriever._tokenize(query)

        stopwords = {
            "what", "which", "when", "where", "why", "how", "is", "are", "was", "were", "the",
            "a", "an", "of", "to", "for", "in", "on", "at", "and", "or", "with", "based",
            "be", "do", "does", "did", "should", "can", "could", "would", "might"
        }

        filtered = [t for t in base_tokens if t not in stopwords and len(t) > 2]
        return filtered

    @staticmethod
    def _score_rule(query_keywords: List[str], rule: Dict[str, Any]) -> float:
        """
        Score a rule against the query keywords.

        Uses query-length-normalized keyword overlap (fraction of query keywords
        matched), so scores are comparable across query lengths. Short structural
        bonuses are ADDITIVE after the normalization to nudge grounded rules.
        """
        search_text = rule.get("search_text", "") or ""
        if not query_keywords:
            return 0.0

        # Count how many query keywords appear in the rule text (fraction of query)
        matches = sum(1 for kw in query_keywords if kw in search_text)
        if matches == 0:
            return 0.0

        # Coverage = fraction of query keywords hit (0..1)
        coverage = matches / len(query_keywords)

        # Bonus: stronger when the rule hits a VARIABLE keyword (e.g. "wind",
        # "crane") since those are more specific than filler words.
        variable_field = (rule.get("variable") or rule.get("sql_variable") or "").lower()
        variable_hit = any(kw in variable_field for kw in query_keywords if len(kw) > 2)

        score = coverage
        if variable_hit:
            score += 0.3
        if rule.get("source_type") == "grounded":
            score += 0.05
        if rule.get("sql_variable"):
            score += 0.03
        if rule.get("operator") is not None:
            score += 0.02
        if (rule.get("value") is not None
                or rule.get("value_min") is not None
                or rule.get("value_max") is not None):
            score += 0.02

        return round(score, 4)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.35,   # normalized score: require >=35% coverage OR variable hit
    ) -> List[RuleMatch]:
        normalized_query = self._normalize_query(query)
        query_keywords = self._extract_query_keywords(normalized_query)

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rule in self.all_rules:
            score = self._score_rule(query_keywords, rule)
            if score >= min_score:
                scored.append((score, rule))

        scored.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            "RULES: query_keywords=%s candidates=%d/%d (above min_score=%.2f)",
            query_keywords[:6], len(scored), len(self.all_rules), min_score,
        )

        results: List[RuleMatch] = []
        for score, rule in scored[:top_k]:
            logger.debug(
                "  rule: score=%.2f var=%s op=%s val=%s text=%.60s",
                score, rule.get("variable"), rule.get("operator"),
                rule.get("value"), (rule.get("rule_text") or "")[:60],
            )
            item: RuleMatch = {
                "rule_text": rule.get("rule_text"),
                "condition": rule.get("condition"),
                "action": rule.get("action"),
                "variable": rule.get("variable"),
                "threshold_raw": rule.get("threshold_raw"),
                "unit_raw": rule.get("unit_raw"),
                "sql_variable": rule.get("sql_variable"),
                "operator": rule.get("operator"),
                "value": rule.get("value"),
                "value_min": rule.get("value_min"),
                "value_max": rule.get("value_max"),
                "canonical_unit": rule.get("canonical_unit"),
                "source_file": rule.get("source_file"),
                "page": rule.get("page"),
                "relevance_score": score,
                "matched": True,
                "triggered": None,
                "trigger_explanation": None,
            }
            results.append(item)

        return results

    def update_state(
        self,
        state: PortQAState,
        top_k: int = 5,
        min_score: float = 0.5,
    ) -> PortQAState:
        query = state.get("user_query", "")
        matched_rules = self.retrieve(query=query, top_k=top_k, min_score=min_score)

        rule_result: RuleEngineResult = {
            "matched_rules": matched_rules,
            "applicable_rule_count": len(matched_rules),
            "triggered_rule_count": 0,
            "execution_ok": True,
            "error": None,
        }

        state["rule_results"] = rule_result

        trace = state.get("reasoning_trace", [])
        trace.append(
            f"RuleRetriever => retrieved {len(matched_rules)} candidate rules from JSON rule stores."
        )
        state["reasoning_trace"] = trace

        return state


def retrieve_rules(
    project_root: str | Path,
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
) -> List[RuleMatch]:
    registry = SourceRegistry.from_project_root(project_root)
    retriever = RuleRetriever(registry=registry)
    return retriever.retrieve(query=query, top_k=top_k, min_score=min_score)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    registry = SourceRegistry.from_project_root(PROJECT_ROOT)
    retriever = RuleRetriever(registry=registry)

    query = "Under what wind conditions should vessel entry be restricted?"
    rules = retriever.retrieve(query=query, top_k=5, min_score=0.5)

    print("=" * 80)
    print("QUERY:", query)
    for i, rule in enumerate(rules, start=1):
        print(f"[{i}] score={rule.get('relevance_score')} source={rule.get('source_file')} page={rule.get('page')}")
        print("variable:", rule.get("variable"))
        print("threshold_raw:", rule.get("threshold_raw"))
        print("operator:", rule.get("operator"))
        print("value:", rule.get("value"))
        print("rule_text:", rule.get("rule_text"))
        print("-" * 80)