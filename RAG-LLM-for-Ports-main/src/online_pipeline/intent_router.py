# src/online_pipeline/intent_router.py

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Set

import numpy as np

import logging

from .llm_client import llm_chat_json
from .state_schema import PortQAState, RouterDecision

logger = logging.getLogger("online_pipeline.intent_router")

# ---------------------------------------------------------------------------
# MLP classifier (loaded once at import time, ~instant)
# ---------------------------------------------------------------------------
_CLASSIFIER_PATH = Path(__file__).resolve().parents[2] / "storage" / "models" / "intent_classifier.pkl"
_MLP_ARTIFACT = None

try:
    with open(_CLASSIFIER_PATH, "rb") as _f:
        _MLP_ARTIFACT = pickle.load(_f)
except Exception:
    pass

# LLM routing system prompt
#
# Design notes (v2 — tightened for over-routing):
#   The previous version of this prompt produced 43% over-routing (set a
#   capability flag to true even when the query had no signal for it).
#   The core fix is the "Minimum Sufficient Routing" principle: each
#   capability must be DISABLED by default and only enabled when the
#   query gives a concrete signal for it. The prompt now includes
#   per-capability trigger rules, hard NOT-triggers, and a negative
#   few-shot example set so the LLM sees what "do not route" looks like.
_LLM_ROUTER_SYSTEM = """You are an intent router for a port operations RAG system.
Given a user query, output which data sources are needed.

Return ONLY a JSON object with these boolean fields:
{
  "needs_vector": true/false,    // retrieve from document store (reports, handbooks, policies)
  "needs_sql": true/false,       // query structured operational data (tables, statistics, time-series)
  "needs_rules": true/false,     // look up policy rules, thresholds, restrictions
  "needs_graph": true/false,     // multi-hop causal reasoning across nodes (concept A → metric B → operation C)
  "question_type": "document_lookup|structured_data|policy_rule|hybrid_reasoning|causal_multihop",
  "answer_mode": "lookup|descriptive|comparison|decision_support|diagnostic",
  "confidence": 0.0-1.0
}

## CORE PRINCIPLE: MINIMUM SUFFICIENT ROUTING
Every flag starts as FALSE. Turn a flag to true ONLY if the query gives a
concrete, textual signal for that capability. If you have to guess whether
a capability is needed, the answer is FALSE. Over-routing is worse than
under-routing because each enabled capability costs an expensive retrieval.

## Trigger rules (enable flag ONLY if condition holds)

needs_vector = true WHEN AND ONLY WHEN:
  - query references a specific document/report/handbook/study/policy
    (e.g. "according to the 2018 handbook", "what does the annual report
    say", "per the VRCA manual")
  - query asks for a DEFINITION, NARRATIVE, or DESCRIPTION that lives in
    prose (e.g. "describe the hurricane response plan")
  - query mentions a noun that is typically discussed in documents only
    (e.g. "financial statements", "treatment steps", "procurement basis")

needs_sql = true WHEN AND ONLY WHEN:
  - query asks for an aggregate / count / average / max / min / sum /
    total / percentage
  - query asks for a specific number from operational tables (crane
    productivity, berth moves, dwell days, vessel transactions, weather
    readings, containers, gate moves)
  - query asks about a TREND over time (yearly, monthly, quarterly)
  - query has explicit "how many / how much / what was the average" phrasing
  - query filters by a specific year, month, terminal, or vessel AND
    asks for a numeric value

needs_rules = true WHEN AND ONLY WHEN:
  - query asks whether something is ALLOWED / PROHIBITED / REQUIRED /
    PERMITTED / RESTRICTED
  - query asks about THRESHOLDS, LIMITS, OR SAFETY CONDITIONS
  - query phrased as "under what conditions should we ... ",
    "is it allowed to ...", "must operations ...", "when should
    crane operations stop"
  - query asks for a policy-defined threshold (e.g. "maximum wind speed")

needs_graph = true WHEN AND ONLY WHEN:
  - query explicitly asks WHY something happened or what FACTORS/CAUSES
    contributed to an outcome
  - query asks to trace a chain of 2+ concept→metric→operation
    relationships (e.g. "how does high tide affect berth productivity
    through pilot scheduling")
  - query requires combining multiple source types to reason about
    causal effects (not just retrieval)

## HARD NOT-TRIGGERS (the single most common over-routing mistake)

- "What does the handbook say about X?" → needs_vector only. NOT sql,
  NOT rules (unless X is explicitly a rule/threshold), NOT graph.
- "What was the average X in 2019?" → needs_sql only. NOT vector, NOT
  rules, NOT graph.
- "Under what wind conditions should crane operations stop?" → needs_rules
  only. NOT sql (no aggregate), NOT vector (not asking for a document
  quote), NOT graph.
- "Describe the maintenance schedule for cranes" → needs_vector only.
- "Who is the port authority chairman?" → needs_vector only.
- "Why is crane productivity low this week?" → needs_sql + needs_graph
  (the "why" is causal, the "this week" wants the actual number), NOT
  vector, NOT rules.
- Single-sentence factoid lookups → ONE source. Never all four.

## Few-shot examples

Q: "According to the 2018 VRCA handbook, what does it say about crane
maintenance?"
A: {"needs_vector": true, "needs_sql": false, "needs_rules": false,
    "needs_graph": false, "question_type": "document_lookup",
    "answer_mode": "lookup", "confidence": 0.92}

Q: "What was the average crane productivity in moves per hour in 2019?"
A: {"needs_vector": false, "needs_sql": true, "needs_rules": false,
    "needs_graph": false, "question_type": "structured_data",
    "answer_mode": "lookup", "confidence": 0.95}

Q: "Under what wind conditions should crane operations be suspended?"
A: {"needs_vector": false, "needs_sql": false, "needs_rules": true,
    "needs_graph": false, "question_type": "policy_rule",
    "answer_mode": "decision_support", "confidence": 0.92}

Q: "Why did berth productivity drop in Q3 2019, and how does that
compare to the operational threshold?"
A: {"needs_vector": false, "needs_sql": true, "needs_rules": true,
    "needs_graph": true, "question_type": "causal_multihop",
    "answer_mode": "diagnostic", "confidence": 0.85}

Q: "What is the relationship between pilot availability, tidal window,
and berth occupancy?"
A: {"needs_vector": false, "needs_sql": false, "needs_rules": false,
    "needs_graph": true, "question_type": "causal_multihop",
    "answer_mode": "diagnostic", "confidence": 0.78}

Return ONLY valid JSON, no markdown, no prose."""


class IntentRouter:
    """
    Rule-based multi-capability intent router for the AI Port Decision-Support System.

    Output is NOT a single label.
    It predicts which capabilities are needed:
    - vector retrieval over documents
    - structured data querying (CSV/SQL-like tables)
    - rule reasoning
    - graph / multi-hop reasoning
    """

    def __init__(self) -> None:
        # ---------- keyword banks ----------
        self.document_keywords = {
            "document", "documents", "report", "reports", "handbook", "policy", "policies",
            "manual", "study", "studies", "guideline", "guidelines", "mention", "mentions",
            "state", "states", "stated", "describe", "describes", "described", "discussion",
            "discuss", "summarize", "summary", "according to", "what does", "what do", "in the report",
            "in the document", "from the handbook", "from the policy"
        }

        self.sql_keywords = {
            "average", "avg", "mean", "sum", "total", "count", "maximum", "minimum", "max", "min",
            "trend", "historical", "history", "in 2015", "over time", "timeline",
            "productivity", "delay", "transactions", "turn time", "dwell", "moves",
            "wave height", "tide", "pressure", "wind", "wind speed", "wind gust",
            "vessel capacity", "loa", "size category", "berth productivity",
            "containers actual", "arrival delay", "crane productivity", "breakdown minutes",
            "teu received", "average dwell days", "total transactions", "average turn time",
            "environment", "berth", "crane", "yard", "gate", "vessel", "table", "dataset", "data"
        }

        self.rule_keywords = {
            "rule", "rules", "policy", "policies", "threshold", "thresholds", "restriction", "restrictions",
            "restrict", "restricted", "limit", "limits", "limitation", "allow", "allowed", "permit",
            "permitted", "must", "must not", "should", "should not", "require", "required",
            "prohibit", "prohibited", "suspend", "pause", "stop", "under what conditions",
            "when should", "when must", "maximum allowable", "compliance", "safety"
        }

        self.graph_keywords = {
            "why", "explain", "reason", "reasons", "factor", "factors", "cause", "causes",
            "causal", "driver", "drivers", "led to", "contributed to", "related to", "relationship",
            "relationships", "impact", "impacts", "interaction", "interactions", "chain",
            "multi-hop", "across sources", "combine evidence", "connect", "link"
        }
        # "how" is checked separately below to avoid false positive with "how many"/"how much"

        # More domain-specific operational nouns that often suggest cross-source reasoning
        self.operation_keywords = {
            "berth operations", "crane operations", "yard operations", "gate operations",
            "vessel calls", "navigation", "entry", "transit", "delay", "slowdown", "congestion",
            "weather conditions", "operational disruption", "disruption"
        }

        self.decision_support_keywords = {
            "should", "can", "whether", "determine if", "is it allowed", "is it permitted",
            "should operations", "should we", "should they", "need to", "must we", "must they"
        }

        self.comparison_keywords = {
            "compare", "difference", "higher than", "lower than", "versus", "vs"
        }

        # Bigram/trigram patterns that strongly indicate data-query intent
        self.sql_bigram_patterns = [
            "how many", "how much", "what was the average", "what is the average",
            "what was the total", "what is the total", "what is the peak",
            "per month", "per day", "per week", "per year", "by month",
            "top 5", "top 10", "top n", "the busiest", "the highest", "the lowest",
            "average of", "sum of", "count of", "number of", "total of",
            "in moves per hour", "moves per hour", "in 2015", "during 2015",
            "across all terminals", "between terminals", "by terminal",
            "what percentage", "how does", "how do",
        ]

    # ----------------------------
    # Public API
    # ----------------------------

    def route(self, query: str) -> RouterDecision:
        normalized = self._normalize(query)

        # ── Strategy: LLM-first, rule-based fallback ──
        # LLM understands semantic intent much better than keyword matching.
        # Rule-based is instant (0ms) and used only when LLM fails.

        routing_method = "llm"
        llm_decision = self._llm_route(query)

        if llm_decision is not None:
            logger.info("LLM router result: %s", llm_decision)
            needs_vector = bool(llm_decision.get("needs_vector", False))
            needs_sql = bool(llm_decision.get("needs_sql", False))
            needs_rules = bool(llm_decision.get("needs_rules", False))
            needs_graph_reasoning = bool(llm_decision.get("needs_graph", False))
            question_type = llm_decision.get("question_type", "unknown")
            answer_mode = llm_decision.get("answer_mode", "lookup")
            llm_conf = llm_decision.get("confidence", 0.7)
            confidence = float(llm_conf) if llm_conf else 0.7
            matched_keywords = []
            rationale = "LLM-routed"
        else:
            # ── Rule-based fallback ──
            logger.warning("LLM router failed => falling back to rule-based")
            routing_method = "rule"

            matched_document = self._match_keywords(normalized, self.document_keywords)
            matched_sql = self._match_keywords(normalized, self.sql_keywords)
            matched_rule = self._match_keywords(normalized, self.rule_keywords)
            matched_graph = self._match_keywords(normalized, self.graph_keywords)
            matched_operations = self._match_keywords(normalized, self.operation_keywords)
            matched_decision = self._match_keywords(normalized, self.decision_support_keywords)
            matched_comparison = self._match_keywords(normalized, self.comparison_keywords)

            matched_keywords = sorted(
                set(
                    matched_document + matched_sql + matched_rule + matched_graph
                    + matched_operations + matched_decision + matched_comparison
                )
            )

            needs_vector = self._infer_needs_vector(
                normalized, matched_document, matched_rule, matched_graph
            )
            needs_sql = self._infer_needs_sql(normalized, matched_sql)
            needs_rules = self._infer_needs_rules(normalized, matched_rule, matched_decision)
            needs_graph_reasoning = self._infer_needs_graph(
                normalized, matched_graph, matched_operations,
                needs_vector, needs_sql, needs_rules,
            )

            question_type = self._infer_question_type(
                needs_vector=needs_vector, needs_sql=needs_sql,
                needs_rules=needs_rules, needs_graph_reasoning=needs_graph_reasoning,
            )
            answer_mode = self._infer_answer_mode(
                normalized=normalized, matched_graph=matched_graph,
                matched_decision=matched_decision, matched_comparison=matched_comparison,
            )
            confidence = self._estimate_confidence(
                matched_keywords=matched_keywords, needs_vector=needs_vector,
                needs_sql=needs_sql, needs_rules=needs_rules,
                needs_graph_reasoning=needs_graph_reasoning,
            )
            rationale = self._build_rationale(
                needs_vector=needs_vector, needs_sql=needs_sql,
                needs_rules=needs_rules, needs_graph_reasoning=needs_graph_reasoning,
                matched_keywords=matched_keywords,
            )

        logger.info(
            "ROUTE [%s]: type=%s mode=%s conf=%.2f vector=%s sql=%s rules=%s graph=%s",
            routing_method, question_type, answer_mode, confidence,
            needs_vector, needs_sql, needs_rules, needs_graph_reasoning,
        )

        return RouterDecision(
            question_type=question_type,
            answer_mode=answer_mode,
            needs_vector=needs_vector,
            needs_sql=needs_sql,
            needs_rules=needs_rules,
            needs_graph_reasoning=needs_graph_reasoning,
            confidence=confidence,
            matched_keywords=matched_keywords,
            rationale=rationale,
        )

    @staticmethod
    def _mlp_classify(query: str) -> dict | None:
        """Use pre-trained MLP classifier for fast intent prediction."""
        if _MLP_ARTIFACT is None:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            clf = _MLP_ARTIFACT["classifier"]
            mlb = _MLP_ARTIFACT["mlb"]
            model_name = _MLP_ARTIFACT["embed_model_name"]
            # Use the same model instance if possible
            if not hasattr(IntentRouter, '_mlp_embed_model'):
                IntentRouter._mlp_embed_model = SentenceTransformer(model_name, device="cuda")
            emb = IntentRouter._mlp_embed_model.encode([query], normalize_embeddings=True)
            pred = clf.predict(emb)
            labels = mlb.inverse_transform(pred)
            if labels and labels[0]:
                active = set(labels[0])
                return {
                    "needs_vector": "documents" in active,
                    "needs_sql": "sql" in active,
                    "needs_rules": "rules" in active,
                    "needs_graph": "graph" in active,
                }
        except Exception:
            pass
        return None

    @staticmethod
    def _llm_route(query: str) -> dict | None:
        """Use LLM to classify intent when rule-based confidence is low."""
        try:
            result = llm_chat_json(
                messages=[
                    {"role": "system", "content": _LLM_ROUTER_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                timeout=30,
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return None

    def update_state(self, state: PortQAState) -> PortQAState:
        query = state.get("user_query", "")
        decision = self.route(query)

        state["router_decision"] = decision
        state["question_type"] = decision["question_type"]
        state["answer_mode"] = decision["answer_mode"]
        state["needs_vector"] = decision["needs_vector"]
        state["needs_sql"] = decision["needs_sql"]
        state["needs_rules"] = decision["needs_rules"]
        state["needs_graph_reasoning"] = decision["needs_graph_reasoning"]

        reasoning_trace = state.get("reasoning_trace", [])
        reasoning_trace.append(
            f"IntentRouter => "
            f"vector={decision['needs_vector']}, "
            f"sql={decision['needs_sql']}, "
            f"rules={decision['needs_rules']}, "
            f"graph={decision['needs_graph_reasoning']}, "
            f"type={decision['question_type']}, "
            f"mode={decision['answer_mode']}"
        )
        state["reasoning_trace"] = reasoning_trace

        return state

    # ----------------------------
    # Internal helpers
    # ----------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _match_keywords(text: str, keywords: Set[str]) -> List[str]:
        matches: List[str] = []
        for kw in keywords:
            if kw in text:
                matches.append(kw)
        return matches

    @staticmethod
    def _contains_temporal_data_request(text: str) -> bool:
        temporal_terms = [
            "in 2015", "in 2016", "in 2017", "in 2018", "in 2019",
            "in 2020", "in 2021", "in 2022", "in 2023", "in 2024",
            "over time", "historical", "timeline", "trend", "during"
        ]
        return any(term in text for term in temporal_terms)

    def _infer_needs_vector(
        self,
        text: str,
        matched_document: List[str],
        matched_rule: List[str],
        matched_graph: List[str],
    ) -> bool:
        # Document retrieval when user asks what documents/policies say
        # or when explanation likely needs textual support.
        if matched_document:
            return True

        # Rule questions often benefit from policy text support
        if matched_rule and ("according to" in text or "policy" in text or "handbook" in text):
            return True

        # Causal explanatory questions often benefit from docs
        if matched_graph and ("report" in text or "document" in text or "policy" in text):
            return True

        return False

    def _infer_needs_sql(self, text: str, matched_sql: List[str]) -> bool:
        if matched_sql:
            return True

        # Bigram/trigram patterns for data-query intent
        if any(p in text for p in self.sql_bigram_patterns):
            return True

        # Explicit numeric / analytic phrasing
        analytic_patterns = [
            "what was the",
            "how many",
            "what is the average",
            "what is the total",
            "which berth",
            "which crane",
            "which vessel",
            "highest",
            "lowest",
        ]
        if any(p in text for p in analytic_patterns):
            return True

        if self._contains_temporal_data_request(text):
            return True

        return False

    def _infer_needs_rules(
        self,
        text: str,
        matched_rule: List[str],
        matched_decision: List[str],
    ) -> bool:
        if matched_rule:
            return True

        # Decision-support queries often imply rule checking
        if matched_decision and any(term in text for term in ["allowed", "permitted", "restrict", "pause", "stop"]):
            return True

        # “Under what conditions” is almost always a rule query
        if "under what conditions" in text:
            return True

        return False

    def _infer_needs_graph(
        self,
        text: str,
        matched_graph: List[str],
        matched_operations: List[str],
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
    ) -> bool:
        # Explicit multi-hop / explanation signals
        if matched_graph:
            return True

        # "how" only triggers graph when explanatory (not "how many", "how much")
        has_explanatory_how = ("how " in text and not any(
            text.count(p) for p in ["how many", "how much", "how does the average", "how does the total"]
            if p in text
        )) or any(p in text for p in ["how did", "how does", "how can", "how might", "how would"])

        # If query likely requires combining multiple sources and asks why/how
        if any(term in text for term in ["why", "explain", "factors", "caused", "led to"]) or has_explanatory_how:
            multi_source_count = sum([needs_vector, needs_sql, needs_rules])
            if multi_source_count >= 2:
                return True

        # Operational causal chain questions
        if matched_operations and any(term in text for term in ["impact", "related", "contributed", "linked"]):
            return True

        return False

    @staticmethod
    def _infer_question_type(
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
        needs_graph_reasoning: bool,
    ) -> str:
        if needs_graph_reasoning:
            return "causal_multihop"

        active_count = sum([needs_vector, needs_sql, needs_rules])

        if active_count >= 2:
            return "hybrid_reasoning"
        if needs_vector:
            return "document_lookup"
        if needs_sql:
            return "structured_data"
        if needs_rules:
            return "policy_rule"

        return "unknown"

    @staticmethod
    def _infer_answer_mode(
        normalized: str,
        matched_graph: List[str],
        matched_decision: List[str],
        matched_comparison: List[str],
    ) -> str:
        # "how" only counts as diagnostic when explanatory
        _how_diag = any(p in normalized for p in ["how did", "how does", "how can", "how might"])
        if matched_graph or any(t in normalized for t in ["why", "explain", "factors"]) or _how_diag:
            return "diagnostic"

        if matched_decision or any(t in normalized for t in ["should", "allowed", "permitted", "must"]):
            return "decision_support"

        if matched_comparison:
            return "comparison"

        if any(t in normalized for t in ["summarize", "describe", "mention", "discuss"]):
            return "descriptive"

        return "lookup"

    @staticmethod
    def _estimate_confidence(
        matched_keywords: List[str],
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
        needs_graph_reasoning: bool,
    ) -> float:
        base = min(0.25 + 0.08 * len(matched_keywords), 0.95)

        # Small penalty if nothing matched clearly
        if not any([needs_vector, needs_sql, needs_rules, needs_graph_reasoning]):
            return 0.30

        return round(base, 2)

    @staticmethod
    def _build_rationale(
        needs_vector: bool,
        needs_sql: bool,
        needs_rules: bool,
        needs_graph_reasoning: bool,
        matched_keywords: List[str],
    ) -> str:
        sources = []
        if needs_vector:
            sources.append("documents")
        if needs_sql:
            sources.append("structured operational data")
        if needs_rules:
            sources.append("policy/rule database")
        if needs_graph_reasoning:
            sources.append("graph-based multi-hop reasoning")

        if not sources:
            return "No strong routing signal was detected from the query."

        source_str = ", ".join(sources)
        keyword_str = ", ".join(matched_keywords[:10]) if matched_keywords else "none"

        return f"Router selected: {source_str}. Matched keywords: {keyword_str}."


def route_query(query: str) -> RouterDecision:
    router = IntentRouter()
    return router.route(query)


def route_state(state: PortQAState) -> PortQAState:
    router = IntentRouter()
    return router.update_state(state)


if __name__ == "__main__":
    demo_queries = [
        "What does the operating handbook say about restricted night navigation?",
        "What was the average crane productivity in 2015?",
        "Under what wind conditions should operations be restricted?",
        "Based on berth productivity and wind speed, should crane operations be paused?",
        "Why did berth operations slow down, and how might weather and policy restrictions explain it?",
    ]

    router = IntentRouter()
    for q in demo_queries:
        print("=" * 80)
        print("QUERY:", q)
        print(router.route(q))