# src/online_pipeline/langgraph_nodes.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, Optional

import logging

from .answer_synthesizer import AnswerSynthesizer

logger = logging.getLogger("online_pipeline.langgraph_nodes")
from .document_retriever import ChromaDocumentRetriever
from .graph_reasoner import Neo4jGraphReasoner
from .hybrid_retriever import HybridDocumentRetriever
from .intent_router import IntentRouter
from .planner import QueryPlanner
from .query_rewriter import QueryRewriter
from .reranker import CrossEncoderReranker
from .rule_retriever import RuleRetriever
from .source_registry import SourceRegistry
from .sql_agent_v2 import SQLAgentV2


class NodeFactory:
    def __init__(
        self,
        project_root: str | Path,
        chroma_collection_name: str | None = None,
        use_llm_sql_planner: bool = False,
        sql_model_name: str | None = None,
    ) -> None:
        self.registry = SourceRegistry.from_project_root(project_root)
        db_path = self.registry.project_root / "storage" / "sql" / "port_ops.duckdb"

        self.router = IntentRouter()
        self.query_rewriter = QueryRewriter()
        self.planner = QueryPlanner()
        self.doc_retriever = HybridDocumentRetriever(
            registry=self.registry,
            collection_name=chroma_collection_name,
        )
        self.reranker = CrossEncoderReranker()
        self.rule_retriever = RuleRetriever(registry=self.registry)
        self.sql_agent = SQLAgentV2(
            db_path=db_path,
            use_llm_sql=use_llm_sql_planner,
            model_name=sql_model_name,  # None -> uses llm_client default
        )
        self.graph_reasoner = Neo4jGraphReasoner()
        self.answer_synthesizer = AnswerSynthesizer(
            use_llm_fallback=True,
        )
        # Per-node latency tracking: {node_name: [elapsed_seconds, ...]}
        self.node_timings: Dict[str, list] = {}

    def _record_timing(self, node_name: str, elapsed: float) -> None:
        self.node_timings.setdefault(node_name, []).append(round(elapsed, 4))

    def _timed(self, node_name: str, func, state: Dict[str, Any]) -> Dict[str, Any]:
        """Wrapper that records per-node latency."""
        t0 = time.time()
        result = func(state)
        elapsed = time.time() - t0
        self._record_timing(node_name, elapsed)
        logger.info("NODE %-22s %.2fs", node_name, elapsed)
        return result

    def route_query_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("route_query", self._route_query_impl, state)

    def _route_query_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("user_query", "")
        decision = self.router.route(query)
        return {
            "router_decision": decision,
            "question_type": decision["question_type"],
            "answer_mode": decision["answer_mode"],
            "needs_vector": decision["needs_vector"],
            "needs_sql": decision["needs_sql"],
            "needs_rules": decision["needs_rules"],
            "needs_graph_reasoning": decision["needs_graph_reasoning"],
            "reasoning_trace": [
                f"route_query_node => vector={decision['needs_vector']}, "
                f"sql={decision['needs_sql']}, "
                f"rules={decision['needs_rules']}, "
                f"graph={decision['needs_graph_reasoning']}"
            ],
        }

    def planner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("planner", self._planner_impl, state)

    def _planner_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # Step 1: query rewrite (abbreviation expansion)
        original_query = state.get("user_query", "")
        rewrite_result = self.query_rewriter.rewrite(original_query)
        rewritten = rewrite_result.get("rewritten_query", original_query)
        expanded_terms = rewrite_result.get("expanded_terms", [])

        # Step 2: planning on rewritten query
        router_decision = state.get("router_decision", {})
        plan = self.planner.plan(user_query=rewritten, router_decision=router_decision)

        trace = []
        if expanded_terms:
            trace.append(f"planner_node => rewrite expanded_terms={expanded_terms}")
        trace.append(
            f"planner_node => sources={plan['source_plan']} "
            f"strategy={plan['execution_strategy']}"
        )

        return {
            "original_query": original_query,
            "user_query": rewritten,
            "source_plan": plan["source_plan"],
            "sub_queries": plan["sub_queries"],
            "execution_strategy": plan["execution_strategy"],
            "reasoning_trace": trace,
        }

    def retrieve_documents_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("retrieve_documents", self._retrieve_documents_impl, state)

    def _retrieve_documents_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._find_subquery(state, source="documents") or state.get("user_query", "")
        docs = self.doc_retriever.retrieve(query=query, top_k=20)
        return {
            "retrieved_docs": docs,
            "reasoning_trace": [f"retrieve_documents_node => retrieved {len(docs)} docs"],
        }

    def rerank_documents_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("rerank_documents", self._rerank_documents_impl, state)

    def _rerank_documents_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("user_query", "")
        docs = state.get("retrieved_docs", [])
        if not docs:
            return {"reasoning_trace": ["rerank_documents_node => no docs to rerank"]}
        reranked = self.reranker.rerank(query, docs, top_k=5)
        return {
            "pre_rerank_docs": docs,  # snapshot before rerank for eval comparison
            "retrieved_docs": reranked,
            "reasoning_trace": [f"rerank_documents_node => reranked {len(docs)} -> {len(reranked)} docs"],
        }

    def retrieve_rules_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("retrieve_rules", self._retrieve_rules_impl, state)

    def _retrieve_rules_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._find_subquery(state, source="rules") or state.get("user_query", "")
        # top_k=3 + min_score=0.4 align with RuleRetriever.update_state
        # defaults. The variable-hit boost floor in _score_rule is 0.45, so
        # min_score must be <= 0.45 for variable-targeted queries to pass.
        matched_rules = self.rule_retriever.retrieve(query=query, top_k=3, min_score=0.4)
        rule_result = {
            "matched_rules": matched_rules,
            "applicable_rule_count": len(matched_rules),
            "triggered_rule_count": 0,
            "execution_ok": True,
            "error": None,
        }
        return {
            "rule_results": rule_result,
            "reasoning_trace": [f"retrieve_rules_node => retrieved {len(matched_rules)} rules"],
        }

    def run_sql_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("run_sql", self._run_sql_impl, state)

    def _run_sql_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._find_subquery(state, source="sql") or state.get("user_query", "")
        result = self.sql_agent.run(query)
        plan = result.get("plan", {})
        return {
            "sql_results": [result],
            "reasoning_trace": [
                f"run_sql_node => mode={plan.get('generation_mode')} "
                f"tables={plan.get('target_tables')} "
                f"rows={result.get('row_count', 0)} ok={result.get('execution_ok')}"
            ],
        }

    def run_graph_reasoner_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("run_graph_reasoner", self._run_graph_impl, state)

    def _run_graph_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = self._find_subquery(state, source="graph") or state.get("user_query", "")
        result = self.graph_reasoner.reason(query)

        # Detailed graph reasoning trace for debugging and demo
        entities = result.get("query_entities", [])
        expanded = result.get("expanded_nodes", [])
        paths = result.get("reasoning_paths", [])
        trace_lines = [
            f"run_graph_reasoner_node => entities={entities} "
            f"expanded={len(expanded)} nodes, {len(paths)} paths"
        ]
        for i, p in enumerate(paths[:5]):
            trace_lines.append(f"  path[{i}]: {p.get('explanation', 'N/A')}")
        if not paths:
            trace_lines.append("  WARNING: no reasoning paths found between extracted entities")
            if result.get("error"):
                trace_lines.append(f"  ERROR: {result['error']}")

        return {
            "graph_results": result,
            "reasoning_trace": trace_lines,
        }

    def merge_evidence_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("merge_evidence", self._merge_evidence_impl, state)

    def _merge_evidence_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        evidence_bundle = {
            "documents": state.get("retrieved_docs", []),
            "sql_results": state.get("sql_results", []),
            "rules": state.get("rule_results", {}),
            "graph": state.get("graph_results", {}),
        }

        # Evidence conflict detection: compare rule thresholds vs SQL actual values
        conflict_annotations = self._detect_evidence_conflicts(
            rule_results=state.get("rule_results", {}),
            sql_results=state.get("sql_results", []),
        )
        evidence_bundle["conflict_annotations"] = conflict_annotations

        # Log merged evidence summary
        doc_count = len(state.get("retrieved_docs", []))
        sql_count = len(state.get("sql_results", []))
        rule_count = len((state.get("rule_results") or {}).get("matched_rules", []) or [])
        graph_paths = len((state.get("graph_results") or {}).get("reasoning_paths", []) or [])
        logger.info(
            "MERGE: docs=%d sql_results=%d rules=%d graph_paths=%d conflicts=%d",
            doc_count, sql_count, rule_count, graph_paths, len(conflict_annotations),
        )
        for ca in conflict_annotations:
            logger.info(
                "  CONFLICT: %s %s %.2f (rule) vs %.4f (actual) => %s",
                ca.get("rule_variable"), ca.get("rule_operator"),
                ca.get("rule_threshold", 0), ca.get("actual_value", 0),
                ca.get("comparison_result"),
            )

        trace = ["merge_evidence_node => merged evidence bundle"]
        if conflict_annotations:
            trace.append(f"merge_evidence_node => detected {len(conflict_annotations)} rule-vs-data conflicts")

        return {
            "evidence_bundle": evidence_bundle,
            "reasoning_trace": trace,
        }

    @staticmethod
    def _detect_evidence_conflicts(
        rule_results: Dict[str, Any],
        sql_results: list,
    ) -> list:
        """
        Compare grounded rule thresholds against actual SQL data values.
        Returns list of conflict annotations.
        """
        conflicts = []
        if not rule_results or not sql_results:
            return conflicts

        matched_rules = rule_results.get("matched_rules", []) or []
        sql_data = sql_results[0] if sql_results else {}
        if not sql_data.get("execution_ok"):
            return conflicts

        rows = sql_data.get("rows", []) or []
        if not rows:
            return conflicts

        # Build a lookup of SQL column values (aggregate or first row)
        sql_values = {}
        for row in rows[:5]:
            data = row.get("data", row) if isinstance(row, dict) else {}
            for k, v in data.items():
                if isinstance(v, (int, float)):
                    if k not in sql_values:
                        sql_values[k] = []
                    sql_values[k].append(v)

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

            # Find matching SQL column
            for col, values in sql_values.items():
                if var in col.lower() or col.lower() in var:
                    actual_avg = sum(values) / len(values)

                    # Compare
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
                        "rule_text": rule.get("rule_text", "")[:120],
                        "rule_variable": var,
                        "rule_operator": op,
                        "rule_threshold": rule_threshold,
                        "sql_column": col,
                        "actual_value": round(actual_avg, 4),
                        "comparison_result": result,
                    })
                    break

        return conflicts

    def synthesize_answer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._timed("synthesize_answer", self._synthesize_impl, state)

    def _synthesize_impl(self, state: Dict[str, Any]) -> Dict[str, Any]:
        final_answer = self.answer_synthesizer.synthesize(state)
        return {
            "final_answer": final_answer,
            "reasoning_trace": ["synthesize_answer_node => produced final answer"],
        }

    @staticmethod
    def _find_subquery(state: Dict[str, Any], source: str) -> Optional[str]:
        sub_queries = state.get("sub_queries", [])
        for item in sub_queries:
            if item.get("source") == source:
                return item.get("query")
        return None