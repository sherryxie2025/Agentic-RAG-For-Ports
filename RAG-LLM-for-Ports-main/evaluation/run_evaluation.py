"""
Port RAG System Evaluation Framework
=====================================
Computes retrieval recall/precision, source routing accuracy,
and end-to-end answer faithfulness metrics.

Usage:
    cd RAG-LLM-for-Ports-main
    python evaluation/run_evaluation.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from online_pipeline.langgraph_workflow import build_langgraph_workflow
from online_pipeline.pipeline_logger import setup_pipeline_logging

# Embedding model for semantic similarity (loaded once)
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("BAAI/bge-small-en", device="cuda")
    return _embed_model


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def keyword_recall(expected_keywords: list, text: str) -> float:
    """Fraction of expected keywords found in text (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    text_lower = text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
    return hits / len(expected_keywords)


def source_routing_accuracy(expected_sources: list, actual_sources: list) -> dict:
    """Precision / recall of source selection (vector, sql, rules, graph)."""
    expected = set(expected_sources)
    actual = set(actual_sources)

    tp = len(expected & actual)
    fp = len(actual - expected)
    fn = len(expected - actual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evidence_quality(evidence_bundle: dict, expected_keywords: list) -> dict:
    """Measure how well retrieved evidence covers expected keywords."""
    all_evidence_text = ""

    # Documents
    for doc in evidence_bundle.get("documents", []):
        if isinstance(doc, dict):
            all_evidence_text += " " + doc.get("text", "")

    # SQL results
    sql_results = evidence_bundle.get("sql_results", [])
    if isinstance(sql_results, list):
        for sr in sql_results:
            if isinstance(sr, dict):
                for row in sr.get("rows", []):
                    all_evidence_text += " " + str(row)

    # Rules
    rules = evidence_bundle.get("rules", {})
    if isinstance(rules, dict):
        for rm in rules.get("matched_rules", []):
            if isinstance(rm, dict):
                all_evidence_text += " " + rm.get("rule_text", "")

    # Graph
    graph = evidence_bundle.get("graph", {})
    if isinstance(graph, dict):
        for rp in graph.get("reasoning_paths", []):
            if isinstance(rp, dict):
                all_evidence_text += " " + rp.get("explanation", "")

    recall = keyword_recall(expected_keywords, all_evidence_text)
    return {"evidence_keyword_recall": recall, "evidence_length": len(all_evidence_text)}


def answer_faithfulness(answer_text: str, evidence_bundle: dict) -> float:
    """
    Simple faithfulness proxy: fraction of answer sentences
    whose key nouns appear in the evidence.
    """
    if not answer_text:
        return 0.0

    all_evidence_text = ""
    for doc in evidence_bundle.get("documents", []):
        if isinstance(doc, dict):
            all_evidence_text += " " + doc.get("text", "")
    sql_results = evidence_bundle.get("sql_results", [])
    if isinstance(sql_results, list):
        for sr in sql_results:
            if isinstance(sr, dict):
                for row in sr.get("rows", []):
                    all_evidence_text += " " + str(row)
    rules = evidence_bundle.get("rules", {})
    if isinstance(rules, dict):
        for rm in rules.get("matched_rules", []):
            if isinstance(rm, dict):
                all_evidence_text += " " + rm.get("rule_text", "")
    graph = evidence_bundle.get("graph", {})
    if isinstance(graph, dict):
        for rp in graph.get("reasoning_paths", []):
            if isinstance(rp, dict):
                all_evidence_text += " " + rp.get("explanation", "")

    evidence_lower = all_evidence_text.lower()

    # Split answer into sentences
    sentences = [s.strip() for s in answer_text.replace("\n", ". ").split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    grounded_count = 0
    for sent in sentences:
        words = [w for w in sent.lower().split() if len(w) > 3]
        if not words:
            continue
        overlap = sum(1 for w in words if w in evidence_lower)
        if overlap / len(words) >= 0.3:
            grounded_count += 1

    return grounded_count / len(sentences)


def claim_level_faithfulness(answer_text: str, evidence_bundle: dict) -> dict:
    """
    Split answer into claims (sentences), check each for:
    1. Has inline citation tag [doc/sql/rule/graph/general knowledge]
    2. If tagged with evidence source, verify content overlap with evidence
    Returns claim-level pass rate.
    """
    if not answer_text:
        return {"total_claims": 0, "cited_claims": 0, "grounded_claims": 0,
                "citation_rate": 0.0, "grounding_rate": 0.0}

    # Split into sentences/claims
    sentences = [s.strip() for s in re.split(r'[.!?\n]', answer_text) if len(s.strip()) > 15]
    if not sentences:
        return {"total_claims": 0, "cited_claims": 0, "grounded_claims": 0,
                "citation_rate": 0.0, "grounding_rate": 0.0}

    # Build evidence text pool
    evidence_text = ""
    for doc in evidence_bundle.get("documents", []):
        if isinstance(doc, dict):
            evidence_text += " " + doc.get("text", "")
    sql_results = evidence_bundle.get("sql_results", [])
    if isinstance(sql_results, list):
        for sr in sql_results:
            if isinstance(sr, dict):
                for row in sr.get("rows", []):
                    evidence_text += " " + str(row)
    rules = evidence_bundle.get("rules", {})
    if isinstance(rules, dict):
        for rm in rules.get("matched_rules", []):
            if isinstance(rm, dict):
                evidence_text += " " + rm.get("rule_text", "")
    graph = evidence_bundle.get("graph", {})
    if isinstance(graph, dict):
        for rp in graph.get("reasoning_paths", []):
            if isinstance(rp, dict):
                evidence_text += " " + rp.get("explanation", "")
    evidence_lower = evidence_text.lower()

    citation_tags = re.compile(r'\[(doc|sql|rule|graph|general knowledge)\]', re.IGNORECASE)

    total = len(sentences)
    cited = 0
    grounded = 0

    for sent in sentences:
        has_citation = bool(citation_tags.search(sent))
        if has_citation:
            cited += 1

        # Check grounding: key words from sentence appear in evidence
        words = [w for w in sent.lower().split() if len(w) > 3
                 and w not in {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'also', 'such', 'these', 'those', 'into', 'more', 'than', 'over', 'each', 'which', 'their', 'about'}]
        if words:
            overlap = sum(1 for w in words if w in evidence_lower)
            if overlap / len(words) >= 0.25:
                grounded += 1

    return {
        "total_claims": total,
        "cited_claims": cited,
        "grounded_claims": grounded,
        "citation_rate": round(cited / total, 4) if total > 0 else 0.0,
        "grounding_rate": round(grounded / total, 4) if total > 0 else 0.0,
    }


def semantic_similarity_score(answer: str, reference: str) -> float:
    """Cosine similarity between answer and reference embeddings."""
    if not answer or not reference:
        return 0.0
    model = _get_embed_model()
    embs = model.encode([answer, reference], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


def mean_reciprocal_rank(retrieved_docs: list, expected_keywords: list) -> float:
    """MRR: 1/rank of first relevant document."""
    if not expected_keywords or not retrieved_docs:
        return 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        text = (doc.get("text", "") if isinstance(doc, dict) else "").lower()
        hits = sum(1 for kw in expected_keywords if kw.lower() in text)
        if hits >= 2 or (hits >= 1 and len(expected_keywords) <= 2):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_docs: list, expected_keywords: list, k: int = 5) -> float:
    """NDCG@k based on keyword relevance per document."""
    if not expected_keywords or not retrieved_docs:
        return 0.0

    docs = retrieved_docs[:k]
    relevances = []
    for doc in docs:
        text = (doc.get("text", "") if isinstance(doc, dict) else "").lower()
        rel = sum(1 for kw in expected_keywords if kw.lower() in text) / len(expected_keywords)
        relevances.append(rel)

    # DCG
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

    # Ideal DCG (sort relevances descending)
    ideal = sorted(relevances, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_docs: list, relevant_chunk_ids: list, k: int) -> float:
    """
    Recall@K: fraction of relevant chunks found in the top-K retrieved docs.

    Args:
        retrieved_docs: list of dicts with 'chunk_id' key
        relevant_chunk_ids: list of ground-truth chunk IDs
        k: cutoff
    Returns:
        recall in [0, 1]
    """
    if not relevant_chunk_ids or not retrieved_docs:
        return 0.0
    top_k_ids = set()
    for doc in retrieved_docs[:k]:
        cid = doc.get("chunk_id", "") if isinstance(doc, dict) else ""
        if cid:
            top_k_ids.add(str(cid))
    relevant_set = set(str(c) for c in relevant_chunk_ids)
    hits = len(top_k_ids & relevant_set)
    return hits / len(relevant_set)


# ---------------------------------------------------------------------------
# Per-source retrieval metrics
# ---------------------------------------------------------------------------

def sql_table_metrics(state: dict, golden_sql: dict) -> dict:
    """Precision/Recall/F1 for SQL table selection."""
    if not golden_sql or not golden_sql.get("expected_tables"):
        return {"precision": None, "recall": None, "f1": None}
    expected = set(golden_sql["expected_tables"].keys())
    sql_results = state.get("sql_results", [])
    actual = set()
    for sr in sql_results:
        plan = sr.get("plan", {})
        actual.update(plan.get("target_tables", []))
    if not actual and not expected:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = len(expected & actual)
    precision = tp / len(actual) if actual else 0.0
    recall = tp / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def rule_variable_metrics(state: dict, golden_rules: dict) -> dict:
    """Recall for rule variable matching."""
    if not golden_rules or not golden_rules.get("expected_rule_variables"):
        return {"recall": None, "matched_count": 0}
    expected_vars = set(v.lower() for v in golden_rules["expected_rule_variables"])
    rule_results = state.get("rule_results", {}) or {}
    matched_rules = rule_results.get("matched_rules", []) or []
    actual_vars = set()
    for r in matched_rules:
        v = r.get("variable") or ""
        actual_vars.add(v.lower())
    hits = len(expected_vars & actual_vars)
    recall = hits / len(expected_vars) if expected_vars else 0.0
    return {"recall": round(recall, 4), "matched_count": len(matched_rules)}


def guardrail_metrics(answer_text: str, state: dict, item: dict) -> dict:
    """
    Evaluate whether answer_synthesizer guardrails worked correctly.

    Tests:
    1. decision_support WITHOUT rules → answer should NOT contain hard stop/go phrases
    2. knowledge_fallback → should be triggered when expected
    3. citation tags → every factual claim should have [doc]/[sql]/[rule]/[graph]/[general knowledge]
    """
    result = {
        "guardrail_type": item.get("expected_guardrail"),
        "guardrail_pass": None,
        "guardrail_detail": "",
    }

    if not item.get("expected_guardrail"):
        return result

    gt = item["expected_guardrail"]
    answer_lower = answer_text.lower() if answer_text else ""
    final_answer = state.get("final_answer", {})

    if gt == "no_source_fallback":
        # Empty source query — system should have weak/no evidence grounding
        grounding = final_answer.get("grounding_status", "")
        kb_used = final_answer.get("knowledge_fallback_used", False)
        passed = grounding in ("weakly_grounded", "fallback_augmented") or kb_used
        result["guardrail_pass"] = passed
        result["guardrail_detail"] = f"grounding={grounding} kb_fallback={kb_used}"

    elif gt == "decision_support_no_rules":
        # Should NOT contain hard stop/go phrases
        forbidden = [
            "proceed with operations", "pause operations now", "suspend operations",
            "safe to continue", "operations should be paused", "restrict vessel entry",
            "operations must be suspended", "you should pause", "must suspend",
        ]
        violations = [p for p in forbidden if p in answer_lower]
        passed = len(violations) == 0
        result["guardrail_pass"] = passed
        result["guardrail_detail"] = f"violations={violations}" if violations else "no forbidden phrases found"

    elif gt == "decision_support_with_rules":
        # Should provide a recommendation (guardrail should NOT block)
        recommendation_signals = [
            "recommend", "should", "advise", "suggest", "based on the rule",
            "threshold", "according to", "within", "exceeds", "below",
        ]
        has_recommendation = any(s in answer_lower for s in recommendation_signals)
        result["guardrail_pass"] = has_recommendation
        result["guardrail_detail"] = f"has_recommendation={has_recommendation}"

    elif gt == "diagnostic_without_graph":
        # Fallback should be triggered (check final_answer metadata)
        kb_used = final_answer.get("knowledge_fallback_used", False)
        result["guardrail_pass"] = kb_used
        result["guardrail_detail"] = f"knowledge_fallback_used={kb_used}"

    elif gt == "descriptive_without_docs":
        kb_used = final_answer.get("knowledge_fallback_used", False)
        result["guardrail_pass"] = kb_used
        result["guardrail_detail"] = f"knowledge_fallback_used={kb_used}"

    elif gt == "knowledge_fallback_test":
        kb_used = final_answer.get("knowledge_fallback_used", False)
        # Either fallback or the system found actual evidence (which is also fine)
        sources = final_answer.get("sources_used", [])
        result["guardrail_pass"] = kb_used or len(sources) >= 2
        result["guardrail_detail"] = f"kb_fallback={kb_used} sources={sources}"

    return result


def graph_entity_metrics(state: dict, golden_graph: dict) -> dict:
    """Recall for graph entity extraction."""
    if not golden_graph or not golden_graph.get("expected_entities"):
        return {"recall": None, "path_count": 0}
    expected = set(golden_graph["expected_entities"])
    graph_results = state.get("graph_results", {}) or {}
    actual_entities = set(graph_results.get("query_entities", []))
    expanded = set(graph_results.get("expanded_nodes", []))
    # Count a hit if entity found in query_entities OR expanded_nodes
    all_found = actual_entities | expanded
    hits = len(expected & all_found)
    recall = hits / len(expected) if expected else 0.0
    paths = graph_results.get("reasoning_paths", []) or []
    return {"recall": round(recall, 4), "path_count": len(paths)}


# ---------------------------------------------------------------------------
# Normalise source names between golden dataset and system output
# ---------------------------------------------------------------------------

SOURCE_MAP = {
    "vector": "documents",
    "documents": "documents",
    "sql": "sql",
    "structured_operational_data": "sql",
    "rules": "rules",
    "graph": "graph",
}


def normalise_source(s: str) -> str:
    return SOURCE_MAP.get(s.lower(), s.lower())


def _infer_intent(expected_sources: list) -> str:
    """Derive intent_type from expected_sources for grouping."""
    s = set(expected_sources)
    if not s:
        return "out_of_domain"
    if "graph" in s:
        return "causal_multihop"
    if len(s) >= 2:
        return "hybrid_reasoning"
    # After normalise_source, "vector" becomes "documents"
    if s == {"documents"} or s == {"vector"}:
        return "document_lookup"
    if s == {"sql"} or s == {"structured_operational_data"}:
        return "structured_data"
    if s == {"rules"}:
        return "policy_rule"
    return "unknown"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(golden_path: str = None, max_queries: int = None, skip_reranker: bool = False):
    # Initialize pipeline logging — writes to file + stderr
    log_file = str(PROJECT_ROOT / "evaluation" / "pipeline_debug.log")
    setup_pipeline_logging(level="INFO", log_file=log_file)

    # Evaluation-level log (captures per-query metrics + summary)
    import logging
    eval_log_path = str(PROJECT_ROOT / "evaluation" / "evaluation_run.log")
    eval_logger = logging.getLogger("evaluation")
    eval_logger.setLevel(logging.INFO)
    _eval_fh = logging.FileHandler(eval_log_path, encoding="utf-8", mode="w")
    _eval_fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    eval_logger.addHandler(_eval_fh)

    if golden_path is None:
        golden_path = str(PROJECT_ROOT / "evaluation" / "golden_dataset.json")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    if max_queries:
        golden = golden[:max_queries]

    mode_label = "WITH reranker" if not skip_reranker else "WITHOUT reranker (ablation)"
    print(f"Loading LangGraph workflow from {PROJECT_ROOT} ... [{mode_label}]")

    from online_pipeline.langgraph_workflow import LangGraphWorkflowBuilder
    builder = LangGraphWorkflowBuilder(
        project_root=PROJECT_ROOT,
        chroma_collection_name=None,
        use_llm_sql_planner=True,
        sql_model_name=None,  # uses centralized llm_client default
    )

    # Ablation: replace reranker with passthrough
    if skip_reranker:
        def _passthrough_rerank(state):
            docs = state.get("retrieved_docs", [])
            return {"retrieved_docs": docs[:5],
                    "reasoning_trace": ["rerank_documents_node => ABLATION: passthrough top-5"]}
        builder.factory.rerank_documents_node = _passthrough_rerank
        print("  [ABLATION] Reranker disabled - using hybrid top-5 directly")

    app = builder.build()

    results = []
    totals = {
        "evidence_keyword_recall": [],
        "source_routing_precision": [],
        "source_routing_recall": [],
        "source_routing_f1": [],
        "answer_faithfulness": [],
        "answer_confidence": [],
        "has_answer": 0,
        "total": 0,
    }

    print(f"\nRunning evaluation on {len(golden)} queries ...\n")
    print("=" * 100)

    for idx, item in enumerate(golden):
        query_id = item["id"]
        query = item["query"]
        expected_sources = [normalise_source(s) for s in item.get("expected_sources", [])]
        expected_keywords = item.get("expected_evidence_keywords", [])

        print(f"\n[{idx+1}/{len(golden)}] {query_id}: {query[:80]}...")

        t0 = time.time()
        try:
            state = app.invoke({
                "user_query": query,
                "reasoning_trace": [],
                "warnings": [],
            })
            elapsed = time.time() - t0
            error = None
        except Exception as e:
            elapsed = time.time() - t0
            error = str(e)
            state = {}

        final_answer = state.get("final_answer", {})
        answer_text = final_answer.get("answer", "")
        confidence = final_answer.get("confidence", 0.0)
        actual_sources = [normalise_source(s) for s in final_answer.get("sources_used", [])]
        evidence_bundle = state.get("evidence_bundle", {})

        # Compute metrics
        routing = source_routing_accuracy(expected_sources, actual_sources)
        evidence = evidence_quality(evidence_bundle, expected_keywords)
        faithfulness = answer_faithfulness(answer_text, evidence_bundle)

        # New metrics
        reference = item.get("reference_answer", "")
        sem_sim = semantic_similarity_score(answer_text, reference) if answer_text else 0.0
        retrieved_docs = state.get("retrieved_docs", [])
        claim_faith = claim_level_faithfulness(answer_text, evidence_bundle)

        # Pre-rerank docs (top-20 from hybrid retrieval, before cross-encoder)
        pre_rerank_docs = state.get("pre_rerank_docs", [])

        # MRR/NDCG only meaningful for queries that involve document retrieval
        has_doc_source = "documents" in expected_sources
        mrr = mean_reciprocal_rank(retrieved_docs, expected_keywords) if has_doc_source else None
        ndcg = ndcg_at_k(retrieved_docs, expected_keywords, k=5) if has_doc_source else None

        # Pre-rerank MRR/NDCG (measures hybrid retrieval quality before cross-encoder)
        pre_mrr = mean_reciprocal_rank(pre_rerank_docs, expected_keywords) if (has_doc_source and pre_rerank_docs) else None
        pre_ndcg = ndcg_at_k(pre_rerank_docs, expected_keywords, k=20) if (has_doc_source and pre_rerank_docs) else None

        # Recall@K with annotated relevant chunk IDs (from golden_vector)
        golden_vec = item.get("golden_vector") or {}
        relevant_chunk_ids = golden_vec.get("relevant_chunk_ids", [])
        if has_doc_source and relevant_chunk_ids:
            recall_at_20 = recall_at_k(pre_rerank_docs, relevant_chunk_ids, k=20) if pre_rerank_docs else None
            recall_at_5 = recall_at_k(retrieved_docs, relevant_chunk_ids, k=5)
        else:
            recall_at_20 = None
            recall_at_5 = None

        # Per-source retrieval metrics
        per_source = {}
        per_source["sql"] = sql_table_metrics(state, item.get("golden_sql"))
        per_source["rules"] = rule_variable_metrics(state, item.get("golden_rules"))
        per_source["graph"] = graph_entity_metrics(state, item.get("golden_graph"))

        # Guardrail metrics
        guard = guardrail_metrics(answer_text, state, item)

        # SQL Result Accuracy: for sql-involving queries, check if answer contains expected numeric keywords
        sql_accuracy = None
        if "sql" in expected_sources and answer_text:
            sql_kws = [kw for kw in expected_keywords if any(c.isdigit() for c in kw)]
            non_num_kws = [kw for kw in expected_keywords if not any(c.isdigit() for c in kw)]
            text_lower = answer_text.lower()
            num_hits = sum(1 for kw in sql_kws if kw.lower() in text_lower) if sql_kws else 0
            term_hits = sum(1 for kw in non_num_kws if kw.lower() in text_lower) if non_num_kws else 0
            total_kws = len(sql_kws) + len(non_num_kws)
            sql_accuracy = (num_hits + term_hits) / total_kws if total_kws > 0 else 0.0

        result = {
            "id": query_id,
            "query": query,
            "intent_type": item.get("intent_type") or _infer_intent(expected_sources),
            "answer_mode": item.get("answer_mode"),
            "difficulty": item.get("difficulty"),
            "elapsed_s": round(elapsed, 2),
            "error": error,
            "has_answer": bool(answer_text),
            "answer_snippet": answer_text[:200] if answer_text else "",
            "confidence": confidence,
            "expected_sources": expected_sources,
            "actual_sources": actual_sources,
            "source_routing": routing,
            "evidence_keyword_recall": evidence["evidence_keyword_recall"],
            "evidence_length": evidence["evidence_length"],
            "answer_faithfulness": faithfulness,
            "semantic_similarity": round(sem_sim, 4),
            "pre_rerank_mrr": round(pre_mrr, 4) if pre_mrr is not None else None,
            "pre_rerank_ndcg_at_20": round(pre_ndcg, 4) if pre_ndcg is not None else None,
            "recall_at_20": round(recall_at_20, 4) if recall_at_20 is not None else None,
            "mrr": round(mrr, 4) if mrr is not None else None,
            "ndcg_at_5": round(ndcg, 4) if ndcg is not None else None,
            "recall_at_5": round(recall_at_5, 4) if recall_at_5 is not None else None,
            "sql_result_accuracy": round(sql_accuracy, 4) if sql_accuracy is not None else None,
            "claim_citation_rate": claim_faith["citation_rate"],
            "claim_grounding_rate": claim_faith["grounding_rate"],
            "claim_total": claim_faith["total_claims"],
            "per_source_metrics": per_source,
            "guardrail": guard,
        }
        results.append(result)

        totals["total"] += 1
        if answer_text:
            totals["has_answer"] += 1
        totals["evidence_keyword_recall"].append(evidence["evidence_keyword_recall"])
        totals["source_routing_precision"].append(routing["precision"])
        totals["source_routing_recall"].append(routing["recall"])
        totals["source_routing_f1"].append(routing["f1"])
        totals["answer_faithfulness"].append(faithfulness)
        totals["answer_confidence"].append(confidence)
        totals.setdefault("semantic_similarity", []).append(sem_sim)
        # Only accumulate MRR/NDCG for doc-involving queries
        if pre_mrr is not None:
            totals.setdefault("pre_rerank_mrr", []).append(pre_mrr)
        if pre_ndcg is not None:
            totals.setdefault("pre_rerank_ndcg_at_20", []).append(pre_ndcg)
        if recall_at_20 is not None:
            totals.setdefault("recall_at_20", []).append(recall_at_20)
        if mrr is not None:
            totals.setdefault("mrr", []).append(mrr)
        if ndcg is not None:
            totals.setdefault("ndcg_at_5", []).append(ndcg)
        if recall_at_5 is not None:
            totals.setdefault("recall_at_5", []).append(recall_at_5)
        if sql_accuracy is not None:
            totals.setdefault("sql_result_accuracy", []).append(sql_accuracy)
        totals.setdefault("claim_citation_rate", []).append(claim_faith["citation_rate"])
        totals.setdefault("claim_grounding_rate", []).append(claim_faith["grounding_rate"])

        # Accumulate guardrail metrics
        if guard["guardrail_pass"] is not None:
            totals.setdefault("guardrail_total", []).append(1)
            totals.setdefault("guardrail_pass", []).append(1 if guard["guardrail_pass"] else 0)

        # Accumulate per-source metrics
        if per_source["sql"]["f1"] is not None:
            totals.setdefault("sql_table_f1", []).append(per_source["sql"]["f1"])
            totals.setdefault("sql_table_recall", []).append(per_source["sql"]["recall"])
        if per_source["rules"]["recall"] is not None:
            totals.setdefault("rule_var_recall", []).append(per_source["rules"]["recall"])
        if per_source["graph"]["recall"] is not None:
            totals.setdefault("graph_entity_recall", []).append(per_source["graph"]["recall"])

        status = "OK" if answer_text else "NO_ANSWER"
        if error:
            status = f"ERROR: {error[:60]}"
        _pre_mrr_s = f"{pre_mrr:.2f}" if pre_mrr is not None else "N/A"
        _mrr_s = f"{mrr:.2f}" if mrr is not None else "N/A"
        _ndcg_s = f"{ndcg:.2f}" if ndcg is not None else "N/A"
        _sqla_s = f"{sql_accuracy:.2f}" if sql_accuracy is not None else "N/A"
        _r20_s = f"{recall_at_20:.2f}" if recall_at_20 is not None else "N/A"
        _r5_s = f"{recall_at_5:.2f}" if recall_at_5 is not None else "N/A"
        per_query_line = (
            f"  {status} | conf={confidence:.2f} | "
            f"sem_sim={sem_sim:.2f} | pre_mrr={_pre_mrr_s} | mrr={_mrr_s} | ndcg={_ndcg_s} | "
            f"R@20={_r20_s} R@5={_r5_s} | sql_acc={_sqla_s} | "
            f"evidence_recall={evidence['evidence_keyword_recall']:.2f} | "
            f"routing_f1={routing['f1']:.2f} | {elapsed:.1f}s"
        )
        print(per_query_line)
        eval_logger.info("[%s] %s exp=%s act=%s %s",
                         query_id, query[:60], sorted(expected_sources), sorted(actual_sources),
                         per_query_line.strip())

        # Log graph reasoning details if graph was invoked
        graph_res = state.get("graph_results", {})
        if graph_res and graph_res.get("query_entities"):
            g_entities = graph_res.get("query_entities", [])
            g_paths = graph_res.get("reasoning_paths", [])
            print(f"    [graph] entities={g_entities}")
            for gp in g_paths[:3]:
                print(f"    [graph] {gp.get('explanation', '')}")
            if not g_paths:
                print(f"    [graph] WARNING: no paths found (error={graph_res.get('error')})")

        # Log reasoning trace
        trace = state.get("reasoning_trace", [])
        if trace:
            for t in trace[-5:]:
                print(f"    [trace] {t[:120]}")

    # ---------------------------------------------------------------------------
    # Aggregate metrics
    # ---------------------------------------------------------------------------

    def safe_avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # Per intent-type breakdown
    intent_breakdown = {}
    for r in results:
        it = r["intent_type"]
        if it not in intent_breakdown:
            intent_breakdown[it] = {
                "count": 0, "evidence_recall": [], "routing_f1": [],
                "faithfulness": [], "confidence": [],
                "semantic_similarity": [], "mrr": [], "ndcg_at_5": [],
            }
        intent_breakdown[it]["count"] += 1
        intent_breakdown[it]["evidence_recall"].append(r["evidence_keyword_recall"])
        intent_breakdown[it]["routing_f1"].append(r["source_routing"]["f1"])
        intent_breakdown[it]["faithfulness"].append(r["answer_faithfulness"])
        intent_breakdown[it]["confidence"].append(r["confidence"])
        intent_breakdown[it]["semantic_similarity"].append(r.get("semantic_similarity", 0))
        if r.get("mrr") is not None:
            intent_breakdown[it].setdefault("mrr_filtered", []).append(r["mrr"])
        if r.get("ndcg_at_5") is not None:
            intent_breakdown[it].setdefault("ndcg_filtered", []).append(r["ndcg_at_5"])
        if r.get("sql_result_accuracy") is not None:
            intent_breakdown[it].setdefault("sql_accuracy", []).append(r["sql_result_accuracy"])

    # Record model name for provenance
    try:
        from online_pipeline.llm_client import get_model_name as _gmn
        _current_model = _gmn()
    except Exception:
        _current_model = "unknown"

    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": _current_model,
        "total_queries": totals["total"],
        "answered": totals["has_answer"],
        "answer_rate": totals["has_answer"] / max(totals["total"], 1),
        "aggregate_metrics": {
            "avg_evidence_keyword_recall": round(safe_avg(totals["evidence_keyword_recall"]), 4),
            "avg_source_routing_precision": round(safe_avg(totals["source_routing_precision"]), 4),
            "avg_source_routing_recall": round(safe_avg(totals["source_routing_recall"]), 4),
            "avg_source_routing_f1": round(safe_avg(totals["source_routing_f1"]), 4),
            "avg_answer_faithfulness": round(safe_avg(totals["answer_faithfulness"]), 4),
            "avg_answer_confidence": round(safe_avg(totals["answer_confidence"]), 4),
            "avg_semantic_similarity": round(safe_avg(totals.get("semantic_similarity", [])), 4),
            "avg_pre_rerank_mrr_doc_only": round(safe_avg(totals.get("pre_rerank_mrr", [])), 4),
            "avg_pre_rerank_ndcg_at_20_doc_only": round(safe_avg(totals.get("pre_rerank_ndcg_at_20", [])), 4),
            "avg_recall_at_20_doc_only": round(safe_avg(totals.get("recall_at_20", [])), 4),
            "avg_mrr_doc_only": round(safe_avg(totals.get("mrr", [])), 4),
            "avg_ndcg_at_5_doc_only": round(safe_avg(totals.get("ndcg_at_5", [])), 4),
            "avg_recall_at_5_doc_only": round(safe_avg(totals.get("recall_at_5", [])), 4),
            "avg_sql_result_accuracy": round(safe_avg(totals.get("sql_result_accuracy", [])), 4),
            "avg_claim_citation_rate": round(safe_avg(totals.get("claim_citation_rate", [])), 4),
            "avg_claim_grounding_rate": round(safe_avg(totals.get("claim_grounding_rate", [])), 4),
        },
        "per_source_retrieval": {
            "vector": {
                "n_queries": len(totals.get("recall_at_20", [])),
                "avg_recall_at_20": round(safe_avg(totals.get("recall_at_20", [])), 4),
                "avg_recall_at_5": round(safe_avg(totals.get("recall_at_5", [])), 4),
                "avg_pre_rerank_mrr": round(safe_avg(totals.get("pre_rerank_mrr", [])), 4),
                "avg_post_rerank_mrr": round(safe_avg(totals.get("mrr", [])), 4),
            },
            "sql": {
                "n_queries": len(totals.get("sql_table_f1", [])),
                "avg_table_recall": round(safe_avg(totals.get("sql_table_recall", [])), 4),
                "avg_table_f1": round(safe_avg(totals.get("sql_table_f1", [])), 4),
            },
            "rules": {
                "n_queries": len(totals.get("rule_var_recall", [])),
                "avg_variable_recall": round(safe_avg(totals.get("rule_var_recall", [])), 4),
            },
            "graph": {
                "n_queries": len(totals.get("graph_entity_recall", [])),
                "avg_entity_recall": round(safe_avg(totals.get("graph_entity_recall", [])), 4),
            },
        },
        "guardrail_evaluation": {
            "total_guardrail_queries": len(totals.get("guardrail_total", [])),
            "guardrail_pass_count": sum(totals.get("guardrail_pass", [])),
            "guardrail_pass_rate": round(
                sum(totals.get("guardrail_pass", [])) / max(len(totals.get("guardrail_total", [])), 1), 4
            ),
            "per_query": [
                {"id": r["id"], "type": r["guardrail"]["guardrail_type"],
                 "pass": r["guardrail"]["guardrail_pass"], "detail": r["guardrail"]["guardrail_detail"]}
                for r in results if r["guardrail"]["guardrail_pass"] is not None
            ],
        },
        "per_intent_type": {},
        "per_query_results": results,
    }

    for it, data in intent_breakdown.items():
        report["per_intent_type"][it] = {
            "count": data["count"],
            "avg_evidence_recall": round(safe_avg(data["evidence_recall"]), 4),
            "avg_routing_f1": round(safe_avg(data["routing_f1"]), 4),
            "avg_faithfulness": round(safe_avg(data["faithfulness"]), 4),
            "avg_confidence": round(safe_avg(data["confidence"]), 4),
            "avg_semantic_similarity": round(safe_avg(data["semantic_similarity"]), 4),
            "avg_mrr_doc_only": round(safe_avg(data.get("mrr_filtered", [])), 4),
            "avg_ndcg_doc_only": round(safe_avg(data.get("ndcg_filtered", [])), 4),
            "avg_sql_accuracy": round(safe_avg(data.get("sql_accuracy", [])), 4),
        }

    # Save report
    report_path = PROJECT_ROOT / "evaluation" / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 100)
    print("EVALUATION REPORT SUMMARY")
    print("=" * 100)
    print(f"Total queries:             {report['total_queries']}")
    print(f"Answered:                  {report['answered']} ({report['answer_rate']*100:.1f}%)")
    agg = report['aggregate_metrics']
    print(f"Avg Evidence Keyword Recall: {agg['avg_evidence_keyword_recall']:.4f}")
    print(f"Avg Source Routing F1:       {agg['avg_source_routing_f1']:.4f}")
    print(f"Avg Answer Faithfulness:     {agg['avg_answer_faithfulness']:.4f}")
    print(f"Avg Semantic Similarity:     {agg['avg_semantic_similarity']:.4f}")
    print(f"--- Retrieval (pre-rerank, top-20 from hybrid) ---")
    print(f"  Avg Pre-Rerank MRR:        {agg['avg_pre_rerank_mrr_doc_only']:.4f}")
    print(f"  Avg Pre-Rerank NDCG@20:    {agg['avg_pre_rerank_ndcg_at_20_doc_only']:.4f}")
    print(f"  Avg Recall@20:             {agg['avg_recall_at_20_doc_only']:.4f}")
    print(f"--- Retrieval (post-rerank, top-5 from cross-encoder) ---")
    print(f"  Avg Post-Rerank MRR:       {agg['avg_mrr_doc_only']:.4f}")
    print(f"  Avg Post-Rerank NDCG@5:    {agg['avg_ndcg_at_5_doc_only']:.4f}")
    print(f"  Avg Recall@5:              {agg['avg_recall_at_5_doc_only']:.4f}")
    print(f"Avg SQL Result Accuracy:     {agg['avg_sql_result_accuracy']:.4f}")
    print(f"Avg Claim Citation Rate:     {agg['avg_claim_citation_rate']:.4f}")
    print(f"Avg Claim Grounding Rate:    {agg['avg_claim_grounding_rate']:.4f}")
    print(f"Avg Answer Confidence:       {report['aggregate_metrics']['avg_answer_confidence']:.4f}")
    print(f"Avg Routing Precision:       {report['aggregate_metrics']['avg_source_routing_precision']:.4f}")
    print(f"Avg Routing Recall:          {report['aggregate_metrics']['avg_source_routing_recall']:.4f}")

    # Per-source retrieval summary
    psr = report["per_source_retrieval"]
    print("\n--- Per-Source Retrieval ---")
    v = psr["vector"]
    print(f"  VECTOR (n={v['n_queries']}): Recall@20={v['avg_recall_at_20']:.4f} | Recall@5={v['avg_recall_at_5']:.4f} | pre_MRR={v['avg_pre_rerank_mrr']:.4f} | post_MRR={v['avg_post_rerank_mrr']:.4f}")
    s = psr["sql"]
    print(f"  SQL    (n={s['n_queries']}): Table Recall={s['avg_table_recall']:.4f} | Table F1={s['avg_table_f1']:.4f}")
    r = psr["rules"]
    print(f"  RULES  (n={r['n_queries']}): Variable Recall={r['avg_variable_recall']:.4f}")
    g = psr["graph"]
    print(f"  GRAPH  (n={g['n_queries']}): Entity Recall={g['avg_entity_recall']:.4f}")

    # Guardrail evaluation summary
    ge = report["guardrail_evaluation"]
    print(f"\n--- Guardrail Evaluation ---")
    print(f"  Total guardrail queries: {ge['total_guardrail_queries']}")
    print(f"  Pass rate: {ge['guardrail_pass_rate']:.2%} ({ge['guardrail_pass_count']}/{ge['total_guardrail_queries']})")
    for gq in ge["per_query"]:
        status = "PASS" if gq["pass"] else "FAIL"
        print(f"    {gq['id']:30s} [{status:4s}] {gq['type']:30s} | {gq['detail'][:60]}")

    print("\nPer Intent Type:")
    for it, data in report["per_intent_type"].items():
        print(f"  {it:25s} n={data['count']:2d} | "
              f"evidence_recall={data['avg_evidence_recall']:.3f} | "
              f"routing_f1={data['avg_routing_f1']:.3f} | "
              f"faithfulness={data['avg_faithfulness']:.3f} | "
              f"confidence={data['avg_confidence']:.3f}")

    # Per-source retrieval stats
    source_stats = {"documents": {"expected": 0, "actual": 0, "tp": 0},
                    "sql": {"expected": 0, "actual": 0, "tp": 0},
                    "rules": {"expected": 0, "actual": 0, "tp": 0},
                    "graph": {"expected": 0, "actual": 0, "tp": 0}}
    for r in results:
        for s in r.get("expected_sources", []):
            if s in source_stats:
                source_stats[s]["expected"] += 1
        for s in r.get("actual_sources", []):
            if s in source_stats:
                source_stats[s]["actual"] += 1
        for s in set(r.get("expected_sources", [])) & set(r.get("actual_sources", [])):
            if s in source_stats:
                source_stats[s]["tp"] += 1

    print("\nPer-Source Retrieval Stats:")
    print(f"  {'Source':12s} | {'Expected':>8s} | {'Actual':>6s} | {'TP':>4s} | {'Precision':>9s} | {'Recall':>6s}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*6}-+-{'-'*4}-+-{'-'*9}-+-{'-'*6}")
    per_source_stats = {}
    for src, st in source_stats.items():
        prec = st["tp"] / st["actual"] if st["actual"] > 0 else 0
        rec = st["tp"] / st["expected"] if st["expected"] > 0 else 0
        print(f"  {src:12s} | {st['expected']:8d} | {st['actual']:6d} | {st['tp']:4d} | {prec:9.3f} | {rec:6.3f}")
        per_source_stats[src] = {"expected": st["expected"], "actual": st["actual"],
                                  "tp": st["tp"], "precision": round(prec, 4), "recall": round(rec, 4)}

    report["per_source_stats"] = per_source_stats

    # Per-node latency profiling
    node_timings = builder.factory.node_timings
    if node_timings:
        print("\nPer-Node Latency Profile:")
        print(f"  {'Node':25s} | {'Count':>5s} | {'P50 (s)':>8s} | {'P95 (s)':>8s} | {'Mean (s)':>8s}")
        print(f"  {'-'*25}-+-{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        latency_report = {}
        for node, times in sorted(node_timings.items()):
            arr = np.array(times)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            mean = float(np.mean(arr))
            print(f"  {node:25s} | {len(times):5d} | {p50:8.2f} | {p95:8.2f} | {mean:8.2f}")
            latency_report[node] = {
                "count": len(times), "p50": round(p50, 3),
                "p95": round(p95, 3), "mean": round(mean, 3),
            }
        report["node_latency"] = latency_report

    print(f"\nFull report saved to: {report_path}")

    # Log summary to eval log
    eval_logger.info("=" * 80)
    eval_logger.info("SUMMARY: %d queries, answered=%d, routing_f1=%.4f, sem_sim=%.4f",
                     report["total_queries"], report["answered"],
                     agg["avg_source_routing_f1"], agg["avg_semantic_similarity"])
    psr = report["per_source_retrieval"]
    for src in ["vector", "sql", "rules", "graph"]:
        eval_logger.info("  %s: %s", src, psr[src])
    ge = report["guardrail_evaluation"]
    eval_logger.info("  guardrail: %d/%d pass (%.0f%%)",
                     ge["guardrail_pass_count"], ge["total_guardrail_queries"],
                     ge["guardrail_pass_rate"] * 100)
    eval_logger.info("Eval log: %s", eval_log_path)

    # Re-save with per-source stats + latency
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Generate markdown report
    generate_markdown_report(report, PROJECT_ROOT / "evaluation" / "EVALUATION_REPORT.md")

    return report


def generate_markdown_report(report: dict, output_path: Path):
    """Generate a human-readable markdown evaluation report."""

    agg = report["aggregate_metrics"]
    lines = [
        "# Port RAG System - Evaluation Report",
        "",
        f"**Date:** {report['timestamp'][:19]}",
        f"**Model:** {report.get('model_name', 'unknown')} (Alibaba DashScope)",
        f"**Embeddings:** BAAI/bge-small-en (384-dim)",
        f"**Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2",
        f"**Total Queries:** {report['total_queries']}",
        f"**Answer Rate:** {report['answer_rate']*100:.1f}%",
        "",
        "## Aggregate Metrics",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        f"| Evidence Keyword Recall | {agg['avg_evidence_keyword_recall']:.4f} |",
        f"| Source Routing Precision | {agg['avg_source_routing_precision']:.4f} |",
        f"| Source Routing Recall | {agg['avg_source_routing_recall']:.4f} |",
        f"| Source Routing F1 | {agg['avg_source_routing_f1']:.4f} |",
        f"| Answer Faithfulness | {agg['avg_answer_faithfulness']:.4f} |",
        f"| Semantic Similarity | {agg.get('avg_semantic_similarity', 0):.4f} |",
        f"| MRR (doc queries only) | {agg.get('avg_mrr_doc_only', 0):.4f} |",
        f"| NDCG@5 (doc queries only) | {agg.get('avg_ndcg_at_5_doc_only', 0):.4f} |",
        f"| SQL Result Accuracy | {agg.get('avg_sql_result_accuracy', 0):.4f} |",
        f"| Claim Citation Rate | {agg.get('avg_claim_citation_rate', 0):.4f} |",
        f"| Claim Grounding Rate | {agg.get('avg_claim_grounding_rate', 0):.4f} |",
        f"| Answer Confidence | {agg['avg_answer_confidence']:.4f} |",
        "",
    ]

    # Per-source stats
    per_source = report.get("per_source_stats", {})
    if per_source:
        lines += [
            "## Per-Source Retrieval Stats",
            "",
            "| Source | Expected | Actual | TP | Precision | Recall |",
            "|--------|----------|--------|----|-----------|--------|",
        ]
        for src, st in per_source.items():
            lines.append(
                f"| {src} | {st['expected']} | {st['actual']} | {st['tp']} | "
                f"{st['precision']:.3f} | {st['recall']:.3f} |"
            )
        lines.append("")

    lines += [
        "## Per Intent Type Breakdown",
        "",
        "| Intent Type | Count | Evidence Recall | Routing F1 | Faithfulness | Sem. Sim. | MRR | NDCG@5 | Confidence |",
        "|-------------|-------|-----------------|------------|--------------|-----------|-----|--------|------------|",
    ]

    for it, data in report["per_intent_type"].items():
        lines.append(
            f"| {it} | {data['count']} | "
            f"{data['avg_evidence_recall']:.3f} | "
            f"{data['avg_routing_f1']:.3f} | "
            f"{data['avg_faithfulness']:.3f} | "
            f"{data.get('avg_semantic_similarity', 0):.3f} | "
            f"{data.get('avg_mrr', 0):.3f} | "
            f"{data.get('avg_ndcg_at_5', 0):.3f} | "
            f"{data['avg_confidence']:.3f} |"
        )

    lines += [
        "",
        "## Per-Query Details",
        "",
        "| ID | Intent | Mode | Difficulty | Confidence | Routing F1 | Evidence Recall | Faithfulness | Time (s) |",
        "|----|--------|------|------------|------------|------------|-----------------|--------------|----------|",
    ]

    for r in report["per_query_results"]:
        lines.append(
            f"| {r['id']} | {r['intent_type']} | {r['answer_mode']} | "
            f"{r.get('difficulty', 'N/A')} | {r['confidence']:.2f} | "
            f"{r['source_routing']['f1']:.2f} | "
            f"{r['evidence_keyword_recall']:.2f} | "
            f"{r['answer_faithfulness']:.2f} | {r['elapsed_s']:.1f} |"
        )

    lines += [
        "",
        "## Metric Definitions",
        "",
        "- **Evidence Keyword Recall**: Fraction of expected keywords found in retrieved evidence",
        "- **Source Routing Precision**: Fraction of selected sources that were expected",
        "- **Source Routing Recall**: Fraction of expected sources that were selected",
        "- **Source Routing F1**: Harmonic mean of routing precision and recall",
        "- **Answer Faithfulness**: Fraction of answer sentences grounded in retrieved evidence",
        "- **Answer Confidence**: System-reported confidence score (0-1)",
        "",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Markdown report saved to: {output_path}")


if __name__ == "__main__":
    # Usage: python run_evaluation.py [max_queries] [--no-reranker]
    args = sys.argv[1:]
    max_q = None
    skip_rr = False
    for arg in args:
        if arg == "--no-reranker":
            skip_rr = True
        elif arg.isdigit():
            max_q = int(arg)
    run_evaluation(max_queries=max_q, skip_reranker=skip_rr)
