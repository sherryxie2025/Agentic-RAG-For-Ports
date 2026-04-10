# src/online_pipeline/state_schema.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


QuestionType = Literal[
    "document_lookup",
    "structured_data",
    "policy_rule",
    "hybrid_reasoning",
    "causal_multihop",
    "unknown",
]

AnswerMode = Literal[
    "descriptive",
    "lookup",
    "comparison",
    "decision_support",
    "diagnostic",
    "unknown",
]


class RouterDecision(TypedDict, total=False):
    question_type: QuestionType
    answer_mode: AnswerMode

    needs_vector: bool
    needs_sql: bool
    needs_rules: bool
    needs_graph_reasoning: bool

    confidence: float
    matched_keywords: List[str]
    rationale: str


class RetrievedDocument(TypedDict, total=False):
    chunk_id: str
    doc_id: Optional[int]
    source_file: str
    page: int
    text: str
    score: Optional[float]


class SQLQueryPlan(TypedDict, total=False):
    nl_query: str
    target_tables: List[str]
    target_columns: List[str]
    filters: Dict[str, Any]
    aggregation: Optional[str]
    generated_sql: Optional[str]


class SQLResultRow(TypedDict, total=False):
    data: Dict[str, Any]


class SQLExecutionResult(TypedDict, total=False):
    plan: SQLQueryPlan
    rows: List[SQLResultRow]
    row_count: int
    execution_ok: bool
    error: Optional[str]


class RuleMatch(TypedDict, total=False):
    rule_text: str
    condition: Optional[str]
    action: Optional[str]

    variable: Optional[str]
    threshold_raw: Optional[str]
    unit_raw: Optional[str]

    sql_variable: Optional[str]
    operator: Optional[str]
    value: Optional[Any]
    value_min: Optional[float]
    value_max: Optional[float]
    canonical_unit: Optional[str]

    source_file: Optional[str]
    page: Optional[int]

    relevance_score: Optional[float]
    matched: Optional[bool]
    triggered: Optional[bool]
    trigger_explanation: Optional[str]


class RuleEngineResult(TypedDict, total=False):
    matched_rules: List[RuleMatch]
    applicable_rule_count: int
    triggered_rule_count: int
    execution_ok: bool
    error: Optional[str]


class GraphReasoningPath(TypedDict, total=False):
    start_node: str
    end_node: str
    path_nodes: List[str]
    path_edges: List[str]
    explanation: str


class GraphReasoningResult(TypedDict, total=False):
    query_entities: List[str]
    expanded_nodes: List[str]
    reasoning_paths: List[GraphReasoningPath]
    execution_ok: bool
    error: Optional[str]


class EvidenceBundle(TypedDict, total=False):
    documents: List[RetrievedDocument]
    sql_results: List[SQLExecutionResult]
    rules: RuleEngineResult
    graph: GraphReasoningResult


class FinalAnswer(TypedDict, total=False):
    answer: str
    confidence: Optional[float]
    sources_used: List[str]
    reasoning_summary: List[str]
    caveats: List[str]

    grounding_status: str
    llm_answer_used: bool
    knowledge_fallback_used: bool
    knowledge_fallback_notes: List[str]


class PortQAState(TypedDict, total=False):
    user_query: str

    router_decision: RouterDecision

    question_type: QuestionType
    answer_mode: AnswerMode
    needs_vector: bool
    needs_sql: bool
    needs_rules: bool
    needs_graph_reasoning: bool

    sub_queries: List[Dict[str, Any]]
    selected_sources: List[str]

    retrieved_docs: List[RetrievedDocument]
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

    evidence_bundle: EvidenceBundle
    final_answer: FinalAnswer

    reasoning_trace: List[str]
    warnings: List[str]
    error: Optional[str]