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


# ---------------------------------------------------------------------------
# Agent-specific types (Plan-and-Execute architecture)
# ---------------------------------------------------------------------------

class PlanStep(TypedDict, total=False):
    """A single step in the agent's execution plan."""
    step_id: int
    tool_name: str          # "document_search", "sql_query", "rule_lookup", "graph_reason", etc.
    query: str              # The sub-query for this tool
    purpose: str            # Why this step is needed
    status: str             # "pending", "running", "completed", "failed"
    result_summary: str     # Brief summary of result after execution


class ToolResult(TypedDict, total=False):
    """Record of a single tool invocation."""
    tool_name: str
    input_query: str
    output: Any
    execution_time_s: float
    success: bool
    error: Optional[str]


# ---------------------------------------------------------------------------
# Multi-turn conversation types
# ---------------------------------------------------------------------------

class ConversationTurn(TypedDict, total=False):
    """A single turn in the conversation history."""
    role: str                                   # "user" | "assistant"
    content: str                                # message text
    timestamp: str                              # ISO-8601
    turn_id: int
    tool_results_summary: Optional[List[str]]   # brief tool summaries
    entities_mentioned: Optional[List[str]]     # extracted domain entities


class ConversationSummary(TypedDict, total=False):
    """Compressed summary of older conversation turns."""
    summary_text: str
    turns_covered: List[int]        # which turn_ids this covers
    key_entities: List[str]
    key_facts: List[str]            # factual claims established


class KeyFactRecord(TypedDict, total=False):
    """
    A durable, atomic factual claim extracted from older conversation turns.

    Lives longer than raw turns and individual summaries — this is the
    bottom layer of the 3-tier short-term memory. Key facts are the only
    thing that survives when a conversation grows past the buffer + summary
    budget.
    """
    fact: str                       # concise one-line claim
    from_turn_ids: List[int]        # which turns this fact was distilled from
    entities: List[str]             # entities mentioned in the fact
    extracted_at: str               # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# ReAct observation types
# ---------------------------------------------------------------------------

class ObservationResult(TypedDict, total=False):
    """LLM observation after a tool execution (ReAct pattern)."""
    step_id: int
    tool_name: str
    observation: str                # what the LLM learned from the result
    action: str                     # "continue" | "modify_next" | "abort_replan"
    modified_query: Optional[str]   # revised query if action == "modify_next"
    reasoning: str                  # why the LLM chose this action


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