# src/online_pipeline/langgraph_state.py

from __future__ import annotations

from operator import add
from typing import Annotated, List, Optional, TypedDict

from .state_schema import (
    RouterDecision,
    RetrievedDocument,
    SQLExecutionResult,
    RuleEngineResult,
    GraphReasoningResult,
    EvidenceBundle,
    FinalAnswer,
)


class LangGraphPortState(TypedDict, total=False):
    user_query: str
    original_query: str

    router_decision: RouterDecision
    question_type: str
    answer_mode: str
    needs_vector: bool
    needs_sql: bool
    needs_rules: bool
    needs_graph_reasoning: bool

    source_plan: List[str]
    sub_queries: List[dict]
    # reasoning_goal removed — answer_mode is used directly by synthesizer
    execution_strategy: str

    pre_rerank_docs: List[RetrievedDocument]
    retrieved_docs: List[RetrievedDocument]
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

    evidence_bundle: EvidenceBundle
    final_answer: FinalAnswer

    reasoning_trace: Annotated[List[str], add]
    warnings: Annotated[List[str], add]
    error: Optional[str]