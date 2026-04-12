# src/online_pipeline/langgraph_state.py

from __future__ import annotations

from operator import add
from typing import Annotated, Dict, List, Optional, TypedDict

from .state_schema import (
    RouterDecision,
    RetrievedDocument,
    SQLExecutionResult,
    RuleEngineResult,
    GraphReasoningResult,
    EvidenceBundle,
    FinalAnswer,
)


def _merge_dicts(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """LangGraph reducer: merge per-node timing dicts across nodes."""
    return {**a, **b}


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
    execution_strategy: str

    pre_rerank_docs: List[RetrievedDocument]
    retrieved_docs: List[RetrievedDocument]
    retrieved_children: List[RetrievedDocument]  # v3: child chunks post-rerank (for eval)
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

    evidence_bundle: EvidenceBundle
    final_answer: FinalAnswer

    reasoning_trace: Annotated[List[str], add]
    warnings: Annotated[List[str], add]
    error: Optional[str]

    # v3: per-invocation node timings (populated by NodeFactory._timed)
    _node_timings: Annotated[Dict[str, float], _merge_dicts]