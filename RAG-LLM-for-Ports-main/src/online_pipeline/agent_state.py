# src/online_pipeline/agent_state.py
"""
Agent state for the Plan-and-Execute architecture.

Extends the original pipeline state with:
- Execution plan (list of PlanStep)
- Iteration tracking for adaptive re-planning
- Tool result accumulation
- Evidence evaluation fields
- Multi-turn conversation history
- Memory context (short-term + long-term)
- ReAct observation results
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from .state_schema import (
    ConversationTurn,
    EvidenceBundle,
    FinalAnswer,
    GraphReasoningResult,
    ObservationResult,
    PlanStep,
    RetrievedDocument,
    RuleEngineResult,
    SQLExecutionResult,
    ToolResult,
)


def _merge_timings(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    """
    Reducer for stage_timings: merge two dicts, summing values for duplicate keys.
    Summing handles re-plan iterations where the same node runs multiple times.
    """
    if not a:
        return dict(b) if b else {}
    if not b:
        return dict(a)
    merged = dict(a)
    for k, v in b.items():
        merged[k] = merged.get(k, 0.0) + v
    return merged


class AgentState(TypedDict, total=False):
    # -- Input --
    user_query: str
    original_query: str

    # -- Multi-turn conversation --
    session_id: Optional[str]
    conversation_history: List[ConversationTurn]     # recent turns injected from memory
    conversation_summary: Optional[str]              # compressed summary of older turns

    # -- Memory --
    memory_context: Optional[str]                    # relevant memories injected as prompt text
    active_entities: Dict[str, Any]                  # entities tracked across turns

    # -- Agent plan & iteration --
    plan: List[PlanStep]
    current_step_index: int
    iteration: int                          # re-plan count (capped at 3)

    # -- Tool results (append-only across iterations) --
    tool_results: Annotated[List[ToolResult], add]
    retrieved_docs: List[RetrievedDocument]
    pre_rerank_docs: List[RetrievedDocument]   # before cross-encoder rerank (for lift metrics)
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

    # -- Per-stage latency tracking --
    stage_timings: Annotated[Dict[str, float], _merge_timings]

    # -- ReAct observations (append-only) --
    observations: Annotated[List[ObservationResult], add]

    # -- Evidence evaluation --
    evidence_bundle: EvidenceBundle
    evidence_sufficient: bool
    evidence_gaps: List[str]

    # -- Output --
    final_answer: FinalAnswer

    # -- Observability --
    reasoning_trace: Annotated[List[str], add]
    warnings: Annotated[List[str], add]
    error: Optional[str]
