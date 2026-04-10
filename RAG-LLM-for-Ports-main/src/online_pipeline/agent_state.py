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
    sql_results: List[SQLExecutionResult]
    rule_results: RuleEngineResult
    graph_results: GraphReasoningResult

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
