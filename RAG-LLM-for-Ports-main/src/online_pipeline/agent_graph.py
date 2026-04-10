# src/online_pipeline/agent_graph.py
"""
Plan-and-Execute Agent built on LangGraph with ReAct-style observations.

Architecture:
    START -> plan_node -> execute_tools_node -> evaluate_evidence_node
                ^              (ReAct loop)              |
                |_____ [insufficient] ___________________|
                                                         |
                              [sufficient] -> synthesize_node -> END

Key agentic features:
- LLM-driven planning (tool selection + sub-query generation)
- Multi-turn conversation context injection
- Short-term + long-term memory integration
- ReAct-style per-tool observation (Think-Act-Observe within execute_tools)
- Self-evaluation with adaptive re-planning (up to 3 iterations)
- Evidence-grounded answer synthesis
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from .agent_prompts import (
    EVALUATE_EVIDENCE_PROMPT,
    OOD_DETECTION_PROMPT,
    PLAN_SYSTEM_PROMPT,
    REPLAN_SYSTEM_PROMPT,
    TOOL_OBSERVATION_PROMPT,
    format_tools_for_prompt,
)
from .agent_state import AgentState
from .agent_tools import AgentToolkit, ToolDescriptor
from .answer_synthesizer import AnswerSynthesizer
from .llm_client import llm_chat_json
from .state_schema import ObservationResult, PlanStep

logger = logging.getLogger("online_pipeline.agent_graph")

# Reduced from 3 to 2: observation from eval showed 56% of queries hit max=3
# but extra iterations rarely added meaningful evidence. Most "real" work
# completes in 1-2 iterations.
MAX_ITERATIONS = 2

# Tools that skip ReAct observation (too simple or always last)
_SKIP_OBSERVATION_TOOLS = {"query_rewrite", "evidence_conflict_check"}


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

class AgentNodes:
    """Encapsulates all node logic; holds references to toolkit + synthesizer."""

    def __init__(
        self,
        toolkit: AgentToolkit,
        enable_react_observations: bool = True,
    ) -> None:
        self.toolkit = toolkit
        self.synthesizer = AnswerSynthesizer(use_llm_fallback=True)
        self.enable_react = enable_react_observations

    # ---- Node 0: OOD Gate ----

    def ood_check_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Pre-plan gate: classifies the query as in-domain / out-of-domain /
        false_premise / too_vague. Sets ood_verdict in state and, if not
        in-domain, builds a refusal FinalAnswer that bypasses the rest of
        the pipeline.
        """
        t0 = time.time()
        user_query = state.get("user_query", "")

        prompt = OOD_DETECTION_PROMPT.format(query=user_query)
        result = llm_chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Classify this query."},
            ],
            temperature=0.0,
            timeout=30,
        )

        elapsed = time.time() - t0

        classification = "in_domain"
        refusal_message = ""
        reasoning = ""
        if isinstance(result, dict):
            classification = result.get("classification", "in_domain")
            refusal_message = result.get("refusal_message", "")
            reasoning = result.get("reasoning", "")

        logger.info(
            "OOD_CHECK: %.2fs classification=%s confidence=%s",
            elapsed, classification,
            (result or {}).get("confidence", "?"),
        )

        update: Dict[str, Any] = {
            "ood_verdict": classification,
            "stage_timings": {"ood_check_node": round(elapsed, 4)},
            "reasoning_trace": [
                f"ood_check: verdict={classification} ({reasoning[:80]})"
            ],
        }

        # If out-of-domain / false premise / too vague, short-circuit with a
        # refusal and skip the expensive pipeline.
        if classification != "in_domain":
            if not refusal_message:
                refusal_message = (
                    "I'm focused on port operations, maritime regulations, and "
                    "sustainability reports. That question falls outside my scope."
                )
            update["final_answer"] = {
                "answer": refusal_message,
                "confidence": 0.95,
                "sources_used": [],
                "reasoning_summary": [f"Query classified as {classification}: {reasoning}"],
                "caveats": [],
                "grounding_status": "refused_ood",
                "llm_answer_used": False,
                "knowledge_fallback_used": False,
                "knowledge_fallback_notes": [],
            }
            update["evidence_sufficient"] = True  # skip re-plan
            update["evidence_bundle"] = {
                "documents": [], "sql_results": [],
                "rules": {}, "graph": {},
            }

        return update

    @staticmethod
    def route_after_ood(state: AgentState) -> str:
        """Bypass plan/execute if OOD refusal triggered."""
        verdict = state.get("ood_verdict", "in_domain")
        if verdict != "in_domain" and state.get("final_answer"):
            return "end"
        return "plan"

    # ---- Node 1: Plan ----

    def plan_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        iteration = state.get("iteration", 0)
        user_query = state.get("user_query", "")

        tools_desc = format_tools_for_prompt(self.toolkit.tools)

        # Build optional conversation context block
        context_block = self._build_context_block(state)

        if iteration == 0:
            # First pass: plan from scratch
            prompt = PLAN_SYSTEM_PROMPT.format(tools_description=tools_desc)
            if context_block:
                prompt += f"\n\n## Conversation Context\n{context_block}"
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query},
            ]
        else:
            # Re-plan: incorporate evidence gaps + observations
            evidence_summary = self._summarize_evidence(state)
            evidence_gaps = state.get("evidence_gaps", [])

            # Include ReAct observations in replan context
            observations = state.get("observations", [])
            obs_text = ""
            if observations:
                obs_lines = [f"- {o.get('tool_name', '?')}: {o.get('observation', '')[:100]}"
                             for o in observations[-5:]]
                obs_text = "\n\n## Tool Observations\n" + "\n".join(obs_lines)

            prompt = REPLAN_SYSTEM_PROMPT.format(
                user_query=user_query,
                evidence_summary=evidence_summary + obs_text,
                evidence_gaps="\n".join(f"- {g}" for g in evidence_gaps),
                tools_description=tools_desc,
            )
            if context_block:
                prompt += f"\n\n## Conversation Context\n{context_block}"
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Fill these evidence gaps for: {user_query}"},
            ]

        raw_plan = llm_chat_json(messages, temperature=0.1)
        steps = self._parse_plan(raw_plan, start_id=self._next_step_id(state))

        elapsed = time.time() - t0
        logger.info("PLAN_NODE (iter=%d): %.2fs, %d steps", iteration + 1, elapsed, len(steps))

        # Merge with existing plan if re-planning
        existing_plan = state.get("plan", [])
        merged_plan = existing_plan + steps

        return {
            "plan": merged_plan,
            "iteration": iteration + 1,
            "stage_timings": {"plan_node": round(elapsed, 4)},
            "reasoning_trace": [
                f"plan_node (iter={iteration + 1}): {len(steps)} new steps — "
                + ", ".join(s.get("tool_name", "?") for s in steps)
            ],
        }

    # ---- Node 2: Execute Tools (with ReAct observation loop) ----

    def execute_tools_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        plan = state.get("plan", [])
        pending = [s for s in plan if s.get("status") == "pending"]

        tool_results = []
        observations = []
        retrieved_docs = state.get("retrieved_docs", [])
        pre_rerank_docs = state.get("pre_rerank_docs", [])
        sql_results = state.get("sql_results", [])
        rule_results = state.get("rule_results", {})
        graph_results = state.get("graph_results", {})
        trace = []

        # Per-tool timing breakdown (observation calls also counted)
        tool_time_sum = 0.0
        observe_time_sum = 0.0

        for step_idx, step in enumerate(pending):
            tool_name = step.get("tool_name", "")
            query = step.get("query", "")
            tool = self.toolkit.tool_map.get(tool_name)

            if not tool:
                step["status"] = "failed"
                trace.append(f"execute_tools: unknown tool '{tool_name}'")
                continue

            step["status"] = "running"

            # Build kwargs based on tool type
            if tool_name == "evidence_conflict_check":
                kwargs = {
                    "rule_results": rule_results,
                    "sql_results": sql_results if isinstance(sql_results, list) else [sql_results],
                }
            elif tool_name in ("document_search", "rule_lookup"):
                kwargs = {"query": query, "top_k": 5}
            else:
                kwargs = {"query": query}

            # ---- ACT: invoke the tool ----
            tool_t0 = time.time()
            result = tool.invoke(**kwargs)
            tool_elapsed = time.time() - tool_t0
            tool_time_sum += tool_elapsed
            tool_results.append(result)

            if result.get("success"):
                step["status"] = "completed"
                output = result.get("output", {})

                # Route output to the appropriate state field
                if tool_name == "document_search":
                    retrieved_docs = output.get("documents", [])
                    # Capture pre-rerank docs for rerank lift evaluation
                    pre_rerank_docs = output.get("pre_rerank_documents", retrieved_docs)
                    step["result_summary"] = f"{output.get('count', 0)} documents retrieved"
                elif tool_name == "sql_query":
                    sql_results = [output] if output else []
                    row_count = output.get("row_count", 0) if output else 0
                    step["result_summary"] = f"{row_count} rows returned"
                elif tool_name == "rule_lookup":
                    rule_results = output
                    step["result_summary"] = f"{output.get('applicable_rule_count', 0)} rules matched"
                elif tool_name == "graph_reason":
                    graph_results = output
                    paths = len(output.get("reasoning_paths", []))
                    step["result_summary"] = f"{paths} reasoning paths found"
                elif tool_name == "query_rewrite":
                    rewritten = output.get("rewritten_query", query)
                    step["result_summary"] = f"Rewritten: {rewritten[:80]}"
                elif tool_name == "evidence_conflict_check":
                    step["result_summary"] = f"{output.get('conflict_count', 0)} conflicts detected"
                else:
                    step["result_summary"] = "completed"

                trace.append(f"execute_tools: {tool_name} => {step.get('result_summary', 'ok')}")

                # ---- OBSERVE: ReAct observation after each tool ----
                remaining = [s for s in pending[step_idx + 1:] if s.get("status") == "pending"]
                if (self.enable_react
                        and remaining
                        and tool_name not in _SKIP_OBSERVATION_TOOLS):
                    obs_t0 = time.time()
                    observation = self._observe_tool_result(
                        user_query=state.get("user_query", ""),
                        step=step,
                        tool_result_summary=step.get("result_summary", ""),
                        remaining_steps=remaining,
                    )
                    observe_time_sum += time.time() - obs_t0
                    observations.append(observation)

                    action = observation.get("action", "continue")

                    if action == "abort_replan":
                        # Mark remaining steps as skipped, break loop
                        for s in remaining:
                            s["status"] = "skipped"
                        trace.append(
                            f"react_observe: ABORT after {tool_name} — "
                            f"{observation.get('reasoning', '?')[:80]}"
                        )
                        break

                    elif action == "modify_next" and remaining:
                        modified_q = observation.get("modified_query")
                        if modified_q:
                            remaining[0]["query"] = modified_q
                            trace.append(
                                f"react_observe: modified next step "
                                f"({remaining[0].get('tool_name', '?')}) query"
                            )
                    # else: continue as planned

            else:
                step["status"] = "failed"
                trace.append(f"execute_tools: {tool_name} FAILED: {result.get('error', '?')}")

        elapsed = time.time() - t0
        logger.info(
            "EXECUTE_TOOLS_NODE: %.2fs (tools=%.2fs, observe=%.2fs), %d tools run",
            elapsed, tool_time_sum, observe_time_sum, len(pending),
        )

        return {
            "plan": plan,
            "tool_results": tool_results,
            "observations": observations,
            "retrieved_docs": retrieved_docs,
            "pre_rerank_docs": pre_rerank_docs,
            "sql_results": sql_results,
            "rule_results": rule_results,
            "graph_results": graph_results,
            "stage_timings": {
                "execute_tools_node": round(elapsed, 4),
                "tools_total": round(tool_time_sum, 4),
                "react_observations": round(observe_time_sum, 4),
            },
            "reasoning_trace": trace,
        }

    # ---- Node 3: Evaluate Evidence ----

    def evaluate_evidence_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        user_query = state.get("user_query", "")
        iteration = state.get("iteration", 0)
        evidence_summary = self._summarize_evidence(state)

        # Build evidence bundle
        evidence_bundle = {
            "documents": state.get("retrieved_docs", []),
            "sql_results": state.get("sql_results", []),
            "rules": state.get("rule_results", {}),
            "graph": state.get("graph_results", {}),
        }

        # Enhanced conflict detection (Rule↔SQL, Doc↔SQL, Doc↔Rule, temporal)
        from .conflict_detector import detect_all_conflicts
        try:
            conflicts = detect_all_conflicts(evidence_bundle)
            evidence_bundle["conflict_annotations"] = conflicts
            if conflicts:
                logger.info(
                    "EVALUATE_NODE: %d conflicts detected (%s)",
                    len(conflicts),
                    ", ".join(set(c.get("conflict_type", "?") for c in conflicts)),
                )
        except Exception as e:
            logger.warning("Conflict detection failed: %s", e)
            evidence_bundle["conflict_annotations"] = []

        # If max iterations reached, skip evaluation — go straight to synthesis
        if iteration >= MAX_ITERATIONS:
            elapsed = time.time() - t0
            logger.info("EVALUATE_NODE: max iterations reached, proceeding to synthesis")
            return {
                "evidence_sufficient": True,
                "evidence_gaps": [],
                "evidence_bundle": evidence_bundle,
                "stage_timings": {"evaluate_evidence_node": round(elapsed, 4)},
                "reasoning_trace": [
                    f"evaluate_evidence (iter={iteration}): max iterations reached, forcing synthesis"
                ],
            }

        # LLM evaluation of evidence sufficiency
        prompt = EVALUATE_EVIDENCE_PROMPT.format(
            user_query=user_query,
            evidence_summary=evidence_summary,
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Evaluate evidence for: {user_query}"},
        ]

        eval_result = llm_chat_json(messages, temperature=0.0)
        sufficient = True
        gaps = []
        if eval_result and isinstance(eval_result, dict):
            sufficient = eval_result.get("sufficient", True)
            gaps = eval_result.get("gaps", [])

        elapsed = time.time() - t0
        logger.info(
            "EVALUATE_NODE: %.2fs sufficient=%s gaps=%d",
            elapsed, sufficient, len(gaps),
        )

        return {
            "evidence_sufficient": sufficient,
            "evidence_gaps": gaps,
            "evidence_bundle": evidence_bundle,
            "stage_timings": {"evaluate_evidence_node": round(elapsed, 4)},
            "reasoning_trace": [
                f"evaluate_evidence (iter={iteration}): sufficient={sufficient}, "
                f"gaps={gaps}"
            ],
        }

    # ---- Node 4: Synthesize Answer ----

    def synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()

        # Inject conversation context into synthesizer state if available
        synth_state = dict(state)
        context_block = self._build_context_block(state)
        if context_block:
            synth_state["_conversation_context"] = context_block

        final_answer = self.synthesizer.synthesize(synth_state)
        elapsed = time.time() - t0
        logger.info("SYNTHESIZE_NODE: %.2fs", elapsed)
        return {
            "final_answer": final_answer,
            "stage_timings": {"synthesize_node": round(elapsed, 4)},
            "reasoning_trace": [f"synthesize_node: produced final answer in {elapsed:.2f}s"],
        }

    # ---- Routing ----

    @staticmethod
    def route_after_evaluation(state: AgentState) -> str:
        if state.get("evidence_sufficient", False):
            return "synthesize"
        if state.get("iteration", 0) >= MAX_ITERATIONS:
            return "synthesize"
        return "re_plan"

    # ---- ReAct: Tool observation ----

    def _observe_tool_result(
        self,
        user_query: str,
        step: Dict[str, Any],
        tool_result_summary: str,
        remaining_steps: List[Dict[str, Any]],
    ) -> ObservationResult:
        """
        ReAct observation: LLM analyzes a tool result and decides whether to
        continue, modify the next step, or abort and replan.
        """
        remaining_desc = "\n".join(
            f"  {i+1}. {s.get('tool_name', '?')}: {s.get('query', '')[:80]}"
            for i, s in enumerate(remaining_steps)
        )

        prompt = TOOL_OBSERVATION_PROMPT.format(
            user_query=user_query,
            tool_name=step.get("tool_name", ""),
            tool_query=step.get("query", ""),
            step_purpose=step.get("purpose", ""),
            tool_result_summary=tool_result_summary,
            remaining_steps=remaining_desc or "(none)",
        )

        result = llm_chat_json(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Observe and decide next action."},
            ],
            temperature=0.0,
            timeout=30,
        )

        if result and isinstance(result, dict):
            obs = ObservationResult(
                step_id=step.get("step_id", 0),
                tool_name=step.get("tool_name", ""),
                observation=result.get("observation", ""),
                action=result.get("action", "continue"),
                modified_query=result.get("modified_query"),
                reasoning=result.get("reasoning", ""),
            )
            logger.info(
                "REACT_OBSERVE: %s -> action=%s",
                step.get("tool_name", "?"), obs.get("action"),
            )
            return obs

        # Default: continue if LLM fails
        return ObservationResult(
            step_id=step.get("step_id", 0),
            tool_name=step.get("tool_name", ""),
            observation="(observation failed, continuing)",
            action="continue",
            reasoning="LLM observation failed, defaulting to continue",
        )

    # ---- Helpers ----

    @staticmethod
    def _build_context_block(state: AgentState) -> str:
        """Build conversation + memory context block for prompt injection."""
        parts = []

        # Memory context (from MemoryManager)
        memory_ctx = state.get("memory_context")
        if memory_ctx:
            parts.append(memory_ctx)

        # Conversation summary (compressed older turns)
        summary = state.get("conversation_summary")
        if summary and not memory_ctx:  # avoid duplication if memory_ctx already includes it
            parts.append(f"[Conversation summary]: {summary}")

        # Recent conversation turns (not already in memory_ctx)
        history = state.get("conversation_history", [])
        if history and not memory_ctx:
            turn_lines = []
            for t in history[-4:]:
                role = t.get("role", "?")
                content = t.get("content", "")[:200]
                turn_lines.append(f"  {role}: {content}")
            parts.append("[Recent conversation]:\n" + "\n".join(turn_lines))

        return "\n\n".join(parts)

    @staticmethod
    def _summarize_evidence(state: AgentState) -> str:
        """Build a text summary of all gathered evidence for LLM prompts."""
        parts = []

        docs = state.get("retrieved_docs", [])
        if docs:
            parts.append(f"## Documents ({len(docs)} retrieved)")
            for d in docs[:5]:
                text_preview = (d.get("text", "") or "")[:200]
                parts.append(f"- [{d.get('source_file', '?')}] {text_preview}...")

        sql = state.get("sql_results", [])
        if sql:
            parts.append(f"\n## SQL Data ({len(sql)} queries)")
            for r in sql:
                if isinstance(r, dict):
                    row_count = r.get("row_count", 0)
                    plan = r.get("plan", {})
                    tables = plan.get("target_tables", []) if isinstance(plan, dict) else []
                    parts.append(f"- Tables: {tables}, Rows: {row_count}")
                    rows = r.get("rows", [])
                    for row in rows[:3]:
                        data = row.get("data", row) if isinstance(row, dict) else row
                        parts.append(f"  {json.dumps(data, default=str)[:200]}")

        rules = state.get("rule_results", {})
        if rules:
            matched = rules.get("matched_rules", []) if isinstance(rules, dict) else []
            if matched:
                parts.append(f"\n## Rules ({len(matched)} matched)")
                for rm in matched[:5]:
                    parts.append(f"- {(rm.get('rule_text', '') or '')[:150]}")

        graph = state.get("graph_results", {})
        if graph and isinstance(graph, dict):
            paths = graph.get("reasoning_paths", [])
            if paths:
                parts.append(f"\n## Graph Reasoning ({len(paths)} paths)")
                for p in paths[:3]:
                    parts.append(f"- {p.get('explanation', 'N/A')[:150]}")

        return "\n".join(parts) if parts else "(no evidence gathered yet)"

    @staticmethod
    def _parse_plan(raw: Any, start_id: int = 1) -> List[PlanStep]:
        """Parse LLM output into PlanStep list."""
        if not raw:
            return []
        steps_raw = raw if isinstance(raw, list) else [raw]
        steps = []
        for i, s in enumerate(steps_raw):
            if not isinstance(s, dict):
                continue
            steps.append(PlanStep(
                step_id=s.get("step_id", start_id + i),
                tool_name=s.get("tool_name", ""),
                query=s.get("query", ""),
                purpose=s.get("purpose", ""),
                status="pending",
                result_summary="",
            ))
        return steps

    @staticmethod
    def _next_step_id(state: AgentState) -> int:
        plan = state.get("plan", [])
        if not plan:
            return 1
        return max(s.get("step_id", 0) for s in plan) + 1


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class AgentGraphBuilder:
    """
    Builds the Plan-and-Execute LangGraph agent.
    Drop-in replacement for LangGraphWorkflowBuilder.
    """

    def __init__(
        self,
        project_root: str | Path,
        chroma_collection_name: str | None = None,
        use_llm_sql_planner: bool = False,
        sql_model_name: str | None = None,
        enable_react_observations: bool = True,
    ) -> None:
        self.toolkit = AgentToolkit(
            project_root=project_root,
            chroma_collection_name=chroma_collection_name,
            use_llm_sql_planner=use_llm_sql_planner,
            sql_model_name=sql_model_name,
        )
        self.nodes = AgentNodes(
            toolkit=self.toolkit,
            enable_react_observations=enable_react_observations,
        )

    def build(self):
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("ood_check", self.nodes.ood_check_node)
        graph.add_node("plan", self.nodes.plan_node)
        graph.add_node("execute_tools", self.nodes.execute_tools_node)
        graph.add_node("evaluate_evidence", self.nodes.evaluate_evidence_node)
        graph.add_node("synthesize", self.nodes.synthesize_node)

        # Flow: ood_check -> [in_domain: plan | ood: END]
        graph.add_edge(START, "ood_check")
        graph.add_conditional_edges(
            "ood_check",
            AgentNodes.route_after_ood,
            {
                "plan": "plan",
                "end": END,  # short-circuit on OOD refusal
            },
        )
        graph.add_edge("plan", "execute_tools")
        graph.add_edge("execute_tools", "evaluate_evidence")

        # Conditional: sufficient -> synthesize, else -> re-plan
        graph.add_conditional_edges(
            "evaluate_evidence",
            AgentNodes.route_after_evaluation,
            {
                "synthesize": "synthesize",
                "re_plan": "plan",
            },
        )

        graph.add_edge("synthesize", END)

        return graph.compile()


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def build_agent_graph(
    project_root: str | Path,
    chroma_collection_name: str | None = None,
    use_llm_sql_planner: bool = False,
    sql_model_name: str | None = None,
    enable_react_observations: bool = True,
):
    """
    Build and compile the Plan-and-Execute agent graph.

    Returns a compiled LangGraph that accepts AgentState with at minimum
    {"user_query": "..."} and returns the full AgentState with final_answer.

    Args:
        enable_react_observations: If True, adds ReAct-style per-tool
            observation loop. Each tool result is analyzed by the LLM which
            can continue, modify the next step, or abort and trigger replan.
    """
    builder = AgentGraphBuilder(
        project_root=project_root,
        chroma_collection_name=chroma_collection_name,
        use_llm_sql_planner=use_llm_sql_planner,
        sql_model_name=sql_model_name,
        enable_react_observations=enable_react_observations,
    )
    return builder.build()
