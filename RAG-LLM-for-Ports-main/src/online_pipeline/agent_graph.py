# src/online_pipeline/agent_graph.py
"""
Plan-and-Execute Agent built on LangGraph.

Architecture:
    START -> plan_node -> execute_tools_node -> evaluate_evidence_node
                ^                                     |
                |_____ [insufficient] ________________|
                                                      |
                              [sufficient] -> synthesize_node -> END

Key agentic features:
- LLM-driven planning (tool selection + sub-query generation)
- Parallel tool execution
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
    PLAN_SYSTEM_PROMPT,
    REPLAN_SYSTEM_PROMPT,
    format_tools_for_prompt,
)
from .agent_state import AgentState
from .agent_tools import AgentToolkit, ToolDescriptor
from .answer_synthesizer import AnswerSynthesizer
from .llm_client import llm_chat_json
from .state_schema import PlanStep

logger = logging.getLogger("online_pipeline.agent_graph")

MAX_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Node implementations
# ---------------------------------------------------------------------------

class AgentNodes:
    """Encapsulates all node logic; holds references to toolkit + synthesizer."""

    def __init__(self, toolkit: AgentToolkit) -> None:
        self.toolkit = toolkit
        self.synthesizer = AnswerSynthesizer(use_llm_fallback=True)

    # ---- Node 1: Plan ----

    def plan_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        iteration = state.get("iteration", 0)
        user_query = state.get("user_query", "")

        tools_desc = format_tools_for_prompt(self.toolkit.tools)

        if iteration == 0:
            # First pass: plan from scratch
            prompt = PLAN_SYSTEM_PROMPT.format(tools_description=tools_desc)
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_query},
            ]
        else:
            # Re-plan: incorporate evidence gaps
            evidence_summary = self._summarize_evidence(state)
            evidence_gaps = state.get("evidence_gaps", [])
            prompt = REPLAN_SYSTEM_PROMPT.format(
                user_query=user_query,
                evidence_summary=evidence_summary,
                evidence_gaps="\n".join(f"- {g}" for g in evidence_gaps),
                tools_description=tools_desc,
            )
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
            "reasoning_trace": [
                f"plan_node (iter={iteration + 1}): {len(steps)} new steps — "
                + ", ".join(s.get("tool_name", "?") for s in steps)
            ],
        }

    # ---- Node 2: Execute Tools ----

    def execute_tools_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        plan = state.get("plan", [])
        pending = [s for s in plan if s.get("status") == "pending"]

        tool_results = []
        retrieved_docs = state.get("retrieved_docs", [])
        sql_results = state.get("sql_results", [])
        rule_results = state.get("rule_results", {})
        graph_results = state.get("graph_results", {})
        trace = []

        for step in pending:
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

            result = tool.invoke(**kwargs)
            tool_results.append(result)

            if result.get("success"):
                step["status"] = "completed"
                output = result.get("output", {})

                # Route output to the appropriate state field
                if tool_name == "document_search":
                    retrieved_docs = output.get("documents", [])
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
                    # Update remaining pending steps with rewritten query
                    for s in plan:
                        if s.get("status") == "pending" and s.get("tool_name") != "query_rewrite":
                            s["query"] = s["query"]  # keep original; rewrite is for doc_search
                elif tool_name == "evidence_conflict_check":
                    step["result_summary"] = f"{output.get('conflict_count', 0)} conflicts detected"
                else:
                    step["result_summary"] = "completed"

                trace.append(f"execute_tools: {tool_name} => {step.get('result_summary', 'ok')}")
            else:
                step["status"] = "failed"
                trace.append(f"execute_tools: {tool_name} FAILED: {result.get('error', '?')}")

        elapsed = time.time() - t0
        logger.info("EXECUTE_TOOLS_NODE: %.2fs, %d tools run", elapsed, len(pending))

        return {
            "plan": plan,
            "tool_results": tool_results,
            "retrieved_docs": retrieved_docs,
            "sql_results": sql_results,
            "rule_results": rule_results,
            "graph_results": graph_results,
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

        # If max iterations reached, skip evaluation — go straight to synthesis
        if iteration >= MAX_ITERATIONS:
            logger.info("EVALUATE_NODE: max iterations reached, proceeding to synthesis")
            return {
                "evidence_sufficient": True,
                "evidence_gaps": [],
                "evidence_bundle": evidence_bundle,
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
            "reasoning_trace": [
                f"evaluate_evidence (iter={iteration}): sufficient={sufficient}, "
                f"gaps={gaps}"
            ],
        }

    # ---- Node 4: Synthesize Answer ----

    def synthesize_node(self, state: AgentState) -> Dict[str, Any]:
        t0 = time.time()
        final_answer = self.synthesizer.synthesize(state)
        elapsed = time.time() - t0
        logger.info("SYNTHESIZE_NODE: %.2fs", elapsed)
        return {
            "final_answer": final_answer,
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

    # ---- Helpers ----

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
    ) -> None:
        self.toolkit = AgentToolkit(
            project_root=project_root,
            chroma_collection_name=chroma_collection_name,
            use_llm_sql_planner=use_llm_sql_planner,
            sql_model_name=sql_model_name,
        )
        self.nodes = AgentNodes(toolkit=self.toolkit)

    def build(self):
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("plan", self.nodes.plan_node)
        graph.add_node("execute_tools", self.nodes.execute_tools_node)
        graph.add_node("evaluate_evidence", self.nodes.evaluate_evidence_node)
        graph.add_node("synthesize", self.nodes.synthesize_node)

        # Edges: linear flow with re-plan loop
        graph.add_edge(START, "plan")
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
):
    """
    Build and compile the Plan-and-Execute agent graph.

    Returns a compiled LangGraph that accepts AgentState with at minimum
    {"user_query": "..."} and returns the full AgentState with final_answer.
    """
    builder = AgentGraphBuilder(
        project_root=project_root,
        chroma_collection_name=chroma_collection_name,
        use_llm_sql_planner=use_llm_sql_planner,
        sql_model_name=sql_model_name,
    )
    return builder.build()
