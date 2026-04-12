# src/online_pipeline/langgraph_workflow.py

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langgraph.graph import START, END, StateGraph

from .langgraph_nodes import NodeFactory
from .langgraph_state import LangGraphPortState

logger = logging.getLogger("online_pipeline.workflow")


class LangGraphWorkflowBuilder:
    def __init__(
        self,
        project_root: str | Path,
        chroma_collection_name: str | None = None,
        use_llm_sql_planner: bool = False,
        sql_model_name: str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.factory = NodeFactory(
            project_root=self.project_root,
            chroma_collection_name=chroma_collection_name,
            use_llm_sql_planner=use_llm_sql_planner,
            sql_model_name=sql_model_name,
        )

    def _route_after_planner(self, state: LangGraphPortState) -> List[str]:
        next_nodes: List[str] = []

        if state.get("needs_vector"):
            next_nodes.append("retrieve_documents")

        if state.get("needs_rules"):
            next_nodes.append("retrieve_rules")

        if state.get("needs_sql"):
            next_nodes.append("run_sql")

        if state.get("needs_graph_reasoning"):
            next_nodes.append("run_graph_reasoner")

        if not next_nodes:
            next_nodes.append("merge_evidence")

        logger.info("WORKFLOW branches: %s", next_nodes)
        return next_nodes

    def build(self):
        graph = StateGraph(LangGraphPortState)

        graph.add_node("route_query", self.factory.route_query_node)
        graph.add_node("planner", self.factory.planner_node)  # includes query rewrite
        graph.add_node("retrieve_documents", self.factory.retrieve_documents_node)
        graph.add_node("rerank_documents", self.factory.rerank_documents_node)
        graph.add_node("retrieve_rules", self.factory.retrieve_rules_node)
        graph.add_node("run_sql", self.factory.run_sql_node)
        graph.add_node("run_graph_reasoner", self.factory.run_graph_reasoner_node)
        graph.add_node("merge_evidence", self.factory.merge_evidence_node)
        graph.add_node("synthesize_answer", self.factory.synthesize_answer_node)

        # Flow: route -> plan (includes rewrite) -> [conditional branches] -> merge -> synthesize
        graph.add_edge(START, "route_query")
        graph.add_edge("route_query", "planner")

        graph.add_conditional_edges("planner", self._route_after_planner)

        # Document branch goes through reranker before merge
        graph.add_edge("retrieve_documents", "rerank_documents")
        graph.add_edge("rerank_documents", "merge_evidence")

        graph.add_edge("retrieve_rules", "merge_evidence")
        graph.add_edge("run_sql", "merge_evidence")
        graph.add_edge("run_graph_reasoner", "merge_evidence")

        graph.add_edge("merge_evidence", "synthesize_answer")
        graph.add_edge("synthesize_answer", END)

        return graph.compile()


def build_langgraph_workflow(
    project_root: str | Path,
    chroma_collection_name: str | None = "port_documents_v2",
    use_llm_sql_planner: bool = True,
    sql_model_name: str | None = None,
):
    """
    Build the Agentic RAG DAG workflow.

    Defaults changed in final version:
    - chroma_collection_name defaults to "port_documents_v2" (BGE + Small-to-Big)
    - use_llm_sql_planner defaults to True (LLM NL2SQL with rule fallback)

    Pass None explicitly to use the v1 collection (for backward compat).
    """
    builder = LangGraphWorkflowBuilder(
        project_root=project_root,
        chroma_collection_name=chroma_collection_name,
        use_llm_sql_planner=use_llm_sql_planner,
        sql_model_name=sql_model_name,
    )
    return builder.build()


def build_langgraph_workflow_presynthesis(
    project_root: str | Path,
    chroma_collection_name: str | None = "port_documents_v2",
    use_llm_sql_planner: bool = True,
    sql_model_name: str | None = None,
):
    """
    Build the DAG up to (and including) merge_evidence, but WITHOUT
    synthesize_answer. Used by the streaming endpoint: run this graph
    to get the fully merged state, then call synthesizer.synthesize_stream()
    externally for SSE streaming.

    Returns (compiled_graph, builder) — the caller needs the builder
    to access builder.factory.answer_synthesizer for streaming.
    """
    builder = LangGraphWorkflowBuilder(
        project_root=project_root,
        chroma_collection_name=chroma_collection_name,
        use_llm_sql_planner=use_llm_sql_planner,
        sql_model_name=sql_model_name,
    )

    graph = StateGraph(LangGraphPortState)

    graph.add_node("route_query", builder.factory.route_query_node)
    graph.add_node("planner", builder.factory.planner_node)
    graph.add_node("retrieve_documents", builder.factory.retrieve_documents_node)
    graph.add_node("rerank_documents", builder.factory.rerank_documents_node)
    graph.add_node("retrieve_rules", builder.factory.retrieve_rules_node)
    graph.add_node("run_sql", builder.factory.run_sql_node)
    graph.add_node("run_graph_reasoner", builder.factory.run_graph_reasoner_node)
    graph.add_node("merge_evidence", builder.factory.merge_evidence_node)
    # No synthesize_answer — streaming handles it externally

    graph.add_edge(START, "route_query")
    graph.add_edge("route_query", "planner")
    graph.add_conditional_edges("planner", builder._route_after_planner)
    graph.add_edge("retrieve_documents", "rerank_documents")
    graph.add_edge("rerank_documents", "merge_evidence")
    graph.add_edge("retrieve_rules", "merge_evidence")
    graph.add_edge("run_sql", "merge_evidence")
    graph.add_edge("run_graph_reasoner", "merge_evidence")
    graph.add_edge("merge_evidence", END)

    return graph.compile(), builder