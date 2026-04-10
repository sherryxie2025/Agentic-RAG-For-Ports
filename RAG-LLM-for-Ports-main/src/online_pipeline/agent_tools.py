# src/online_pipeline/agent_tools.py
"""
Agent tool definitions for the Plan-and-Execute architecture.

Each tool wraps an existing retrieval/reasoning component and exposes
a uniform callable interface.  Tool metadata (name, description, parameters)
is used by the planner LLM to decide which tools to invoke.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .source_registry import SourceRegistry
from .state_schema import ToolResult

logger = logging.getLogger("online_pipeline.agent_tools")


# ---------------------------------------------------------------------------
# Tool descriptor — lightweight metadata for each tool
# ---------------------------------------------------------------------------

@dataclass
class ToolDescriptor:
    """Metadata + callable for a single agent tool."""
    name: str
    description: str
    parameters: Dict[str, Any]       # JSON-Schema-like parameter spec
    fn: Callable[..., Any] = field(repr=False)

    def to_openai_tool(self) -> Dict[str, Any]:
        """Format as OpenAI function-calling tool spec."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_mcp_tool(self) -> Dict[str, Any]:
        """Format as MCP tool spec."""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.parameters,
        }

    def invoke(self, **kwargs) -> ToolResult:
        """Call the tool and return a ToolResult."""
        t0 = time.time()
        try:
            output = self.fn(**kwargs)
            elapsed = time.time() - t0
            logger.info("TOOL %s: %.2fs success", self.name, elapsed)
            return ToolResult(
                tool_name=self.name,
                input_query=kwargs.get("query", str(kwargs)),
                output=output,
                execution_time_s=round(elapsed, 4),
                success=True,
                error=None,
            )
        except Exception as e:
            elapsed = time.time() - t0
            logger.error("TOOL %s: %.2fs FAILED: %s", self.name, elapsed, e)
            return ToolResult(
                tool_name=self.name,
                input_query=kwargs.get("query", str(kwargs)),
                output=None,
                execution_time_s=round(elapsed, 4),
                success=False,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Tool factory — build all tools from project configuration
# ---------------------------------------------------------------------------

class AgentToolkit:
    """
    Instantiates all retrieval/reasoning components and exposes them
    as a list of ToolDescriptors.
    """

    def __init__(
        self,
        project_root: str | Path,
        chroma_collection_name: str | None = None,
        use_llm_sql_planner: bool = False,
        sql_model_name: str | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.registry = SourceRegistry.from_project_root(self.project_root)
        db_path = self.registry.project_root / "storage" / "sql" / "port_ops.duckdb"

        # Lazy-import heavy components to avoid circular imports
        from .hybrid_retriever import HybridDocumentRetriever
        from .reranker import CrossEncoderReranker
        from .rule_retriever import RuleRetriever
        from .sql_agent_v2 import SQLAgentV2
        from .graph_reasoner import Neo4jGraphReasoner
        from .query_rewriter import QueryRewriter

        self._hybrid_retriever = HybridDocumentRetriever(
            registry=self.registry,
            collection_name=chroma_collection_name,
        )
        self._reranker = CrossEncoderReranker()
        self._rule_retriever = RuleRetriever(registry=self.registry)
        self._sql_agent = SQLAgentV2(
            db_path=db_path,
            use_llm_sql=use_llm_sql_planner,
            model_name=sql_model_name,
        )
        self._graph_reasoner = Neo4jGraphReasoner()
        self._query_rewriter = QueryRewriter()

        self._tools: List[ToolDescriptor] = self._build_tools()

    @property
    def tools(self) -> List[ToolDescriptor]:
        return self._tools

    @property
    def tool_map(self) -> Dict[str, ToolDescriptor]:
        return {t.name: t for t in self._tools}

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """All tools in OpenAI function-calling format."""
        return [t.to_openai_tool() for t in self._tools]

    def get_mcp_tools(self) -> List[Dict[str, Any]]:
        """All tools in MCP tool format."""
        return [t.to_mcp_tool() for t in self._tools]

    def close(self) -> None:
        """Cleanup resources (Neo4j connections, etc.)."""
        try:
            self._graph_reasoner.close()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Tool implementations
    # -----------------------------------------------------------------------

    def _document_search(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Hybrid BM25+dense retrieval with cross-encoder reranking."""
        docs = self._hybrid_retriever.retrieve(query=query, top_k=20)
        reranked = self._reranker.rerank(query, docs, top_k=top_k)
        return {
            "documents": reranked,
            "pre_rerank_documents": docs,   # for rerank lift evaluation
            "count": len(reranked),
            "source": "hybrid_retrieval+reranker",
        }

    def _sql_query(self, query: str) -> Dict[str, Any]:
        """Natural-language to SQL on port operational data."""
        result = self._sql_agent.run(query)
        return result

    def _rule_lookup(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search grounded rules and policy thresholds."""
        matched = self._rule_retriever.retrieve(query=query, top_k=top_k, min_score=0.5)
        return {
            "matched_rules": matched,
            "applicable_rule_count": len(matched),
            "execution_ok": True,
            "error": None,
        }

    def _graph_reason(self, query: str) -> Dict[str, Any]:
        """Multi-hop causal reasoning on Neo4j knowledge graph."""
        result = self._graph_reasoner.reason(query)
        return result

    def _query_rewrite(self, query: str) -> Dict[str, Any]:
        """Expand abbreviations and rewrite query for better retrieval."""
        return self._query_rewriter.rewrite(query)

    def _evidence_conflict_check(
        self,
        rule_results: Dict[str, Any],
        sql_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compare rule thresholds against actual SQL data to detect conflicts."""
        from .langgraph_nodes import NodeFactory
        conflicts = NodeFactory._detect_evidence_conflicts(rule_results, sql_results)
        return {"conflicts": conflicts, "conflict_count": len(conflicts)}

    # -----------------------------------------------------------------------
    # Build tool descriptors
    # -----------------------------------------------------------------------

    def _build_tools(self) -> List[ToolDescriptor]:
        return [
            ToolDescriptor(
                name="document_search",
                description=(
                    "Search port documents, reports, handbooks, and policies "
                    "using hybrid BM25+dense retrieval with cross-encoder reranking. "
                    "Use for questions about procedures, guidelines, or factual port information."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language search query about port topics",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top results to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                fn=self._document_search,
            ),
            ToolDescriptor(
                name="sql_query",
                description=(
                    "Query structured port operational data (berth ops, crane ops, "
                    "yard ops, gate ops, vessel calls, environment). "
                    "Use for questions about metrics, statistics, or operational numbers."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language query about port operational data",
                        },
                    },
                    "required": ["query"],
                },
                fn=self._sql_query,
            ),
            ToolDescriptor(
                name="rule_lookup",
                description=(
                    "Look up port operational rules, safety thresholds, and policy constraints. "
                    "Use for questions about limits, regulations, or compliance requirements."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query about rules, thresholds, or policies",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of rules to return",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
                fn=self._rule_lookup,
            ),
            ToolDescriptor(
                name="graph_reason",
                description=(
                    "Perform multi-hop causal reasoning on the port knowledge graph (Neo4j). "
                    "Use for 'why' questions, cause-effect analysis, or tracing relationships "
                    "between port entities (berths, cranes, weather, delays)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Causal or relationship query",
                        },
                    },
                    "required": ["query"],
                },
                fn=self._graph_reason,
            ),
            ToolDescriptor(
                name="query_rewrite",
                description=(
                    "Expand port/maritime abbreviations (TEU, LOA, ISPS, etc.) and "
                    "rewrite the query for better retrieval. Use before document_search "
                    "if the query contains domain-specific abbreviations."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Query that may contain abbreviations",
                        },
                    },
                    "required": ["query"],
                },
                fn=self._query_rewrite,
            ),
            ToolDescriptor(
                name="evidence_conflict_check",
                description=(
                    "Compare rule thresholds against actual SQL data values to detect "
                    "conflicts (e.g., wind speed exceeds safety limit). "
                    "Use after obtaining both rule_lookup and sql_query results."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "rule_results": {
                            "type": "object",
                            "description": "Results from rule_lookup tool",
                        },
                        "sql_results": {
                            "type": "array",
                            "description": "Results from sql_query tool",
                        },
                    },
                    "required": ["rule_results", "sql_results"],
                },
                fn=self._evidence_conflict_check,
            ),
        ]
