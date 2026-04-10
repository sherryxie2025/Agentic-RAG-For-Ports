# src/online_pipeline/mcp_server.py
"""
MCP (Model Context Protocol) Server for the Port Decision-Support Agent.

Exposes the agent's retrieval and reasoning tools over the MCP standard,
allowing external LLM clients (e.g. Claude Desktop, other MCP-compatible
agents) to use port decision-support capabilities as tools.

Run:
    python -m src.online_pipeline.mcp_server

Supports stdio transport (default) for local use with Claude Desktop,
or can be adapted for SSE transport for remote use.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("online_pipeline.mcp_server")

# ---------------------------------------------------------------------------
# MCP Server implementation
# ---------------------------------------------------------------------------

try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import TextContent, Tool

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Run: pip install mcp")


def _build_mcp_tools() -> List[Dict[str, Any]]:
    """
    Define MCP tool specs without instantiating heavy components.
    Actual component instantiation happens on first tool call.
    """
    return [
        {
            "name": "document_search",
            "description": (
                "Search port documents, reports, handbooks, and policies "
                "using hybrid BM25+dense retrieval with cross-encoder reranking. "
                "Returns the top-k most relevant document chunks."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query about port topics",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "sql_query",
            "description": (
                "Query structured port operational data including berth operations, "
                "crane operations, yard operations, gate operations, vessel calls, "
                "and environmental data. Translates natural language to SQL."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query about port operational data",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "rule_lookup",
            "description": (
                "Look up port operational rules, safety thresholds, and policy "
                "constraints. Returns matching rules with conditions, thresholds, "
                "and compliance requirements."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query about rules, thresholds, or policies",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of rules to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "graph_reason",
            "description": (
                "Perform multi-hop causal reasoning on the port knowledge graph "
                "(Neo4j). Use for 'why' questions, cause-effect analysis, or "
                "tracing relationships between port entities."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Causal or relationship query",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "query_rewrite",
            "description": (
                "Expand port/maritime abbreviations (TEU, LOA, ISPS, etc.) and "
                "rewrite the query for better retrieval accuracy."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Query that may contain abbreviations",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "ask_agent",
            "description": (
                "Send a question to the full Plan-and-Execute agent pipeline. "
                "The agent will automatically plan which tools to use, execute "
                "them, evaluate evidence, and synthesize a comprehensive answer. "
                "Use this for complex questions that may require multiple tools."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user's question about port operations",
                    },
                },
                "required": ["query"],
            },
        },
    ]


class PortMCPServer:
    """
    MCP server that lazily initializes the AgentToolkit on first tool call.
    """

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root)
        self._toolkit = None
        self._agent = None

    @property
    def toolkit(self):
        if self._toolkit is None:
            from .agent_tools import AgentToolkit
            self._toolkit = AgentToolkit(
                project_root=self.project_root,
                use_llm_sql_planner=True,
            )
        return self._toolkit

    @property
    def agent(self):
        if self._agent is None:
            from .agent_graph import build_agent_graph
            self._agent = build_agent_graph(
                project_root=self.project_root,
                use_llm_sql_planner=True,
            )
        return self._agent

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Dispatch a tool call and return JSON result."""
        if name == "ask_agent":
            # Run the full agent pipeline
            state = self.agent.invoke({
                "user_query": arguments.get("query", ""),
                "reasoning_trace": [],
                "warnings": [],
                "tool_results": [],
            })
            final = state.get("final_answer", {})
            return json.dumps({
                "answer": final.get("answer", ""),
                "confidence": final.get("confidence"),
                "sources_used": final.get("sources_used", []),
                "iterations": state.get("iteration", 0),
                "reasoning_trace": state.get("reasoning_trace", []),
            }, default=str, ensure_ascii=False)
        else:
            # Call individual tool
            tool = self.toolkit.tool_map.get(name)
            if not tool:
                return json.dumps({"error": f"Unknown tool: {name}"})
            result = tool.invoke(**arguments)
            output = result.get("output", {})
            return json.dumps(output, default=str, ensure_ascii=False)


def create_mcp_app(project_root: str | Path):
    """Create and configure the MCP Server application."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP SDK is required. Install with: pip install mcp"
        )

    app = Server("port-decision-support")
    port_server = PortMCPServer(project_root)
    tool_specs = _build_mcp_tools()

    @app.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=spec["name"],
                description=spec["description"],
                inputSchema=spec["inputSchema"],
            )
            for spec in tool_specs
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        # Run synchronous tool in executor to not block the event loop
        loop = asyncio.get_event_loop()
        result_text = await loop.run_in_executor(
            None, port_server.call_tool, name, arguments
        )
        return [TextContent(type="text", text=result_text)]

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _run_stdio(project_root: Path) -> None:
    """Run MCP server with stdio transport."""
    app = create_mcp_app(project_root)
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    """Entry point: python -m src.online_pipeline.mcp_server"""
    logging.basicConfig(level=logging.INFO)
    project_root = Path(__file__).resolve().parents[2]

    if not MCP_AVAILABLE:
        print("ERROR: MCP SDK not installed. Run: pip install mcp")
        print("Then restart: python -m src.online_pipeline.mcp_server")
        return

    print(f"Starting Port Decision-Support MCP Server...")
    print(f"Project root: {project_root}")
    print(f"Transport: stdio")
    print(f"Tools: {len(_build_mcp_tools())}")
    asyncio.run(_run_stdio(project_root))


if __name__ == "__main__":
    main()
