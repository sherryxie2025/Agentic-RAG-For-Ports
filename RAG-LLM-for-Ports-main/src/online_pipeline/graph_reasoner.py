# src/online_pipeline/graph_reasoner.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import logging

from .graph_entity_index import GraphEntityIndex
from .llm_client import llm_chat_json
from .neo4j_client import Neo4jClient
from .state_schema import GraphReasoningResult

logger = logging.getLogger("online_pipeline.graph_reasoner")

_ENTITY_EXTRACT_SYSTEM = """You are an entity extractor for a port operations knowledge graph.

The graph contains these node names (pick ONLY from this list):
  Metrics: wind_speed_ms, wind_gust_ms, wave_height_m, tide_level_m, atmospheric_pressure_hpa,
    berth_productivity_mph, arrival_delay_hours, crane_productivity_mph, breakdown_minutes,
    average_dwell_days, average_turn_time_minutes, teu_received, containers_actual,
    vessel_capacity, total_transactions, yard_occupancy_pct
  Operations: vessel_entry, navigation, berth_operations, crane_operations, yard_operations,
    gate_operations, operational_pause, delay, slowdown, vessel_scheduling, container_logistics
  Concepts: weather_conditions, environmental_conditions, wind_restriction, navigation_restriction,
    operational_threshold, crane_slowdown, operational_disruption, congestion, safety, compliance,
    gate_congestion, yard_overflow, storm_event

Given a user query, extract the most relevant entity names FROM THE LIST ABOVE.
Return a JSON list of 2-6 entities that would help answer the query via graph traversal.

Few-shot examples:
Q: "Why might berth delays be related to weather conditions and crane slowdown?"
A: ["arrival_delay_hours", "weather_conditions", "crane_slowdown", "berth_operations"]

Q: "Explain how storm events cascade through port operations"
A: ["storm_event", "weather_conditions", "vessel_scheduling", "operational_pause", "delay"]

Q: "What drives the relationship between vessel capacity and berth delay?"
A: ["vessel_capacity", "berth_operations", "arrival_delay_hours", "crane_operations"]

Q: "How do crane breakdowns affect the entire port logistics chain?"
A: ["breakdown_minutes", "crane_slowdown", "crane_operations", "delay", "yard_operations"]

Q: "Why do arrival delays increase during high wind and high tide?"
A: ["arrival_delay_hours", "wind_speed_ms", "tide_level_m", "weather_conditions", "vessel_entry"]

Return ONLY the JSON list, nothing else."""


class Neo4jGraphReasoner:
    """
    Neo4j-backed graph reasoner with LLM-assisted entity extraction.

    Flow:
    1. LLM extracts query entities (with rule-based fallback)
    2. map entities to graph nodes
    3. expand 1-2 hop neighborhoods
    4. search for short reasoning paths
    """

    def __init__(self) -> None:
        self.client = Neo4jClient()
        # Embedding-based entity index (loaded once, ~0.5s)
        self._entity_index: GraphEntityIndex | None = None

        # Kept as fallback when embedding + LLM both fail
        self.entity_alias_map: Dict[str, str] = {
            "wind": "wind_speed_ms",
            "wind speed": "wind_speed_ms",
            "wind gust": "wind_gust_ms",
            "wave": "wave_height_m",
            "wave height": "wave_height_m",
            "berth delay": "arrival_delay_hours",
            "berth delays": "arrival_delay_hours",
            "delay": "delay",
            "delays": "delay",
            "berth productivity": "berth_productivity_mph",
            "crane productivity": "crane_productivity_mph",
            "crane slowdown": "crane_slowdown",
            "breakdown": "breakdown_minutes",
            "weather": "weather_conditions",
            "weather conditions": "weather_conditions",
            "vessel entry": "vessel_entry",
            "navigation": "navigation",
            "pause": "operational_pause",
            "operations": "berth_operations",
            "congestion": "congestion",
            "disruption": "disruption",
            "tide": "tide_level_m",
            "pressure": "atmospheric_pressure_hpa",
            "dwell": "average_dwell_days",
            "turn time": "average_turn_time_minutes",
            "teu": "teu_received",
            "containers": "containers_actual",
            "vessel capacity": "vessel_capacity",
            "safety": "safety",
            "compliance": "compliance",
            "yard": "yard_operations",
            "gate": "gate_operations",
            "crane": "crane_operations",
            "environmental": "environmental_conditions",
        }

    def close(self) -> None:
        self.client.close()

    def reason(self, query: str) -> GraphReasoningResult:
        # R4 strategy: LLM-first entity extraction with rule-based fallback
        mapped_nodes = self._extract_entities_llm(query)
        extraction_method = "llm"

        # Fallback: rule-based
        if not mapped_nodes:
            logger.info("LLM entity extraction returned empty => falling back to rule-based")
            query_entities = self._extract_entities_rules(query)
            mapped_nodes = self._map_entities_to_nodes(query_entities)
            extraction_method = "rule"

        query_entities = mapped_nodes[:]
        logger.info(
            "GRAPH: entities=%s method=%s",
            query_entities, extraction_method,
        )

        try:
            expanded_nodes = self._expand_neighbors(mapped_nodes)
            reasoning_paths = self._find_reasoning_paths(mapped_nodes)
            logger.info(
                "GRAPH: expanded=%d nodes, paths=%d",
                len(expanded_nodes), len(reasoning_paths),
            )
            for i, p in enumerate(reasoning_paths[:5]):
                logger.debug("  path[%d]: %s", i, p.get("explanation", "N/A"))
            if not reasoning_paths:
                logger.warning("GRAPH: no reasoning paths found between entities %s", query_entities)
            return {
                "query_entities": query_entities,
                "expanded_nodes": expanded_nodes,
                "reasoning_paths": reasoning_paths,
                "execution_ok": True,
                "error": None,
            }
        except Exception as e:
            logger.error("GRAPH: Neo4j query failed: %s", e)
            return {
                "query_entities": query_entities,
                "expanded_nodes": [],
                "reasoning_paths": [],
                "execution_ok": False,
                "error": str(e),
            }

    def _extract_entities_embedding(self, query: str) -> tuple[List[str], List[float]]:
        """Use embedding cosine similarity to find matching graph entities."""
        try:
            matches = self.entity_index.match(query, top_k=4, threshold=0.35)
            nodes = [name for name, _ in matches]
            scores = [score for _, score in matches]
            return nodes, scores
        except Exception:
            return [], []

    def _extract_entities_llm(self, query: str) -> List[str]:
        """Use LLM to extract graph entity names from the query."""
        try:
            result = llm_chat_json(
                messages=[
                    {"role": "system", "content": _ENTITY_EXTRACT_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                timeout=30,
            )
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                entities = list(dict.fromkeys(result))  # unique, preserve order
                logger.debug("LLM entity extraction: %s", entities)
                return entities
        except Exception as e:
            logger.warning("LLM entity extraction failed: %s", e)
        return []

    def _extract_entities_rules(self, query: str) -> List[str]:
        """Rule-based fallback entity extraction."""
        q = query.lower()
        found = []
        for alias in sorted(self.entity_alias_map.keys(), key=len, reverse=True):
            if alias in q:
                found.append(alias)
        seen = set()
        result = []
        for x in found:
            if x not in seen:
                result.append(x)
                seen.add(x)
        return result

    def _map_entities_to_nodes(self, query_entities: List[str]) -> List[str]:
        nodes = []
        for e in query_entities:
            mapped = self.entity_alias_map.get(e)
            if mapped:
                nodes.append(mapped)
        seen = set()
        out = []
        for n in nodes:
            if n not in seen:
                out.append(n)
                seen.add(n)
        return out

    def _expand_neighbors(self, mapped_nodes: List[str]) -> List[str]:
        expanded: List[str] = []

        # Undirected neighbor expansion is fine here — we just need to find
        # related nodes regardless of direction for entity coverage
        cypher = """
        UNWIND $names AS name
        MATCH (n {name: name})-[r]-(m)
        RETURN DISTINCT m.name AS neighbor
        LIMIT 50
        """
        rows = self.client.run_query(cypher, {"names": mapped_nodes})

        for row in rows:
            neighbor = row.get("neighbor")
            if neighbor:
                expanded.append(neighbor)

        return sorted(list(set(mapped_nodes + expanded)))

    def _find_reasoning_paths(self, mapped_nodes: List[str]) -> List[Dict[str, Any]]:
        paths: List[Dict[str, Any]] = []

        if not mapped_nodes:
            return paths

        # Single-anchor mode: many diagnostic queries only extract one
        # entity (e.g. "berth_operations"). Instead of returning no
        # paths, enumerate all 2-hop chains anchored at that node —
        # these represent "what factors influence X" or "what does X
        # affect", which is usually what the user wanted anyway.
        if len(mapped_nodes) == 1:
            anchor = mapped_nodes[0]
            cypher_single = """
            MATCH p = (a {name: $anchor})-[*1..2]-(b)
            WHERE a <> b
            RETURN
              a.name AS start_node,
              b.name AS end_node,
              [n IN nodes(p) | n.name] AS path_nodes,
              [rel IN relationships(p) | type(rel)] AS path_edges,
              [rel IN relationships(p) | startNode(rel).name] AS edge_starts,
              [rel IN relationships(p) | endNode(rel).name] AS edge_ends
            LIMIT 8
            """
            rows = self.client.run_query(cypher_single, {"anchor": anchor})
            for row in rows:
                paths.append({
                    "start_node": row.get("start_node"),
                    "end_node": row.get("end_node"),
                    "path_nodes": row.get("path_nodes", []),
                    "path_edges": row.get("path_edges", []),
                    "explanation": self._explain_path(
                        row.get("path_nodes", []),
                        row.get("path_edges", []),
                        row.get("edge_starts", []),
                        row.get("edge_ends", []),
                    ),
                })
            return paths

        pairs = self._build_pairs(mapped_nodes)

        # Direction-aware query: return startNode/endNode of each relationship
        cypher = """
        MATCH p = shortestPath((a {name: $start})-[*..4]-(b {name: $end}))
        RETURN
          a.name AS start_node,
          b.name AS end_node,
          [n IN nodes(p) | n.name] AS path_nodes,
          [rel IN relationships(p) | type(rel)] AS path_edges,
          [rel IN relationships(p) | startNode(rel).name] AS edge_starts,
          [rel IN relationships(p) | endNode(rel).name] AS edge_ends
        LIMIT 5
        """

        for start, end in pairs:
            rows = self.client.run_query(cypher, {"start": start, "end": end})
            for row in rows:
                path_nodes = row.get("path_nodes", [])
                path_edges = row.get("path_edges", [])
                edge_starts = row.get("edge_starts", [])
                edge_ends = row.get("edge_ends", [])
                paths.append({
                    "start_node": row.get("start_node"),
                    "end_node": row.get("end_node"),
                    "path_nodes": path_nodes,
                    "path_edges": path_edges,
                    "explanation": self._explain_path(
                        path_nodes, path_edges, edge_starts, edge_ends,
                    ),
                })

        return paths

    @staticmethod
    def _build_pairs(nodes: List[str]) -> List[Tuple[str, str]]:
        pairs = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                pairs.append((nodes[i], nodes[j]))
        return pairs[:8]

    @staticmethod
    def _explain_path(
        path_nodes: List[str],
        path_edges: List[str],
        edge_starts: List[str] | None = None,
        edge_ends: List[str] | None = None,
    ) -> str:
        """
        Generate a direction-aware explanation using true Neo4j edge directions.
        Each edge is rendered as "src -[REL]-> tgt" regardless of traversal order.
        """
        if not path_nodes or not path_edges:
            return ""

        # If we have direction info, list each edge with its true direction
        if edge_starts and edge_ends and len(edge_starts) == len(path_edges):
            edge_strs = []
            for rel, src, tgt in zip(path_edges, edge_starts, edge_ends):
                edge_strs.append(f"{src} -[{rel}]-> {tgt}")
            return "; ".join(edge_strs)

        # Fallback: no direction info, use neutral phrasing
        return f"Path: {' -- '.join(path_nodes)} (relations: {', '.join(path_edges)})"


if __name__ == "__main__":
    reasoner = Neo4jGraphReasoner()
    try:
        q = "Why might berth delays be related to weather conditions and crane slowdown?"
        result = reasoner.reason(q)
        print(result)
    finally:
        reasoner.close()