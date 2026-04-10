# src/online_pipeline/graph_entity_index.py
"""
Embedding-based entity matching for graph reasoner.

Pre-computes BGE-small-en embeddings for all Neo4j node names + aliases,
then uses cosine similarity to find top-k matching entities for a query.
Falls back to LLM when similarity is below threshold.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer

# Node names and their human-readable descriptions for embedding
GRAPH_NODES = {
    # Metrics
    "wind_speed_ms": ["wind speed", "wind velocity", "wind meters per second"],
    "wind_gust_ms": ["wind gust", "gust speed", "wind gust meters per second"],
    "wave_height_m": ["wave height", "wave", "sea state", "wave meters"],
    "tide_level_m": ["tide level", "tide", "tidal height", "tide meters"],
    "atmospheric_pressure_hpa": ["atmospheric pressure", "barometric pressure", "pressure hPa"],
    "berth_productivity_mph": ["berth productivity", "berth moves per hour", "berth efficiency"],
    "arrival_delay_hours": ["arrival delay", "berth delay", "vessel delay", "delay hours"],
    "crane_productivity_mph": ["crane productivity", "crane moves per hour", "crane efficiency"],
    "breakdown_minutes": ["crane breakdown", "breakdown minutes", "equipment downtime", "crane downtime"],
    "average_dwell_days": ["dwell time", "dwell days", "container dwell", "yard dwell"],
    "average_turn_time_minutes": ["turn time", "truck turn time", "gate turn time", "gate processing"],
    "teu_received": ["TEU received", "container throughput", "TEU volume"],
    "containers_actual": ["actual containers", "container moves", "container handling"],
    "vessel_capacity": ["vessel capacity", "ship capacity", "vessel size", "DWT", "TEU capacity"],
    "total_transactions": ["gate transactions", "total transactions", "truck movements"],
    "yard_occupancy_pct": ["yard occupancy", "yard utilization", "yard capacity"],
    # Operations
    "vessel_entry": ["vessel entry", "ship entry", "port entry", "vessel arrival"],
    "navigation": ["navigation", "vessel navigation", "channel transit"],
    "berth_operations": ["berth operations", "berthing", "berth activity"],
    "crane_operations": ["crane operations", "crane activity", "crane handling"],
    "yard_operations": ["yard operations", "yard activity", "container yard"],
    "gate_operations": ["gate operations", "gate activity", "truck gate"],
    "operational_pause": ["operational pause", "operations suspended", "work stoppage"],
    "delay": ["delay", "waiting time", "hold up", "schedule delay"],
    "slowdown": ["slowdown", "reduced speed", "deceleration"],
    "vessel_scheduling": ["vessel scheduling", "ship schedule", "berth window", "vessel queue"],
    "container_logistics": ["container logistics", "cargo flow", "container movement"],
    # Concepts
    "weather_conditions": ["weather conditions", "weather", "meteorological conditions"],
    "environmental_conditions": ["environmental conditions", "environment", "natural conditions"],
    "wind_restriction": ["wind restriction", "wind limit", "wind threshold", "wind policy"],
    "navigation_restriction": ["navigation restriction", "channel restriction", "transit restriction"],
    "operational_threshold": ["operational threshold", "safety threshold", "operating limit"],
    "crane_slowdown": ["crane slowdown", "crane speed reduction", "crane derating"],
    "operational_disruption": ["operational disruption", "disruption", "service interruption"],
    "congestion": ["congestion", "port congestion", "terminal congestion"],
    "safety": ["safety", "port safety", "operational safety", "maritime safety"],
    "compliance": ["compliance", "regulatory compliance", "rule compliance"],
    "gate_congestion": ["gate congestion", "gate queue", "gate backup", "truck queue"],
    "yard_overflow": ["yard overflow", "yard full", "yard capacity exceeded"],
    "storm_event": ["storm", "storm event", "severe weather", "storm conditions"],
}


class GraphEntityIndex:
    """Pre-computed embedding index for fast entity matching."""

    def __init__(self, embed_model_name: str = "BAAI/bge-small-en", device: str = "cuda"):
        self.model = SentenceTransformer(embed_model_name, device=device)
        self.node_names: List[str] = []
        self.alias_texts: List[str] = []
        self.alias_to_node: Dict[int, str] = {}
        self.embeddings: np.ndarray | None = None

        self._build_index()

    def _build_index(self) -> None:
        """Build embedding vectors for all node aliases."""
        texts = []
        for node_name, aliases in GRAPH_NODES.items():
            for alias in aliases:
                idx = len(texts)
                texts.append(alias)
                self.alias_to_node[idx] = node_name
                self.alias_texts.append(alias)
            self.node_names.append(node_name)

        self.embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        # Shape: (n_aliases, 384)

    def match(self, query: str, top_k: int = 3, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find top-k matching graph node names for a query.
        Returns list of (node_name, similarity_score) above threshold.
        """
        query_emb = self.model.encode([query], normalize_embeddings=True)
        # Cosine similarity (embeddings are normalized)
        sims = query_emb @ self.embeddings.T  # (1, n_aliases)
        sims = sims[0]

        # For each node, take the best alias score
        node_best: Dict[str, float] = {}
        for idx, score in enumerate(sims):
            node = self.alias_to_node[idx]
            if node not in node_best or score > node_best[node]:
                node_best[node] = float(score)

        # Sort by score, filter by threshold
        ranked = sorted(node_best.items(), key=lambda x: x[1], reverse=True)
        results = [(name, score) for name, score in ranked if score >= threshold]

        return results[:top_k]
