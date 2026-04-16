# src/offline_pipeline/build_neo4j_graph.py


from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load .env from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


class Neo4jGraphBuilder:
    """
    Offline graph builder for the AI Port Decision-Support System.

    Graph schema (initial MVP):
    - Metric nodes
    - Operation nodes
    - Concept nodes
    - Source nodes
    - Relationships connecting operational semantics
    """

    def __init__(self) -> None:
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        self.database = os.getenv("NEO4J_DATABASE", "neo4j")

        if not self.uri or not self.username or not self.password:
            raise ValueError(
                "Missing Neo4j credentials. Please set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD."
            )

        self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))

    def close(self) -> None:
        self.driver.close()

    def build(self, wipe_existing: bool = False) -> None:
        with self.driver.session(database=self.database) as session:
            if wipe_existing:
                session.run("MATCH (n) DETACH DELETE n")

            self._create_constraints(session)
            self._load_nodes(session)
            self._load_relationships(session)

    def _create_constraints(self, session) -> None:
        queries = [
            "CREATE CONSTRAINT metric_name_unique IF NOT EXISTS FOR (n:Metric) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT operation_name_unique IF NOT EXISTS FOR (n:Operation) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (n:Concept) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT source_name_unique IF NOT EXISTS FOR (n:Source) REQUIRE n.name IS UNIQUE",
        ]
        for q in queries:
            session.run(q)

    def _load_nodes(self, session) -> None:
        metric_nodes = [
            {"name": "wind_speed_ms", "description": "Wind speed in meters per second"},
            {"name": "wind_gust_ms", "description": "Wind gust speed in meters per second"},
            {"name": "wave_height_m", "description": "Wave height in meters"},
            {"name": "tide_level_m", "description": "Tide level in meters"},
            {"name": "atmospheric_pressure_hpa", "description": "Atmospheric pressure in hPa"},
            {"name": "berth_productivity_mph", "description": "Berth productivity moves per hour"},
            {"name": "arrival_delay_hours", "description": "Arrival delay in hours"},
            {"name": "crane_productivity_mph", "description": "Crane productivity moves per hour"},
            {"name": "breakdown_minutes", "description": "Crane breakdown minutes"},
            {"name": "average_dwell_days", "description": "Average yard dwell days"},
            {"name": "average_turn_time_minutes", "description": "Average gate turn time"},
            {"name": "teu_received", "description": "TEU containers received"},
            {"name": "containers_actual", "description": "Actual container moves"},
            {"name": "vessel_capacity", "description": "Vessel capacity (TEU/DWT)"},
            {"name": "total_transactions", "description": "Total gate transactions"},
            {"name": "yard_occupancy_pct", "description": "Yard occupancy percentage"},
        ]

        operation_nodes = [
            {"name": "vessel_entry"},
            {"name": "navigation"},
            {"name": "berth_operations"},
            {"name": "crane_operations"},
            {"name": "yard_operations"},
            {"name": "gate_operations"},
            {"name": "operational_pause"},
            {"name": "delay"},
            {"name": "slowdown"},
            {"name": "vessel_scheduling"},
            {"name": "container_logistics"},
        ]

        concept_nodes = [
            {"name": "weather_conditions"},
            {"name": "environmental_conditions"},
            {"name": "wind_restriction"},
            {"name": "navigation_restriction"},
            {"name": "operational_threshold"},
            {"name": "crane_slowdown"},
            {"name": "operational_disruption"},
            {"name": "congestion"},
            {"name": "safety"},
            {"name": "compliance"},
            {"name": "gate_congestion"},
            {"name": "yard_overflow"},
            {"name": "storm_event"},
        ]

        source_nodes = [
            {"name": "environment"},
            {"name": "berth_operations_table"},
            {"name": "crane_operations_table"},
            {"name": "yard_operations_table"},
            {"name": "gate_operations_table"},
            {"name": "rules_store"},
            {"name": "documents_store"},
        ]

        for node in metric_nodes:
            session.run(
                """
                MERGE (n:Metric {name: $name})
                SET n.description = $description
                """,
                node,
            )

        for node in operation_nodes:
            session.run(
                """
                MERGE (n:Operation {name: $name})
                """,
                node,
            )

        for node in concept_nodes:
            session.run(
                """
                MERGE (n:Concept {name: $name})
                """,
                node,
            )

        for node in source_nodes:
            session.run(
                """
                MERGE (n:Source {name: $name})
                """,
                node,
            )

    def _load_relationships(self, session) -> None:
        rel_queries = [
            # metrics -> concepts
            """
            MATCH (a:Metric {name:'wind_speed_ms'}), (b:Concept {name:'wind_restriction'})
            MERGE (a)-[:INDICATES]->(b)
            """,
            """
            MATCH (a:Metric {name:'wind_gust_ms'}), (b:Concept {name:'wind_restriction'})
            MERGE (a)-[:INDICATES]->(b)
            """,
            """
            MATCH (a:Metric {name:'wave_height_m'}), (b:Concept {name:'operational_threshold'})
            MERGE (a)-[:INDICATES]->(b)
            """,
            """
            MATCH (a:Metric {name:'breakdown_minutes'}), (b:Concept {name:'crane_slowdown'})
            MERGE (a)-[:INDICATES]->(b)
            """,

            # concepts -> operations
            """
            MATCH (a:Concept {name:'wind_restriction'}), (b:Operation {name:'vessel_entry'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Concept {name:'wind_restriction'}), (b:Operation {name:'navigation'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Concept {name:'navigation_restriction'}), (b:Operation {name:'navigation'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Concept {name:'operational_threshold'}), (b:Operation {name:'operational_pause'})
            MERGE (a)-[:CAN_TRIGGER]->(b)
            """,
            """
            MATCH (a:Concept {name:'crane_slowdown'}), (b:Operation {name:'crane_operations'})
            MERGE (a)-[:AFFECTS]->(b)
            """,

            # metrics -> operations
            """
            MATCH (a:Metric {name:'crane_productivity_mph'}), (b:Operation {name:'crane_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Metric {name:'berth_productivity_mph'}), (b:Operation {name:'berth_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Metric {name:'arrival_delay_hours'}), (b:Operation {name:'delay'})
            MERGE (a)-[:MEASURES]->(b)
            """,

            # concept chains
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Concept {name:'crane_slowdown'})
            MERGE (a)-[:CONTRIBUTES_TO]->(b)
            """,
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Concept {name:'operational_disruption'})
            MERGE (a)-[:CONTRIBUTES_TO]->(b)
            """,
            """
            MATCH (a:Concept {name:'crane_slowdown'}), (b:Operation {name:'delay'})
            MERGE (a)-[:CAN_LEAD_TO]->(b)
            """,
            """
            MATCH (a:Operation {name:'crane_operations'}), (b:Operation {name:'berth_operations'})
            MERGE (a)-[:INFLUENCES]->(b)
            """,
            """
            MATCH (a:Operation {name:'berth_operations'}), (b:Operation {name:'delay'})
            MERGE (a)-[:INFLUENCES]->(b)
            """,

            # --- NEW causal chains ---

            # storm/weather cascade: storm -> weather -> wind restriction -> vessel entry
            """
            MATCH (a:Concept {name:'storm_event'}), (b:Concept {name:'weather_conditions'})
            MERGE (a)-[:INTENSIFIES]->(b)
            """,
            """
            MATCH (a:Concept {name:'storm_event'}), (b:Operation {name:'operational_pause'})
            MERGE (a)-[:CAN_TRIGGER]->(b)
            """,
            """
            MATCH (a:Concept {name:'storm_event'}), (b:Operation {name:'vessel_scheduling'})
            MERGE (a)-[:DISRUPTS]->(b)
            """,

            # environmental conditions umbrella
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Concept {name:'environmental_conditions'})
            MERGE (a)-[:IS_PART_OF]->(b)
            """,
            """
            MATCH (a:Metric {name:'tide_level_m'}), (b:Concept {name:'environmental_conditions'})
            MERGE (a)-[:INDICATES]->(b)
            """,
            """
            MATCH (a:Metric {name:'atmospheric_pressure_hpa'}), (b:Concept {name:'environmental_conditions'})
            MERGE (a)-[:INDICATES]->(b)
            """,
            """
            MATCH (a:Metric {name:'tide_level_m'}), (b:Operation {name:'vessel_scheduling'})
            MERGE (a)-[:AFFECTS]->(b)
            """,

            # gate congestion chain
            """
            MATCH (a:Metric {name:'total_transactions'}), (b:Operation {name:'gate_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Operation {name:'gate_operations'}), (b:Concept {name:'gate_congestion'})
            MERGE (a)-[:CAN_LEAD_TO]->(b)
            """,
            """
            MATCH (a:Concept {name:'gate_congestion'}), (b:Concept {name:'congestion'})
            MERGE (a)-[:IS_PART_OF]->(b)
            """,
            """
            MATCH (a:Metric {name:'average_turn_time_minutes'}), (b:Operation {name:'gate_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,

            # yard overflow chain
            """
            MATCH (a:Metric {name:'average_dwell_days'}), (b:Operation {name:'yard_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Metric {name:'yard_occupancy_pct'}), (b:Operation {name:'yard_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Operation {name:'yard_operations'}), (b:Concept {name:'yard_overflow'})
            MERGE (a)-[:CAN_LEAD_TO]->(b)
            """,
            """
            MATCH (a:Concept {name:'yard_overflow'}), (b:Concept {name:'congestion'})
            MERGE (a)-[:IS_PART_OF]->(b)
            """,
            """
            MATCH (a:Concept {name:'congestion'}), (b:Operation {name:'delay'})
            MERGE (a)-[:CONTRIBUTES_TO]->(b)
            """,

            # vessel capacity -> berth operations chain
            """
            MATCH (a:Metric {name:'vessel_capacity'}), (b:Operation {name:'berth_operations'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Metric {name:'containers_actual'}), (b:Operation {name:'crane_operations'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Metric {name:'teu_received'}), (b:Operation {name:'container_logistics'})
            MERGE (a)-[:MEASURES]->(b)
            """,
            """
            MATCH (a:Operation {name:'container_logistics'}), (b:Operation {name:'yard_operations'})
            MERGE (a)-[:INFLUENCES]->(b)
            """,

            # weather -> vessel scheduling
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Operation {name:'vessel_scheduling'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Operation {name:'vessel_scheduling'}), (b:Operation {name:'vessel_entry'})
            MERGE (a)-[:INFLUENCES]->(b)
            """,

            # crane breakdown -> berth delay cascade
            """
            MATCH (a:Concept {name:'crane_slowdown'}), (b:Concept {name:'operational_disruption'})
            MERGE (a)-[:CONTRIBUTES_TO]->(b)
            """,
            """
            MATCH (a:Concept {name:'operational_disruption'}), (b:Operation {name:'delay'})
            MERGE (a)-[:CAN_LEAD_TO]->(b)
            """,

            # safety and compliance
            """
            MATCH (a:Concept {name:'wind_restriction'}), (b:Concept {name:'safety'})
            MERGE (a)-[:ENFORCES]->(b)
            """,
            """
            MATCH (a:Concept {name:'navigation_restriction'}), (b:Concept {name:'safety'})
            MERGE (a)-[:ENFORCES]->(b)
            """,
            """
            MATCH (a:Concept {name:'safety'}), (b:Concept {name:'compliance'})
            MERGE (a)-[:REQUIRES]->(b)
            """,

            # weather -> operational metrics
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Metric {name:'arrival_delay_hours'})
            MERGE (a)-[:AFFECTS]->(b)
            """,
            """
            MATCH (a:Concept {name:'weather_conditions'}), (b:Metric {name:'berth_productivity_mph'})
            MERGE (a)-[:AFFECTS]->(b)
            """,

            # source bindings
            """
            MATCH (a:Source {name:'environment'}), (b:Metric {name:'wind_speed_ms'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'environment'}), (b:Metric {name:'wave_height_m'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'environment'}), (b:Metric {name:'tide_level_m'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'environment'}), (b:Metric {name:'atmospheric_pressure_hpa'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'berth_operations_table'}), (b:Metric {name:'berth_productivity_mph'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'berth_operations_table'}), (b:Metric {name:'arrival_delay_hours'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'crane_operations_table'}), (b:Metric {name:'crane_productivity_mph'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'crane_operations_table'}), (b:Metric {name:'breakdown_minutes'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'yard_operations_table'}), (b:Metric {name:'average_dwell_days'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'gate_operations_table'}), (b:Metric {name:'average_turn_time_minutes'})
            MERGE (a)-[:CONTAINS]->(b)
            """,
            """
            MATCH (a:Source {name:'rules_store'}), (b:Concept {name:'wind_restriction'})
            MERGE (a)-[:SUPPORTS]->(b)
            """,
            """
            MATCH (a:Source {name:'documents_store'}), (b:Concept {name:'navigation_restriction'})
            MERGE (a)-[:SUPPORTS]->(b)
            """,
            """
            MATCH (a:Source {name:'rules_store'}), (b:Concept {name:'safety'})
            MERGE (a)-[:SUPPORTS]->(b)
            """,
            """
            MATCH (a:Source {name:'documents_store'}), (b:Concept {name:'compliance'})
            MERGE (a)-[:SUPPORTS]->(b)
            """,
        ]

        for q in rel_queries:
            session.run(q)


if __name__ == "__main__":
    builder = Neo4jGraphBuilder()
    try:
        builder.build(wipe_existing=True)
        print("Neo4j graph build completed.")
    finally:
        builder.close()