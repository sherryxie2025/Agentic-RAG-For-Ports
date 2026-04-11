"""
Build Golden Dataset v2: data-source-first approach.

Strategy:
1. Read actual chunks, SQL data, rules, graph nodes
2. For each source type, craft queries grounded in real data
3. Attach golden metadata per source for precision recall evaluation

Output schema per query:
{
  "id": "VEC_001",
  "query": "...",
  "answer_mode": "lookup|descriptive|comparison|decision_support|diagnostic",
  "expected_sources": ["vector", "sql", ...],
  "needs_vector": true/false,
  "needs_sql": true/false,
  "needs_rules": true/false,
  "needs_graph": true/false,
  "expected_evidence_keywords": [...],
  "reference_answer": "...",
  "difficulty": "easy|medium|hard",
  "golden_vector": {                          # null if not vector query
    "relevant_chunk_ids": ["0_1_0", ...],
    "relevant_source_files": ["file.pdf", ...],
    "relevant_pages": [2, 3, ...]
  },
  "golden_sql": {                             # null if not sql query
    "expected_tables": {"berth_operations": ["berth_productivity_mph", ...]},
    "expected_aggregation": "AVG|SUM|COUNT|MAX|MIN|null"
  },
  "golden_rules": {                           # null if not rules query
    "expected_rule_variables": ["wind_speed_ms", ...],
    "expected_rules": [{"variable": ..., "operator": ..., "value": ...}]
  },
  "golden_graph": {                           # null if not graph query
    "expected_entities": ["weather_conditions", "crane_slowdown", ...],
    "expected_relationships": ["AFFECTS", "CONTRIBUTES_TO", ...]
  }
}
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_chunks():
    path = PROJECT_ROOT / "data" / "chunks" / "chunks_v1.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rules():
    rules = []
    for name in ["grounded_rules.json", "policy_rules.json"]:
        path = PROJECT_ROOT / "data" / "rules" / name
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data:
                r["_source"] = name
            rules.extend(data)
    return rules


def find_chunks(chunks_by_file, source_file, keywords, max_results=10):
    """Find chunks in a specific file that contain keywords."""
    if source_file not in chunks_by_file:
        return []
    results = []
    for c in chunks_by_file[source_file]:
        text_lower = c["text"].lower()
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 1:
            results.append((hits, c))
    results.sort(key=lambda x: -x[0])
    return [c for _, c in results[:max_results]]


def build_golden_dataset():
    chunks = load_chunks()
    rules = load_rules()

    # Index chunks by source file
    chunks_by_file = defaultdict(list)
    for c in chunks:
        chunks_by_file[c["source_file"]].append(c)

    golden = []

    # ================================================================
    # PART 1: PURE VECTOR QUERIES (from real chunks)
    # ================================================================

    # VEC_001: Sustainability from POV report (real chunks exist)
    kws = ["sustainability", "net-zero", "emissions"]
    matched = find_chunks(chunks_by_file, "2023-POV-Sustainability-Report-3.pdf", kws)
    golden.append({
        "id": "VEC_001",
        "query": "What sustainability commitments has the Port of Virginia made regarding emissions reduction?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["sustainability", "net-zero", "emissions", "2040"],
        "reference_answer": "The Port of Virginia committed to Net-Zero by 2040, with initiatives including automated stacking cranes, channel optimization, and emission reduction programs.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["2023-POV-Sustainability-Report-3.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_002: Green Port Policy (POLB)
    kws = ["green port", "environmental", "emissions", "clean"]
    matched = find_chunks(chunks_by_file, "Green-Port-Policy-White-Paper.pdf", kws)
    golden.append({
        "id": "VEC_002",
        "query": "What is the Port of Long Beach's Green Port Policy?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["green port", "environmental", "Long Beach", "clean air"],
        "reference_answer": "The Port of Long Beach's Green Port Policy addresses air quality, water quality, soil and sediment contamination, and community outreach, with a strong commitment to environmental stewardship.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["Green-Port-Policy-White-Paper.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_003: Cybersecurity
    kws = ["cybersecurity", "cyber", "attack", "operational technology"]
    matched = find_chunks(chunks_by_file,
        "MSC-104-7-1-IAPH-Cybersecurity-Guidelines-for-Ports-and-Port-Facilities-IAPH.pdf", kws)
    golden.append({
        "id": "VEC_003",
        "query": "What cybersecurity threats do ports face and what guidelines exist?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["cybersecurity", "cyber-attacks", "operational technology", "IAPH"],
        "reference_answer": "Ports face a fourfold increase in cyber-attacks since 2020, especially targeting operational technology. IAPH guidelines recommend implementing cybersecurity protection, detection, and mitigation measures.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": [
                "MSC-104-7-1-IAPH-Cybersecurity-Guidelines-for-Ports-and-Port-Facilities-IAPH.pdf"
            ],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_004: Risk and Resilience
    kws = ["risk", "resilience", "hazard", "stakeholder"]
    matched = find_chunks(chunks_by_file, "IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf", kws)
    golden.append({
        "id": "VEC_004",
        "query": "What are the IAPH guidelines for port risk management and resilience?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["risk", "resilience", "IAPH", "stakeholder", "hazard"],
        "reference_answer": "IAPH risk and resilience guidelines provide a structured process for defining risk, inventorising hazards, managing stakeholders, and building resilient port operations across economic and operational domains.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_005: Container Performance Index
    kws = ["container", "performance", "index", "ranking", "efficiency"]
    matched = find_chunks(chunks_by_file,
        "The Container Port Performance Index 2020 to 2024 - Trends and Lessons Learned.pdf", kws)
    golden.append({
        "id": "VEC_005",
        "query": "What does the Container Port Performance Index measure and what trends does it show?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["container", "performance", "index", "efficiency", "ranking"],
        "reference_answer": "The Container Port Performance Index ranks ports based on container handling efficiency, measuring throughput speed, vessel turnaround time, and operational productivity.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": [
                "The Container Port Performance Index 2020 to 2024 - Trends and Lessons Learned.pdf"
            ],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_006: Green Infrastructure Design
    kws = ["stormwater", "green infrastructure", "permeable", "rain garden"]
    matched = find_chunks(chunks_by_file, "PANYNJ-Green-Infrastructure-Design-Manual.pdf", kws)
    golden.append({
        "id": "VEC_006",
        "query": "What green infrastructure design practices are recommended for port facilities?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["stormwater", "green infrastructure", "permeable", "design"],
        "reference_answer": "Green infrastructure design for ports includes stormwater management with permeable surfaces, rain gardens, bioswales, and green roofs to manage runoff and improve environmental performance.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["PANYNJ-Green-Infrastructure-Design-Manual.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_007: Decarbonisation blueprint
    kws = ["decarbonisation", "carbon", "fuel", "ammonia", "hydrogen"]
    matched = find_chunks(chunks_by_file, "Maritime-Singapore-Decarbonisation-blueprint.pdf", kws)
    golden.append({
        "id": "VEC_007",
        "query": "What is Singapore's maritime decarbonisation strategy?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["decarbonisation", "Singapore", "maritime", "fuel", "2050"],
        "reference_answer": "Singapore's Maritime Decarbonisation Blueprint outlines a pathway to 2050, including alternative fuels like ammonia and hydrogen, green financing, and training programs.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["Maritime-Singapore-Decarbonisation-blueprint.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_008: GHG emissions inventory
    kws = ["greenhouse", "GHG", "emissions", "inventory", "scope"]
    matched = find_chunks(chunks_by_file, "EY2023_GHG_CAP_Inventory_Report-Final.pdf", kws)
    golden.append({
        "id": "VEC_008",
        "query": "What does the 2023 GHG emissions inventory show for the Port Authority?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["greenhouse", "GHG", "emissions", "inventory", "Port Authority"],
        "reference_answer": "The 2023 GHG inventory reports criteria air pollutant and greenhouse gas emissions for Port Authority facilities, tracking progress against reduction targets.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["EY2023_GHG_CAP_Inventory_Report-Final.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_009: ESPO environmental report
    kws = ["environmental", "port", "european", "priority", "air quality"]
    matched = find_chunks(chunks_by_file, "espo-environmental-report-2023.pdf", kws)
    golden.append({
        "id": "VEC_009",
        "query": "What are the top environmental priorities identified by European ports?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["environmental", "priority", "European", "air quality", "climate"],
        "reference_answer": "The ESPO environmental report identifies air quality, climate change, energy efficiency, and noise as top priorities for European ports.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["espo-environmental-report-2023.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_010: Port of Rotterdam highlights
    kws = ["rotterdam", "throughput", "container", "energy transition"]
    matched = find_chunks(chunks_by_file,
        "Highlights Annual Report 2024 Port of Rotterdam Authority.pdf", kws)
    golden.append({
        "id": "VEC_010",
        "query": "What are the key highlights from the Port of Rotterdam's 2024 annual report?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["Rotterdam", "throughput", "container", "energy"],
        "reference_answer": "The Port of Rotterdam's 2024 report highlights container throughput volumes, energy transition initiatives, and strategic investments.",
        "difficulty": "easy",
        "golden_vector": {
            "relevant_chunk_ids": [c["chunk_id"] for c in matched],
            "relevant_source_files": ["Highlights Annual Report 2024 Port of Rotterdam Authority.pdf"],
            "relevant_pages": list(set(c["page"] for c in matched)),
        },
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # ================================================================
    # PART 2: PURE SQL QUERIES (from actual data)
    # ================================================================

    def sql_golden(tables_cols, agg=None):
        return {"expected_tables": tables_cols, "expected_aggregation": agg}

    # SQL_001-015: grounded in real schema and data
    sql_queries = [
        {
            "id": "SQL_001", "difficulty": "easy",
            "query": "What was the average crane productivity in moves per hour?",
            "answer_mode": "lookup",
            "keywords": ["crane_productivity_mph", "average", "moves per hour"],
            "reference": "The average crane productivity across all operations was approximately 25.27 moves per hour.",
            "tables": {"crane_operations": ["crane_productivity_mph"]}, "agg": "AVG",
        },
        {
            "id": "SQL_002", "difficulty": "easy",
            "query": "What was the average wave height in meters?",
            "answer_mode": "lookup",
            "keywords": ["wave_height_m", "average"],
            "reference": "The average wave height was approximately 0.85 meters.",
            "tables": {"environment": ["wave_height_m"]}, "agg": "AVG",
        },
        {
            "id": "SQL_003", "difficulty": "easy",
            "query": "What was the average gate turn time in minutes?",
            "answer_mode": "lookup",
            "keywords": ["average_turn_time_minutes", "gate"],
            "reference": "The average gate turn time was approximately 39.87 minutes.",
            "tables": {"gate_operations": ["average_turn_time_minutes"]}, "agg": "AVG",
        },
        {
            "id": "SQL_004", "difficulty": "easy",
            "query": "What is the maximum wind speed recorded in the environment data?",
            "answer_mode": "lookup",
            "keywords": ["wind_speed_ms", "maximum", "max"],
            "reference": "The maximum recorded wind speed was 16.57 m/s.",
            "tables": {"environment": ["wind_speed_ms"]}, "agg": "MAX",
        },
        {
            "id": "SQL_005", "difficulty": "easy",
            "query": "What was the average yard dwell time in days?",
            "answer_mode": "lookup",
            "keywords": ["average_dwell_days", "yard"],
            "reference": "The average yard dwell time was approximately 4.74 days.",
            "tables": {"yard_operations": ["average_dwell_days"]}, "agg": "AVG",
        },
        {
            "id": "SQL_006", "difficulty": "medium",
            "query": "What are the top 5 highest arrival delay cases?",
            "answer_mode": "lookup",
            "keywords": ["arrival_delay_hours", "top", "highest"],
            "reference": "The top 5 arrival delay cases show the longest delays by call_id and terminal.",
            "tables": {"berth_operations": ["arrival_delay_hours", "call_id", "terminal_code"]}, "agg": "MAX",
        },
        {
            "id": "SQL_007", "difficulty": "medium",
            "query": "How many total crane operations are recorded in the dataset?",
            "answer_mode": "lookup",
            "keywords": ["count", "crane_operations", "total"],
            "reference": "There are 3,148 crane operation records in the dataset.",
            "tables": {"crane_operations": ["crane_operation_id"]}, "agg": "COUNT",
        },
        {
            "id": "SQL_008", "difficulty": "medium",
            "query": "Compare berth productivity across different terminals.",
            "answer_mode": "comparison",
            "keywords": ["berth_productivity_mph", "terminal_code", "compare"],
            "reference": "Berth productivity varies across terminals (PCT, TTI, SSA_C, LBCT, SSA_A, ITS), with different average moves per hour.",
            "tables": {"berth_operations": ["berth_productivity_mph", "terminal_code"]}, "agg": "AVG",
        },
        {
            "id": "SQL_009", "difficulty": "medium",
            "query": "What is the average crane breakdown time in minutes?",
            "answer_mode": "lookup",
            "keywords": ["breakdown_minutes", "average", "crane"],
            "reference": "The average crane breakdown time was approximately 2.45 minutes per operation.",
            "tables": {"crane_operations": ["breakdown_minutes"]}, "agg": "AVG",
        },
        {
            "id": "SQL_010", "difficulty": "medium",
            "query": "What is the average peak yard occupancy percentage?",
            "answer_mode": "lookup",
            "keywords": ["peak_occupancy_pct", "average", "yard"],
            "reference": "The average peak yard occupancy was approximately 79.77%.",
            "tables": {"yard_operations": ["peak_occupancy_pct"]}, "agg": "AVG",
        },
        {
            "id": "SQL_011", "difficulty": "medium",
            "query": "Which terminal has the highest average berth productivity?",
            "answer_mode": "comparison",
            "keywords": ["berth_productivity_mph", "terminal_code", "highest"],
            "reference": "Terminal-level comparison shows which has the highest average berth productivity in moves per hour.",
            "tables": {"berth_operations": ["berth_productivity_mph", "terminal_code"]}, "agg": "MAX",
        },
        {
            "id": "SQL_012", "difficulty": "easy",
            "query": "How many vessel calls are in the dataset?",
            "answer_mode": "lookup",
            "keywords": ["vessel_calls", "count"],
            "reference": "There are 946 vessel calls recorded in the dataset.",
            "tables": {"vessel_calls": ["visit_id"]}, "agg": "COUNT",
        },
        {
            "id": "SQL_013", "difficulty": "medium",
            "query": "What was the average number of containers handled per berth call?",
            "answer_mode": "lookup",
            "keywords": ["containers_actual", "average", "berth"],
            "reference": "The average containers per berth call is calculated from containers_actual in berth_operations.",
            "tables": {"berth_operations": ["containers_actual"]}, "agg": "AVG",
        },
        {
            "id": "SQL_014", "difficulty": "hard",
            "query": "What is the relationship between arrival delay and berth productivity?",
            "answer_mode": "diagnostic",
            "keywords": ["arrival_delay_hours", "berth_productivity_mph", "correlation"],
            "reference": "Analysis of arrival_delay_hours vs berth_productivity_mph shows the operational relationship between vessel delays and berth efficiency.",
            "tables": {"berth_operations": ["arrival_delay_hours", "berth_productivity_mph"]}, "agg": None,
        },
        {
            "id": "SQL_015", "difficulty": "medium",
            "query": "What are the average daily gate transactions?",
            "answer_mode": "lookup",
            "keywords": ["total_transactions", "average", "gate"],
            "reference": "The average daily gate transactions are approximately 708.",
            "tables": {"gate_operations": ["total_transactions"]}, "agg": "AVG",
        },
    ]

    for sq in sql_queries:
        golden.append({
            "id": sq["id"],
            "query": sq["query"],
            "answer_mode": sq["answer_mode"],
            "expected_sources": ["sql"],
            "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
            "expected_evidence_keywords": sq["keywords"],
            "reference_answer": sq["reference"],
            "difficulty": sq["difficulty"],
            "golden_vector": None,
            "golden_sql": sql_golden(sq["tables"], sq["agg"]),
            "golden_rules": None, "golden_graph": None,
        })

    # ================================================================
    # PART 3: PURE RULE QUERIES (from actual rules)
    # ================================================================

    # Find real rules and build queries from them
    rule_queries = [
        {
            "id": "RULE_001", "difficulty": "easy",
            "query": "What is the maximum wind speed limit for vessel entry at Point Richards?",
            "answer_mode": "lookup",
            "keywords": ["wind", "speed", "knots", "Point Richards", "entry"],
            "reference": "The maximum wind speed limit for a Point Richards entry is 30-35 knots based on steady winds.",
            "variables": ["Wind Speed"],
            "rules": [r for r in rules if "Point Richards entry" in (r.get("rule_text") or "")][:3],
        },
        {
            "id": "RULE_002", "difficulty": "easy",
            "query": "At what wave height should external crane operations be suspended?",
            "answer_mode": "lookup",
            "keywords": ["wave_height_m", "crane", "suspend", "safety"],
            "reference": "External crane operations should be suspended when significant wave height exceeds 2.5 meters.",
            "variables": ["wave_height_m"],
            "rules": [r for r in rules if r.get("variable") == "wave_height_m"][:3],
        },
        {
            "id": "RULE_003", "difficulty": "easy",
            "query": "What pressure threshold triggers storm watch procedures?",
            "answer_mode": "lookup",
            "keywords": ["pressure_hpa", "storm", "watch", "threshold"],
            "reference": "Storm watch procedures should be initiated when atmospheric pressure drops below 980 hPa.",
            "variables": ["pressure_hpa"],
            "rules": [r for r in rules if r.get("variable") == "pressure_hpa"][:3],
        },
        {
            "id": "RULE_004", "difficulty": "medium",
            "query": "What are the tug requirements during high wind conditions?",
            "answer_mode": "descriptive",
            "keywords": ["tug", "wind", "requirement", "risk assessment"],
            "reference": "Tug requirements for higher wind speeds must be subject to a risk assessment by the Vessel's Master/Pilot and the Harbour Master.",
            "variables": ["Wind Speed"],
            "rules": [r for r in rules if "tug" in (r.get("rule_text") or "").lower()][:3],
        },
        {
            "id": "RULE_005", "difficulty": "easy",
            "query": "What wind speed limit applies for Geelong departures?",
            "answer_mode": "lookup",
            "keywords": ["wind", "speed", "Geelong", "departure", "knots"],
            "reference": "The maximum wind speed limit for a Geelong departure is 25-30 knots based on steady winds.",
            "variables": ["Wind Speed"],
            "rules": [r for r in rules if "Geelong departure" in (r.get("rule_text") or "")][:3],
        },
        {
            "id": "RULE_006", "difficulty": "medium",
            "query": "What are the rules regarding VHF communication requirements for vessels?",
            "answer_mode": "descriptive",
            "keywords": ["VHF", "communication", "vessel", "monitoring"],
            "reference": "Vessels must maintain VHF monitoring on designated channels as specified in port operating procedures.",
            "variables": ["VHF monitoring", "VHF Channel"],
            "rules": [r for r in rules if "vhf" in (r.get("variable") or "").lower() or "vhf" in (r.get("rule_text") or "").lower()][:3],
        },
        {
            "id": "RULE_007", "difficulty": "medium",
            "query": "What rules govern hazardous cargo handling in port terminals?",
            "answer_mode": "descriptive",
            "keywords": ["hazardous", "cargo", "handling", "segregation", "safety"],
            "reference": "Hazardous cargo must follow segregation rules, designated handling procedures, and comply with fire safety requirements.",
            "variables": ["hazardous_cargo_handling", "hazardous_segregation"],
            "rules": [r for r in rules if "hazard" in (r.get("variable") or "").lower()][:3],
        },
        {
            "id": "RULE_008", "difficulty": "medium",
            "query": "What are the ISPS compliance requirements for port facilities?",
            "answer_mode": "descriptive",
            "keywords": ["ISPS", "compliance", "security", "facility"],
            "reference": "ISPS compliance requires security screening, access controls, and security plan implementation for all port facilities.",
            "variables": ["isps_compliance", "security_screening"],
            "rules": [r for r in rules if "isps" in (r.get("variable") or "").lower()][:3],
        },
        {
            "id": "RULE_009", "difficulty": "easy",
            "query": "What are the crane inspection requirements?",
            "answer_mode": "lookup",
            "keywords": ["crane", "inspection", "requirement", "safety"],
            "reference": "Cranes must undergo periodic inspections as specified in the equipment safety requirements.",
            "variables": ["crane_inspection"],
            "rules": [r for r in rules if "crane" in (r.get("variable") or "").lower() and "inspect" in (r.get("variable") or "").lower()][:3],
        },
        {
            "id": "RULE_010", "difficulty": "medium",
            "query": "What pilotage requirements apply for vessel entry?",
            "answer_mode": "descriptive",
            "keywords": ["pilotage", "vessel", "entry", "pilot", "mandatory"],
            "reference": "Pilotage is mandatory for vessel entry, with requirements based on vessel size and port conditions.",
            "variables": ["pilotage_mandate"],
            "rules": [r for r in rules if "pilot" in (r.get("variable") or "").lower()][:3],
        },
    ]

    for rq in rule_queries:
        rule_meta = []
        for r in rq["rules"]:
            rule_meta.append({
                "variable": r.get("variable"),
                "operator": r.get("operator"),
                "value": r.get("value"),
                "threshold": r.get("threshold"),
                "rule_text": (r.get("rule_text") or "")[:120],
                "source_file": r.get("source_file"),
                "page": r.get("page"),
            })

        golden.append({
            "id": rq["id"],
            "query": rq["query"],
            "answer_mode": rq["answer_mode"],
            "expected_sources": ["rules"],
            "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
            "expected_evidence_keywords": rq["keywords"],
            "reference_answer": rq["reference"],
            "difficulty": rq["difficulty"],
            "golden_vector": None, "golden_sql": None,
            "golden_rules": {
                "expected_rule_variables": rq["variables"],
                "expected_rules": rule_meta,
            },
            "golden_graph": None,
        })

    # ================================================================
    # PART 4: MULTI-SOURCE QUERIES (SQL + Rules)
    # ================================================================

    multi_sr = [
        {
            "id": "MIX_SR_001", "difficulty": "hard",
            "query": "Based on current wind speed data, should vessel entry be restricted?",
            "answer_mode": "decision_support",
            "keywords": ["wind_speed_ms", "vessel", "entry", "restriction", "knots"],
            "reference": "Decision requires comparing current wind speed data against the 30-35 knot threshold for vessel entry restriction.",
            "sql": {"environment": ["wind_speed_ms", "wind_gust_ms"]}, "agg": "MAX",
            "rule_vars": ["Wind Speed"],
        },
        {
            "id": "MIX_SR_002", "difficulty": "hard",
            "query": "Should crane operations be paused given current wave height conditions?",
            "answer_mode": "decision_support",
            "keywords": ["wave_height_m", "crane", "pause", "threshold", "2.5"],
            "reference": "If current wave height exceeds 2.5 meters, crane operations should be suspended per safety rules.",
            "sql": {"environment": ["wave_height_m"]}, "agg": "MAX",
            "rule_vars": ["wave_height_m"],
        },
        {
            "id": "MIX_SR_003", "difficulty": "hard",
            "query": "Is the atmospheric pressure low enough to trigger storm watch procedures?",
            "answer_mode": "decision_support",
            "keywords": ["pressure_hpa", "storm", "watch", "980"],
            "reference": "Storm watch is triggered when atmospheric pressure falls below 980 hPa.",
            "sql": {"environment": ["pressure_hpa"]}, "agg": "MIN",
            "rule_vars": ["pressure_hpa"],
        },
        {
            "id": "MIX_SR_004", "difficulty": "hard",
            "query": "Are wind gust levels within safe limits for crane operations?",
            "answer_mode": "decision_support",
            "keywords": ["wind_gust_ms", "crane", "safe", "limit"],
            "reference": "Wind gust limits for crane operations must be assessed against operational safety thresholds.",
            "sql": {"environment": ["wind_gust_ms"]}, "agg": "MAX",
            "rule_vars": ["crane_wind_limit"],
        },
        {
            "id": "MIX_SR_005", "difficulty": "medium",
            "query": "What wind conditions have been recorded and what are the port rules for vessel entry?",
            "answer_mode": "descriptive",
            "keywords": ["wind_speed_ms", "wind_gust_ms", "vessel", "entry", "rules"],
            "reference": "The environment data shows wind statistics, while port rules specify entry restrictions at 30-35 knots.",
            "sql": {"environment": ["wind_speed_ms", "wind_gust_ms"]}, "agg": "AVG",
            "rule_vars": ["Wind Speed"],
        },
    ]

    for m in multi_sr:
        golden.append({
            "id": m["id"],
            "query": m["query"],
            "answer_mode": m["answer_mode"],
            "expected_sources": ["sql", "rules"],
            "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
            "expected_evidence_keywords": m["keywords"],
            "reference_answer": m["reference"],
            "difficulty": m["difficulty"],
            "golden_vector": None,
            "golden_sql": sql_golden(m["sql"], m["agg"]),
            "golden_rules": {
                "expected_rule_variables": m["rule_vars"],
                "expected_rules": [r for r in rules if r.get("variable") in m["rule_vars"]][:3],
            },
            "golden_graph": None,
        })

    # ================================================================
    # PART 5: MULTI-SOURCE QUERIES (Vector + SQL)
    # ================================================================

    mix_vs = [
        {
            "id": "MIX_VS_001", "difficulty": "hard",
            "query": "What does port documentation say about emissions reduction and how do environmental metrics support this?",
            "answer_mode": "descriptive",
            "keywords": ["emissions", "GHG", "environmental", "reduction", "wind_speed_ms"],
            "reference": "Port sustainability reports describe GHG reduction targets, while environmental sensor data provides actual metrics.",
            "vec_files": ["2023-POV-Sustainability-Report-3.pdf", "EY2023_GHG_CAP_Inventory_Report-Final.pdf"],
            "vec_kws": ["emissions", "GHG", "reduction"],
            "sql": {"environment": ["wind_speed_ms", "air_temp_c"]}, "agg": None,
        },
        {
            "id": "MIX_VS_002", "difficulty": "medium",
            "query": "How does actual crane productivity compare to performance benchmarks in port reports?",
            "answer_mode": "comparison",
            "keywords": ["crane_productivity_mph", "performance", "benchmark", "moves per hour"],
            "reference": "Actual crane productivity averages 25.27 mph, which can be compared against industry benchmarks in port performance reports.",
            "vec_files": ["The Container Port Performance Index 2020 to 2024 - Trends and Lessons Learned.pdf"],
            "vec_kws": ["crane", "productivity", "performance"],
            "sql": {"crane_operations": ["crane_productivity_mph"]}, "agg": "AVG",
        },
        {
            "id": "MIX_VS_003", "difficulty": "medium",
            "query": "What do strategic plans say about berth efficiency and what does the data show?",
            "answer_mode": "comparison",
            "keywords": ["berth_productivity_mph", "strategic", "efficiency", "improvement"],
            "reference": "Strategic plans outline berth efficiency targets, while berth_operations data shows actual productivity.",
            "vec_files": ["Strategic-Plan-2018-2022.pdf", "2019-Port-of-Long-Beach-Strategic-Plan-042319.pdf"],
            "vec_kws": ["strategic", "berth", "efficiency"],
            "sql": {"berth_operations": ["berth_productivity_mph"]}, "agg": "AVG",
        },
        {
            "id": "MIX_VS_004", "difficulty": "hard",
            "query": "How do yard dwell time metrics align with best management practices described in port literature?",
            "answer_mode": "diagnostic",
            "keywords": ["average_dwell_days", "yard", "management", "best practice"],
            "reference": "Yard dwell averages 4.74 days, which can be assessed against best management practices from port literature.",
            "vec_files": ["manualBestManagementPorts.pdf"],
            "vec_kws": ["yard", "dwell", "management"],
            "sql": {"yard_operations": ["average_dwell_days"]}, "agg": "AVG",
        },
        {
            "id": "MIX_VS_005", "difficulty": "hard",
            "query": "What smart port technologies are described in reports and how do gate transaction volumes reflect their impact?",
            "answer_mode": "diagnostic",
            "keywords": ["smart port", "technology", "total_transactions", "gate", "appointment"],
            "reference": "Smart port reports describe automation and appointment systems; gate transaction data shows actual volumes and appointment adoption rates.",
            "vec_files": ["SmartPortDevelopment_Feb2021.pdf"],
            "vec_kws": ["smart", "technology", "automation"],
            "sql": {"gate_operations": ["total_transactions", "appointment_transactions"]}, "agg": "AVG",
        },
    ]

    for m in mix_vs:
        vec_chunks = []
        for sf in m["vec_files"]:
            vec_chunks.extend(find_chunks(chunks_by_file, sf, m["vec_kws"]))

        golden.append({
            "id": m["id"],
            "query": m["query"],
            "answer_mode": m["answer_mode"],
            "expected_sources": ["vector", "sql"],
            "needs_vector": True, "needs_sql": True, "needs_rules": False, "needs_graph": False,
            "expected_evidence_keywords": m["keywords"],
            "reference_answer": m["reference"],
            "difficulty": m["difficulty"],
            "golden_vector": {
                "relevant_chunk_ids": [c["chunk_id"] for c in vec_chunks[:15]],
                "relevant_source_files": m["vec_files"],
                "relevant_pages": list(set(c["page"] for c in vec_chunks[:15])),
            },
            "golden_sql": sql_golden(m["sql"], m["agg"]),
            "golden_rules": None, "golden_graph": None,
        })

    # ================================================================
    # PART 6: CAUSAL/GRAPH QUERIES (SQL + Graph)
    # ================================================================

    graph_queries = [
        {
            "id": "GRAPH_001", "difficulty": "hard",
            "query": "Why might berth delays be related to weather conditions and crane slowdown?",
            "answer_mode": "diagnostic",
            "keywords": ["berth", "delay", "weather", "crane", "slowdown", "causal"],
            "reference": "Weather conditions affect crane operations through slowdown, which in turn affects berth productivity and arrival delays.",
            "entities": ["weather_conditions", "crane_slowdown", "arrival_delay_hours", "berth_operations", "crane_operations"],
            "rels": ["AFFECTS", "CONTRIBUTES_TO", "CAN_LEAD_TO"],
            "sql": {"berth_operations": ["arrival_delay_hours", "berth_productivity_mph"],
                    "crane_operations": ["crane_productivity_mph", "breakdown_minutes"],
                    "environment": ["wind_speed_ms", "wave_height_m"]}, "agg": None,
        },
        {
            "id": "GRAPH_002", "difficulty": "hard",
            "query": "How does weather impact crane operations and berth productivity?",
            "answer_mode": "diagnostic",
            "keywords": ["weather", "crane", "berth", "productivity", "impact"],
            "reference": "Weather conditions (wind, waves) contribute to crane slowdown and breakdowns, which reduce berth productivity.",
            "entities": ["weather_conditions", "wind_speed_ms", "crane_slowdown", "berth_productivity_mph"],
            "rels": ["AFFECTS", "CONTRIBUTES_TO"],
            "sql": {"environment": ["wind_speed_ms", "wave_height_m"],
                    "crane_operations": ["crane_productivity_mph"]}, "agg": None,
        },
        {
            "id": "GRAPH_003", "difficulty": "hard",
            "query": "What factors contribute to gate congestion at port terminals?",
            "answer_mode": "diagnostic",
            "keywords": ["gate", "congestion", "factors", "terminal", "transaction"],
            "reference": "Gate congestion is influenced by yard overflow, high transaction volumes, and limited appointment availability.",
            "entities": ["gate_congestion", "gate_operations", "yard_overflow", "congestion"],
            "rels": ["CONTRIBUTES_TO", "CAN_LEAD_TO"],
            "sql": {"gate_operations": ["total_transactions", "peak_hour_volume", "appointment_transactions"]}, "agg": None,
        },
        {
            "id": "GRAPH_004", "difficulty": "hard",
            "query": "How can a storm event cascade through port operations?",
            "answer_mode": "diagnostic",
            "keywords": ["storm", "cascade", "operations", "disruption", "delay"],
            "reference": "Storm events trigger operational pauses, cause vessel scheduling disruptions, lead to berth delays and crane slowdowns.",
            "entities": ["storm_event", "weather_conditions", "operational_disruption", "operational_pause", "crane_slowdown", "delay"],
            "rels": ["CAN_TRIGGER", "CAN_LEAD_TO", "DISRUPTS"],
            "sql": {"environment": ["event_storm", "wind_speed_ms"]}, "agg": None,
        },
        {
            "id": "GRAPH_005", "difficulty": "hard",
            "query": "What is the causal chain from high winds to berth delays?",
            "answer_mode": "diagnostic",
            "keywords": ["wind", "berth", "delay", "causal", "chain"],
            "reference": "High winds → crane slowdown → reduced berth productivity → increased arrival delays.",
            "entities": ["wind_speed_ms", "weather_conditions", "crane_slowdown", "berth_operations", "arrival_delay_hours"],
            "rels": ["AFFECTS", "CONTRIBUTES_TO", "CAN_LEAD_TO"],
            "sql": {"environment": ["wind_speed_ms"],
                    "berth_operations": ["arrival_delay_hours"]}, "agg": None,
        },
    ]

    for gq in graph_queries:
        golden.append({
            "id": gq["id"],
            "query": gq["query"],
            "answer_mode": gq["answer_mode"],
            "expected_sources": ["sql", "graph"],
            "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": True,
            "expected_evidence_keywords": gq["keywords"],
            "reference_answer": gq["reference"],
            "difficulty": gq["difficulty"],
            "golden_vector": None,
            "golden_sql": sql_golden(gq["sql"], gq["agg"]),
            "golden_rules": None,
            "golden_graph": {
                "expected_entities": gq["entities"],
                "expected_relationships": gq["rels"],
            },
        })

    # ================================================================
    # PART 7: COMPLEX MULTI-SOURCE (3+ sources)
    # ================================================================

    complex_queries = [
        {
            "id": "COMPLEX_001", "difficulty": "hard",
            "query": "Based on current weather data, port rules, and operational documents, should we restrict vessel entry?",
            "answer_mode": "decision_support",
            "sources": ["vector", "sql", "rules"],
            "keywords": ["wind_speed_ms", "vessel", "entry", "restriction", "knots", "policy"],
            "reference": "Requires cross-referencing wind speed data against entry thresholds from rules and operational handbook.",
            "vec_files": ["2018_VRCA_Port_Operating_Handbook_31_5_18.pdf"],
            "vec_kws": ["vessel", "entry", "wind"],
            "sql": {"environment": ["wind_speed_ms", "wind_gust_ms"]}, "agg": "MAX",
            "rule_vars": ["Wind Speed"],
        },
        {
            "id": "COMPLEX_002", "difficulty": "hard",
            "query": "How do weather-related crane slowdowns affect berth delays, and what do safety rules say about operational limits?",
            "answer_mode": "diagnostic",
            "sources": ["sql", "rules", "graph"],
            "keywords": ["weather", "crane", "slowdown", "berth", "delay", "safety", "threshold"],
            "reference": "Weather causes crane slowdowns (graph), data shows correlation (sql), and safety rules define operational limits (rules).",
            "sql": {"environment": ["wind_speed_ms"], "crane_operations": ["crane_productivity_mph"],
                    "berth_operations": ["arrival_delay_hours"]}, "agg": None,
            "rule_vars": ["crane_wind_limit", "wave_height_m"],
            "entities": ["weather_conditions", "crane_slowdown", "arrival_delay_hours"],
            "rels": ["AFFECTS", "CONTRIBUTES_TO"],
        },
        {
            "id": "COMPLEX_003", "difficulty": "hard",
            "query": "What do sustainability reports say about emissions targets, and how do environmental data and safety rules support green port operations?",
            "answer_mode": "descriptive",
            "sources": ["vector", "sql", "rules"],
            "keywords": ["sustainability", "emissions", "environmental", "safety", "green port"],
            "reference": "Sustainability reports set targets; environmental data provides metrics; safety rules enforce operational standards.",
            "vec_files": ["Green-Port-Policy-White-Paper.pdf", "2023-POV-Sustainability-Report-3.pdf"],
            "vec_kws": ["sustainability", "emissions", "green"],
            "sql": {"environment": ["air_temp_c", "wind_speed_ms"]}, "agg": "AVG",
            "rule_vars": ["pollution_prevention_measures"],
        },
        {
            "id": "COMPLEX_004", "difficulty": "hard",
            "query": "Using weather data, graph reasoning, and yard rules, assess if yard overflow risk is high.",
            "answer_mode": "decision_support",
            "sources": ["sql", "rules", "graph"],
            "keywords": ["yard", "overflow", "weather", "occupancy", "risk"],
            "reference": "Yard overflow risk depends on current occupancy data, weather disruption chains (graph), and operational thresholds (rules).",
            "sql": {"yard_operations": ["peak_occupancy_pct", "teu_received"],
                    "environment": ["event_storm"]}, "agg": None,
            "rule_vars": ["operational_threshold"],
            "entities": ["yard_overflow", "yard_operations", "weather_conditions", "congestion"],
            "rels": ["CAN_LEAD_TO", "CONTRIBUTES_TO"],
        },
        {
            "id": "COMPLEX_005", "difficulty": "hard",
            "query": "Based on all available evidence, what is the full operational impact of severe weather on port throughput?",
            "answer_mode": "diagnostic",
            "sources": ["vector", "sql", "rules", "graph"],
            "keywords": ["weather", "throughput", "impact", "crane", "berth", "delay", "safety"],
            "reference": "Full assessment requires documents (policies), SQL (weather/ops data), rules (thresholds), and graph (causal chains).",
            "vec_files": ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"],
            "vec_kws": ["weather", "risk", "resilience"],
            "sql": {"environment": ["wind_speed_ms", "event_storm"],
                    "crane_operations": ["crane_productivity_mph"],
                    "berth_operations": ["berth_productivity_mph"]}, "agg": None,
            "rule_vars": ["crane_wind_limit", "wave_height_m"],
            "entities": ["weather_conditions", "storm_event", "crane_slowdown", "operational_disruption", "berth_operations"],
            "rels": ["AFFECTS", "CAN_TRIGGER", "CAN_LEAD_TO", "DISRUPTS"],
        },
    ]

    for cq in complex_queries:
        needs_v = "vector" in cq["sources"]
        needs_s = "sql" in cq["sources"]
        needs_r = "rules" in cq["sources"]
        needs_g = "graph" in cq["sources"]

        gv = None
        if needs_v:
            vec_chunks = []
            for sf in cq.get("vec_files", []):
                vec_chunks.extend(find_chunks(chunks_by_file, sf, cq.get("vec_kws", [])))
            gv = {
                "relevant_chunk_ids": [c["chunk_id"] for c in vec_chunks[:15]],
                "relevant_source_files": cq.get("vec_files", []),
                "relevant_pages": list(set(c["page"] for c in vec_chunks[:15])),
            }

        gs = sql_golden(cq["sql"], cq.get("agg")) if needs_s else None

        gr = None
        if needs_r:
            gr = {
                "expected_rule_variables": cq.get("rule_vars", []),
                "expected_rules": [],
            }

        gg = None
        if needs_g:
            gg = {
                "expected_entities": cq.get("entities", []),
                "expected_relationships": cq.get("rels", []),
            }

        golden.append({
            "id": cq["id"],
            "query": cq["query"],
            "answer_mode": cq["answer_mode"],
            "expected_sources": cq["sources"],
            "needs_vector": needs_v, "needs_sql": needs_s, "needs_rules": needs_r, "needs_graph": needs_g,
            "expected_evidence_keywords": cq["keywords"],
            "reference_answer": cq["reference"],
            "difficulty": cq["difficulty"],
            "golden_vector": gv, "golden_sql": gs, "golden_rules": gr, "golden_graph": gg,
        })

    # ================================================================
    # Save
    # ================================================================

    out_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"Golden Dataset v2: {len(golden)} queries")
    print(f"Saved to {out_path}")
    print(f"{'='*80}")

    # Stats
    from collections import Counter
    src_combos = Counter(tuple(sorted(g["expected_sources"])) for g in golden)
    print("\nBy source combination:")
    for combo, cnt in src_combos.most_common():
        print(f"  {combo}: {cnt}")

    diff_dist = Counter(g["difficulty"] for g in golden)
    print("\nBy difficulty:")
    for d, cnt in diff_dist.most_common():
        print(f"  {d}: {cnt}")

    mode_dist = Counter(g["answer_mode"] for g in golden)
    print("\nBy answer_mode:")
    for m, cnt in mode_dist.most_common():
        print(f"  {m}: {cnt}")

    # Per-source annotation coverage
    for src in ["vector", "sql", "rules", "graph"]:
        key = f"golden_{src}"
        annotated = sum(1 for g in golden if g.get(key) is not None)
        print(f"\ngolden_{src}: {annotated} queries annotated")
        if src == "vector":
            total_chunks = sum(len(g[key]["relevant_chunk_ids"]) for g in golden if g.get(key))
            print(f"  Total relevant chunks: {total_chunks}")


if __name__ == "__main__":
    build_golden_dataset()
