"""
Expand golden dataset v2 from 55→~95 queries.
Queries crafted by reading actual data source content via LLM.

Fills coverage gaps:
- More vector docs (Antwerp, MPA, Port Houston, Rotterdam facts, KPIs)
- vessel_calls table, multi-table JOINs, time-based SQL
- fog/departure/vessel-size/crane-productivity/yard-overflow rules
- yard→gate cascade, vessel scheduling graph
- Vec+Rules, Graph+Vec new combos
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def find_chunks(chunks_by_file, source_file, keywords, max_results=10):
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


def vec_meta(chunks, files):
    return {
        "relevant_chunk_ids": [c["chunk_id"] for c in chunks],
        "relevant_source_files": files,
        "relevant_pages": list(set(c["page"] for c in chunks)),
    }


def sql_meta(tables, agg=None):
    return {"expected_tables": tables, "expected_aggregation": agg}


def rule_meta(variables, rules_list=None):
    return {
        "expected_rule_variables": variables,
        "expected_rules": rules_list or [],
    }


def graph_meta(entities, rels):
    return {"expected_entities": entities, "expected_relationships": rels}


def main():
    # Load existing golden
    golden_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)
    print(f"Existing: {len(golden)} queries")

    # Load chunks
    with open(PROJECT_ROOT / "data" / "chunks" / "chunks_v1.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    by_src = defaultdict(list)
    for c in chunks:
        by_src[c["source_file"]].append(c)

    # Load rules
    with open(PROJECT_ROOT / "data" / "rules" / "grounded_rules.json", "r", encoding="utf-8") as f:
        grounded = json.load(f)

    new_queries = []

    # ================================================================
    # A. NEW VECTOR QUERIES — from docs not yet covered
    # ================================================================

    # VEC_011: Port of Antwerp-Bruges hydrogen strategy
    matched = find_chunks(by_src, "Port-of-Antwerp-Bruges-2023-ENG-1.pdf",
                          ["hydrogen", "green", "energy", "import"])
    new_queries.append({
        "id": "VEC_011", "difficulty": "easy",
        "query": "What is the Port of Antwerp-Bruges' strategy for green hydrogen?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["hydrogen", "green", "Antwerp", "import hub", "energy"],
        "reference_answer": "The Port of Antwerp-Bruges aims to become Europe's leading import hub for green hydrogen, with its port platform serving as an important link in the fuel value chain.",
        "golden_vector": vec_meta(matched, ["Port-of-Antwerp-Bruges-2023-ENG-1.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_012: MPA Singapore sustainability — ammonia marine fuel
    matched = find_chunks(by_src, "mpa-sustainability-report-2023.pdf",
                          ["ammonia", "marine fuel", "harbour", "net-zero"])
    new_queries.append({
        "id": "VEC_012", "difficulty": "medium",
        "query": "What progress has Singapore's MPA made on ammonia as a marine fuel?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["ammonia", "marine fuel", "Singapore", "MPA", "trial"],
        "reference_answer": "MPA completed a successful trial of the world's first use of ammonia combined with diesel as a marine fuel on the Fortescue Green Pioneer at the Port of Singapore.",
        "golden_vector": vec_meta(matched, ["mpa-sustainability-report-2023.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_013: Port Houston carbon neutrality and crane expansion
    matched = find_chunks(by_src, "2023-Annual-Report_Final_Web.pdf",
                          ["carbon", "neutrality", "crane", "TEU", "Bayport"])
    new_queries.append({
        "id": "VEC_013", "difficulty": "easy",
        "query": "What are Port Houston's carbon neutrality goals and terminal expansion plans?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["carbon neutrality", "2050", "Bayport", "crane", "TEU"],
        "reference_answer": "Port Houston is committed to reaching carbon neutrality by 2050, with Bayport Container Terminal equipped with cranes sized for 15,000 TEU ships.",
        "golden_vector": vec_meta(matched, ["2023-Annual-Report_Final_Web.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_014: Rotterdam shore-based power and digitisation
    matched = find_chunks(by_src, "facts-and-figures-port-of-rotterdam.pdf",
                          ["shore-based power", "digitisation", "smart", "innovation"])
    new_queries.append({
        "id": "VEC_014", "difficulty": "medium",
        "query": "What smart port and shore-based power initiatives is the Port of Rotterdam implementing?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["shore-based power", "Rotterdam", "digitisation", "smart", "innovation"],
        "reference_answer": "Rotterdam is rolling out shore-based power for sea-going vessels by 2030, with 100+ innovative pilots in digitisation, sustainability, and smart port operations.",
        "golden_vector": vec_meta(matched, ["facts-and-figures-port-of-rotterdam.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_015: Strategic plan KPIs — truck turn time targets
    matched = find_chunks(by_src, "strategic_plan_kpis.pdf",
                          ["truck", "turn time", "KPI", "goal", "minute"])
    new_queries.append({
        "id": "VEC_015", "difficulty": "medium",
        "query": "What are the truck turn time KPI targets in the port strategic plan?",
        "answer_mode": "lookup",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["truck", "turn time", "KPI", "goal", "minutes"],
        "reference_answer": "The strategic plan sets a day goal of below 55 minutes for average truck turn times.",
        "golden_vector": vec_meta(matched, ["strategic_plan_kpis.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_016: Port Houston federal funding and sustainability partnerships
    matched = find_chunks(by_src, "Corporate-Sustainability-Update_2024_Final.pdf",
                          ["federal funding", "PIDP", "partnership", "sustainability"])
    new_queries.append({
        "id": "VEC_016", "difficulty": "easy",
        "query": "What federal funding has Port Houston received for sustainability projects?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["federal funding", "PIDP", "Port Houston", "sustainability"],
        "reference_answer": "Port Houston received over $25 million in PIDP federal funding and nearly $3 million in CPP funding for sustainability and infrastructure projects.",
        "golden_vector": vec_meta(matched, ["Corporate-Sustainability-Update_2024_Final.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_017: Sustainable infrastructure guidelines — Envision framework
    matched = find_chunks(by_src, "Sustainable-infrastructure-guidelines.pdf",
                          ["Envision", "sustainable", "infrastructure", "track"])
    new_queries.append({
        "id": "VEC_017", "difficulty": "medium",
        "query": "What sustainability framework does the Port Authority use for infrastructure projects?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["Envision", "sustainable", "infrastructure", "guidelines", "track"],
        "reference_answer": "The Port Authority uses the Envision framework for sustainable infrastructure, with Track 1 for standard projects and Track 2 for large, high-profile projects.",
        "golden_vector": vec_meta(matched, ["Sustainable-infrastructure-guidelines.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # VEC_018: Seattle 2024 environment — PFAS cleanup
    matched = find_chunks(by_src, "2024 Environment and Sustainability Annual Report.pdf",
                          ["PFAS", "contamination", "cleanup", "environmental"])
    new_queries.append({
        "id": "VEC_018", "difficulty": "medium",
        "query": "What environmental contamination cleanup efforts are described in the 2024 sustainability report?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector"],
        "needs_vector": True, "needs_sql": False, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["PFAS", "contamination", "cleanup", "environmental", "sustainability"],
        "reference_answer": "The 2024 report describes PFAS cleanup from fire trucks, PCB contamination remediation at Terminal 25, and comprehensive environmental sustainability efforts.",
        "golden_vector": vec_meta(matched, ["2024 Environment and Sustainability Annual Report.pdf"]),
        "golden_sql": None, "golden_rules": None, "golden_graph": None,
    })

    # ================================================================
    # B. NEW SQL QUERIES — vessel_calls, multi-table, time-based
    # ================================================================

    new_queries.append({
        "id": "SQL_016", "difficulty": "easy",
        "query": "How many ULCV (ultra-large container vessels) are in the dataset?",
        "answer_mode": "lookup",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["ULCV", "SizeCategory", "count", "vessel"],
        "reference_answer": "There are 18 ULCV class vessels in the dataset.",
        "golden_vector": None,
        "golden_sql": sql_meta({"vessel_calls": ["SizeCategory", "VesselCapacityTEU"]}, "COUNT"),
        "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "SQL_017", "difficulty": "medium",
        "query": "What is the average vessel capacity in TEU by size category?",
        "answer_mode": "comparison",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["vessel_capacity_teu", "SizeCategory", "average", "Panamax", "Neo-Panamax"],
        "reference_answer": "Average TEU capacity varies: Small ~3,433, Panamax ~6,727, Post-Panamax ~10,703, Neo-Panamax ~15,449, ULCV ~21,100.",
        "golden_vector": None,
        "golden_sql": sql_meta({"vessel_calls": ["SizeCategory", "VesselCapacityTEU"]}, "AVG"),
        "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "SQL_018", "difficulty": "hard",
        "query": "Which terminal has the highest average berth productivity and the lowest average arrival delay?",
        "answer_mode": "comparison",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["terminal_code", "berth_productivity_mph", "arrival_delay_hours", "highest", "lowest"],
        "reference_answer": "LBCT has the highest average berth productivity (~27.45 mph). Delay patterns vary by terminal.",
        "golden_vector": None,
        "golden_sql": sql_meta({"berth_operations": ["terminal_code", "berth_productivity_mph", "arrival_delay_hours"]}, "AVG"),
        "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "SQL_019", "difficulty": "medium",
        "query": "In which month is the average wind speed the highest?",
        "answer_mode": "lookup",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["wind_speed_ms", "month", "highest", "average"],
        "reference_answer": "Wind speed varies by month, with winter months (January) showing the highest average wind speeds around 4.89 m/s.",
        "golden_vector": None,
        "golden_sql": sql_meta({"environment": ["wind_speed_ms", "timestamp"]}, "AVG"),
        "golden_rules": None, "golden_graph": None,
    })

    new_queries.append({
        "id": "SQL_020", "difficulty": "hard",
        "query": "What is the average crane productivity and breakdown minutes for each terminal?",
        "answer_mode": "comparison",
        "expected_sources": ["sql"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": False,
        "expected_evidence_keywords": ["crane_productivity_mph", "breakdown_minutes", "terminal_code", "average"],
        "reference_answer": "Crane productivity and breakdown minutes vary by terminal, showing the maintenance-performance tradeoff across different operators.",
        "golden_vector": None,
        "golden_sql": sql_meta({"crane_operations": ["terminal_code", "crane_productivity_mph", "breakdown_minutes"]}, "AVG"),
        "golden_rules": None, "golden_graph": None,
    })

    # ================================================================
    # C. NEW RULE QUERIES — unused variables
    # ================================================================

    new_queries.append({
        "id": "RULE_011", "difficulty": "easy",
        "query": "At what crane productivity level should operations be paused for review?",
        "answer_mode": "lookup",
        "expected_sources": ["rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["crane_productivity_mph", "15", "paused", "review"],
        "reference_answer": "Crane operations must be paused when crane productivity drops below 15 moves per hour.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["crane_productivity_mph"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "RULE_012", "difficulty": "easy",
        "query": "When does a crane breakdown require immediate maintenance dispatch?",
        "answer_mode": "lookup",
        "expected_sources": ["rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["breakdown_minutes", "60", "maintenance", "dispatch"],
        "reference_answer": "Crane breakdown exceeding 60 minutes requires immediate maintenance dispatch and reassignment.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["breakdown_minutes"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "RULE_013", "difficulty": "easy",
        "query": "What yard occupancy percentage triggers the overflow protocol?",
        "answer_mode": "lookup",
        "expected_sources": ["rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["peak_occupancy_pct", "85", "overflow", "protocol"],
        "reference_answer": "Yard peak occupancy exceeding 85% triggers overflow protocol and diverts containers.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["peak_occupancy_pct"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "RULE_014", "difficulty": "easy",
        "query": "What is the maximum vessel LOA that triggers safety protocol?",
        "answer_mode": "lookup",
        "expected_sources": ["rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["vessel_loa_meters", "366", "400", "safety protocol", "ultra-large"],
        "reference_answer": "Vessels with LOA >= 366 meters trigger safety protocol for ULCVs, and those exceeding 400 meters require manual review.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["vessel_loa_meters"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "RULE_015", "difficulty": "easy",
        "query": "What arrival delay threshold triggers berth schedule reassignment?",
        "answer_mode": "lookup",
        "expected_sources": ["rules"],
        "needs_vector": False, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["arrival_delay_hours", "6", "berth", "reassignment"],
        "reference_answer": "Vessel arrival delay exceeding 6 hours requires berth schedule reassignment.",
        "golden_vector": None, "golden_sql": None,
        "golden_rules": rule_meta(["arrival_delay_hours"]),
        "golden_graph": None,
    })

    # ================================================================
    # D. NEW SQL+RULES — using numeric threshold rules
    # ================================================================

    new_queries.append({
        "id": "MIX_SR_006", "difficulty": "hard",
        "query": "Is current crane productivity below the minimum threshold requiring operational pause?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["crane_productivity_mph", "15", "threshold", "pause"],
        "reference_answer": "Compare actual crane_productivity_mph against the 15 mph threshold. If below, pause is required.",
        "golden_vector": None,
        "golden_sql": sql_meta({"crane_operations": ["crane_productivity_mph"]}, "AVG"),
        "golden_rules": rule_meta(["crane_productivity_mph"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "MIX_SR_007", "difficulty": "hard",
        "query": "Does the current yard occupancy exceed the overflow protocol threshold?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["peak_occupancy_pct", "85", "overflow", "yard"],
        "reference_answer": "Compare actual peak_occupancy_pct against the 85% threshold. Average is ~79.8%, close but below.",
        "golden_vector": None,
        "golden_sql": sql_meta({"yard_operations": ["peak_occupancy_pct"]}, "MAX"),
        "golden_rules": rule_meta(["peak_occupancy_pct"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "MIX_SR_008", "difficulty": "hard",
        "query": "Are there vessels in the dataset that exceed the maximum terminal LOA limit?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["vessel_loa_meters", "400", "vessel_calls", "exceed", "limit"],
        "reference_answer": "Check vessel_calls for vessels with LOA > 400 meters against the safety protocol threshold.",
        "golden_vector": None,
        "golden_sql": sql_meta({"vessel_calls": ["LOA_meters", "VesselName"]}, "MAX"),
        "golden_rules": rule_meta(["vessel_loa_meters"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "MIX_SR_009", "difficulty": "hard",
        "query": "Are there arrival delays exceeding the 6-hour berth reassignment threshold?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["arrival_delay_hours", "6", "berth", "reassignment", "threshold"],
        "reference_answer": "Check berth_operations for cases where arrival_delay_hours > 6.",
        "golden_vector": None,
        "golden_sql": sql_meta({"berth_operations": ["arrival_delay_hours", "call_id"]}, "MAX"),
        "golden_rules": rule_meta(["arrival_delay_hours"]),
        "golden_graph": None,
    })

    new_queries.append({
        "id": "MIX_SR_010", "difficulty": "hard",
        "query": "Does the wind gust data exceed the crane boom-down threshold of 25 m/s?",
        "answer_mode": "decision_support",
        "expected_sources": ["sql", "rules"],
        "needs_vector": False, "needs_sql": True, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["wind_gust_ms", "25", "crane", "boom-down"],
        "reference_answer": "Wind gust exceeding 25 m/s triggers immediate crane boom-down. Check environment data for exceedances.",
        "golden_vector": None,
        "golden_sql": sql_meta({"environment": ["wind_gust_ms"]}, "MAX"),
        "golden_rules": rule_meta(["wind_gust_ms"]),
        "golden_graph": None,
    })

    # ================================================================
    # E. NEW COMBOS: Vec+Rules (not previously covered)
    # ================================================================

    matched = find_chunks(by_src, "Port Information Manual 1.4.5.pdf",
                          ["vessel", "pilot", "entry", "requirement", "berth"])
    new_queries.append({
        "id": "MIX_VR_001", "difficulty": "medium",
        "query": "What does the Port Information Manual say about pilotage requirements, and what are the specific rules?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector", "rules"],
        "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["pilot", "vessel", "entry", "requirement", "mandatory"],
        "reference_answer": "The Port Information Manual describes pilotage procedures, and specific rules mandate pilotage for vessel entry based on vessel size and port conditions.",
        "golden_vector": vec_meta(matched, ["Port Information Manual 1.4.5.pdf"]),
        "golden_sql": None,
        "golden_rules": rule_meta(["pilotage_mandate"]),
        "golden_graph": None,
    })

    matched = find_chunks(by_src, "Green-Port-Progress-Report-August-2025 (1).pdf",
                          ["emissions", "fuel", "zero", "clean air"])
    new_queries.append({
        "id": "MIX_VR_002", "difficulty": "medium",
        "query": "What does the Green Port Progress Report say about emission reduction, and what pollution rules apply?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector", "rules"],
        "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["emissions", "pollution", "prevention", "green port", "clean"],
        "reference_answer": "The Green Port Progress Report describes decarbonisation efforts, while pollution prevention rules enforce operational standards.",
        "golden_vector": vec_meta(matched, ["Green-Port-Progress-Report-August-2025 (1).pdf"]),
        "golden_sql": None,
        "golden_rules": rule_meta(["pollution_prevention_measures"]),
        "golden_graph": None,
    })

    matched = find_chunks(by_src, "IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf",
                          ["hazard", "security", "risk", "terrorism"])
    new_queries.append({
        "id": "MIX_VR_003", "difficulty": "hard",
        "query": "What security hazards do the IAPH risk guidelines identify, and what ISPS compliance rules apply?",
        "answer_mode": "descriptive",
        "expected_sources": ["vector", "rules"],
        "needs_vector": True, "needs_sql": False, "needs_rules": True, "needs_graph": False,
        "expected_evidence_keywords": ["security", "hazard", "ISPS", "compliance", "risk"],
        "reference_answer": "IAPH guidelines identify terrorism, crime, and cyber threats; ISPS compliance rules require security screening and access controls.",
        "golden_vector": vec_meta(matched, ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"]),
        "golden_sql": None,
        "golden_rules": rule_meta(["isps_compliance", "security_screening"]),
        "golden_graph": None,
    })

    # ================================================================
    # F. NEW GRAPH QUERIES — yard→gate cascade, vessel scheduling
    # ================================================================

    new_queries.append({
        "id": "GRAPH_006", "difficulty": "hard",
        "query": "How does yard overflow cascade to gate congestion?",
        "answer_mode": "diagnostic",
        "expected_sources": ["sql", "graph"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["yard", "overflow", "gate", "congestion", "cascade"],
        "reference_answer": "When yard occupancy is high, container retrieval slows down, increasing truck turn times and causing gate congestion.",
        "golden_vector": None,
        "golden_sql": sql_meta({"yard_operations": ["peak_occupancy_pct"],
                                "gate_operations": ["average_turn_time_minutes"]}, None),
        "golden_rules": None,
        "golden_graph": graph_meta(
            ["yard_overflow", "yard_operations", "gate_congestion", "gate_operations", "congestion"],
            ["CAN_LEAD_TO", "CONTRIBUTES_TO"],
        ),
    })

    new_queries.append({
        "id": "GRAPH_007", "difficulty": "hard",
        "query": "What is the causal relationship between vessel scheduling disruptions and berth delays?",
        "answer_mode": "diagnostic",
        "expected_sources": ["sql", "graph"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["vessel", "scheduling", "berth", "delay", "disruption"],
        "reference_answer": "Weather conditions affect vessel scheduling, which impacts berth availability and arrival delays.",
        "golden_vector": None,
        "golden_sql": sql_meta({"berth_operations": ["arrival_delay_hours"],
                                "environment": ["wind_speed_ms"]}, None),
        "golden_rules": None,
        "golden_graph": graph_meta(
            ["vessel_scheduling", "weather_conditions", "arrival_delay_hours", "berth_operations"],
            ["AFFECTS", "CAN_LEAD_TO"],
        ),
    })

    new_queries.append({
        "id": "GRAPH_008", "difficulty": "hard",
        "query": "How do crane breakdowns contribute to operational disruption?",
        "answer_mode": "diagnostic",
        "expected_sources": ["sql", "graph"],
        "needs_vector": False, "needs_sql": True, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["crane", "breakdown", "disruption", "delay", "productivity"],
        "reference_answer": "Crane breakdowns reduce crane productivity, contribute to berth delays, and can cascade into wider operational disruption.",
        "golden_vector": None,
        "golden_sql": sql_meta({"crane_operations": ["breakdown_minutes", "crane_productivity_mph"]}, None),
        "golden_rules": None,
        "golden_graph": graph_meta(
            ["breakdown_minutes", "crane_operations", "crane_slowdown", "operational_disruption"],
            ["CONTRIBUTES_TO", "CAN_LEAD_TO", "DISRUPTS"],
        ),
    })

    # ================================================================
    # G. NEW COMPLEX — Graph+Vec (new combo)
    # ================================================================

    matched = find_chunks(by_src, "IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf",
                          ["storm", "disruption", "resilience", "weather"])
    new_queries.append({
        "id": "COMPLEX_006", "difficulty": "hard",
        "query": "What do risk guidelines say about storm disruption, and how does the causal chain work from weather to port-wide delays?",
        "answer_mode": "diagnostic",
        "expected_sources": ["vector", "sql", "graph"],
        "needs_vector": True, "needs_sql": True, "needs_rules": False, "needs_graph": True,
        "expected_evidence_keywords": ["storm", "disruption", "resilience", "weather", "delay", "cascade"],
        "reference_answer": "IAPH risk guidelines describe storm disruption scenarios; the causal graph shows weather → crane slowdown → berth delay cascades.",
        "golden_vector": vec_meta(matched, ["IAPH-Risk-and-Resilience-Guidelines-for-Ports-BD.pdf"]),
        "golden_sql": sql_meta({"environment": ["event_storm", "wind_speed_ms"],
                                "berth_operations": ["arrival_delay_hours"]}, None),
        "golden_rules": None,
        "golden_graph": graph_meta(
            ["storm_event", "weather_conditions", "crane_slowdown", "operational_disruption", "arrival_delay_hours"],
            ["CAN_TRIGGER", "AFFECTS", "CAN_LEAD_TO"],
        ),
    })

    # ================================================================
    # Merge and save
    # ================================================================

    golden.extend(new_queries)

    out_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nExpanded: {len(golden) - len(new_queries)} + {len(new_queries)} = {len(golden)} queries")

    from collections import Counter
    combos = Counter(tuple(sorted(g["expected_sources"])) for g in golden)
    print("\nBy source combination:")
    for c, n in combos.most_common():
        print(f"  {str(c):50s} {n}")

    diffs = Counter(g["difficulty"] for g in golden)
    print("\nBy difficulty:")
    for d, n in diffs.most_common():
        print(f"  {d}: {n}")

    modes = Counter(g["answer_mode"] for g in golden)
    print("\nBy answer_mode:")
    for m, n in modes.most_common():
        print(f"  {m}: {n}")


if __name__ == "__main__":
    main()
