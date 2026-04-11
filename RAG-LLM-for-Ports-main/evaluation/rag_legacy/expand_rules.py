"""
Rule Expansion Script
======================
Analyzes coverage gaps in existing rules, then uses LLM to synthesize
additional grounded_rules and policy_rules based on existing documents
and port operations domain knowledge.

Usage:
    cd RAG-LLM-for-Ports-main
    python evaluation/expand_rules.py
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from online_pipeline.llm_client import llm_chat_raw_post


def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    result = llm_chat_raw_post(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        timeout=(10, 180),
    )
    return result or "[]"


def extract_json_list(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return []


# ---------------------------------------------------------------------------
# 1. Coverage Gap Analysis
# ---------------------------------------------------------------------------

def analyze_coverage_gaps():
    """Identify which port operation scenarios lack rule coverage."""

    grounded_path = PROJECT_ROOT / "data" / "rules" / "grounded_rules.json"
    policy_path = PROJECT_ROOT / "data" / "rules" / "policy_rules.json"

    # Try v1 files as fallback
    if not grounded_path.exists():
        grounded_path = PROJECT_ROOT / "data" / "rules" / "grounded_rules_v1.json"
    if not policy_path.exists():
        policy_path = PROJECT_ROOT / "data" / "rules" / "policy_rules_v1.json"

    grounded = json.loads(grounded_path.read_text()) if grounded_path.exists() else []
    policy = json.loads(policy_path.read_text()) if policy_path.exists() else []

    # Catalog existing coverage
    grounded_vars = set()
    for r in grounded:
        v = (r.get("variable") or "").lower()
        sv = (r.get("sql_variable") or "").lower()
        if v:
            grounded_vars.add(v)
        if sv:
            grounded_vars.add(sv)

    policy_topics = set()
    for r in policy:
        v = (r.get("variable") or "").lower()
        if v:
            policy_topics.add(v)

    # Schema variables that SHOULD have grounded rules
    schema_path = PROJECT_ROOT / "data" / "sql_data" / "schema_metadata.json"
    schema = json.loads(schema_path.read_text()) if schema_path.exists() else {}

    all_rule_candidate_cols = []
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            if col.get("rule_candidate"):
                all_rule_candidate_cols.append({
                    "table": table["table_name"],
                    "column": col["name"],
                    "semantic_type": col.get("semantic_type"),
                    "has_grounded_rule": col["name"].lower() in grounded_vars,
                })

    uncovered_cols = [c for c in all_rule_candidate_cols if not c["has_grounded_rule"]]

    # Port operations categories that should have policy rules
    expected_policy_categories = [
        "vessel_entry_procedures", "vessel_departure_procedures",
        "pilotage_requirements", "tug_operations",
        "berth_allocation", "berth_safety",
        "crane_operations_safety", "crane_wind_limits",
        "yard_operations", "yard_capacity_management",
        "gate_operations", "truck_management",
        "hazardous_cargo", "dangerous_goods",
        "environmental_compliance", "emission_controls",
        "security_protocols", "access_control",
        "emergency_procedures", "incident_response",
        "navigation_rules", "channel_management",
        "mooring_operations", "bollard_requirements",
        "bunkering_operations", "fuel_management",
        "waste_management", "pollution_prevention",
        "communication_protocols", "VHF_requirements",
        "crew_requirements", "watchkeeping",
        "weather_restrictions", "storm_protocols",
        "maintenance_standards", "equipment_inspection",
    ]

    report = {
        "existing_grounded_rules": len(grounded),
        "existing_policy_rules": len(policy),
        "grounded_variables_covered": list(grounded_vars),
        "uncovered_schema_columns": uncovered_cols[:20],
        "expected_policy_categories": expected_policy_categories,
        "existing_policy_topics": list(policy_topics)[:30],
    }

    print("\n=== COVERAGE GAP ANALYSIS ===")
    print(f"Existing grounded rules: {len(grounded)}")
    print(f"Existing policy rules: {len(policy)}")
    print(f"Schema columns needing rules: {len(all_rule_candidate_cols)}")
    print(f"Uncovered columns: {len(uncovered_cols)}")
    print(f"Uncovered columns (sample): {[c['column'] for c in uncovered_cols[:10]]}")

    return report


# ---------------------------------------------------------------------------
# 2. Synthesize Grounded Rules
# ---------------------------------------------------------------------------

GROUNDED_SYSTEM = """You are a port operations expert. Generate operational threshold rules
that can be validated against SQL data. Each rule must reference a measurable variable
that exists in port operations databases.

Return ONLY a JSON array. Each rule must follow this schema:
{
  "rule_text": "Full text of the operational rule",
  "condition": "When this rule applies",
  "action": "What must be done",
  "variable": "The measurable variable name (must match the SQL column name provided)",
  "sql_variable": "Same as variable - the exact SQL column name",
  "operator": "one of: > < >= <= = !=",
  "value": "threshold numeric value as string",
  "unit": "unit of measurement",
  "source_file": "synthetic_rule",
  "page": 0
}
"""


def synthesize_grounded_rules(gap_report: dict, target_count: int = 18) -> list:
    """Generate grounded rules for uncovered schema variables."""

    uncovered = gap_report["uncovered_schema_columns"]
    existing_count = gap_report["existing_grounded_rules"]
    needed = max(target_count - existing_count, 5)

    # Group uncovered by table
    by_table = {}
    for c in uncovered:
        t = c["table"]
        if t not in by_table:
            by_table[t] = []
        by_table[t].append(c["column"])

    rules = []

    for table, columns in by_table.items():
        col_list = ", ".join(columns[:8])

        prompt = f"""Generate {min(needed, 5)} operational threshold rules for port operations.

Table: {table}
Available columns (SQL names): {col_list}

Requirements:
- Each rule must reference one of the columns above as its variable/sql_variable
- Use realistic port operations thresholds based on industry standards
- operator must be one of: > < >= <= = !=
- value must be a numeric threshold
- Rules should be actionable (e.g., "If wind_speed_ms > 15, restrict vessel entry")

Return ONLY a JSON array."""

        print(f"\n  Generating grounded rules for table: {table} ({col_list[:60]}...)")

        response = call_llm(GROUNDED_SYSTEM, prompt)
        batch = extract_json_list(response)

        for r in batch:
            # Validate structure
            if r.get("variable") and r.get("operator") and r.get("value") is not None:
                if "sql_variable" not in r:
                    r["sql_variable"] = r["variable"]
                if "source_file" not in r:
                    r["source_file"] = "synthetic_rule"
                if "page" not in r:
                    r["page"] = 0
                rules.append(r)

        time.sleep(1)  # Rate limit

        if len(rules) >= needed:
            break

    print(f"\n  Synthesized {len(rules)} new grounded rules")
    return rules


# ---------------------------------------------------------------------------
# 3. Synthesize Policy Rules
# ---------------------------------------------------------------------------

POLICY_SYSTEM = """You are a port operations and maritime safety expert.
Generate operational policy rules for port management. These are qualitative rules
that govern procedures, safety protocols, and operational requirements.

Return ONLY a JSON array. Each rule must follow this schema:
{
  "rule_text": "Full text of the policy rule",
  "condition": "When/where this rule applies",
  "action": "What must/must not be done",
  "variable": "Category or topic of the rule",
  "operator": null,
  "value": null,
  "unit": "",
  "source_file": "synthetic_rule",
  "page": 0
}
"""

POLICY_CATEGORIES = {
    "vessel_operations": [
        "vessel entry and departure procedures",
        "pilotage and tug requirements",
        "mooring and unmooring operations",
        "anchorage management",
        "vessel speed limits in port waters",
    ],
    "berth_and_crane": [
        "berth allocation priorities",
        "crane operation safety procedures",
        "crane wind speed operational limits",
        "loading and unloading protocols",
        "berth maintenance requirements",
    ],
    "yard_and_gate": [
        "yard container stacking regulations",
        "yard capacity management protocols",
        "gate access control procedures",
        "truck appointment systems",
        "reefer container monitoring",
    ],
    "safety_and_environment": [
        "hazardous cargo handling procedures",
        "environmental compliance requirements",
        "emergency response protocols",
        "fire safety in terminal areas",
        "pollution prevention measures",
    ],
    "security_and_communication": [
        "port facility security requirements (ISPS)",
        "VHF communication protocols",
        "crew and personnel requirements",
        "visitor and contractor access",
        "cybersecurity for port systems",
    ],
    "weather_and_navigation": [
        "storm preparation and response",
        "fog and low visibility procedures",
        "tidal window restrictions",
        "channel navigation rules",
        "weather monitoring requirements",
    ],
    "maintenance_and_quality": [
        "equipment inspection schedules",
        "infrastructure maintenance standards",
        "quality control for cargo handling",
        "documentation and record-keeping",
        "training and certification requirements",
    ],
}


def synthesize_policy_rules(gap_report: dict, target_count: int = 130) -> list:
    """Generate policy rules across port operations categories."""

    existing_count = gap_report["existing_policy_rules"]
    needed = max(target_count - existing_count, 50)
    per_category = max(needed // len(POLICY_CATEGORIES), 5)

    rules = []

    for category, topics in POLICY_CATEGORIES.items():
        topic_list = "\n".join(f"- {t}" for t in topics)

        prompt = f"""Generate {per_category} port operations policy rules for the category: {category}

Topics to cover:
{topic_list}

Requirements:
- Rules must be specific and actionable
- Use regulatory language (must, shall, must not, prohibited)
- Cover different aspects of each topic
- Be consistent with IMO, SOLAS, ISPS, and port authority regulations
- variable field should be a short category label

Return ONLY a JSON array."""

        print(f"\n  Generating policy rules for: {category}")

        response = call_llm(POLICY_SYSTEM, prompt)
        batch = extract_json_list(response)

        for r in batch:
            if r.get("rule_text") and r.get("condition"):
                if "source_file" not in r:
                    r["source_file"] = "synthetic_rule"
                if "page" not in r:
                    r["page"] = 0
                rules.append(r)

        time.sleep(1)

    print(f"\n  Synthesized {len(rules)} new policy rules")
    return rules


# ---------------------------------------------------------------------------
# 4. Merge and Save
# ---------------------------------------------------------------------------

def merge_and_save(new_grounded: list, new_policy: list):
    """Merge new rules with existing and save."""

    grounded_path = PROJECT_ROOT / "data" / "rules" / "grounded_rules.json"
    policy_path = PROJECT_ROOT / "data" / "rules" / "policy_rules.json"

    # Load existing (try non-v1 first, then v1)
    if grounded_path.exists():
        existing_grounded = json.loads(grounded_path.read_text())
    else:
        v1_path = PROJECT_ROOT / "data" / "rules" / "grounded_rules_v1.json"
        existing_grounded = json.loads(v1_path.read_text()) if v1_path.exists() else []

    if policy_path.exists():
        existing_policy = json.loads(policy_path.read_text())
    else:
        v1_path = PROJECT_ROOT / "data" / "rules" / "policy_rules_v1.json"
        existing_policy = json.loads(v1_path.read_text()) if v1_path.exists() else []

    # Deduplicate by rule_text
    existing_grounded_texts = {r.get("rule_text", "").lower() for r in existing_grounded}
    existing_policy_texts = {r.get("rule_text", "").lower() for r in existing_policy}

    added_grounded = 0
    for r in new_grounded:
        if r.get("rule_text", "").lower() not in existing_grounded_texts:
            existing_grounded.append(r)
            existing_grounded_texts.add(r.get("rule_text", "").lower())
            added_grounded += 1

    added_policy = 0
    for r in new_policy:
        if r.get("rule_text", "").lower() not in existing_policy_texts:
            existing_policy.append(r)
            existing_policy_texts.add(r.get("rule_text", "").lower())
            added_policy += 1

    # Save
    with open(grounded_path, "w", encoding="utf-8") as f:
        json.dump(existing_grounded, f, indent=2, ensure_ascii=False)

    with open(policy_path, "w", encoding="utf-8") as f:
        json.dump(existing_policy, f, indent=2, ensure_ascii=False)

    print(f"\n=== MERGE RESULTS ===")
    print(f"Grounded rules: {len(existing_grounded)} total ({added_grounded} new)")
    print(f"Policy rules: {len(existing_policy)} total ({added_policy} new)")
    print(f"Saved to: {grounded_path}")
    print(f"Saved to: {policy_path}")

    return len(existing_grounded), len(existing_policy)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("PORT RAG - RULE EXPANSION")
    print("=" * 70)

    # Step 1: Analyze gaps
    gap_report = analyze_coverage_gaps()

    # Step 2: Synthesize grounded rules
    print("\n" + "=" * 70)
    print("SYNTHESIZING GROUNDED RULES (target: 15-20)")
    print("=" * 70)
    new_grounded = synthesize_grounded_rules(gap_report, target_count=20)

    # Step 3: Synthesize policy rules
    print("\n" + "=" * 70)
    print("SYNTHESIZING POLICY RULES (target: 120+)")
    print("=" * 70)
    new_policy = synthesize_policy_rules(gap_report, target_count=130)

    # Step 4: Merge and save
    print("\n" + "=" * 70)
    print("MERGING AND SAVING")
    print("=" * 70)
    grounded_total, policy_total = merge_and_save(new_grounded, new_policy)

    print("\n" + "=" * 70)
    print("RULE EXPANSION COMPLETE")
    print(f"Grounded rules: {grounded_total} (target: 15-20)")
    print(f"Policy rules: {policy_total} (target: 120+)")
    print("=" * 70)


if __name__ == "__main__":
    main()
