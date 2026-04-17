"""
Cross-session multi-session dataset builder.

Purpose: evaluate the LONG-TERM memory retrieval path that the single-session
multi-turn dataset leaves untouched. Each "conversation" here is actually a
*sequence of sessions*. Session N builds knowledge (and gets its summary +
key_facts persisted to the DuckDB long-term store on end_session); session
N+1 is a fresh session whose queries should trigger long-term recall of
session-N facts.

This is the dataset designed to make the `cross_session_hit_rate` metric
actually discriminate between Phase-A keyword and Phase-B vector retrieval.

Format mirrors `golden_dataset_v3_multi_turn.json` but with an extra
`sessions` layer:

    {
      "conversation_id": "CS3_001",
      "pattern": "cross_session_entity_recall",
      "description": "...",
      "sessions": [
         {
           "session_order": 1,
           "description": "establishes berth tide + wind rule",
           "turns": [ {turn spec}, ... ]
         },
         {
           "session_order": 2,
           "description": "fresh session queries prior knowledge",
           "turns": [ {turn spec with expected_cross_session_hit}, ... ]
         }
      ]
    }

Each turn derives from an existing 205-sample base (`derived_from_sample_id`)
exactly as in the single-session dataset, so source/mode/guardrail coverage
is inherited.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Conversation templates — hand-written for reproducibility
# ---------------------------------------------------------------------------
# Each conversation = 2 sessions. Session 1 builds knowledge (2-3 turns).
# Session 2 (fresh session_id) asks queries that should retrieve session-1
# content from the long-term store.

# -------------------------------------------------------------------
# Helpers to reduce template boilerplate
# -------------------------------------------------------------------

def _pos_2sess(cid, pattern, desc, s1_ids, s2_id, rephrase, key_fact,
               from_order=1):
    """Shorthand for a standard 2-session positive conversation."""
    s1_turns = [{"from_sample_id": sid,
                 "evaluation_focus": "session1_establish"} for sid in s1_ids]
    return {
        "conversation_id": cid,
        "pattern": pattern,
        "description": desc,
        "sessions": [
            {"session_order": 1, "description": "establish", "turns": s1_turns},
            {"session_order": 2, "description": "recall via long-term",
             "turns": [{
                 "from_sample_id": s2_id,
                 "rephrase_as": rephrase,
                 "evaluation_focus": f"cross_session_{pattern}",
                 "expected_cross_session_hit": True,
                 "expected_from_session_order": from_order,
                 "expected_memory_recall": {
                     "from_session": from_order,
                     "key_fact": key_fact,
                 },
             }]},
        ],
    }


def _neg_2sess(cid, desc, s1_ids, s2_id):
    """Shorthand for a 2-session NEGATIVE conversation (topic drift)."""
    s1_turns = [{"from_sample_id": sid,
                 "evaluation_focus": "session1_unrelated_topic"} for sid in s1_ids]
    return {
        "conversation_id": cid,
        "pattern": "cross_session_topic_drift",
        "description": desc,
        "sessions": [
            {"session_order": 1, "description": "topic A", "turns": s1_turns},
            {"session_order": 2, "description": "topic B (unrelated — negative)",
             "turns": [{
                 "from_sample_id": s2_id,
                 "evaluation_focus": "cross_session_negative_no_leak",
                 "expected_cross_session_hit": False,
                 "expected_from_session_order": None,
             }]},
        ],
    }


CONVERSATIONS: List[Dict[str, Any]] = [

    # ===================================================================
    #  POSITIVE — cross_session_entity_recall  (5)
    # ===================================================================
    # Session 1 establishes a specific entity / metric.
    # Session 2 (fresh sid) asks about the same entity using different
    # vocabulary — tests whether long-term retrieve can bridge the gap.

    # -----------------------------------------------------------
    # Pattern 1: cross_session_entity_recall
    # Session 1 discusses a specific entity (tide/berth).
    # Session 2 asks about the same entity from scratch —
    # should hit long-term memory via entity overlap + semantic.
    # -----------------------------------------------------------
    {
        "conversation_id": "CS3_001",
        "pattern": "cross_session_entity_recall",
        "description": (
            "Session 1 establishes 2016 tide level + related policy rules. "
            "Session 2 (new session) asks a semantically similar but "
            "lexically different question. Tests whether long-term retrieve "
            "can bridge the vocabulary gap (wind vs gust, tide vs sea-level)."
        ),
        "sessions": [
            {
                "session_order": 1,
                "description": "Establishes tide + rule context",
                "turns": [
                    {"from_sample_id": "V3_SQL_001",
                     "evaluation_focus": "session1_establish_fact"},
                    {"from_sample_id": "V3_RUL_001",
                     "rephrase_as": "And what are the policy rules around that?",
                     "expected_resolved_query_contains": ["tide", "rule"],
                     "evaluation_focus": "session1_establish_rule"},
                ],
            },
            {
                "session_order": 2,
                "description": "Fresh session — should retrieve session 1 via long-term",
                "turns": [
                    {
                        "from_sample_id": "V3_SQL_001",
                        "rephrase_as": "What do we know about sea level data at the port?",
                        "evaluation_focus": "cross_session_recall_entity",
                        "expected_cross_session_hit": True,
                        "expected_from_session_order": 1,
                        "expected_memory_recall": {
                            "from_session": 1,
                            "key_fact": "tide",
                        },
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------
    # Pattern 2: cross_session_paraphrase
    # Session 2 uses heavy paraphrase — pure keyword matching fails,
    # vector retrieval should succeed.
    # -----------------------------------------------------------
    {
        "conversation_id": "CS3_002",
        "pattern": "cross_session_paraphrase",
        "description": (
            "Session 1 discusses berth productivity in Q3 2015. "
            "Session 2 asks 'how efficient were terminal operations last "
            "year' — no lexical overlap with 'berth productivity Q3 2015'. "
            "Phase-A keyword scoring should fail; Phase-B vector should win."
        ),
        "sessions": [
            {
                "session_order": 1,
                "description": "Establishes productivity + wind data",
                "turns": [
                    {"from_sample_id": "V3_SQL_011",
                     "evaluation_focus": "session1_establish_productivity"},
                    {"from_sample_id": "V3_SQL_012",
                     "evaluation_focus": "session1_establish_related_metric"},
                ],
            },
            {
                "session_order": 2,
                "description": "Heavy paraphrase — tests vector vs keyword",
                "turns": [
                    {
                        "from_sample_id": "V3_SQL_011",
                        "rephrase_as": "How efficient were terminal operations last year?",
                        "evaluation_focus": "cross_session_paraphrase",
                        "expected_cross_session_hit": True,
                        "expected_from_session_order": 1,
                        "expected_memory_recall": {
                            "from_session": 1,
                            "key_fact": "productivity",
                        },
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------
    # Pattern 3: cross_session_rule_followup
    # Session 1 establishes a rule. Session 2 asks to apply that rule.
    # -----------------------------------------------------------
    {
        "conversation_id": "CS3_003",
        "pattern": "cross_session_rule_followup",
        "description": (
            "Session 1 retrieves a wind-speed rule. Session 2 asks 'what "
            "should we do when gusts are high' — tests whether the wind "
            "rule is retrieved across session boundary."
        ),
        "sessions": [
            {
                "session_order": 1,
                "description": "Retrieve wind policy rule",
                "turns": [
                    {"from_sample_id": "V3_RUL_002",
                     "evaluation_focus": "session1_establish_wind_rule"},
                    {"from_sample_id": "V3_SQL_005",
                     "rephrase_as": "And what are typical wind speeds in 2015?",
                     "expected_resolved_query_contains": ["wind", "2015"],
                     "evaluation_focus": "session1_support_data"},
                ],
            },
            {
                "session_order": 2,
                "description": "Apply rule in fresh session",
                "turns": [
                    {
                        "from_sample_id": "V3_RUL_002",
                        "rephrase_as": "What should we do when strong gusts are present?",
                        "evaluation_focus": "cross_session_rule_application",
                        "expected_cross_session_hit": True,
                        "expected_from_session_order": 1,
                        "expected_memory_recall": {
                            "from_session": 1,
                            "key_fact": "wind",
                        },
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------
    # Pattern 4: cross_session_topic_drift
    # Session 1 discusses topic A. Session 2 discusses topic B (unrelated).
    # Tests that long-term memory does NOT incorrectly surface session-1
    # content when it's irrelevant (negative test).
    # -----------------------------------------------------------
    {
        "conversation_id": "CS3_004",
        "pattern": "cross_session_topic_drift",
        "description": (
            "Session 1 discusses tide levels. Session 2 discusses gate "
            "operations (completely unrelated). cross_session_hit SHOULD "
            "NOT fire — long-term retrieve must return nothing relevant."
        ),
        "sessions": [
            {
                "session_order": 1,
                "description": "Tide topic",
                "turns": [
                    {"from_sample_id": "V3_SQL_002",
                     "evaluation_focus": "session1_tide"},
                    {"from_sample_id": "V3_SQL_003",
                     "evaluation_focus": "session1_tide_followup"},
                ],
            },
            {
                "session_order": 2,
                "description": "Unrelated gate topic — negative test",
                "turns": [
                    {
                        "from_sample_id": "V3_VEC_003",
                        "evaluation_focus": "cross_session_negative_no_leak",
                        "expected_cross_session_hit": False,
                        "expected_from_session_order": None,
                    },
                ],
            },
        ],
    },

    # -----------------------------------------------------------
    # Pattern 5: cross_session_multi_hop
    # Three sessions. Session 3 needs facts from both session 1 and 2.
    # -----------------------------------------------------------
    {
        "conversation_id": "CS3_005",
        "pattern": "cross_session_multi_hop",
        "description": (
            "Session 1: tide facts. Session 2: wind facts. Session 3: "
            "'compare weather impact on operations' — should retrieve "
            "from BOTH prior sessions."
        ),
        "sessions": [
            {
                "session_order": 1,
                "description": "Tide data",
                "turns": [
                    {"from_sample_id": "V3_SQL_002",
                     "evaluation_focus": "session1_tide_data"},
                ],
            },
            {
                "session_order": 2,
                "description": "Wind data",
                "turns": [
                    {"from_sample_id": "V3_SQL_005",
                     "evaluation_focus": "session2_wind_data"},
                ],
            },
            {
                "session_order": 3,
                "description": "Multi-hop retrieval — needs both prior sessions",
                "turns": [
                    {
                        "from_sample_id": "V3_MULTI_001",
                        "rephrase_as": "Can you summarise what weather factors we discussed affect operations?",
                        "evaluation_focus": "cross_session_multi_hop",
                        "expected_cross_session_hit": True,
                        "expected_from_session_order": [1, 2],
                        "expected_memory_recall": {
                            "from_session": 1,
                            "key_fact": "tide",
                        },
                    },
                ],
            },
        ],
    },

    # ===================================================================
    #  POSITIVE — cross_session_entity_recall  (4 more = 5 total)
    # ===================================================================

    _pos_2sess("CS3_006", "entity_recall",
               "Session 1: crane productivity. Session 2: 'terminal equipment throughput'.",
               ["V3_SQL_011", "V3_SQL_012"], "V3_SQL_011",
               "What do we know about terminal equipment throughput?",
               "crane"),
    _pos_2sess("CS3_007", "entity_recall",
               "Session 1: berth delays Q3 2015. Session 2: 'port congestion last summer'.",
               ["V3_SQL_006"], "V3_SQL_006",
               "Was there port congestion during the summer months?",
               "delay"),
    _pos_2sess("CS3_008", "entity_recall",
               "Session 1: vessel calls data. Session 2: 'ship arrivals record'.",
               ["V3_SQL_007", "V3_SQL_008"], "V3_SQL_007",
               "What's the ship arrivals record we discussed?",
               "vessel"),
    _pos_2sess("CS3_009", "entity_recall",
               "Session 1: yard occupancy. Session 2: 'container storage utilisation'.",
               ["V3_SQL_009"], "V3_SQL_009",
               "How full was the container storage area?",
               "yard"),

    # ===================================================================
    #  POSITIVE — cross_session_paraphrase  (5)
    # ===================================================================
    # Heavy paraphrase with near-zero lexical overlap.

    _pos_2sess("CS3_010", "paraphrase",
               "Session 1: 'average wind speed 2015'. Session 2: 'how strong were the gusts recorded?'",
               ["V3_SQL_005"], "V3_SQL_005",
               "How strong were the gusts recorded at the facility?",
               "wind"),
    _pos_2sess("CS3_011", "paraphrase",
               "Session 1: 'gate transactions'. Session 2: 'truck movement volume at entry points'.",
               ["V3_SQL_010"], "V3_SQL_010",
               "What was the truck movement volume at the entry points?",
               "gate"),
    _pos_2sess("CS3_012", "paraphrase",
               "Session 1: 'berth productivity'. Session 2: 'quayside loading rate'.",
               ["V3_SQL_013", "V3_SQL_014"], "V3_SQL_013",
               "What was the quayside loading rate?",
               "berth"),
    _pos_2sess("CS3_013", "paraphrase",
               "Session 1: environmental data. Session 2: 'weather observations at the harbour'.",
               ["V3_SQL_002", "V3_SQL_003"], "V3_SQL_002",
               "Any weather observations recorded at the harbour?",
               "tide"),
    _pos_2sess("CS3_014", "paraphrase",
               "Session 1: crane breakdown data. Session 2: 'equipment downtime incidents'.",
               ["V3_SQL_015"], "V3_SQL_015",
               "Were there equipment downtime incidents recently?",
               "crane"),

    # ===================================================================
    #  POSITIVE — cross_session_rule_followup  (4)
    # ===================================================================
    # Session 1 establishes a rule / policy. Session 2 asks to apply it.

    _pos_2sess("CS3_015", "rule_followup",
               "Session 1: wind restriction policy. Session 2: 'operational limits during storms'.",
               ["V3_RUL_002", "V3_RUL_003"], "V3_RUL_002",
               "What are the operational limits during severe weather?",
               "wind"),
    _pos_2sess("CS3_016", "rule_followup",
               "Session 1: vessel entry restrictions. Session 2: 'ship access conditions'.",
               ["V3_RUL_004", "V3_RUL_005"], "V3_RUL_004",
               "Under what conditions is ship access restricted?",
               "vessel"),
    _pos_2sess("CS3_017", "rule_followup",
               "Session 1: crane safety thresholds. Session 2: 'when should crane ops halt?'",
               ["V3_RUL_006"], "V3_RUL_006",
               "When should crane operations be halted for safety?",
               "crane"),
    _pos_2sess("CS3_018", "rule_followup",
               "Session 1: berth allocation policy. Session 2: 'how are berths assigned?'.",
               ["V3_RUL_007", "V3_RUL_008"], "V3_RUL_007",
               "How are berths assigned to incoming vessels?",
               "berth"),

    # ===================================================================
    #  POSITIVE — cross_session_multi_hop  (3 more = 4 total)
    # ===================================================================
    # 3 sessions; session 3 needs facts from sessions 1 AND 2.

    {
        "conversation_id": "CS3_019",
        "pattern": "cross_session_multi_hop",
        "description": "S1: crane data. S2: berth data. S3: correlate crane + berth.",
        "sessions": [
            {"session_order": 1, "description": "crane metrics",
             "turns": [{"from_sample_id": "V3_SQL_011",
                        "evaluation_focus": "session1_crane"}]},
            {"session_order": 2, "description": "berth metrics",
             "turns": [{"from_sample_id": "V3_SQL_013",
                        "evaluation_focus": "session2_berth"}]},
            {"session_order": 3, "description": "needs both",
             "turns": [{
                 "from_sample_id": "V3_MULTI_002",
                 "rephrase_as": "How do crane and berth performance relate to each other?",
                 "evaluation_focus": "cross_session_multi_hop",
                 "expected_cross_session_hit": True,
                 "expected_from_session_order": [1, 2],
                 "expected_memory_recall": {"from_session": 1, "key_fact": "crane"},
             }]},
        ],
    },
    {
        "conversation_id": "CS3_020",
        "pattern": "cross_session_multi_hop",
        "description": "S1: rules. S2: SQL data. S3: 'does actual data comply with rules?'",
        "sessions": [
            {"session_order": 1, "description": "rule context",
             "turns": [{"from_sample_id": "V3_RUL_001",
                        "evaluation_focus": "session1_rule"}]},
            {"session_order": 2, "description": "actual data",
             "turns": [{"from_sample_id": "V3_SQL_001",
                        "evaluation_focus": "session2_data"}]},
            {"session_order": 3, "description": "compliance check",
             "turns": [{
                 "from_sample_id": "V3_MULTI_003",
                 "rephrase_as": "Does the actual operational data comply with the rules we reviewed?",
                 "evaluation_focus": "cross_session_multi_hop",
                 "expected_cross_session_hit": True,
                 "expected_from_session_order": [1, 2],
                 "expected_memory_recall": {"from_session": 1, "key_fact": "rule"},
             }]},
        ],
    },
    {
        "conversation_id": "CS3_021",
        "pattern": "cross_session_multi_hop",
        "description": "S1: graph causal. S2: SQL numeric. S3: 'explain the root cause with data'.",
        "sessions": [
            {"session_order": 1, "description": "causal graph",
             "turns": [{"from_sample_id": "V3_GRA_001",
                        "evaluation_focus": "session1_graph"}]},
            {"session_order": 2, "description": "numeric data",
             "turns": [{"from_sample_id": "V3_SQL_004",
                        "evaluation_focus": "session2_sql"}]},
            {"session_order": 3, "description": "combine both",
             "turns": [{
                 "from_sample_id": "V3_MULTI_004",
                 "rephrase_as": "Can you explain the root cause using both the causal model and the numbers?",
                 "evaluation_focus": "cross_session_multi_hop",
                 "expected_cross_session_hit": True,
                 "expected_from_session_order": [1, 2],
                 "expected_memory_recall": {"from_session": 2, "key_fact": "value"},
             }]},
        ],
    },

    # ===================================================================
    #  NEGATIVE — cross_session_topic_drift  (10)
    # ===================================================================
    # Session 1 and session 2 are on COMPLETELY DIFFERENT topics.
    # expected_cross_session_hit = False.

    _neg_2sess("CS3_NEG_001", "S1: tide data.      S2: gate operations.",
               ["V3_SQL_001", "V3_SQL_002"], "V3_VEC_003"),
    _neg_2sess("CS3_NEG_002", "S1: crane metrics.  S2: environmental policy doc.",
               ["V3_SQL_011"], "V3_VEC_010"),
    _neg_2sess("CS3_NEG_003", "S1: wind speed.     S2: vessel scheduling doc.",
               ["V3_SQL_005"], "V3_VEC_015"),
    _neg_2sess("CS3_NEG_004", "S1: berth delays.   S2: sustainability report.",
               ["V3_SQL_006"], "V3_VEC_020"),
    _neg_2sess("CS3_NEG_005", "S1: yard occupancy. S2: noise policy.",
               ["V3_SQL_009"], "V3_VEC_004"),
    _neg_2sess("CS3_NEG_006", "S1: vessel calls.   S2: HOT lane feasibility.",
               ["V3_SQL_007"], "V3_VEC_001"),
    _neg_2sess("CS3_NEG_007", "S1: gate transactions. S2: graph causal analysis.",
               ["V3_SQL_010"], "V3_GRA_005"),
    _neg_2sess("CS3_NEG_008", "S1: rule R-14 wind. S2: annual report overview.",
               ["V3_RUL_002"], "V3_VEC_008"),
    _neg_2sess("CS3_NEG_009", "S1: crane safety.   S2: tide historical data.",
               ["V3_RUL_006"], "V3_SQL_002"),
    _neg_2sess("CS3_NEG_010", "S1: graph weather impact. S2: berth allocation rule.",
               ["V3_GRA_001"], "V3_RUL_007"),
]


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _index_base_samples(base_samples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {s["id"]: s for s in base_samples}


def _materialise_turn(
    turn_spec: Dict[str, Any],
    base: Dict[str, Dict[str, Any]],
    turn_id: int,
) -> Dict[str, Any]:
    sample_id = turn_spec["from_sample_id"]
    if sample_id not in base:
        raise KeyError(
            f"unknown base sample id '{sample_id}' — open "
            f"evaluation/golden_dataset_v3_rag.json and pick a real id"
        )
    base_sample = base[sample_id]
    raw_query = turn_spec.get("rephrase_as") or base_sample["query"]

    inherited = {
        k: base_sample.get(k)
        for k in (
            "expected_sources", "needs_vector", "needs_sql", "needs_rules",
            "needs_graph", "answer_mode", "expected_evidence_keywords",
            "reference_answer", "difficulty",
            "golden_vector", "golden_sql", "golden_rules", "golden_graph",
        )
        if k in base_sample
    }

    out: Dict[str, Any] = {
        "turn_id": turn_id,
        "raw_query": raw_query,
        "derived_from_sample_id": sample_id,
        "evaluation_focus": turn_spec.get("evaluation_focus", "unspecified"),
        **inherited,
    }
    for k in (
        "expected_resolved_query_contains",
        "expected_resolved_query_should_not_contain",
        "expected_memory_recall",
        "expected_cross_session_hit",
        "expected_from_session_order",
    ):
        if k in turn_spec:
            out[k] = turn_spec[k]
    return out


def build_cross_session_dataset(
    base_path: Path,
    out_path: Path,
    skip_invalid: bool = False,
) -> Dict[str, Any]:
    with open(base_path, "r", encoding="utf-8") as f:
        base = json.load(f)
    by_id = _index_base_samples(base["samples"])

    materialised: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for conv in CONVERSATIONS:
        try:
            sessions = []
            for sess in conv["sessions"]:
                sess_out = {
                    "session_order": sess["session_order"],
                    "description": sess["description"],
                    "turns": [
                        _materialise_turn(t, by_id, turn_id=i + 1)
                        for i, t in enumerate(sess["turns"])
                    ],
                }
                sessions.append(sess_out)
        except KeyError as e:
            if skip_invalid:
                skipped.append(f"{conv['conversation_id']}: {e}")
                continue
            raise

        materialised.append({
            "conversation_id": conv["conversation_id"],
            "pattern": conv["pattern"],
            "description": conv["description"],
            "sessions": sessions,
        })

    pattern_counts: Dict[str, int] = defaultdict(int)
    total_sessions = 0
    total_turns = 0
    for c in materialised:
        pattern_counts[c["pattern"]] += 1
        total_sessions += len(c["sessions"])
        total_turns += sum(len(s["turns"]) for s in c["sessions"])

    output = {
        "description": (
            "Multi-session (cross-session) conversations composed from "
            "golden_dataset_v3_rag.json. Each conversation is a SEQUENCE "
            "of sessions — session N writes facts into long-term memory on "
            "end_session; session N+1 (fresh sid) queries them back. "
            "Designed to exercise the long-term retrieval path the single-"
            "session multi-turn dataset leaves untouched."
        ),
        "version": "v3.0-cross-session",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dataset": str(base_path.name),
        "total_conversations": len(materialised),
        "total_sessions": total_sessions,
        "total_turns": total_turns,
        "pattern_counts": dict(pattern_counts),
        "skipped_conversations": skipped,
        "conversations": materialised,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=Path("evaluation/golden_dataset_v3_rag.json"))
    parser.add_argument("--out", type=Path,
                        default=Path("evaluation/golden_dataset_v3_cross_session.json"))
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()

    out = build_cross_session_dataset(args.base, args.out, skip_invalid=args.skip_invalid)
    print(f"Wrote {args.out}")
    print(f"  conversations: {out['total_conversations']}")
    print(f"  sessions:      {out['total_sessions']}")
    print(f"  turns:         {out['total_turns']}")
    print(f"  patterns:      {out['pattern_counts']}")
    if out["skipped_conversations"]:
        print(f"  skipped:       {len(out['skipped_conversations'])}")
        for s in out["skipped_conversations"]:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
