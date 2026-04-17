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

CONVERSATIONS: List[Dict[str, Any]] = [

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
