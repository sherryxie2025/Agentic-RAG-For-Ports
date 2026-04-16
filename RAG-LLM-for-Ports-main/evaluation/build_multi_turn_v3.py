"""
Multi-turn conversation dataset builder for the agentic-RAG DAG.

Composes conversations *on top of* the 205-sample single-turn golden set
(`golden_dataset_v3_rag.json`) so that:

- Every turn is **derived from** an existing single-turn sample (linked via
  `derived_from_sample_id`), inheriting that sample's `expected_sources`,
  `needs_*`, `answer_mode`, `expected_evidence_keywords`, `reference_answer`,
  and golden retrievals. Nothing is invented from thin air.

- Across the turns of one conversation, the source combos and answer modes
  *vary* deliberately (a real user won't ask 5 SQL lookups in a row),
  so the multi-turn set inherits the full 2^4 source-combo coverage of
  the base set rather than collapsing to one slice.

- Each multi-turn conversation falls into one of six **patterns** that
  stress different memory capabilities:
    1. entity_anchored          - same berth/crane/vessel across turns
    2. mode_progression         - lookup -> comparison -> decision -> diagnostic
    3. cross_source_verification - SQL fact then policy/graph follow-up about it
    4. topic_switch             - turn N pivots to an unrelated topic (negative test)
    5. long_summarisation       - 6+ turns to trigger short-term summarisation
    6. guardrail_in_conversation - a guardrail sample is embedded mid-chat

- Each follow-up turn carries:
    - `expected_resolved_query_contains` / `expected_resolved_query_should_not_contain`
      for the co-reference resolver evaluation
    - `evaluation_focus` tag describing the specific memory ability tested
    - (optional) `expected_memory_recall` for "what should the agent remember
      from turn K" assertions

Run from project root:
    python evaluation/build_multi_turn_v3.py \
        --base evaluation/golden_dataset_v3_rag.json \
        --out  evaluation/golden_dataset_v3_multi_turn.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Pattern templates
# ---------------------------------------------------------------------------
# Each conversation is fully specified by hand at the level of (a) the base
# sample IDs used per turn and (b) the follow-up phrasing for turn >= 2.
# This makes the dataset reproducible and reviewable.

# A "turn spec" is:
#   {
#       "from_sample_id": "V3_SQL_001",
#       "rephrase_as": "And in 2017?",         # optional — overrides raw query
#       "expected_resolved_query_contains": ["tide", "2017"],
#       "expected_resolved_query_should_not_contain": [],
#       "evaluation_focus": "ellipsis_resolution",
#       "expected_memory_recall": {"from_turn": 1, "key_fact": "tide level"},  # optional
#   }
#
# A pattern is a list of turn specs + a description + pattern type.

CONVERSATIONS: List[Dict[str, Any]] = [

    # ---------------------------------------------------------------
    # Pattern 1: entity_anchored — berth productivity thread
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_001",
        "pattern": "entity_anchored",
        "description": "Berth productivity SQL lookup, followed by a rules check then a graph causal probe — all anchored on the same berth.",
        "turns": [
            {
                "from_sample_id": "V3_SQL_001",
                "evaluation_focus": "single_turn_baseline",
            },
            {
                "from_sample_id": "V3_RUL_001",
                "rephrase_as": "And what are the policy rules around that?",
                "expected_resolved_query_contains": ["tide", "rule"],
                "evaluation_focus": "ellipsis_with_topic_carry",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "tide"},
            },
            {
                "from_sample_id": "V3_GRA_001",
                "rephrase_as": "If that threshold is breached, which port operations are affected?",
                "expected_resolved_query_contains": ["threshold", "operations"],
                "evaluation_focus": "demonstrative_resolution",
                "expected_memory_recall": {"from_turn": 2, "key_fact": "threshold"},
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 2: mode_progression — lookup -> comparison -> decision
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_002",
        "pattern": "mode_progression",
        "description": "User starts with a fact lookup, then asks for a comparison, then a decision-support framing.",
        "turns": [
            {
                "from_sample_id": "V3_SQL_002",
                "evaluation_focus": "single_turn_baseline",
            },
            {
                "from_sample_id": "V3_SQL_003",
                "rephrase_as": "How does that compare with the previous year?",
                "expected_resolved_query_contains": ["compare", "year"],
                "evaluation_focus": "ellipsis_with_metric_carry",
            },
            {
                "from_sample_id": "V3_MULTI_001",
                "rephrase_as": "Given those numbers, should we adjust operations?",
                "expected_resolved_query_contains": ["operations"],
                "evaluation_focus": "decision_support_with_history",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "value"},
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 3: cross_source_verification — SQL fact then rule check
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_003",
        "pattern": "cross_source_verification",
        "description": "User gets a SQL number, then asks whether port policy is consistent with it (forces rules), then whether documentation explains the gap (forces vector).",
        "turns": [
            {
                "from_sample_id": "V3_SQL_004",
                "evaluation_focus": "single_turn_baseline",
            },
            {
                "from_sample_id": "V3_RUL_002",
                "rephrase_as": "Is that consistent with current port policy?",
                "expected_resolved_query_contains": ["policy"],
                "evaluation_focus": "policy_grounding",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "value"},
            },
            {
                "from_sample_id": "V3_VEC_002",
                "rephrase_as": "What does the documentation say about the discrepancy?",
                "expected_resolved_query_contains": ["documentation"],
                "evaluation_focus": "doc_grounding_with_history",
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 4: topic_switch — agent must NOT carry old context
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_004",
        "pattern": "topic_switch",
        "description": "Two turns about port productivity, then a hard pivot to a different operational area. Tests that old entities are NOT injected into the resolved third query.",
        "turns": [
            {
                "from_sample_id": "V3_SQL_005",
                "evaluation_focus": "single_turn_baseline",
            },
            {
                "from_sample_id": "V3_SQL_006",
                "rephrase_as": "Same question for the previous quarter.",
                "expected_resolved_query_contains": ["quarter"],
                "evaluation_focus": "temporal_carry",
            },
            {
                "from_sample_id": "V3_VEC_003",
                "evaluation_focus": "topic_switch_no_carry",
                "expected_resolved_query_should_not_contain": ["productivity", "quarter"],
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 5: long_summarisation — 6 turns to trigger summary
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_005",
        "pattern": "long_summarisation",
        "description": "Six diverse turns: triggers ShortTermMemory.summaries population (max_raw_turns=10/2). Final turn must recall a fact from turn 1 after summarisation.",
        "turns": [
            {"from_sample_id": "V3_VEC_004", "evaluation_focus": "single_turn_baseline"},
            {"from_sample_id": "V3_SQL_007", "evaluation_focus": "single_turn_baseline"},
            {"from_sample_id": "V3_RUL_003", "evaluation_focus": "single_turn_baseline"},
            {"from_sample_id": "V3_GRA_002", "evaluation_focus": "single_turn_baseline"},
            {"from_sample_id": "V3_SQL_008", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_VEC_005",
                "rephrase_as": "Going back to what we discussed first — does that source explain it?",
                "expected_resolved_query_contains": ["source"],
                "evaluation_focus": "long_range_recall",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "first_topic"},
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 6: guardrail_in_conversation — OOD mid-chat
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_006",
        "pattern": "guardrail_in_conversation",
        "description": "Two normal port turns, then an out-of-domain query in the middle of the chat, then a return to the original topic. Tests that the OOD refusal does not contaminate downstream turns.",
        "turns": [
            {"from_sample_id": "V3_SQL_009", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_GUARD_OOD_001",
                "evaluation_focus": "ood_mid_conversation",
            },
            {
                "from_sample_id": "V3_SQL_010",
                "rephrase_as": "Back to the earlier topic — what was the trend?",
                "expected_resolved_query_contains": ["trend"],
                "evaluation_focus": "post_guardrail_recovery",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "topic"},
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 1 (additional): entity_anchored — vessel/crane thread
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_007",
        "pattern": "entity_anchored",
        "description": "Crane operations question with two follow-ups, all anchored on the same crane.",
        "turns": [
            {"from_sample_id": "V3_SQL_011", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_RUL_004",
                "rephrase_as": "Are there safety rules tied to it?",
                "expected_resolved_query_contains": ["safety", "rule"],
                "evaluation_focus": "pronoun_resolution",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "crane"},
            },
            {
                "from_sample_id": "V3_VEC_006",
                "rephrase_as": "Is this documented anywhere?",
                "expected_resolved_query_contains": ["document"],
                "evaluation_focus": "doc_grounding_with_history",
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 3 (additional): cross_source_verification with conflict
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_008",
        "pattern": "cross_source_verification",
        "description": "User checks a SQL number and asks if it conflicts with documented thresholds.",
        "turns": [
            {"from_sample_id": "V3_SQL_012", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_RUL_005",
                "rephrase_as": "Does that exceed the documented threshold?",
                "expected_resolved_query_contains": ["threshold", "exceed"],
                "evaluation_focus": "conflict_query_with_carry",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "value"},
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 4 (additional): topic switch
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_009",
        "pattern": "topic_switch",
        "description": "Berth thread followed by an unrelated graph query about gate operations.",
        "turns": [
            {"from_sample_id": "V3_VEC_007", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_GRA_003",
                "evaluation_focus": "topic_switch_no_carry",
                "expected_resolved_query_should_not_contain": ["berth"],
            },
        ],
    },

    # ---------------------------------------------------------------
    # Pattern 2 (additional): mode progression — diagnostic chain
    # ---------------------------------------------------------------
    {
        "conversation_id": "MT3_010",
        "pattern": "mode_progression",
        "description": "Diagnostic question sequence: descriptive -> diagnostic -> decision_support over 3 turns.",
        "turns": [
            {"from_sample_id": "V3_VEC_008", "evaluation_focus": "single_turn_baseline"},
            {
                "from_sample_id": "V3_MULTI_002",
                "rephrase_as": "Why might that be happening?",
                "expected_resolved_query_contains": ["why"],
                "evaluation_focus": "diagnostic_with_carry",
                "expected_memory_recall": {"from_turn": 1, "key_fact": "phenomenon"},
            },
            {
                "from_sample_id": "V3_MULTI_003",
                "rephrase_as": "What should we recommend doing about it?",
                "expected_resolved_query_contains": ["recommend"],
                "evaluation_focus": "decision_with_diagnostic_history",
                "expected_memory_recall": {"from_turn": 2, "key_fact": "cause"},
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
            f"turn references unknown base sample id '{sample_id}'. "
            f"Open evaluation/golden_dataset_v3_rag.json and pick a real id."
        )
    base_sample = base[sample_id]
    raw_query = turn_spec.get("rephrase_as") or base_sample["query"]

    # Inherit golden fields verbatim — the eval pipeline reuses base scoring.
    inherited = {
        k: base_sample.get(k)
        for k in (
            "expected_sources",
            "needs_vector",
            "needs_sql",
            "needs_rules",
            "needs_graph",
            "answer_mode",
            "expected_evidence_keywords",
            "reference_answer",
            "difficulty",
            "golden_vector",
            "golden_sql",
            "golden_rules",
            "golden_graph",
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
    # Multi-turn-specific extras
    for k in (
        "expected_resolved_query_contains",
        "expected_resolved_query_should_not_contain",
        "expected_memory_recall",
    ):
        if k in turn_spec:
            out[k] = turn_spec[k]
    return out


def build_multi_turn_dataset(
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
            turns = [
                _materialise_turn(t, by_id, turn_id=i + 1)
                for i, t in enumerate(conv["turns"])
            ]
        except KeyError as e:
            if skip_invalid:
                skipped.append(f"{conv['conversation_id']}: {e}")
                continue
            raise

        materialised.append({
            "conversation_id": conv["conversation_id"],
            "pattern": conv["pattern"],
            "description": conv["description"],
            "turns": turns,
        })

    pattern_counts: Dict[str, int] = defaultdict(int)
    mode_counts: Dict[str, int] = defaultdict(int)
    source_combo_counts: Dict[str, int] = defaultdict(int)
    for c in materialised:
        pattern_counts[c["pattern"]] += 1
        for t in c["turns"]:
            mode_counts[t.get("answer_mode", "unknown")] += 1
            combo = tuple(sorted(t.get("expected_sources", []) or []))
            source_combo_counts[str(combo)] += 1

    output = {
        "description": (
            "Multi-turn conversations composed from golden_dataset_v3_rag.json. "
            "Each turn inherits its golden retrieval/answer fields from the linked "
            "base sample (`derived_from_sample_id`)."
        ),
        "version": "v3.0",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dataset": str(base_path.name),
        "total_conversations": len(materialised),
        "total_turns": sum(len(c["turns"]) for c in materialised),
        "pattern_counts": dict(pattern_counts),
        "answer_mode_counts": dict(mode_counts),
        "source_combo_counts": dict(source_combo_counts),
        "skipped_conversations": skipped,
        "conversations": materialised,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("evaluation/golden_dataset_v3_rag.json"),
        help="Path to the 205-sample base golden set.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evaluation/golden_dataset_v3_multi_turn.json"),
        help="Output path for the multi-turn JSON.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip (rather than fail on) conversations referencing unknown sample IDs.",
    )
    args = parser.parse_args()

    out = build_multi_turn_dataset(args.base, args.out, skip_invalid=args.skip_invalid)
    print(f"Wrote {args.out}")
    print(f"  conversations: {out['total_conversations']}")
    print(f"  turns:         {out['total_turns']}")
    print(f"  patterns:      {out['pattern_counts']}")
    print(f"  modes:         {out['answer_mode_counts']}")
    if out["skipped_conversations"]:
        print(f"  skipped:       {len(out['skipped_conversations'])}")
        for s in out["skipped_conversations"]:
            print(f"    - {s}")


if __name__ == "__main__":
    main()
