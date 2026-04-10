"""
Multi-Turn Conversation Evaluation.

Evaluates the new multi-turn capabilities added to the agent:
1. Query resolution quality (co-reference, ellipsis)
2. Context preservation across turns
3. Topic switch detection
4. Memory retrieval from long-term store
5. Entity tracking consistency
6. Turn-to-turn coherence (LLM judge)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from online_pipeline.llm_client import llm_chat_json
    from online_pipeline.agent_memory import MemoryManager, extract_entities
    from online_pipeline.session_manager import SessionManager
except ImportError:
    llm_chat_json = None
    MemoryManager = None
    SessionManager = None
    extract_entities = None


# ---------------------------------------------------------------------------
# LLM judge prompts
# ---------------------------------------------------------------------------

COHERENCE_JUDGE_PROMPT = """\
You are evaluating coherence across turns in a multi-turn conversation
about port operations.

## Conversation
{conversation}

## Evaluation Criteria (score 1-5 each)
1. **consistency**: Are statements across turns consistent (no self-contradiction)?
2. **context_use**: Does the assistant properly use context from earlier turns when needed?
3. **reference_resolution**: Are pronouns and references ("that", "it", "same") resolved correctly?
4. **topic_handling**: When topic changes, is the transition clean (old context not wrongly reused)?

Return ONLY JSON:
```json
{{
  "consistency": 1-5,
  "context_use": 1-5,
  "reference_resolution": 1-5,
  "topic_handling": 1-5,
  "rationale": "<brief>"
}}
```
"""


# ---------------------------------------------------------------------------
# Resolution quality evaluation
# ---------------------------------------------------------------------------

def evaluate_query_resolution(
    raw_query: str,
    resolved_query: str,
    expected_contains: List[str],
    expected_not_contains: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Check if a resolved query properly includes context.

    Returns:
        {
          "contains_hits": fraction of expected_contains found,
          "not_contains_hits": fraction of should-not-contain NOT found (1.0 = perfect),
          "was_rewritten": bool,
          "length_ratio": resolved_len / raw_len
        }
    """
    resolved_lower = resolved_query.lower()
    raw_lower = raw_query.lower()

    contains_hits = 0
    for kw in expected_contains:
        if kw.lower() in resolved_lower:
            contains_hits += 1
    contains_score = contains_hits / len(expected_contains) if expected_contains else 1.0

    not_contains_score = 1.0
    if expected_not_contains:
        bad_hits = sum(1 for kw in expected_not_contains if kw.lower() in resolved_lower)
        not_contains_score = 1.0 - (bad_hits / len(expected_not_contains))

    was_rewritten = resolved_lower.strip() != raw_lower.strip()
    length_ratio = len(resolved_query) / max(len(raw_query), 1)

    return {
        "contains_hits": round(contains_score, 4),
        "not_contains_score": round(not_contains_score, 4),
        "was_rewritten": was_rewritten,
        "length_ratio": round(length_ratio, 3),
    }


# ---------------------------------------------------------------------------
# Entity tracking evaluation
# ---------------------------------------------------------------------------

def evaluate_entity_tracking(
    conversation_turns: List[Dict[str, Any]],
    tracked_entities_per_turn: List[List[str]],
) -> Dict[str, Any]:
    """
    Check if the memory system correctly tracks entities across turns.

    Args:
        conversation_turns: list of {turn_id, query, answer} from the run
        tracked_entities_per_turn: list of active_entities at each turn
    """
    if extract_entities is None:
        return {"error": "agent_memory not available"}

    total_expected = 0
    total_tracked = 0
    persistence_score = 0.0

    for i, turn in enumerate(conversation_turns):
        text = turn.get("query", "") + " " + turn.get("answer", "")
        expected_ents = set(extract_entities(text))
        total_expected += len(expected_ents)

        if i < len(tracked_entities_per_turn):
            tracked = set(tracked_entities_per_turn[i])
            total_tracked += len(expected_ents & tracked)

            # Persistence: entities from earlier turns should still be tracked
            if i > 0:
                prior_text = " ".join(
                    t.get("query", "") + " " + t.get("answer", "")
                    for t in conversation_turns[:i]
                )
                prior_ents = set(extract_entities(prior_text))
                if prior_ents:
                    persistence_score += len(prior_ents & tracked) / len(prior_ents)

    n_turns = len(conversation_turns)
    return {
        "entity_tracking_recall": round(
            total_tracked / total_expected, 4
        ) if total_expected else 1.0,
        "persistence_rate": round(
            persistence_score / max(n_turns - 1, 1), 4
        ) if n_turns > 1 else 1.0,
        "total_expected_entities": total_expected,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge coherence scoring
# ---------------------------------------------------------------------------

def coherence_judge(conversation_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Call LLM judge on a full conversation to score coherence."""
    if llm_chat_json is None:
        return {"error": "llm_client not available"}

    convo_lines = []
    for t in conversation_turns:
        convo_lines.append(f"User (turn {t.get('turn_id', '?')}): {t.get('query', '')}")
        convo_lines.append(f"Assistant: {(t.get('answer', '') or '')[:500]}")
    convo_text = "\n".join(convo_lines)

    prompt = COHERENCE_JUDGE_PROMPT.format(conversation=convo_text[:4000])
    result = llm_chat_json(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Score this conversation."},
        ],
        temperature=0.0,
        timeout=60,
    )
    if not isinstance(result, dict):
        return {"error": "judge failed"}
    return {
        "consistency": int(result.get("consistency", 0)),
        "context_use": int(result.get("context_use", 0)),
        "reference_resolution": int(result.get("reference_resolution", 0)),
        "topic_handling": int(result.get("topic_handling", 0)),
        "rationale": result.get("rationale", ""),
    }


# ---------------------------------------------------------------------------
# End-to-end multi-turn evaluation driver
# ---------------------------------------------------------------------------

@dataclass
class MultiTurnMetrics:
    # Resolution
    avg_resolution_contains: float = 0.0
    avg_resolution_not_contains: float = 0.0
    resolution_rewrite_rate: float = 0.0

    # Entity tracking
    avg_entity_tracking_recall: float = 0.0
    avg_persistence_rate: float = 0.0

    # Coherence (LLM judge)
    avg_consistency: float = 0.0
    avg_context_use: float = 0.0
    avg_reference_resolution: float = 0.0
    avg_topic_handling: float = 0.0

    # Memory retrieval
    long_term_hit_rate: float = 0.0

    conversations_evaluated: int = 0
    turns_evaluated: int = 0
    conversations_judged: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": {
                "avg_contains_score": round(self.avg_resolution_contains, 4),
                "avg_not_contains_score": round(self.avg_resolution_not_contains, 4),
                "rewrite_rate": round(self.resolution_rewrite_rate, 4),
            },
            "entity_tracking": {
                "recall": round(self.avg_entity_tracking_recall, 4),
                "persistence_rate": round(self.avg_persistence_rate, 4),
            },
            "coherence_llm_judge": {
                "consistency": round(self.avg_consistency, 3),
                "context_use": round(self.avg_context_use, 3),
                "reference_resolution": round(self.avg_reference_resolution, 3),
                "topic_handling": round(self.avg_topic_handling, 3),
            },
            "memory": {
                "long_term_hit_rate": round(self.long_term_hit_rate, 4),
            },
            "counts": {
                "conversations_evaluated": self.conversations_evaluated,
                "turns_evaluated": self.turns_evaluated,
                "conversations_judged": self.conversations_judged,
            },
        }


def evaluate_multi_turn_conversation(
    conversation_spec: Dict[str, Any],
    conversation_run: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate a single multi-turn conversation.

    Args:
        conversation_spec: from golden dataset (turns with expected fields)
        conversation_run: actual run output containing:
            - turns: list of {turn_id, raw_query, resolved_query, answer,
                              tracked_entities, tool_results_summary}
            - session_id: str
            - long_term_hits: int (optional)

    Returns:
        Per-conversation metrics dict.
    """
    spec_turns = conversation_spec.get("turns", [])
    run_turns = conversation_run.get("turns", [])

    resolutions = []
    for spec_turn, run_turn in zip(spec_turns, run_turns):
        expected_contains = spec_turn.get("expected_resolved_query_contains", [])
        expected_not = spec_turn.get("expected_resolved_query_should_not_contain", [])

        if expected_contains or expected_not:
            res = evaluate_query_resolution(
                raw_query=run_turn.get("raw_query", ""),
                resolved_query=run_turn.get("resolved_query", run_turn.get("raw_query", "")),
                expected_contains=expected_contains,
                expected_not_contains=expected_not,
            )
            resolutions.append(res)

    # Entity tracking
    tracked_per_turn = [
        t.get("tracked_entities", []) for t in run_turns
    ]
    conversation_turns_for_eval = [
        {"turn_id": t.get("turn_id"), "query": t.get("raw_query", ""), "answer": t.get("answer", "")}
        for t in run_turns
    ]
    entity_eval = evaluate_entity_tracking(
        conversation_turns_for_eval, tracked_per_turn
    )

    return {
        "conversation_id": conversation_spec.get("conversation_id"),
        "turns": len(run_turns),
        "resolutions": resolutions,
        "entity_tracking": entity_eval,
    }


def aggregate_multi_turn(
    per_conversation: List[Dict[str, Any]],
    llm_judge_per_conv: Optional[List[Dict[str, Any]]] = None,
) -> MultiTurnMetrics:
    """Aggregate per-conversation results into summary metrics."""
    resolutions_flat = []
    entity_recalls = []
    persistences = []

    for conv in per_conversation:
        resolutions_flat.extend(conv.get("resolutions", []))
        et = conv.get("entity_tracking", {})
        if "entity_tracking_recall" in et:
            entity_recalls.append(et["entity_tracking_recall"])
        if "persistence_rate" in et:
            persistences.append(et["persistence_rate"])

    n_resolutions = len(resolutions_flat) or 1
    avg_contains = sum(r["contains_hits"] for r in resolutions_flat) / n_resolutions
    avg_not_contains = sum(r["not_contains_score"] for r in resolutions_flat) / n_resolutions
    rewrite_rate = sum(1 for r in resolutions_flat if r["was_rewritten"]) / n_resolutions

    avg_et_recall = sum(entity_recalls) / len(entity_recalls) if entity_recalls else 0.0
    avg_persistence = sum(persistences) / len(persistences) if persistences else 0.0

    judges = llm_judge_per_conv or []
    valid_judges = [j for j in judges if "error" not in j]
    n_judges = len(valid_judges) or 1
    avg_consistency = sum(j["consistency"] for j in valid_judges) / n_judges
    avg_context_use = sum(j["context_use"] for j in valid_judges) / n_judges
    avg_ref_res = sum(j["reference_resolution"] for j in valid_judges) / n_judges
    avg_topic = sum(j["topic_handling"] for j in valid_judges) / n_judges

    total_turns = sum(c.get("turns", 0) for c in per_conversation)

    return MultiTurnMetrics(
        avg_resolution_contains=avg_contains,
        avg_resolution_not_contains=avg_not_contains,
        resolution_rewrite_rate=rewrite_rate,
        avg_entity_tracking_recall=avg_et_recall,
        avg_persistence_rate=avg_persistence,
        avg_consistency=avg_consistency,
        avg_context_use=avg_context_use,
        avg_reference_resolution=avg_ref_res,
        avg_topic_handling=avg_topic,
        conversations_evaluated=len(per_conversation),
        turns_evaluated=total_turns,
        conversations_judged=len(valid_judges),
    )


def print_multi_turn_report(metrics: MultiTurnMetrics) -> None:
    print("\n" + "=" * 70)
    print("  MULTI-TURN EVALUATION")
    print("=" * 70)
    d = metrics.to_dict()
    print(f"  Conversations: {d['counts']['conversations_evaluated']}")
    print(f"  Total turns:   {d['counts']['turns_evaluated']}")

    print("\n  Query Resolution:")
    r = d["resolution"]
    print(f"    Contains score:       {r['avg_contains_score']:.2%}")
    print(f"    Not-contains score:   {r['avg_not_contains_score']:.2%}")
    print(f"    Rewrite rate:         {r['rewrite_rate']:.2%}")

    print("\n  Entity Tracking:")
    e = d["entity_tracking"]
    print(f"    Recall:               {e['recall']:.2%}")
    print(f"    Persistence rate:     {e['persistence_rate']:.2%}")

    if d["counts"]["conversations_judged"] > 0:
        print("\n  LLM-judge coherence (1-5):")
        c = d["coherence_llm_judge"]
        print(f"    Consistency:          {c['consistency']:.2f}")
        print(f"    Context use:          {c['context_use']:.2f}")
        print(f"    Reference resolution: {c['reference_resolution']:.2f}")
        print(f"    Topic handling:       {c['topic_handling']:.2f}")
