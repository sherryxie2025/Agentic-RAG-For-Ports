"""
Memory evaluation for the multi-turn agentic-RAG DAG.

Industry-aligned metrics (mapped to where they come from):

| Metric                      | Inspired by                              | What it measures                                                                            |
|-----------------------------|------------------------------------------|---------------------------------------------------------------------------------------------|
| coref_resolution_contains   | LangChain conv-eval, MT-Bench-Conv       | After follow-up rewrite, did the standalone query include the expected referents?            |
| coref_resolution_exclusion  | LongChat / TopicSwitch                   | After a topic switch, did the rewrite NOT carry over old (now-irrelevant) entities?          |
| memory_recall@k             | MemGPT, ChatRAG-Bench                    | Of the gold "must-recall" facts at turn N, how many appear in the prompt's memory_context?  |
| memory_precision            | RAGAS context_precision (LLM judge)      | Of the items injected via memory_context, how many are actually relevant to the current turn?|
| answer_faithfulness_to_mem  | RAGAS faithfulness                       | Are answer claims that reference earlier turns consistent with the recorded turn content?    |
| temporal_recall_decay       | LongMemEval                              | Recall@k for facts vs. their age in turns (shows the forgetting curve).                      |
| entity_persistence          | DialDoc, Multi-Doc QA                    | Fraction of entities mentioned in turn 1..N-1 still in active_entities at turn N.            |
| topic_shift_detected_rate   | TIAGE, TopiOCQA                          | Heuristic: did the system NOT carry old entities when the gold says topic switched?          |
| cross_session_hit_rate      | LangChain VectorStoreRetrieverMemory     | When a new session asks about a prior session's topic, does long-term retrieve return it?    |
| context_token_overhead      | MemGPT efficiency table                  | Tokens added by memory injection / tokens of base prompt (rough char ratio used as proxy).   |
| latency_overhead_ms         | Production observability                 | resolve_followup + build_context wall time per turn.                                          |

The LLM-judge metrics (memory_precision, answer_faithfulness_to_mem) are
optional — pass `--skip-llm-judge` to skip them if you only want the
deterministic metrics.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from online_pipeline.llm_client import llm_chat_json
    from online_pipeline.conversation_memory import extract_entities
except ImportError:                                    # pragma: no cover
    llm_chat_json = None
    extract_entities = None


# ---------------------------------------------------------------------------
# Deterministic metrics
# ---------------------------------------------------------------------------

def coref_resolution_score(
    raw_query: str,
    resolved_query: str,
    expected_contains: Optional[List[str]],
    expected_not_contains: Optional[List[str]],
) -> Dict[str, float]:
    """
    Returns:
      contains_score    fraction of expected terms found in the resolved query
      exclusion_score   fraction of forbidden terms NOT found (1.0 = perfect)
      was_rewritten     0/1 flag
      length_ratio      len(resolved) / len(raw)
    """
    rl = (resolved_query or "").lower()
    rw = (raw_query or "").lower()

    if expected_contains:
        hits = sum(1 for kw in expected_contains if kw.lower() in rl)
        contains = hits / len(expected_contains)
    else:
        contains = 1.0

    if expected_not_contains:
        bad = sum(1 for kw in expected_not_contains if kw.lower() in rl)
        exclusion = 1.0 - bad / len(expected_not_contains)
    else:
        exclusion = 1.0

    return {
        "contains_score": round(contains, 4),
        "exclusion_score": round(exclusion, 4),
        "was_rewritten": 1.0 if rl.strip() != rw.strip() else 0.0,
        "length_ratio": round(len(resolved_query or "") / max(len(raw_query or ""), 1), 3),
    }


def memory_recall_at_k(
    expected_recall: Optional[Dict[str, Any]],
    memory_context: str,
) -> Optional[float]:
    """
    For turns annotated with expected_memory_recall = {from_turn, key_fact},
    check whether the key_fact appears (case-insensitive substring) in the
    injected memory_context.
    """
    if not expected_recall:
        return None
    fact = (expected_recall.get("key_fact") or "").lower()
    if not fact:
        return None
    return 1.0 if fact in (memory_context or "").lower() else 0.0


def entity_persistence(
    prior_turn_entities: List[List[str]],
    active_entities_now: List[str],
) -> float:
    """
    What fraction of entities that appeared in earlier turns are still in the
    short-term active_entities set at the current turn.
    """
    prior = set()
    for ents in prior_turn_entities:
        prior.update(ents or [])
    if not prior:
        return 1.0
    return round(len(prior & set(active_entities_now or [])) / len(prior), 4)


def topic_shift_detected(
    coref_was_rewritten: bool,
    expected_not_contains: Optional[List[str]],
    resolved_query: str,
) -> Optional[float]:
    """
    Pass when (a) the gold flags this turn as a topic switch (via
    expected_resolved_query_should_not_contain) AND (b) none of the
    forbidden terms appear in the resolved query.
    Returns None if the turn is not a topic-switch case.
    """
    if not expected_not_contains:
        return None
    rl = (resolved_query or "").lower()
    leaked = any(kw.lower() in rl for kw in expected_not_contains)
    return 0.0 if leaked else 1.0


def temporal_recall_buckets(
    per_turn_recall: List[Tuple[int, Optional[float]]],
) -> Dict[str, float]:
    """
    Group recall measurements by the *age* of the fact being recalled (i.e.,
    `current_turn - from_turn`). Returns avg recall per age bucket. This is
    the shape of the forgetting curve.

    Input: list of (age_in_turns, recall_score).
    """
    by_age: Dict[int, List[float]] = {}
    for age, score in per_turn_recall:
        if score is None:
            continue
        by_age.setdefault(age, []).append(score)
    return {
        f"age_{age}_turns": round(sum(scores) / len(scores), 4)
        for age, scores in sorted(by_age.items())
    }


def context_token_overhead(memory_context: str, base_query: str) -> float:
    base = max(len(base_query or ""), 1)
    return round(len(memory_context or "") / base, 3)


# ---------------------------------------------------------------------------
# LLM-judge metrics (optional)
# ---------------------------------------------------------------------------

_PRECISION_PROMPT = """\
You are evaluating a memory-augmented port-operations assistant.

The assistant's MEMORY CONTEXT injected into the prompt for this turn was:
---
{memory_context}
---

The current user question is:
"{current_query}"

Score (1-5) how RELEVANT the memory context is to answering the current
question (memory precision):
- 5: every line in memory is needed to answer this question
- 3: about half the memory is relevant
- 1: memory is unrelated to the question

Return ONLY JSON: {{"precision": 1-5, "rationale": "<brief>"}}"""


_FAITHFULNESS_PROMPT = """\
You are evaluating whether a multi-turn assistant's answer is FAITHFUL to
its conversation memory.

CONVERSATION SO FAR:
{conversation}

CURRENT TURN'S ANSWER:
"{answer}"

Score (1-5):
- consistency: does the current answer contradict any earlier turn? (5 = no
  contradictions, 1 = direct contradiction)
- attribution: when the answer references earlier facts, does it represent
  them correctly? (5 = perfect attribution, 1 = misattribution / made up)

Return ONLY JSON: {{"consistency": 1-5, "attribution": 1-5, "rationale": "<brief>"}}"""


def llm_judge_memory_precision(
    memory_context: str,
    current_query: str,
) -> Optional[Dict[str, Any]]:
    if llm_chat_json is None or not memory_context:
        return None
    try:
        out = llm_chat_json(
            messages=[
                {"role": "system", "content": _PRECISION_PROMPT.format(
                    memory_context=memory_context[:2000],
                    current_query=current_query[:400],
                )},
                {"role": "user", "content": "Score it."},
            ],
            temperature=0.0,
            timeout=45,
        )
        if isinstance(out, dict) and "precision" in out:
            return {"precision": int(out["precision"]), "rationale": out.get("rationale", "")}
    except Exception:                                  # pragma: no cover
        pass
    return None


def llm_judge_faithfulness(
    conversation: List[Dict[str, Any]],
    current_answer: str,
) -> Optional[Dict[str, Any]]:
    if llm_chat_json is None or not conversation or not current_answer:
        return None
    convo_lines = []
    for t in conversation:
        role = t.get("role", "?")
        content = (t.get("content") or "")[:300]
        convo_lines.append(f"{role}: {content}")
    try:
        out = llm_chat_json(
            messages=[
                {"role": "system", "content": _FAITHFULNESS_PROMPT.format(
                    conversation="\n".join(convo_lines)[:3000],
                    answer=(current_answer or "")[:600],
                )},
                {"role": "user", "content": "Score it."},
            ],
            temperature=0.0,
            timeout=60,
        )
        if isinstance(out, dict):
            return {
                "consistency": int(out.get("consistency", 0)),
                "attribution": int(out.get("attribution", 0)),
                "rationale": out.get("rationale", ""),
            }
    except Exception:                                  # pragma: no cover
        pass
    return None


# ---------------------------------------------------------------------------
# Cross-session hit rate
# ---------------------------------------------------------------------------

def cross_session_hit_rate(
    new_session_queries: List[str],
    long_term_retrieve_fn: Callable[[str, int], List[Dict[str, Any]]],
    expected_session_ids: List[str],
    top_k: int = 3,
) -> float:
    """
    Given a list of "new-session" queries that should hit memories from the
    listed prior sessions, compute the fraction of queries whose top-k LT
    retrieval contains an entry whose `session_id` is in `expected_session_ids`.
    """
    if not new_session_queries:
        return 0.0
    expected = set(expected_session_ids)
    hits = 0
    for q in new_session_queries:
        results = long_term_retrieve_fn(q, top_k) or []
        if any((r.get("session_id") or "") in expected for r in results):
            hits += 1
    return round(hits / len(new_session_queries), 4)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

@dataclass
class MemoryMetrics:
    # Co-reference quality
    avg_coref_contains: float = 0.0
    avg_coref_exclusion: float = 0.0
    coref_rewrite_rate: float = 0.0

    # Recall family
    overall_memory_recall: float = 0.0
    forgetting_curve: Dict[str, float] = field(default_factory=dict)

    # Active-entity tracking
    avg_entity_persistence: float = 0.0

    # Topic-switch correctness
    topic_shift_correct_rate: float = 0.0

    # Efficiency
    avg_context_token_overhead: float = 0.0
    avg_resolve_latency_ms: float = 0.0
    avg_build_ctx_latency_ms: float = 0.0

    # LLM-judge (optional)
    avg_memory_precision: Optional[float] = None
    avg_faithfulness_consistency: Optional[float] = None
    avg_faithfulness_attribution: Optional[float] = None

    # Cross-session
    cross_session_hit_rate: Optional[float] = None

    # Counts
    conversations: int = 0
    turns: int = 0
    judged_turns: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "co_reference": {
                "contains_score": round(self.avg_coref_contains, 4),
                "exclusion_score": round(self.avg_coref_exclusion, 4),
                "rewrite_rate": round(self.coref_rewrite_rate, 4),
            },
            "recall": {
                "overall_memory_recall_at_k": round(self.overall_memory_recall, 4),
                "forgetting_curve_by_age": self.forgetting_curve,
            },
            "entities": {
                "avg_persistence_rate": round(self.avg_entity_persistence, 4),
            },
            "topic_switch": {
                "correct_no_carry_rate": round(self.topic_shift_correct_rate, 4),
            },
            "efficiency": {
                "avg_context_token_overhead": round(self.avg_context_token_overhead, 3),
                "avg_resolve_followup_ms": round(self.avg_resolve_latency_ms, 1),
                "avg_build_context_ms": round(self.avg_build_ctx_latency_ms, 1),
            },
            "counts": {
                "conversations": self.conversations,
                "turns": self.turns,
                "llm_judged_turns": self.judged_turns,
            },
        }
        if self.avg_memory_precision is not None:
            d["llm_judge"] = {
                "memory_precision_1to5": round(self.avg_memory_precision, 3),
                "faithfulness_consistency_1to5": round(
                    self.avg_faithfulness_consistency or 0.0, 3
                ),
                "faithfulness_attribution_1to5": round(
                    self.avg_faithfulness_attribution or 0.0, 3
                ),
            }
        if self.cross_session_hit_rate is not None:
            d["cross_session"] = {"hit_rate": round(self.cross_session_hit_rate, 4)}
        return d


def aggregate(
    per_turn_records: List[Dict[str, Any]],
    n_conversations: int,
) -> MemoryMetrics:
    """
    `per_turn_records` schema (one entry per turn evaluated):
      {
        "coref": {contains_score, exclusion_score, was_rewritten, length_ratio},
        "memory_recall": float | None,
        "recall_age_in_turns": int | None,
        "entity_persistence": float | None,
        "topic_switch": float | None,
        "context_token_overhead": float,
        "resolve_latency_ms": float,
        "build_ctx_latency_ms": float,
        "judge_precision": int | None,
        "judge_consistency": int | None,
        "judge_attribution": int | None,
      }
    """
    n = max(len(per_turn_records), 1)

    contains = sum(r["coref"]["contains_score"] for r in per_turn_records) / n
    exclusion = sum(r["coref"]["exclusion_score"] for r in per_turn_records) / n
    rewrite = sum(r["coref"]["was_rewritten"] for r in per_turn_records) / n

    recalls = [r for r in per_turn_records if r.get("memory_recall") is not None]
    overall_recall = (
        sum(r["memory_recall"] for r in recalls) / len(recalls) if recalls else 0.0
    )
    forgetting = temporal_recall_buckets(
        [(r["recall_age_in_turns"], r["memory_recall"]) for r in recalls
         if r.get("recall_age_in_turns") is not None]
    )

    pers = [r["entity_persistence"] for r in per_turn_records
            if r.get("entity_persistence") is not None]
    avg_pers = sum(pers) / len(pers) if pers else 0.0

    topic = [r["topic_switch"] for r in per_turn_records
             if r.get("topic_switch") is not None]
    avg_topic = sum(topic) / len(topic) if topic else 0.0

    overhead = sum(r.get("context_token_overhead", 0.0) for r in per_turn_records) / n
    rl = sum(r.get("resolve_latency_ms", 0.0) for r in per_turn_records) / n
    bl = sum(r.get("build_ctx_latency_ms", 0.0) for r in per_turn_records) / n

    # Judge subfields are independent — precision judge can succeed while
    # faithfulness judge times out or is skipped (first turn has no history
    # to judge against). Filter each subfield separately.
    precs = [r["judge_precision"] for r in per_turn_records
             if r.get("judge_precision") is not None]
    conss = [r["judge_consistency"] for r in per_turn_records
             if r.get("judge_consistency") is not None]
    attrs = [r["judge_attribution"] for r in per_turn_records
             if r.get("judge_attribution") is not None]
    prec = sum(precs) / len(precs) if precs else None
    cons = sum(conss) / len(conss) if conss else None
    attr = sum(attrs) / len(attrs) if attrs else None
    judges = precs  # for count reporting

    return MemoryMetrics(
        avg_coref_contains=contains,
        avg_coref_exclusion=exclusion,
        coref_rewrite_rate=rewrite,
        overall_memory_recall=overall_recall,
        forgetting_curve=forgetting,
        avg_entity_persistence=avg_pers,
        topic_shift_correct_rate=avg_topic,
        avg_context_token_overhead=overhead,
        avg_resolve_latency_ms=rl,
        avg_build_ctx_latency_ms=bl,
        avg_memory_precision=prec,
        avg_faithfulness_consistency=cons,
        avg_faithfulness_attribution=attr,
        conversations=n_conversations,
        turns=len(per_turn_records),
        judged_turns=len(judges),
    )


def print_memory_report(m: MemoryMetrics) -> None:
    d = m.to_dict()
    print("\n" + "=" * 70)
    print("  MEMORY EVALUATION (multi-turn)")
    print("=" * 70)
    print(f"  Conversations: {d['counts']['conversations']}  Turns: {d['counts']['turns']}")
    print("\n  Co-reference resolution:")
    cr = d["co_reference"]
    print(f"    contains_score (carry expected refs):  {cr['contains_score']:.2%}")
    print(f"    exclusion_score (drop wrong context):  {cr['exclusion_score']:.2%}")
    print(f"    rewrite_rate:                          {cr['rewrite_rate']:.2%}")
    print("\n  Recall:")
    rc = d["recall"]
    print(f"    overall memory_recall@k:               {rc['overall_memory_recall_at_k']:.2%}")
    if rc["forgetting_curve_by_age"]:
        print(f"    forgetting curve by fact age:")
        for k, v in rc["forgetting_curve_by_age"].items():
            print(f"      {k}: {v:.2%}")
    print("\n  Entities & topic switch:")
    print(f"    entity persistence:                    {d['entities']['avg_persistence_rate']:.2%}")
    print(f"    topic-switch correct (no carry):       {d['topic_switch']['correct_no_carry_rate']:.2%}")
    print("\n  Efficiency:")
    eff = d["efficiency"]
    print(f"    context overhead (chars / query):      {eff['avg_context_token_overhead']}")
    print(f"    resolve_followup latency:              {eff['avg_resolve_followup_ms']} ms")
    print(f"    build_context latency:                 {eff['avg_build_context_ms']} ms")
    if "llm_judge" in d:
        j = d["llm_judge"]
        print(f"\n  LLM judge ({d['counts']['llm_judged_turns']} turns):")
        print(f"    memory_precision (1-5):                {j['memory_precision_1to5']:.2f}")
        print(f"    faithfulness consistency (1-5):        {j['faithfulness_consistency_1to5']:.2f}")
        print(f"    faithfulness attribution (1-5):        {j['faithfulness_attribution_1to5']:.2f}")
    if "cross_session" in d:
        print(f"\n  Cross-session hit rate:                  {d['cross_session']['hit_rate']:.2%}")
