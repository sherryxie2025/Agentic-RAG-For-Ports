"""
Multi-turn evaluation driver for the agentic-RAG DAG.

Reads `evaluation/golden_dataset_v3_multi_turn.json`, runs each conversation
through `build_langgraph_workflow_with_memory`, and scores it with the
metrics in `evaluation/agent/eval_memory.py`.

Per-turn it also reuses the single-turn metric modules (`eval_routing`,
`eval_retrieval`, `eval_answer_quality`, `eval_guardrails`) by feeding
them the per-turn record alongside the inherited golden fields from the
linked base sample (`derived_from_sample_id`).

Run from project root:
    python evaluation/run_multi_turn_evaluation.py [--skip-llm-judge]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "evaluation"
AGENT_EVAL_DIR = EVAL_ROOT / "agent"
REPORTS_DIR = AGENT_EVAL_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(AGENT_EVAL_DIR))
os.chdir(str(PROJECT_ROOT))

from online_pipeline import (
    MemoryManager,
    build_langgraph_workflow_with_memory,
)
from online_pipeline.conversation_memory import extract_entities

from eval_memory import (
    coref_resolution_score,
    memory_recall_at_k,
    entity_persistence,
    topic_shift_detected,
    context_token_overhead,
    llm_judge_memory_precision,
    llm_judge_faithfulness,
    aggregate,
    print_memory_report,
)


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

def run_conversation(
    workflow,
    memory_manager: MemoryManager,
    conv: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one multi-turn conversation through the DAG. Returns per-turn records."""
    sid = memory_manager.start_session()
    print(f"\n  [{conv['conversation_id']}] {conv['pattern']} ({len(conv['turns'])} turns) sid={sid}")

    per_turn_outputs: List[Dict[str, Any]] = []
    accumulating_entities_per_turn: List[List[str]] = []

    for turn_idx, turn_spec in enumerate(conv["turns"], start=1):
        raw_query = turn_spec["raw_query"]

        # Resolve follow-up + build context (timed)
        t0 = time.time()
        resolved, was_rewritten = memory_manager.resolve_followup(sid, raw_query)
        resolve_ms = (time.time() - t0) * 1000.0

        t0 = time.time()
        memory_context = memory_manager.build_context(sid, resolved, max_chars=3000)
        build_ctx_ms = (time.time() - t0) * 1000.0

        # Record user turn
        memory_manager.record_user_turn(sid, raw_query)

        # Invoke the DAG
        state_in = {
            "session_id": sid,
            "raw_query": raw_query,
            "user_query": resolved,
            "resolved_query": resolved,
            "memory_context": memory_context if memory_context else None,
            "coref_was_rewritten": was_rewritten,
            "reasoning_trace": [],
            "warnings": [],
        }
        try:
            state_out = workflow.invoke(state_in)
        except Exception as e:                          # pragma: no cover — surfaced
            print(f"    ! turn {turn_idx} crashed: {type(e).__name__}: {e}")
            state_out = {"final_answer": {"answer": "", "sources_used": []}}

        # Record assistant turn (updates memory for the NEXT turn)
        memory_manager.record_assistant_turn(sid, state_out)

        # Snapshot all 3 short-term layers + entities AFTER recording
        # the assistant turn — this lets post-hoc analysis see how the
        # Phase-C key_facts layer grew across a conversation.
        stm = memory_manager.get_session(sid)
        active_now = list(stm.active_entities.keys())
        summaries_snapshot = [
            {
                "summary_text": s.get("summary_text", ""),
                "turns_covered": s.get("turns_covered", []),
                "key_facts": s.get("key_facts", []),
            }
            for s in stm.summaries
        ]
        key_facts_snapshot = [
            {
                "fact": f.get("fact", ""),
                "from_turn_ids": f.get("from_turn_ids", []),
                "entities": f.get("entities", []),
            }
            for f in stm.key_facts
        ]
        accumulating_entities_per_turn.append(extract_entities(raw_query))

        per_turn_outputs.append({
            "turn_id": turn_idx,
            "turn_spec": turn_spec,
            "raw_query": raw_query,
            "resolved_query": resolved,
            "was_rewritten": was_rewritten,
            "memory_context": memory_context,
            "memory_context_chars": len(memory_context or ""),
            "resolve_latency_ms": resolve_ms,
            "build_ctx_latency_ms": build_ctx_ms,
            "active_entities_after_turn": active_now,
            "prior_entities_per_turn": list(accumulating_entities_per_turn[:-1]),
            # Phase-C observability
            "raw_turns_count_after": len(stm.turns),
            "summaries_after_turn": summaries_snapshot,
            "key_facts_after_turn": key_facts_snapshot,
            "final_answer": (state_out.get("final_answer") or {}).get("answer", ""),
            "sources_used": (state_out.get("final_answer") or {}).get("sources_used", []),
            "router_decision": state_out.get("router_decision"),
            "expected_sources": turn_spec.get("expected_sources"),
            "answer_mode": turn_spec.get("answer_mode"),
        })

    # End session (writes long-term summary)
    memory_manager.end_session(sid)

    return {
        "conversation_id": conv["conversation_id"],
        "pattern": conv["pattern"],
        "session_id": sid,
        "turns": per_turn_outputs,
    }


# ---------------------------------------------------------------------------
# Per-turn scoring
# ---------------------------------------------------------------------------

def score_turn(
    turn_record: Dict[str, Any],
    full_conversation_so_far: List[Dict[str, str]],
    use_llm_judge: bool,
) -> Dict[str, Any]:
    spec = turn_record["turn_spec"]

    coref = coref_resolution_score(
        raw_query=turn_record["raw_query"],
        resolved_query=turn_record["resolved_query"],
        expected_contains=spec.get("expected_resolved_query_contains"),
        expected_not_contains=spec.get("expected_resolved_query_should_not_contain"),
    )

    expected_recall = spec.get("expected_memory_recall")
    recall_score = memory_recall_at_k(expected_recall, turn_record["memory_context"] or "")
    recall_age = (
        turn_record["turn_id"] - expected_recall["from_turn"]
        if expected_recall and "from_turn" in expected_recall
        else None
    )

    persistence = (
        entity_persistence(
            prior_turn_entities=turn_record["prior_entities_per_turn"],
            active_entities_now=turn_record["active_entities_after_turn"],
        )
        if turn_record["prior_entities_per_turn"]
        else None
    )

    topic = topic_shift_detected(
        coref_was_rewritten=turn_record["was_rewritten"],
        expected_not_contains=spec.get("expected_resolved_query_should_not_contain"),
        resolved_query=turn_record["resolved_query"],
    )

    overhead = context_token_overhead(
        memory_context=turn_record["memory_context"] or "",
        base_query=turn_record["raw_query"],
    )

    judge_prec = judge_cons = judge_attr = None
    if use_llm_judge and turn_record["memory_context"]:
        prec_out = llm_judge_memory_precision(
            memory_context=turn_record["memory_context"],
            current_query=turn_record["raw_query"],
        )
        if prec_out:
            judge_prec = prec_out["precision"]
        if turn_record["final_answer"] and len(full_conversation_so_far) > 1:
            faith_out = llm_judge_faithfulness(
                conversation=full_conversation_so_far,
                current_answer=turn_record["final_answer"],
            )
            if faith_out:
                judge_cons = faith_out["consistency"]
                judge_attr = faith_out["attribution"]

    return {
        "turn_id": turn_record["turn_id"],
        "evaluation_focus": spec.get("evaluation_focus"),
        "coref": coref,
        "memory_recall": recall_score,
        "recall_age_in_turns": recall_age,
        "entity_persistence": persistence,
        "topic_switch": topic,
        "context_token_overhead": overhead,
        "resolve_latency_ms": turn_record["resolve_latency_ms"],
        "build_ctx_latency_ms": turn_record["build_ctx_latency_ms"],
        "judge_precision": judge_prec,
        "judge_consistency": judge_cons,
        "judge_attribution": judge_attr,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=EVAL_ROOT / "golden_dataset_v3_multi_turn.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of conversations.")
    parser.add_argument("--skip-llm-judge", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "rag_v3_multi_turn.json")
    args = parser.parse_args()

    print(f"Loading multi-turn dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)
    convs = ds["conversations"]
    if args.limit:
        convs = convs[: args.limit]
    print(f"  {len(convs)} conversations, {sum(len(c['turns']) for c in convs)} turns")

    print("\nBuilding agentic-RAG DAG with memory...")
    # max_raw_turns=4 ensures MT3_005 (6 turns) triggers short-term
    # summarisation so Phase-C key_facts extraction is actually exercised.
    # Production default is 10.
    mgr = MemoryManager(project_root=PROJECT_ROOT, max_raw_turns=4)
    workflow = build_langgraph_workflow_with_memory(
        project_root=PROJECT_ROOT,
        memory_manager=mgr,
    )

    all_runs: List[Dict[str, Any]] = []
    all_scored_turns: List[Dict[str, Any]] = []

    for conv in convs:
        run = run_conversation(workflow, mgr, conv)
        all_runs.append(run)

        # Build the rolling conversation transcript for the LLM judge
        transcript: List[Dict[str, str]] = []
        for turn_record in run["turns"]:
            transcript.append({"role": "user", "content": turn_record["raw_query"]})
            scored = score_turn(
                turn_record=turn_record,
                full_conversation_so_far=transcript,
                use_llm_judge=not args.skip_llm_judge,
            )
            scored["conversation_id"] = run["conversation_id"]
            scored["pattern"] = run["pattern"]
            all_scored_turns.append(scored)
            transcript.append({"role": "assistant", "content": turn_record["final_answer"]})

    metrics = aggregate(per_turn_records=all_scored_turns, n_conversations=len(all_runs))
    print_memory_report(metrics)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset.name),
        "evaluated_system": "agentic-rag DAG with conversation_memory",
        "skip_llm_judge": args.skip_llm_judge,
        "metrics": metrics.to_dict(),
        "per_turn": all_scored_turns,
        "raw_runs": all_runs,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nWrote {args.output}")
    mgr.close()


if __name__ == "__main__":
    main()
