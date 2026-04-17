"""
Cross-session evaluation driver for long-term memory retrieval.

Dataset: `golden_dataset_v3_cross_session.json` (sessions-per-conversation
format). Each conversation = sequence of sessions; session N writes its
summary + embeddings to the DuckDB long-term store on `end_session()`;
session N+1 is a fresh session whose queries should retrieve session-N
content via long-term retrieve.

Metrics (in addition to the multi-turn ones):
  - `cross_session_hit_rate`:
        for turns tagged `expected_cross_session_hit=True`, did the memory
        context actually contain content from a prior session?
  - `cross_session_leak_rate`:
        for turns tagged `expected_cross_session_hit=False`, did we wrongly
        surface prior-session content? (negative test)
  - `expected_session_recall_rate`:
        when the gold names a specific `expected_from_session_order`, did
        we hit THAT session's content specifically?

Usage:
    python evaluation/run_cross_session_evaluation.py \\
        [--no-embeddings] [--weights "cos=0.55,entity=0.25,..."] \\
        --output evaluation/agent/reports/rag_v3_cross_session_<tag>.json
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


# ---------------------------------------------------------------------------
# Weight-string parser
# ---------------------------------------------------------------------------

def parse_weights(text: Optional[str]) -> Optional[Dict[str, float]]:
    """Parse `--weights "cos=0.55,entity=0.25,..."` into dict."""
    if not text:
        return None
    out = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=")
        out[k.strip()] = float(v.strip())
    return out


# ---------------------------------------------------------------------------
# Conversation runner
# ---------------------------------------------------------------------------

def run_conversation(
    workflow,
    memory_manager: MemoryManager,
    conv: Dict[str, Any],
) -> Dict[str, Any]:
    """Run all sessions of one cross-session conversation sequentially."""
    print(f"\n  [{conv['conversation_id']}] {conv['pattern']} "
          f"({len(conv['sessions'])} sessions)")

    session_records: List[Dict[str, Any]] = []
    sid_by_order: Dict[int, str] = {}

    for sess in conv["sessions"]:
        order = sess["session_order"]
        sid = memory_manager.start_session()
        sid_by_order[order] = sid
        print(f"    session {order} sid={sid}  ({len(sess['turns'])} turns)")

        per_turn_outputs: List[Dict[str, Any]] = []
        for turn_idx, turn_spec in enumerate(sess["turns"], start=1):
            raw_query = turn_spec["raw_query"]

            t0 = time.time()
            resolved, was_rewritten = memory_manager.resolve_followup(sid, raw_query)
            resolve_ms = (time.time() - t0) * 1000.0

            t0 = time.time()
            memory_context = memory_manager.build_context(sid, resolved, max_chars=3000)
            build_ctx_ms = (time.time() - t0) * 1000.0

            memory_manager.record_user_turn(sid, raw_query)

            # Also query long-term directly so we can score cross-session
            # recall independently of what build_context actually injected.
            lt_hits = memory_manager.long_term.retrieve(resolved, top_k=5)
            lt_session_ids_hit = [h.get("session_id") for h in lt_hits]

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
            except Exception as e:
                print(f"      ! turn {turn_idx} crashed: {type(e).__name__}: {e}")
                state_out = {"final_answer": {"answer": "", "sources_used": []}}

            memory_manager.record_assistant_turn(sid, state_out)

            per_turn_outputs.append({
                "turn_id": turn_idx,
                "session_order": order,
                "turn_spec": turn_spec,
                "raw_query": raw_query,
                "resolved_query": resolved,
                "was_rewritten": was_rewritten,
                "memory_context": memory_context,
                "memory_context_chars": len(memory_context or ""),
                "resolve_latency_ms": resolve_ms,
                "build_ctx_latency_ms": build_ctx_ms,
                "lt_hit_session_ids": lt_session_ids_hit,
                "lt_hit_count": len(lt_hits),
                "lt_hit_top_score": lt_hits[0]["score"] if lt_hits else 0.0,
                "final_answer": (state_out.get("final_answer") or {}).get("answer", ""),
                "sources_used": (state_out.get("final_answer") or {}).get("sources_used", []),
                "router_decision": state_out.get("router_decision"),
                "expected_sources": turn_spec.get("expected_sources"),
                "answer_mode": turn_spec.get("answer_mode"),
            })

        session_records.append({
            "session_order": order,
            "session_id": sid,
            "description": sess.get("description", ""),
            "turns": per_turn_outputs,
        })

        # End session → writes LLM summary + embedding to long-term
        memory_manager.end_session(sid)

    return {
        "conversation_id": conv["conversation_id"],
        "pattern": conv["pattern"],
        "sid_by_session_order": sid_by_order,
        "sessions": session_records,
    }


# ---------------------------------------------------------------------------
# Cross-session scoring
# ---------------------------------------------------------------------------

def score_conversation(conv_run: Dict[str, Any]) -> Dict[str, Any]:
    """Produce the cross-session-specific metrics on one conversation run."""
    sid_by_order = conv_run["sid_by_session_order"]

    hit_pos = hit_pos_correct = 0     # expected True + actually hit + session correct
    hit_pos_total = 0                  # expected True (positive cases)
    hit_neg_leaked = 0                 # expected False + wrongly leaked
    hit_neg_total = 0                  # expected False (negative cases)
    lt_top_scores_pos: List[float] = []
    lt_top_scores_neg: List[float] = []

    per_turn_scored: List[Dict[str, Any]] = []
    for sess in conv_run["sessions"]:
        for t in sess["turns"]:
            spec = t["turn_spec"]
            expected_hit = spec.get("expected_cross_session_hit")
            if expected_hit is None:
                continue
            lt_sids = t.get("lt_hit_session_ids") or []

            # The "prior session sids" are all sids BEFORE the current one.
            prior_sids = {sid_by_order[o] for o in sid_by_order
                          if o < t["session_order"]}
            hit_any_prior = any(s in prior_sids for s in lt_sids)

            if expected_hit:
                hit_pos_total += 1
                lt_top_scores_pos.append(t.get("lt_hit_top_score", 0.0))
                if hit_any_prior:
                    hit_pos += 1
                    # If expected_from_session_order specified, check specific session
                    exp_order = spec.get("expected_from_session_order")
                    if exp_order is not None:
                        if isinstance(exp_order, list):
                            exp_sids = {sid_by_order[o] for o in exp_order}
                        else:
                            exp_sids = {sid_by_order.get(exp_order)}
                        if any(s in exp_sids for s in lt_sids):
                            hit_pos_correct += 1
            else:
                hit_neg_total += 1
                lt_top_scores_neg.append(t.get("lt_hit_top_score", 0.0))
                if hit_any_prior:
                    hit_neg_leaked += 1

            per_turn_scored.append({
                "conversation_id": conv_run["conversation_id"],
                "session_order": t["session_order"],
                "turn_id": t["turn_id"],
                "evaluation_focus": spec.get("evaluation_focus"),
                "expected_hit": expected_hit,
                "actually_hit_prior_session": hit_any_prior,
                "lt_top_score": t.get("lt_hit_top_score", 0.0),
                "lt_hit_count": t.get("lt_hit_count", 0),
            })

    return {
        "conversation_id": conv_run["conversation_id"],
        "pattern": conv_run["pattern"],
        "hit_pos": hit_pos,
        "hit_pos_total": hit_pos_total,
        "hit_pos_correct_session": hit_pos_correct,
        "hit_neg_leaked": hit_neg_leaked,
        "hit_neg_total": hit_neg_total,
        "avg_lt_top_score_positive": (
            sum(lt_top_scores_pos) / len(lt_top_scores_pos)
            if lt_top_scores_pos else 0.0
        ),
        "avg_lt_top_score_negative": (
            sum(lt_top_scores_neg) / len(lt_top_scores_neg)
            if lt_top_scores_neg else 0.0
        ),
        "per_turn": per_turn_scored,
    }


def aggregate(per_conv_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate cross-session metrics across all conversations."""
    tot_pos = sum(c["hit_pos_total"] for c in per_conv_scores)
    tot_pos_hit = sum(c["hit_pos"] for c in per_conv_scores)
    tot_pos_correct = sum(c["hit_pos_correct_session"] for c in per_conv_scores)
    tot_neg = sum(c["hit_neg_total"] for c in per_conv_scores)
    tot_neg_leaked = sum(c["hit_neg_leaked"] for c in per_conv_scores)

    all_pos_scores = []
    all_neg_scores = []
    for c in per_conv_scores:
        for t in c["per_turn"]:
            (all_pos_scores if t["expected_hit"] else all_neg_scores).append(
                t["lt_top_score"]
            )

    return {
        "counts": {
            "conversations": len(per_conv_scores),
            "positive_turns": tot_pos,
            "negative_turns": tot_neg,
        },
        "cross_session_hit_rate": (
            round(tot_pos_hit / tot_pos, 4) if tot_pos else None
        ),
        "correct_session_recall_rate": (
            round(tot_pos_correct / tot_pos, 4) if tot_pos else None
        ),
        "cross_session_leak_rate": (
            round(tot_neg_leaked / tot_neg, 4) if tot_neg else None
        ),
        "avg_lt_top_score_positive": (
            round(sum(all_pos_scores) / len(all_pos_scores), 4)
            if all_pos_scores else 0.0
        ),
        "avg_lt_top_score_negative": (
            round(sum(all_neg_scores) / len(all_neg_scores), 4)
            if all_neg_scores else 0.0
        ),
        "score_gap_pos_minus_neg": (
            round(
                (sum(all_pos_scores) / len(all_pos_scores) if all_pos_scores else 0)
                - (sum(all_neg_scores) / len(all_neg_scores) if all_neg_scores else 0),
                4,
            )
        ),
    }


def print_report(agg: Dict[str, Any], config_label: str) -> None:
    print("\n" + "=" * 70)
    print(f"  CROSS-SESSION EVALUATION  [{config_label}]")
    print("=" * 70)
    c = agg["counts"]
    print(f"  Conversations: {c['conversations']}")
    print(f"  Positive turns (expect hit):  {c['positive_turns']}")
    print(f"  Negative turns (expect miss): {c['negative_turns']}")
    print()
    print(f"  cross_session_hit_rate         :  {agg['cross_session_hit_rate']}")
    print(f"  correct_session_recall_rate    :  {agg['correct_session_recall_rate']}")
    print(f"  cross_session_leak_rate (neg)  :  {agg['cross_session_leak_rate']}")
    print()
    print(f"  avg LT top-score (positive)    :  {agg['avg_lt_top_score_positive']}")
    print(f"  avg LT top-score (negative)    :  {agg['avg_lt_top_score_negative']}")
    print(f"  score gap (pos - neg)          :  {agg['score_gap_pos_minus_neg']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=EVAL_ROOT / "golden_dataset_v3_cross_session.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path,
                        default=REPORTS_DIR / "rag_v3_cross_session.json")
    parser.add_argument("--no-embeddings", action="store_true")
    parser.add_argument("--memory-db", type=Path, default=None)
    parser.add_argument(
        "--weights", type=str, default=None,
        help='Scoring weights as "cos=0.55,entity=0.25,decay=0.12,'
             'access=0.03,importance=0.05". Must use "word" instead of "cos" '
             "when --no-embeddings is set.",
    )
    parser.add_argument(
        "--config-label", type=str, default="baseline",
        help="Label printed in the report header (e.g. 'cos_only', 'no_decay')",
    )
    args = parser.parse_args()

    print(f"Loading cross-session dataset: {args.dataset}")
    with open(args.dataset, "r", encoding="utf-8") as f:
        ds = json.load(f)
    convs = ds["conversations"]
    if args.limit:
        convs = convs[: args.limit]
    print(f"  {len(convs)} conversations, "
          f"{sum(len(c['sessions']) for c in convs)} sessions, "
          f"{sum(sum(len(s['turns']) for s in c['sessions']) for c in convs)} turns")

    weights = parse_weights(args.weights)
    print("\nBuilding agentic-RAG DAG with memory...")
    print(f"  config:      {args.config_label}")
    print(f"  embeddings:  {'OFF (Phase A)' if args.no_embeddings else 'ON (Phase B)'}")
    print(f"  memory_db:   {args.memory_db or '<default>'}")
    print(f"  weights:     {weights or '<default>'}")

    mgr = MemoryManager(
        project_root=PROJECT_ROOT,
        max_raw_turns=4,
        use_embeddings=not args.no_embeddings,
        db_path=args.memory_db,
        scoring_weights=weights,
    )
    workflow = build_langgraph_workflow_with_memory(
        project_root=PROJECT_ROOT,
        memory_manager=mgr,
    )

    all_runs: List[Dict[str, Any]] = []
    all_conv_scores: List[Dict[str, Any]] = []

    for conv in convs:
        run = run_conversation(workflow, mgr, conv)
        all_runs.append(run)
        conv_score = score_conversation(run)
        all_conv_scores.append(conv_score)

    # Crash-safe: persist raw_runs first
    partial_report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(args.dataset.name),
        "config_label": args.config_label,
        "use_embeddings": not args.no_embeddings,
        "weights": weights or mgr.long_term.scoring_weights,
        "per_conversation_scores": all_conv_scores,
        "raw_runs": all_runs,
        "aggregate": None,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(partial_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[checkpoint] Raw + per-conv written to {args.output}")

    try:
        agg = aggregate(all_conv_scores)
        print_report(agg, args.config_label)
        partial_report["aggregate"] = agg
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(partial_report, f, ensure_ascii=False, indent=2, default=str)
        print(f"[final] Aggregated metrics appended to {args.output}")
    except Exception as exc:
        print(f"\n[warning] aggregate() failed: {type(exc).__name__}: {exc}")
    mgr.close()


if __name__ == "__main__":
    main()
