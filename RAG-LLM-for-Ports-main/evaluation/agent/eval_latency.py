"""
Latency Evaluation: Per-stage and end-to-end latency statistics.

Computes p50/p95/p99 for each pipeline node plus derived metrics:
- Total end-to-end latency
- Per-node stage breakdown (v3: 9 DAG nodes)
- TTFT (time-to-first-token) when streaming data is available
- Re-plan iteration distribution (legacy Agent v1 compat)
- ReAct observation call count (legacy Agent v1 compat)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List


# Agent v1 stages (kept for backward compat with old reports)
AGENT_STAGES = [
    "resolve_query",
    "plan_node",
    "execute_tools_node",
    "evaluate_evidence_node",
    "synthesize_node",
    "end_to_end",
]

# DAG v3 stages (matches NodeFactory._timed node names)
# Top-level nodes come first, then planner sub-stages, then end_to_end.
DAG_STAGES = [
    "route_query",
    "planner",
    "retrieve_documents",
    "rerank_documents",
    "retrieve_rules",
    "run_sql",
    "run_graph_reasoner",
    "merge_evidence",
    "synthesize_answer",
    "end_to_end",
]

# Planner sub-stage keys emitted by langgraph_nodes + planner.py
PLANNER_SUB_STAGES = [
    "planner__query_rewrite",
    "planner__plan_total",
    "planner__sub_queries__llm_call",
    "planner__sub_queries__rule_fallback",
    "planner__sub_query__documents__method",
    "planner__sub_query__sql__method",
    "planner__sub_query__rules__method",
    "planner__sub_query__graph__method",
]

# Streaming metric
TTFT_KEY = "ttft"


def _percentile(values: List[float], p: float) -> float:
    """Compute percentile (0-100) from a list."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 3),
        "p50": round(_percentile(values, 50), 3),
        "p95": round(_percentile(values, 95), 3),
        "p99": round(_percentile(values, 99), 3),
        "max": round(max(values), 3),
    }


def _detect_stage_set(timings: Dict[str, float]) -> List[str]:
    """Auto-detect whether the report uses DAG or Agent stage names."""
    dag_hits = sum(1 for s in DAG_STAGES[:-1] if s in timings)
    agent_hits = sum(1 for s in AGENT_STAGES[:-1] if s in timings)
    return DAG_STAGES if dag_hits >= agent_hits else AGENT_STAGES


@dataclass
class LatencyMetrics:
    per_stage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ttft: Dict[str, float] = field(default_factory=dict)
    iteration_dist: Dict[int, int] = field(default_factory=dict)
    observation_calls: Dict[str, float] = field(default_factory=dict)
    replan_trigger_rate: float = 0.0
    react_abort_rate: float = 0.0
    react_modify_rate: float = 0.0
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "per_stage_seconds": self.per_stage,
            "iteration_distribution": self.iteration_dist,
            "observation_call_count": self.observation_calls,
            "replan_trigger_rate": round(self.replan_trigger_rate, 4),
            "react_abort_rate": round(self.react_abort_rate, 4),
            "react_modify_rate": round(self.react_modify_rate, 4),
            "total_samples": self.total_samples,
        }
        if self.ttft and self.ttft.get("count", 0) > 0:
            d["ttft"] = self.ttft
        return d


def evaluate_latency(runs: List[Dict[str, Any]]) -> LatencyMetrics:
    """
    Compute latency metrics from a list of run records.

    Auto-detects whether the report uses DAG stage names or Agent v1 names
    and collects the appropriate set. Always collects end_to_end and TTFT.
    """
    if not runs:
        return LatencyMetrics()

    # Detect stage set from the first run that has timings
    detected_stages = AGENT_STAGES
    for run in runs:
        timings = run.get("stage_timings", {})
        if timings:
            detected_stages = _detect_stage_set(timings)
            break

    # Collect all unique stage names across runs (handles mixed reports)
    all_stage_names = set(detected_stages)
    for run in runs:
        all_stage_names.update(run.get("stage_timings", {}).keys())
    all_stage_names.add("end_to_end")

    stage_values: Dict[str, List[float]] = {s: [] for s in all_stage_names}
    ttft_values: List[float] = []
    iteration_counts: Dict[int, int] = {}
    observation_counts: List[int] = []
    abort_count = 0
    modify_count = 0
    replan_count = 0
    total_observations = 0

    for run in runs:
        timings = run.get("stage_timings", {})
        for stage, elapsed in timings.items():
            if stage in stage_values:
                stage_values[stage].append(float(elapsed))
            else:
                stage_values[stage] = [float(elapsed)]
        if "total_time" in run:
            stage_values["end_to_end"].append(float(run["total_time"]))

        # TTFT (streaming metric)
        if TTFT_KEY in run and run[TTFT_KEY] is not None:
            ttft_values.append(float(run[TTFT_KEY]))

        iteration = run.get("iteration", 1)
        iteration_counts[iteration] = iteration_counts.get(iteration, 0) + 1
        if iteration > 1:
            replan_count += 1

        observations = run.get("observations", []) or []
        observation_counts.append(len(observations))
        total_observations += len(observations)
        for obs in observations:
            action = obs.get("action", "continue")
            if action == "abort_replan":
                abort_count += 1
            elif action == "modify_next":
                modify_count += 1

    per_stage = {s: _summarize(v) for s, v in stage_values.items()}

    return LatencyMetrics(
        per_stage=per_stage,
        ttft=_summarize(ttft_values),
        iteration_dist=iteration_counts,
        observation_calls={
            "total": total_observations,
            "avg_per_query": round(total_observations / len(runs), 3) if runs else 0,
        },
        replan_trigger_rate=replan_count / len(runs) if runs else 0.0,
        react_abort_rate=abort_count / max(total_observations, 1),
        react_modify_rate=modify_count / max(total_observations, 1),
        total_samples=len(runs),
    )


# Preferred display order: top-level nodes → planner sub-stages → extras
_DISPLAY_ORDER = DAG_STAGES + PLANNER_SUB_STAGES


def print_latency_report(metrics: LatencyMetrics) -> None:
    print("\n" + "=" * 70)
    print("  LATENCY EVALUATION")
    print("=" * 70)
    print(f"  Samples: {metrics.total_samples}")

    # Collect stages that have data, in preferred order
    stages_with_data = [
        s for s in _DISPLAY_ORDER
        if metrics.per_stage.get(s, {}).get("count", 0) > 0
    ]
    # Add any extra stages not in _DISPLAY_ORDER
    extra = sorted(
        s for s in metrics.per_stage
        if s not in _DISPLAY_ORDER and metrics.per_stage[s].get("count", 0) > 0
    )
    stages_with_data.extend(extra)

    if stages_with_data:
        print(f"\n  {'Stage':<25} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  (seconds)")
        print(f"  {'-' * 75}")
        for stage in stages_with_data:
            s = metrics.per_stage[stage]
            print(f"  {stage:<25} "
                  f"{s.get('mean', 0):>8.2f} {s.get('p50', 0):>8.2f} "
                  f"{s.get('p95', 0):>8.2f} {s.get('p99', 0):>8.2f} {s.get('max', 0):>8.2f}")

    # TTFT
    if metrics.ttft and metrics.ttft.get("count", 0) > 0:
        t = metrics.ttft
        print(f"\n  TTFT (streaming, n={t['count']}):")
        print(f"    mean={t['mean']:.2f}s  p50={t['p50']:.2f}s  p95={t['p95']:.2f}s  max={t['max']:.2f}s")

    print(f"\n  Iteration distribution: {dict(sorted(metrics.iteration_dist.items()))}")
    print(f"  Re-plan trigger rate:   {metrics.replan_trigger_rate:.2%}")
    print(f"\n  ReAct observation stats:")
    print(f"    Total observations:   {metrics.observation_calls.get('total', 0)}")
    print(f"    Avg per query:        {metrics.observation_calls.get('avg_per_query', 0):.2f}")
    print(f"    Abort-replan rate:    {metrics.react_abort_rate:.2%}")
    print(f"    Modify-next rate:     {metrics.react_modify_rate:.2%}")
