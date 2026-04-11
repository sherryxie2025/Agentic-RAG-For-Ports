"""
Latency Evaluation: Per-stage and end-to-end latency statistics.

Computes p50/p95/p99 for each agent node plus derived metrics:
- Total end-to-end latency
- Re-plan iteration distribution
- ReAct observation call count
- LLM call count and token usage (if available)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List


# Stages we track
AGENT_STAGES = [
    "resolve_query",
    "plan_node",
    "execute_tools_node",
    "evaluate_evidence_node",
    "synthesize_node",
    "end_to_end",
]


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


@dataclass
class LatencyMetrics:
    per_stage: Dict[str, Dict[str, float]] = field(default_factory=dict)
    iteration_dist: Dict[int, int] = field(default_factory=dict)
    observation_calls: Dict[str, float] = field(default_factory=dict)
    replan_trigger_rate: float = 0.0
    react_abort_rate: float = 0.0
    react_modify_rate: float = 0.0
    total_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_stage_seconds": self.per_stage,
            "iteration_distribution": self.iteration_dist,
            "observation_call_count": self.observation_calls,
            "replan_trigger_rate": round(self.replan_trigger_rate, 4),
            "react_abort_rate": round(self.react_abort_rate, 4),
            "react_modify_rate": round(self.react_modify_rate, 4),
            "total_samples": self.total_samples,
        }


def evaluate_latency(runs: List[Dict[str, Any]]) -> LatencyMetrics:
    """
    Compute latency metrics from a list of agent run records.

    Each run should contain:
        - stage_timings: {stage_name: seconds}
        - total_time: float
        - iteration: int
        - observations: list of ObservationResult dicts
    """
    stage_values: Dict[str, List[float]] = {s: [] for s in AGENT_STAGES}
    iteration_counts: Dict[int, int] = {}
    observation_counts: List[int] = []
    abort_count = 0
    modify_count = 0
    replan_count = 0
    total_observations = 0

    for run in runs:
        timings = run.get("stage_timings", {})
        for stage in AGENT_STAGES[:-1]:
            if stage in timings:
                stage_values[stage].append(float(timings[stage]))
        if "total_time" in run:
            stage_values["end_to_end"].append(float(run["total_time"]))

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


def print_latency_report(metrics: LatencyMetrics) -> None:
    print("\n" + "=" * 70)
    print("  LATENCY EVALUATION")
    print("=" * 70)
    print(f"  Samples: {metrics.total_samples}")

    print(f"\n  {'Stage':<25} {'mean':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}  (seconds)")
    print(f"  {'-' * 75}")
    for stage in AGENT_STAGES:
        s = metrics.per_stage.get(stage, {})
        if s.get("count", 0) == 0:
            continue
        print(f"  {stage:<25} "
              f"{s.get('mean', 0):>8.2f} {s.get('p50', 0):>8.2f} "
              f"{s.get('p95', 0):>8.2f} {s.get('p99', 0):>8.2f} {s.get('max', 0):>8.2f}")

    print(f"\n  Iteration distribution: {dict(sorted(metrics.iteration_dist.items()))}")
    print(f"  Re-plan trigger rate:   {metrics.replan_trigger_rate:.2%}")
    print(f"\n  ReAct observation stats:")
    print(f"    Total observations:   {metrics.observation_calls.get('total', 0)}")
    print(f"    Avg per query:        {metrics.observation_calls.get('avg_per_query', 0):.2f}")
    print(f"    Abort-replan rate:    {metrics.react_abort_rate:.2%}")
    print(f"    Modify-next rate:     {metrics.react_modify_rate:.2%}")
