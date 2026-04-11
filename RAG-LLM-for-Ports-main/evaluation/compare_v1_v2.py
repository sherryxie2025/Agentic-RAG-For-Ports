"""Quick comparison script for v1 vs v2 evaluation reports."""
import json
from pathlib import Path

BASE = Path(__file__).parent

with open(BASE / "evaluation_report_v2_n115.json", encoding="utf-8") as f:
    v2 = json.load(f)
with open(BASE / "evaluation_report_full_single.json", encoding="utf-8") as f:
    v1 = json.load(f)

v1s = v1["single_turn"]
v2s = v2["single_turn"]

print("=" * 76)
print("FINAL COMPARISON: v1 (n=114) vs v2 (n=115) — SAME DATASET, ALL FIXES")
print("=" * 76)
print()

print("ROUTING")
print("-" * 76)
rt1 = v1s["routing"]
rt2 = v2s["routing"]
print(f'  samples            v1={rt1["total_samples"]}  v2={rt2["total_samples"]}')
print(f'  exact_match        v1={rt1["exact_match_rate"]:.2%}   v2={rt2["exact_match_rate"]:.2%}   ({(rt2["exact_match_rate"]-rt1["exact_match_rate"])*100:+.1f} pp)')
print(f'  over_routing       v1={rt1["over_routing_rate"]:.2%}   v2={rt2["over_routing_rate"]:.2%}   ({(rt2["over_routing_rate"]-rt1["over_routing_rate"])*100:+.1f} pp)')
print(f'  under_routing      v1={rt1["under_routing_rate"]:.2%}   v2={rt2["under_routing_rate"]:.2%}')
print(f'  micro_f1           v1={rt1["micro"]["f1"]:.3f}    v2={rt2["micro"]["f1"]:.3f}   ({rt2["micro"]["f1"]-rt1["micro"]["f1"]:+.3f})')
print(f'  macro_f1           v1={rt1["macro"]["f1"]:.3f}    v2={rt2["macro"]["f1"]:.3f}')
print()
for cap in ["vector", "sql", "rules", "graph"]:
    m1 = rt1["per_capability"][cap]
    m2 = rt2["per_capability"][cap]
    print(f'  {cap:<10}  v1: P={m1["precision"]:.3f} R={m1["recall"]:.3f} F1={m1["f1"]:.3f}  |  v2: P={m2["precision"]:.3f} R={m2["recall"]:.3f} F1={m2["f1"]:.3f}')

print()
print("RETRIEVAL")
print("-" * 76)
for src in ["vector", "sql", "rules", "graph"]:
    m1 = v1s["retrieval"].get(src, {})
    m2 = v2s["retrieval"].get(src, {})
    c1 = m1.get("count", 0)
    c2 = m2.get("count", 0)
    print(f'  {src.upper()} (v1 n={c1}, v2 n={c2}):')
    if src == "vector":
        v1_r5 = m1.get("recall@5", 0)
        v2_cr5 = m2.get("chunk_recall@5", 0)
        v2_sr5 = m2.get("source_recall@5")
        v2_sr20 = m2.get("source_recall@20")
        print(f'    chunk_recall@5     v1={v1_r5:.2%}  v2={v2_cr5:.2%}')
        if v2_sr5 is not None:
            print(f'    source_recall@5    v2={v2_sr5:.2%}  (cross-format)')
            print(f'    source_recall@20   v2={v2_sr20:.2%}')
        v1_mrr = m1.get("mrr", 0)
        v2_mrr = m2.get("chunk_mrr", 0) or m2.get("source_mrr", 0)
        print(f'    mrr                v1={v1_mrr:.3f}  v2={v2_mrr:.3f}')
    elif src == "sql":
        print(f'    table_f1           v1={m1.get("table_f1",0):.3f}  v2={m2.get("table_f1",0):.3f}')
        print(f'    execution_ok_rate  v1={m1.get("execution_ok_rate",0):.2%}  v2={m2.get("execution_ok_rate",0):.2%}')
    elif src == "rules":
        print(f'    variable_recall    v1={m1.get("variable_recall",0):.2%}  v2={m2.get("variable_recall",0):.2%}')
        print(f'    variable_precision v1={m1.get("variable_precision",0):.2%}  v2={m2.get("variable_precision",0):.2%}')
    elif src == "graph":
        print(f'    entity_recall      v1={m1.get("entity_recall",0):.2%}  v2={m2.get("entity_recall",0):.2%}')
        print(f'    path_found_rate    v1={m1.get("path_found_rate",0):.2%}  v2={m2.get("path_found_rate",0):.2%}')

print()
print("RERANK LIFT")
print("-" * 76)
rl1 = v1s["retrieval"].get("reranking_lift", {})
rl2 = v2s["retrieval"].get("reranking_lift", {})
print(f'  samples          v1={rl1.get("count",0)}  v2={rl2.get("count",0)}')
print(f'  ndcg@5_lift      v1={rl1.get("ndcg@5_lift",0):+.3f}  v2={rl2.get("ndcg@5_lift",0):+.3f}')
print(f'  top1_before      v1={rl1.get("top1_hit_before",0):.2%}  v2={rl2.get("top1_hit_before",0):.2%}')
print(f'  top1_after       v1={rl1.get("top1_hit_after",0):.2%}  v2={rl2.get("top1_hit_after",0):.2%}')
print(f'  top1_lift        v1={rl1.get("top1_lift",0):+.3f}  v2={rl2.get("top1_lift",0):+.3f}')

print()
print("ANSWER QUALITY")
print("-" * 76)
a1 = v1s["answer_quality"]
a2 = v2s["answer_quality"]
print(f'  keyword_coverage   v1={a1["avg_keyword_coverage"]:.2%}  v2={a2["avg_keyword_coverage"]:.2%}   ({(a2["avg_keyword_coverage"]-a1["avg_keyword_coverage"])*100:+.1f} pp)')
print(f'  citation_validity  v1={a1["avg_citation_validity"]:.2%}  v2={a2["avg_citation_validity"]:.2%}   ({(a2["avg_citation_validity"]-a1["avg_citation_validity"])*100:+.1f} pp)')
if a1.get("avg_numerical_accuracy") is not None and a2.get("avg_numerical_accuracy") is not None:
    print(f'  numerical_accuracy v1={a1["avg_numerical_accuracy"]:.2%}  v2={a2["avg_numerical_accuracy"]:.2%}')
print(f'  grounding v1: {a1["grounding_distribution"]}')
print(f'  grounding v2: {a2["grounding_distribution"]}')

print()
print("GUARDRAILS (v2 should show major improvement on OOD!)")
print("-" * 76)
gr1 = v1s["guardrails"]["pass_rates"]
gr2 = v2s["guardrails"]["pass_rates"]
cnt1 = v1s["guardrails"]["counts"]
cnt2 = v2s["guardrails"]["counts"]
all_types = sorted(set(gr1.keys()) | set(gr2.keys()))
print(f'  {"Type":<28} {"v1 pass":>10} {"v2 pass":>10} {"delta":>10}  samples')
for gtype in all_types:
    v1_rate = gr1.get(gtype)
    v2_rate = gr2.get(gtype)
    v1_str = f'{v1_rate:.2%}' if v1_rate is not None else "-"
    v2_str = f'{v2_rate:.2%}' if v2_rate is not None else "-"
    delta = f'{(v2_rate-v1_rate)*100:+.0f}pp' if v1_rate is not None and v2_rate is not None else "-"
    v2_count = cnt2.get(gtype, 0)
    print(f'  {gtype:<28} {v1_str:>10} {v2_str:>10} {delta:>10}  n={v2_count}')

print()
print("LATENCY (seconds)")
print("-" * 76)
print(f'  {"Stage":<24} {"v1 p50":>10} {"v1 p95":>10} {"v2 p50":>10} {"v2 p95":>10}')
for stage in ["ood_check_node", "plan_node", "execute_tools_node", "evaluate_evidence_node", "synthesize_node", "end_to_end"]:
    s1 = v1s["latency"]["per_stage_seconds"].get(stage, {})
    s2 = v2s["latency"]["per_stage_seconds"].get(stage, {})
    if s1.get("count", 0) == 0 and s2.get("count", 0) == 0:
        continue
    v1p50 = f'{s1.get("p50",0):.1f}s' if s1.get("count", 0) > 0 else "-"
    v1p95 = f'{s1.get("p95",0):.1f}s' if s1.get("count", 0) > 0 else "-"
    v2p50 = f'{s2.get("p50",0):.1f}s' if s2.get("count", 0) > 0 else "-"
    v2p95 = f'{s2.get("p95",0):.1f}s' if s2.get("count", 0) > 0 else "-"
    print(f'  {stage:<24} {v1p50:>10} {v1p95:>10} {v2p50:>10} {v2p95:>10}')
print()
print(f'  iterations v1: {v1s["latency"]["iteration_distribution"]}')
print(f'  iterations v2: {v2s["latency"]["iteration_distribution"]}')
print(f'  replan_rate  v1={v1s["latency"]["replan_trigger_rate"]:.2%}  v2={v2s["latency"]["replan_trigger_rate"]:.2%}')
print(f'  react_abort  v1={v1s["latency"]["react_abort_rate"]:.2%}  v2={v2s["latency"]["react_abort_rate"]:.2%}')
print(f'  react_modify v1={v1s["latency"]["react_modify_rate"]:.2%}  v2={v2s["latency"]["react_modify_rate"]:.2%}')
