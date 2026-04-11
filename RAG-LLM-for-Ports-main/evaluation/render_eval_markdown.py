"""
Render a Chinese Markdown evaluation report from an eval JSON.

Used both by `run_rag_evaluation.py` (auto-rendered after each run) and
as a standalone CLI to re-render an existing JSON report.

Usage:
  # Standalone
  python evaluation/render_eval_markdown.py \
      --input evaluation/agent/reports/rag_v2_n205_final.json \
      --output evaluation/agent/reports/rag_v2_n205_final.md

  # From Python (runner does this automatically)
  from evaluation.render_eval_markdown import render_report_md
  render_report_md(report_dict, Path("out.md"))
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _fmt(v, digits: int = 4, pct: bool = False) -> str:
    """Pretty-format a numeric metric for markdown output."""
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if pct:
            return f"{v * 100:.2f}%"
        return f"{v:.{digits}f}"
    return str(v)


def _routing_md(r: Dict[str, Any]) -> str:
    """Render routing sub-report."""
    out = ["## 1. 路由 (Routing)", ""]
    micro = r.get("micro", {})
    macro = r.get("macro", {})
    out.append("| 指标 | 值 |")
    out.append("|---|---|")
    out.append(f"| Micro F1 | {_fmt(micro.get('f1'))} |")
    out.append(f"| Micro Precision | {_fmt(micro.get('precision'))} |")
    out.append(f"| Micro Recall | {_fmt(micro.get('recall'))} |")
    out.append(f"| Macro F1 | {_fmt(macro.get('f1'))} |")
    out.append(f"| Exact match rate | {_fmt(r.get('exact_match_rate'), pct=True)} |")
    out.append(f"| Over-routing rate | {_fmt(r.get('over_routing_rate'), pct=True)} |")
    out.append(f"| Under-routing rate | {_fmt(r.get('under_routing_rate'), pct=True)} |")
    out.append("")

    out.append("### Per-capability F1")
    out.append("")
    out.append("| Capability | Precision | Recall | F1 | TP | FP | FN |")
    out.append("|---|---|---|---|---|---|---|")
    for cap, v in (r.get("per_capability") or {}).items():
        out.append(
            f"| {cap} | {_fmt(v.get('precision'))} | {_fmt(v.get('recall'))}"
            f" | {_fmt(v.get('f1'))} | {v.get('tp', 0)}"
            f" | {v.get('fp', 0)} | {v.get('fn', 0)} |"
        )
    out.append("")
    return "\n".join(out)


def _retrieval_md(r: Dict[str, Any]) -> str:
    """Render retrieval sub-report."""
    out = ["## 2. 检索 (Retrieval)", ""]

    vec = r.get("vector", {}) or {}
    if vec:
        out.append("### 2.1 Vector")
        out.append("")
        out.append("| 指标 | chunk 级 | source 级 |")
        out.append("|---|---|---|")
        out.append(f"| Recall@5 | {_fmt(vec.get('chunk_recall@5'))} | {_fmt(vec.get('source_recall@5'))} |")
        out.append(f"| Recall@20 | {_fmt(vec.get('chunk_recall@20'))} | {_fmt(vec.get('source_recall@20'))} |")
        out.append(f"| Precision@5 | {_fmt(vec.get('chunk_precision@5'))} | {_fmt(vec.get('source_precision@5'))} |")
        out.append(f"| MRR | {_fmt(vec.get('chunk_mrr'))} | {_fmt(vec.get('source_mrr'))} |")
        out.append(f"| NDCG@10 | {_fmt(vec.get('chunk_ndcg@10'))} | {_fmt(vec.get('source_ndcg@10'))} |")
        out.append(f"| Samples | {vec.get('chunk_samples', '?')} | {vec.get('source_samples', '?')} |")
        out.append("")

    sql = r.get("sql", {}) or {}
    if sql:
        out.append("### 2.2 SQL")
        out.append("")
        out.append("| 指标 | 值 |")
        out.append("|---|---|")
        out.append(f"| Table F1 | {_fmt(sql.get('table_f1'))} |")
        out.append(f"| Execution OK rate | {_fmt(sql.get('execution_ok_rate'), pct=True)} |")
        out.append(f"| Samples | {sql.get('count', '?')} |")
        out.append("")

    rules = r.get("rules", {}) or {}
    if rules:
        out.append("### 2.3 Rules")
        out.append("")
        out.append("| 指标 | 值 |")
        out.append("|---|---|")
        out.append(f"| Variable Precision | {_fmt(rules.get('variable_precision'))} |")
        out.append(f"| Variable Recall | {_fmt(rules.get('variable_recall'))} |")
        out.append(f"| Samples | {rules.get('count', '?')} |")
        out.append("")

    graph = r.get("graph", {}) or {}
    if graph:
        out.append("### 2.4 Graph")
        out.append("")
        out.append("| 指标 | 值 |")
        out.append("|---|---|")
        out.append(f"| path_found_rate | {_fmt(graph.get('path_found_rate'), pct=True)} |")
        out.append(f"| Samples | {graph.get('count', '?')} |")
        out.append("")

    rerank = r.get("reranking_lift", {}) or {}
    if rerank:
        out.append("### 2.5 Rerank")
        out.append("")
        out.append("| 指标 | 值 |")
        out.append("|---|---|")
        out.append(f"| NDCG@5 lift | {_fmt(rerank.get('ndcg@5_lift'))} |")
        out.append(f"| Recall@5 lift | {_fmt(rerank.get('recall@5_lift'))} |")
        out.append(f"| Top-1 hit before | {_fmt(rerank.get('top1_hit_before'), pct=True)} |")
        out.append(f"| Top-1 hit after | {_fmt(rerank.get('top1_hit_after'), pct=True)} |")
        out.append(f"| Top-1 lift | {_fmt(rerank.get('top1_lift'), pct=True)} |")
        out.append(f"| Samples | {rerank.get('count', '?')} |")
        out.append("")
    return "\n".join(out)


def _answer_md(a: Dict[str, Any]) -> str:
    """Render answer quality sub-report."""
    out = ["## 3. 答案质量 (Answer Quality)", ""]
    out.append("| 指标 | 值 |")
    out.append("|---|---|")
    out.append(f"| Keyword coverage | {_fmt(a.get('avg_keyword_coverage'), pct=True)} |")
    out.append(f"| Citation validity | {_fmt(a.get('avg_citation_validity'), pct=True)} |")
    out.append(f"| Numerical accuracy | {_fmt(a.get('avg_numerical_accuracy'), pct=True)} |")
    out.append(f"| Embedding cosine similarity (BGE) | {_fmt(a.get('avg_embedding_similarity'))} |")
    out.append(f"| ROUGE-L F1 | {_fmt(a.get('avg_rougeL_f1'))} |")
    out.append(f"| Similarity 样本数 | {a.get('samples_similarity_scored', 0)} |")
    out.append("")
    out.append("### LLM Judge")
    out.append("")
    out.append("| 维度 (1-5) | 值 |")
    out.append("|---|---|")
    out.append(f"| Faithfulness | {_fmt(a.get('avg_faithfulness'), digits=2)} |")
    out.append(f"| Relevance | {_fmt(a.get('avg_relevance'), digits=2)} |")
    out.append(f"| Completeness | {_fmt(a.get('avg_completeness'), digits=2)} |")
    out.append(f"| LLM judged samples | {a.get('samples_llm_judged', 0)} |")
    out.append("")
    gd = a.get("grounding_distribution") or {}
    if gd:
        out.append("### Grounding distribution")
        out.append("")
        for k, v in gd.items():
            out.append(f"- `{k}`: {v}")
        out.append("")
    return "\n".join(out)


def _guardrails_md(g: Dict[str, Any]) -> str:
    """Render guardrail sub-report."""
    out = ["## 4. 护栏 (Guardrails)", ""]
    pr = g.get("pass_rates", {}) or {}
    ct = g.get("counts", {}) or {}
    out.append("| 类型 | Pass Rate | 样本数 |")
    out.append("|---|---|---|")
    for k in sorted(pr.keys()):
        out.append(f"| {k} | {_fmt(pr.get(k), pct=True)} | {ct.get(k, 0)} |")
    out.append("")
    return "\n".join(out)


def _latency_md(l: Dict[str, Any]) -> str:
    """Render latency sub-report."""
    out = ["## 5. 延迟 (Latency)", ""]
    stages = l.get("per_stage_seconds", {}) or {}
    e2e = stages.get("end_to_end", {})
    if e2e and e2e.get("count", 0):
        out.append("| 阶段 | count | mean | p50 | p95 | p99 | max |")
        out.append("|---|---|---|---|---|---|---|")
        for name, s in stages.items():
            if not s or not s.get("count"):
                continue
            out.append(
                f"| {name} | {s.get('count', 0)}"
                f" | {_fmt(s.get('mean'))}"
                f" | {_fmt(s.get('p50'))}"
                f" | {_fmt(s.get('p95'))}"
                f" | {_fmt(s.get('p99'))}"
                f" | {_fmt(s.get('max'))} |"
            )
    out.append("")
    return "\n".join(out)


def render_report_md(
    report: Dict[str, Any],
    out_path: Path,
    source_json: Optional[Path] = None,
) -> None:
    """Render an eval report dict to a Markdown file."""
    st = report.get("single_turn", {}) or {}
    lines = []
    lines.append(f"# RAG 评测报告 — {report.get('architecture', 'N/A')}")
    lines.append("")
    lines.append(f"- 生成时间: {report.get('timestamp', datetime.now().isoformat())}")
    lines.append(f"- 数据管线: {report.get('data_pipeline', '?')}")
    lines.append(f"- Golden 数据集: {report.get('golden_dataset', '?')}")
    lines.append(f"- 样本数: {report.get('evaluated_samples', 0)} / {report.get('dataset_size', 0)}")
    if "workers" in report:
        lines.append(f"- 并发 workers: {report.get('workers')}")
    if source_json:
        lines.append(f"- 原始 JSON: `{source_json.name}`")
    per = len(report.get("per_sample_results") or [])
    lines.append(f"- per_sample_results 保存: {'是 (' + str(per) + ' 条)' if per else '否 (历史报告)'}")
    lines.append("")

    lines.append(_routing_md(st.get("routing", {})))
    lines.append(_retrieval_md(st.get("retrieval", {})))
    lines.append(_answer_md(st.get("answer_quality", {})))
    lines.append(_guardrails_md(st.get("guardrails", {})))
    lines.append(_latency_md(st.get("latency", {})))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to eval JSON")
    parser.add_argument("--output", default=None,
                        help="Path to output .md (default: same dir, same basename)")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".md")
    report = json.loads(in_path.read_text(encoding="utf-8"))
    render_report_md(report, out_path, source_json=in_path)
    print(f">> Wrote {out_path}")


if __name__ == "__main__":
    main()
