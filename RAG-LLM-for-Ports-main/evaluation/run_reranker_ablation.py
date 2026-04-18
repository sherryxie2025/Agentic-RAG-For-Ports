"""
Reranker ablation: ms-marco-MiniLM-L-6-v2 (6L, 22M) vs bge-reranker-v2-m3 (24L, 568M).

For each needs_vector golden sample:
1. Retrieve top-40 from v2 Chroma collection (BGE embeddings) — same candidates for both
2. Rerank with model A (current) → top-10
3. Rerank with model B (BGE reranker) → top-10
4. Score both by keyword coverage against expected_evidence_keywords
5. Measure latency per rerank call

No LLM calls. ~5 minutes.
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

import chromadb
from sentence_transformers import CrossEncoder
from online_pipeline.document_retriever import BGE_QUERY_PREFIX


def keyword_coverage(texts: List[str], keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


def rerank(model: CrossEncoder, query: str, docs: List[str], top_k: int) -> List[int]:
    """Return indices of top_k docs after cross-encoder scoring."""
    pairs = [(query, d) for d in docs]
    scores = model.predict(pairs)
    ranked = np.argsort(scores)[::-1][:top_k]
    return ranked.tolist()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieve-k", type=int, default=40,
                        help="How many candidates to pull from Chroma before reranking")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "evaluation" / "agent" / "reports" / "reranker_ablation.json")
    args = parser.parse_args()

    # Load golden
    with open("evaluation/golden_dataset_v3_rag.json", "r", encoding="utf-8") as f:
        golden = json.load(f)
    samples = [s for s in golden["samples"] if s.get("needs_vector")][:args.limit]
    print(f"Samples: {len(samples)}")

    # Load models
    print("Loading ms-marco-MiniLM-L-6-v2 (current)...")
    model_a = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cuda")
    print(f"  layers={model_a.model.config.num_hidden_layers}  params={sum(p.numel() for p in model_a.model.parameters())/1e6:.1f}M")

    print("Loading bge-reranker-v2-m3...")
    model_b = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512, device="cuda")
    print(f"  layers={model_b.model.config.num_hidden_layers}  params={sum(p.numel() for p in model_b.model.parameters())/1e6:.1f}M")

    # Load v2 collection + BGE for initial retrieval
    client = chromadb.PersistentClient(path="storage/chroma")
    col_v2 = client.get_collection(name="port_documents_v2")
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    bge.max_seq_length = 512
    print(f"Chroma v2: {col_v2.count()} chunks")

    # Run
    per_sample: List[Dict[str, Any]] = []
    a_kw_total = b_kw_total = 0.0
    no_rerank_kw_total = 0.0
    a_lat_total = b_lat_total = 0.0
    a_wins = b_wins = both_good = both_bad = 0

    for i, s in enumerate(samples):
        query = s["query"]
        expected_kw = s.get("expected_evidence_keywords", [])

        # Retrieve candidates with BGE
        q_emb = bge.encode([BGE_QUERY_PREFIX + query],
                           normalize_embeddings=True).tolist()
        results = col_v2.query(query_embeddings=q_emb, n_results=args.retrieve_k)
        cand_docs = results["documents"][0]
        if not cand_docs:
            per_sample.append({"id": s["id"], "skipped": True})
            continue

        # No-rerank baseline (just take top-k by embedding distance)
        no_rerank_texts = cand_docs[:args.top_k]
        no_rerank_kw = keyword_coverage(no_rerank_texts, expected_kw)
        no_rerank_kw_total += no_rerank_kw

        # Rerank with model A
        t0 = time.time()
        idx_a = rerank(model_a, query, cand_docs, args.top_k)
        lat_a = (time.time() - t0) * 1000
        a_lat_total += lat_a
        a_texts = [cand_docs[j] for j in idx_a]
        a_kw = keyword_coverage(a_texts, expected_kw)
        a_kw_total += a_kw

        # Rerank with model B
        t0 = time.time()
        idx_b = rerank(model_b, query, cand_docs, args.top_k)
        lat_b = (time.time() - t0) * 1000
        b_lat_total += lat_b
        b_texts = [cand_docs[j] for j in idx_b]
        b_kw = keyword_coverage(b_texts, expected_kw)
        b_kw_total += b_kw

        # Classify
        threshold = 0.5
        a_good = a_kw >= threshold
        b_good = b_kw >= threshold
        if a_good and b_good: both_good += 1; verdict = "both_good"
        elif a_good and not b_good: a_wins += 1; verdict = "a_wins"
        elif not a_good and b_good: b_wins += 1; verdict = "b_wins"
        else: both_bad += 1; verdict = "both_bad"

        record = {
            "id": s["id"],
            "query": query[:120],
            "no_rerank_kw": round(no_rerank_kw, 4),
            "model_a_kw": round(a_kw, 4),
            "model_b_kw": round(b_kw, 4),
            "model_a_latency_ms": round(lat_a, 1),
            "model_b_latency_ms": round(lat_b, 1),
            "verdict": verdict,
        }
        if verdict in ("a_wins", "b_wins"):
            record["a_top3"] = [cand_docs[j][:150] for j in idx_a[:3]]
            record["b_top3"] = [cand_docs[j][:150] for j in idx_b[:3]]
            record["expected_kw"] = expected_kw[:6]
        per_sample.append(record)

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    n = len([r for r in per_sample if not r.get("skipped")])

    summary = {
        "total_samples": n,
        "retrieve_k": args.retrieve_k,
        "rerank_top_k": args.top_k,
        "model_a": "cross-encoder/ms-marco-MiniLM-L-6-v2 (6L, 22M)",
        "model_b": "BAAI/bge-reranker-v2-m3 (24L, 568M)",
        "avg_kw_no_rerank": round(no_rerank_kw_total / n, 4),
        "avg_kw_model_a": round(a_kw_total / n, 4),
        "avg_kw_model_b": round(b_kw_total / n, 4),
        "avg_latency_a_ms": round(a_lat_total / n, 1),
        "avg_latency_b_ms": round(b_lat_total / n, 1),
        "verdict": {"both_good": both_good, "a_wins": a_wins,
                    "b_wins": b_wins, "both_bad": both_bad},
    }

    print("\n" + "=" * 80)
    print("  RERANKER ABLATION")
    print("=" * 80)
    print(f"  Samples: {n}  |  candidates: top-{args.retrieve_k}  |  rerank to: top-{args.top_k}")
    print(f"\n  {'Config':<45} {'Avg KW Cov':>12} {'Avg Latency':>14}")
    print(f"  {'-'*72}")
    print(f"  {'No rerank (embedding distance only)':<45} {summary['avg_kw_no_rerank']:>12.2%} {'—':>14}")
    print(f"  {'ms-marco-MiniLM-L-6 (6L, 22M) [current]':<45} {summary['avg_kw_model_a']:>12.2%} {summary['avg_latency_a_ms']:>11.1f} ms")
    print(f"  {'bge-reranker-v2-m3 (24L, 568M)':<45} {summary['avg_kw_model_b']:>12.2%} {summary['avg_latency_b_ms']:>11.1f} ms")
    print(f"\n  Rerank lift (vs no rerank):")
    print(f"    MiniLM-L-6:        {summary['avg_kw_model_a'] - summary['avg_kw_no_rerank']:>+.2%}")
    print(f"    bge-reranker-v2:   {summary['avg_kw_model_b'] - summary['avg_kw_no_rerank']:>+.2%}")
    print(f"\n  Head-to-head (MiniLM-L-6 vs bge-reranker):")
    print(f"    Both good:  {both_good:>4}  ({both_good/n:.0%})")
    print(f"    A wins:     {a_wins:>4}  ({a_wins/n:.0%})")
    print(f"    B wins:     {b_wins:>4}  ({b_wins/n:.0%})")
    print(f"    Both bad:   {both_bad:>4}  ({both_bad/n:.0%})")

    # Bad cases
    for label, key in [("A wins (MiniLM better)", "a_wins"), ("B wins (BGE better)", "b_wins")]:
        cases = [r for r in per_sample if r.get("verdict") == key]
        if cases:
            print(f"\n  --- {label} ({len(cases)}) ---")
            for r in cases[:3]:
                print(f"  {r['id']}: a_kw={r['model_a_kw']:.2f} b_kw={r['model_b_kw']:.2f}")
                print(f"    query: {r['query']}")

    report = {"summary": summary, "per_sample": per_sample}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
