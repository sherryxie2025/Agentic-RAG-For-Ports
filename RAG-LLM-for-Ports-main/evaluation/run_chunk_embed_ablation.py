"""
Chunking + Embedding ablation: v1 (all-MiniLM + 400-char fixed) vs v2 (BGE + Small-to-Big 250-word).

Compares retrieval quality on golden_dataset_v3_rag.json samples that
require vector retrieval (needs_vector=True). For each query:

1. Retrieve top-K from BOTH Chroma collections
2. Score by keyword coverage against `expected_evidence_keywords`
3. Score by source-file recall against `golden_vector.expected_source_files`
4. Identify bad cases (v1 wins, v2 wins, both fail)
5. Output a side-by-side comparison table + per-query bad-case analysis

No LLM calls — pure retrieval + statistics. Runs in ~2 minutes.

Usage:
    python evaluation/run_chunk_embed_ablation.py [--limit N] [--top-k 10]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(str(PROJECT_ROOT))

import chromadb
from online_pipeline.document_retriever import ChromaDocumentRetriever, BGE_QUERY_PREFIX
from online_pipeline.source_registry import SourceRegistry


def keyword_coverage(retrieved_texts: List[str], keywords: List[str]) -> float:
    """Fraction of expected keywords found in any retrieved chunk."""
    if not keywords:
        return 1.0
    combined = " ".join(retrieved_texts).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return hits / len(keywords)


def source_recall(retrieved_sources: List[str],
                  expected_sources: List[str]) -> float:
    """Fraction of expected source files found in retrieved chunks."""
    if not expected_sources:
        return 1.0
    retrieved_set = set(s.lower() for s in retrieved_sources)
    hits = sum(1 for s in expected_sources if s.lower() in retrieved_set)
    return hits / len(expected_sources)


def run_retrieval(collection, query: str, top_k: int,
                  bge_model=None) -> List[Dict[str, Any]]:
    """Query a Chroma collection and return results with text + metadata."""
    if bge_model is not None:
        prefixed = BGE_QUERY_PREFIX + query
        emb = bge_model.encode([prefixed], normalize_embeddings=True,
                               convert_to_numpy=True).tolist()
        results = collection.query(query_embeddings=emb, n_results=top_k)
    else:
        results = collection.query(query_texts=[query], n_results=top_k)

    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    out = []
    for i in range(len(ids)):
        out.append({
            "chunk_id": ids[i],
            "text": docs[i] if i < len(docs) else "",
            "source_file": (metas[i] or {}).get("source_file", "") if i < len(metas) else "",
            "distance": dists[i] if i < len(dists) else 999,
        })
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "evaluation" / "agent" / "reports" / "chunk_embed_ablation.json")
    args = parser.parse_args()

    # Load golden dataset — filter to needs_vector=True
    with open("evaluation/golden_dataset_v3_rag.json", "r", encoding="utf-8") as f:
        golden = json.load(f)
    samples = [s for s in golden["samples"] if s.get("needs_vector")]
    if args.limit:
        samples = samples[:args.limit]
    print(f"Samples with needs_vector=True: {len(samples)}")

    # Load both collections
    client = chromadb.PersistentClient(path="storage/chroma")
    col_v1 = client.get_collection(name="port_documents")
    col_v2 = client.get_collection(name="port_documents_v2")
    print(f"v1 collection: {col_v1.count()} chunks (all-MiniLM + fixed 400-char)")
    print(f"v2 collection: {col_v2.count()} chunks (BGE + Small-to-Big 250-word)")

    # Load BGE model for v2 queries
    from sentence_transformers import SentenceTransformer
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    bge.max_seq_length = 512
    print("BGE model loaded")

    # Run comparisons
    per_sample: List[Dict[str, Any]] = []
    v1_kw_total = v2_kw_total = 0.0
    v1_src_total = v2_src_total = 0.0
    v1_wins = v2_wins = both_good = both_bad = 0

    for i, s in enumerate(samples):
        query = s["query"]
        expected_kw = s.get("expected_evidence_keywords", [])

        # Get expected source files from golden_vector if available
        gv = s.get("golden_vector", {}) or {}
        expected_src = gv.get("expected_source_files", [])

        # Retrieve from both collections
        r_v1 = run_retrieval(col_v1, query, args.top_k, bge_model=None)  # v1 uses Chroma default embed
        r_v2 = run_retrieval(col_v2, query, args.top_k, bge_model=bge)   # v2 uses BGE

        # Score
        v1_texts = [r["text"] for r in r_v1]
        v2_texts = [r["text"] for r in r_v2]
        v1_srcs = [r["source_file"] for r in r_v1]
        v2_srcs = [r["source_file"] for r in r_v2]

        v1_kw = keyword_coverage(v1_texts, expected_kw)
        v2_kw = keyword_coverage(v2_texts, expected_kw)
        v1_sr = source_recall(v1_srcs, expected_src)
        v2_sr = source_recall(v2_srcs, expected_src)

        v1_kw_total += v1_kw
        v2_kw_total += v2_kw
        v1_src_total += v1_sr
        v2_src_total += v2_sr

        # Classify
        threshold = 0.3
        v1_good = v1_kw >= threshold
        v2_good = v2_kw >= threshold
        if v1_good and v2_good:
            both_good += 1
            verdict = "both_good"
        elif v1_good and not v2_good:
            v1_wins += 1
            verdict = "v1_wins"
        elif not v1_good and v2_good:
            v2_wins += 1
            verdict = "v2_wins"
        else:
            both_bad += 1
            verdict = "both_bad"

        record = {
            "id": s["id"],
            "query": query[:120],
            "answer_mode": s.get("answer_mode"),
            "v1_keyword_coverage": round(v1_kw, 4),
            "v2_keyword_coverage": round(v2_kw, 4),
            "v1_source_recall": round(v1_sr, 4),
            "v2_source_recall": round(v2_sr, 4),
            "verdict": verdict,
        }

        # For bad cases, include chunk snippets for analysis
        if verdict in ("v1_wins", "v2_wins"):
            record["v1_top3_snippets"] = [r["text"][:200] for r in r_v1[:3]]
            record["v2_top3_snippets"] = [r["text"][:200] for r in r_v2[:3]]
            record["expected_keywords"] = expected_kw
            record["expected_sources"] = expected_src

        per_sample.append(record)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] processed")

    n = len(samples)
    summary = {
        "total_samples": n,
        "top_k": args.top_k,
        "v1_config": "all-MiniLM-L6-v2 (384d) + fixed 400-char chunks (130K)",
        "v2_config": "BGE-base-en-v1.5 (768d) + Small-to-Big 250-word children (16K)",
        "avg_keyword_coverage_v1": round(v1_kw_total / n, 4),
        "avg_keyword_coverage_v2": round(v2_kw_total / n, 4),
        "avg_source_recall_v1": round(v1_src_total / n, 4),
        "avg_source_recall_v2": round(v2_src_total / n, 4),
        "verdict_distribution": {
            "both_good": both_good,
            "v1_wins": v1_wins,
            "v2_wins": v2_wins,
            "both_bad": both_bad,
        },
    }

    # Print report
    print("\n" + "=" * 80)
    print("  CHUNKING + EMBEDDING ABLATION")
    print("=" * 80)
    print(f"  Samples: {n}  |  top_k: {args.top_k}")
    print(f"\n  v1: {summary['v1_config']}")
    print(f"  v2: {summary['v2_config']}")
    print(f"\n  {'Metric':<30} {'v1':>10} {'v2':>10} {'Delta':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Avg keyword coverage':<30} {summary['avg_keyword_coverage_v1']:>10.2%} {summary['avg_keyword_coverage_v2']:>10.2%} {summary['avg_keyword_coverage_v2']-summary['avg_keyword_coverage_v1']:>+10.2%}")
    print(f"  {'Avg source recall':<30} {summary['avg_source_recall_v1']:>10.2%} {summary['avg_source_recall_v2']:>10.2%} {summary['avg_source_recall_v2']-summary['avg_source_recall_v1']:>+10.2%}")
    print(f"\n  Verdict distribution:")
    print(f"    Both good:  {both_good:>4}  ({both_good/n:.0%})")
    print(f"    v1 wins:    {v1_wins:>4}  ({v1_wins/n:.0%})")
    print(f"    v2 wins:    {v2_wins:>4}  ({v2_wins/n:.0%})")
    print(f"    Both bad:   {both_bad:>4}  ({both_bad/n:.0%})")

    # Print bad cases
    bad_v1 = [r for r in per_sample if r["verdict"] == "v1_wins"]
    bad_v2 = [r for r in per_sample if r["verdict"] == "v2_wins"]
    if bad_v1:
        print(f"\n  --- BAD CASES: v1 wins ({len(bad_v1)}) ---")
        for r in bad_v1[:5]:
            print(f"  {r['id']}: v1_kw={r['v1_keyword_coverage']:.2f} v2_kw={r['v2_keyword_coverage']:.2f}")
            print(f"    query: {r['query']}")
            print(f"    expected_kw: {r.get('expected_keywords', [])[:5]}")
            print(f"    v1 top-1: {r.get('v1_top3_snippets', [''])[0][:100]}")
            print(f"    v2 top-1: {r.get('v2_top3_snippets', [''])[0][:100]}")
            print()

    if bad_v2:
        print(f"\n  --- BAD CASES: v2 wins ({len(bad_v2)}) ---")
        for r in bad_v2[:5]:
            print(f"  {r['id']}: v1_kw={r['v1_keyword_coverage']:.2f} v2_kw={r['v2_keyword_coverage']:.2f}")
            print(f"    query: {r['query']}")
            print(f"    expected_kw: {r.get('expected_keywords', [])[:5]}")
            print(f"    v1 top-1: {r.get('v1_top3_snippets', [''])[0][:100]}")
            print(f"    v2 top-1: {r.get('v2_top3_snippets', [''])[0][:100]}")
            print()

    # Save
    report = {"summary": summary, "per_sample": per_sample}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
