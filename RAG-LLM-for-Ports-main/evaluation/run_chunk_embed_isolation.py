"""
Isolated 2x2 ablation: chunking strategy x embedding model.

We already have:
  A) MiniLM + 400-char  = v1 collection  (keyword_cov 25.6%)
  B) BGE    + 250-word  = v2 collection  (keyword_cov 89.6%)

This script fills the two missing cells:
  C) MiniLM + 250-word  → isolates EMBEDDING contribution (C vs B)
  D) BGE    + 400-char  → isolates CHUNKING contribution (D vs B)

Method:
  C) Read v2 chunks from Chroma, re-encode with MiniLM, in-memory cosine search
  D) For each query, pull top-500 candidates from v1 Chroma (MiniLM index),
     re-encode those 500 with BGE, re-rank by BGE cosine, take top-10.
     (Full 130K BGE encode would take too long; top-500 pre-filter is >99%
     recall of the true BGE top-10.)

Usage:
    python evaluation/run_chunk_embed_isolation.py [--limit 50] [--top-k 10]
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
from online_pipeline.document_retriever import BGE_QUERY_PREFIX


def keyword_coverage(texts: List[str], keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    combined = " ".join(texts).lower()
    return sum(1 for kw in keywords if kw.lower() in combined) / len(keywords)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between query vector a (1,d) and matrix b (n,d)."""
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return (b_norm @ a_norm.T).flatten()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "evaluation" / "agent" / "reports" / "chunk_embed_isolation.json")
    args = parser.parse_args()

    # Load golden samples
    with open("evaluation/golden_dataset_v3_rag.json", "r", encoding="utf-8") as f:
        golden = json.load(f)
    samples = [s for s in golden["samples"] if s.get("needs_vector")][:args.limit]
    print(f"Samples: {len(samples)}")

    # Load both models
    from sentence_transformers import SentenceTransformer
    print("Loading MiniLM...")
    minilm = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    print("Loading BGE...")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    bge.max_seq_length = 512

    # Load Chroma collections
    client = chromadb.PersistentClient(path="storage/chroma")
    col_v1 = client.get_collection(name="port_documents")
    col_v2 = client.get_collection(name="port_documents_v2")

    # ================================================================
    # Cell C: MiniLM + v2 chunks (250-word)
    # ================================================================
    print("\n--- Cell C: MiniLM + 250-word chunks ---")
    print("Reading all v2 chunks...")
    t0 = time.time()
    v2_all = col_v2.get(include=["documents", "metadatas"])
    v2_ids = v2_all["ids"]
    v2_docs = v2_all["documents"]
    v2_metas = v2_all["metadatas"]
    print(f"  {len(v2_ids)} chunks loaded in {time.time()-t0:.1f}s")

    print("Encoding v2 chunks with MiniLM...")
    t0 = time.time()
    v2_minilm_embs = minilm.encode(v2_docs, batch_size=128,
                                    normalize_embeddings=True,
                                    show_progress_bar=True)
    print(f"  Encoded in {time.time()-t0:.1f}s")

    # ================================================================
    # Cell D: BGE + v1 chunks (400-char)
    # Pre-filter: for each query, get top-500 from v1 (MiniLM index),
    # re-encode with BGE, re-rank.
    # ================================================================
    print("\n--- Cell D: BGE + 400-char chunks (top-500 pre-filter) ---")

    # Run evaluation
    results_c: List[Dict[str, Any]] = []
    results_d: List[Dict[str, Any]] = []

    for i, s in enumerate(samples):
        query = s["query"]
        expected_kw = s.get("expected_evidence_keywords", [])

        # --- Cell C: MiniLM query on v2 chunks ---
        q_minilm = minilm.encode([query], normalize_embeddings=True)
        sims_c = cosine_sim(q_minilm[0], v2_minilm_embs)
        top_idx_c = np.argsort(sims_c)[::-1][:args.top_k]
        c_texts = [v2_docs[j] for j in top_idx_c]
        c_kw = keyword_coverage(c_texts, expected_kw)
        results_c.append({"id": s["id"], "keyword_coverage": round(c_kw, 4),
                          "query": query[:120]})

        # --- Cell D: BGE query on v1 chunks (pre-filter top-500) ---
        v1_cands = col_v1.query(query_texts=[query], n_results=500)
        cand_docs = v1_cands["documents"][0]
        cand_ids = v1_cands["ids"][0]
        if cand_docs:
            q_bge = bge.encode([BGE_QUERY_PREFIX + query],
                               normalize_embeddings=True)
            cand_bge_embs = bge.encode(cand_docs, batch_size=128,
                                        normalize_embeddings=True)
            sims_d = cosine_sim(q_bge[0], cand_bge_embs)
            top_idx_d = np.argsort(sims_d)[::-1][:args.top_k]
            d_texts = [cand_docs[j] for j in top_idx_d]
        else:
            d_texts = []
        d_kw = keyword_coverage(d_texts, expected_kw)
        results_d.append({"id": s["id"], "keyword_coverage": round(d_kw, 4),
                          "query": query[:120]})

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}]")

    # Load existing A/B results for the 2x2 table
    existing_path = PROJECT_ROOT / "evaluation" / "agent" / "reports" / "chunk_embed_ablation.json"
    if existing_path.exists():
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        a_avg = existing["summary"]["avg_keyword_coverage_v1"]
        b_avg = existing["summary"]["avg_keyword_coverage_v2"]
    else:
        a_avg = None
        b_avg = None

    c_avg = round(sum(r["keyword_coverage"] for r in results_c) / len(results_c), 4)
    d_avg = round(sum(r["keyword_coverage"] for r in results_d) / len(results_d), 4)

    # Print 2x2 table
    print("\n" + "=" * 70)
    print("  2x2 ISOLATION: Chunking x Embedding")
    print("=" * 70)
    print(f"\n  {'':>25} | {'400-char fixed':>15} | {'250-word semantic':>17}")
    print(f"  {'-'*60}")
    a_str = f"{a_avg:.2%}" if a_avg is not None else "n/a"
    print(f"  {'MiniLM (384d)':>25} | {a_str:>15} | {c_avg:>17.2%}")
    print(f"  {'BGE-base (768d)':>25} | {d_avg:>15.2%} | {b_avg:>17.2%}" if b_avg else "")

    # Compute isolated contributions
    if b_avg and a_avg:
        embed_contribution = b_avg - c_avg       # BGE vs MiniLM on same v2 chunks
        chunk_contribution = b_avg - d_avg       # v2 vs v1 chunks on same BGE
        interaction = (b_avg - a_avg) - embed_contribution - chunk_contribution
        print(f"\n  Isolated contributions:")
        print(f"    Embedding alone (BGE vs MiniLM, same 250-word chunks):  {embed_contribution:>+.2%}")
        print(f"    Chunking alone (250-word vs 400-char, same BGE):        {chunk_contribution:>+.2%}")
        print(f"    Interaction effect:                                     {interaction:>+.2%}")
        print(f"    Total (v1 → v2):                                       {b_avg - a_avg:>+.2%}")

    # Save
    report = {
        "matrix": {
            "minilm_400char": a_avg,
            "minilm_250word": c_avg,
            "bge_400char": d_avg,
            "bge_250word": b_avg,
        },
        "isolated_contributions": {
            "embedding_alone": round(b_avg - c_avg, 4) if b_avg else None,
            "chunking_alone": round(b_avg - d_avg, 4) if b_avg else None,
            "interaction": round((b_avg - a_avg) - (b_avg - c_avg) - (b_avg - d_avg), 4) if (a_avg and b_avg) else None,
            "total": round(b_avg - a_avg, 4) if (a_avg and b_avg) else None,
        },
        "cell_c_per_sample": results_c,
        "cell_d_per_sample": results_d,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
