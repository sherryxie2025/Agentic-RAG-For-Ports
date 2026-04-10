"""
Annotate golden dataset with relevant chunk IDs for Recall@K evaluation.

Strategy:
1. Load all chunks and encode them with BGE-small-en
2. For each doc-involving golden query:
   a. Compute embedding similarity (query vs all chunks)
   b. Compute keyword overlap score
   c. Combine scores: 0.7 * embedding + 0.3 * keyword
   d. Take top-30 as relevant_chunk_ids (ground truth pool)
3. Save annotated golden dataset

This gives us true Recall@20 (pre-rerank) and Recall@5 (post-rerank).
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def keyword_relevance(text: str, keywords: list) -> float:
    """Fraction of expected keywords found in chunk text."""
    if not keywords:
        return 0.0
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords)


def annotate():
    # Load chunks
    chunks_path = PROJECT_ROOT / "data" / "chunks" / "chunks_v1.json"
    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded")

    # Load golden dataset
    golden_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    # Filter to doc-involving queries
    doc_queries = [g for g in golden if "vector" in g.get("expected_sources", [])]
    print(f"  {len(doc_queries)} doc-involving queries to annotate")

    # Load embedding model
    print("Loading embedding model ...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")

    # Encode all chunk texts (batched for memory efficiency)
    print("Encoding chunks (this may take a minute) ...")
    chunk_texts = [c["text"] for c in chunks]
    batch_size = 512
    chunk_embeddings = model.encode(
        chunk_texts, batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True,
    )
    print(f"  Chunk embeddings shape: {chunk_embeddings.shape}")

    # Annotate each query
    annotations = {}  # query_id -> list of relevant chunk_ids
    for i, item in enumerate(doc_queries):
        query_id = item["id"]
        query = item["query"]
        keywords = item.get("expected_evidence_keywords", [])

        # Encode query
        query_emb = model.encode([query], normalize_embeddings=True)[0]

        # Cosine similarity (embeddings already normalized)
        emb_scores = chunk_embeddings @ query_emb  # shape: (n_chunks,)

        # Keyword relevance scores
        kw_scores = np.array([
            keyword_relevance(c["text"], keywords) for c in chunks
        ])

        # Combined score
        combined = 0.7 * emb_scores + 0.3 * kw_scores

        # Get top-30 indices
        top_indices = np.argsort(combined)[::-1][:30]

        relevant_chunks = []
        for idx in top_indices:
            score = float(combined[idx])
            # Only include if score is above a minimum threshold
            if score < 0.3:
                break
            relevant_chunks.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "source_file": chunks[idx]["source_file"],
                "page": chunks[idx].get("page"),
                "score": round(score, 4),
            })

        annotations[query_id] = [c["chunk_id"] for c in relevant_chunks]

        print(
            f"  [{i+1}/{len(doc_queries)}] {query_id}: "
            f"{len(relevant_chunks)} relevant chunks "
            f"(top_score={relevant_chunks[0]['score']:.3f} "
            f"from {relevant_chunks[0]['source_file'][:40]})"
        )

    # Merge annotations back into golden dataset
    for item in golden:
        query_id = item["id"]
        if query_id in annotations:
            item["relevant_chunk_ids"] = annotations[query_id]
        else:
            item["relevant_chunk_ids"] = []

    # Save annotated golden dataset
    out_path = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=2, ensure_ascii=False)

    print(f"\nAnnotated golden dataset saved to {out_path}")
    print(f"  {len(annotations)} queries annotated with relevant_chunk_ids")

    # Stats
    chunk_counts = [len(v) for v in annotations.values()]
    print(f"  Chunks per query: min={min(chunk_counts)} max={max(chunk_counts)} avg={np.mean(chunk_counts):.1f}")


if __name__ == "__main__":
    annotate()
