# src/offline_pipeline/build_embeddings_v2.py
"""
Build embeddings for chunks_v2.json using BGE-base.

Upgrades from v1:
- Model: `BAAI/bge-small-en` -> `BAAI/bge-base-en-v1.5`
  (768 dim vs 384, better MTEB retrieval score: 53.3 vs 43.8)
- BGE expects a special query prefix for retrieval (passages do NOT get prefix)
- Writes embeddings into a new Chroma collection `port_documents_v2`
  (keeping v1 intact for A/B comparison)

Usage:
    python -m src.offline_pipeline.build_embeddings_v2

After running, update document_retriever.py to use collection_name='port_documents_v2'
and add the query prefix when embedding queries.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

logger = logging.getLogger("offline_pipeline.build_embeddings_v2")

CHUNK_PATH = "data/chunks/chunks_v2.json"
EMBEDDINGS_PATH = "data/chunks/chunks_v2_with_embeddings.json"
CHROMA_COLLECTION_NAME = "port_documents_v2"

# BGE-base: 768 dim, 110MB, MTEB retrieval 53.3
EMBED_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# BGE query prefix (only for queries, not passages)
# See: https://huggingface.co/BAAI/bge-base-en-v1.5
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def build_embeddings_v2() -> None:
    """Embed chunks_v2 and write Chroma collection v2."""
    from sentence_transformers import SentenceTransformer
    import chromadb

    chunk_file = Path(CHUNK_PATH)
    if not chunk_file.exists():
        raise FileNotFoundError(
            f"{chunk_file} not found. Run semantic_chunker_v2.py first."
        )

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {chunk_file}")

    # Load BGE model
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")
    model.max_seq_length = 512  # BGE max

    # Encode passages (no prefix for passages)
    print("Encoding passages (no prefix)...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # BGE recommends normalization
        convert_to_numpy=True,
    )

    # Save embeddings to JSON (optional backup)
    print(f"Saving embeddings to {EMBEDDINGS_PATH}...")
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()
    with open(EMBEDDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    # Build Chroma collection v2
    print(f"\nBuilding Chroma collection: {CHROMA_COLLECTION_NAME}")
    chroma_path = Path("storage/chroma")
    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete old v2 collection if exists
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        print(f"  Deleting existing collection '{CHROMA_COLLECTION_NAME}'")
        client.delete_collection(name=CHROMA_COLLECTION_NAME)

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine works best for normalized BGE
    )

    # Add in batches (Chroma has size limits)
    batch_size = 1000
    print(f"Inserting {len(chunks)} chunks in batches of {batch_size}...")
    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            metadatas=[
                {
                    "doc_id": c.get("doc_id", 0),
                    "source_file": c.get("source_file", ""),
                    "page": c.get("page", 0),
                    "section_number": c.get("section_number", ""),
                    "section_title": c.get("section_title", ""),
                    "doc_type": c.get("doc_type", "document"),
                    "is_table": bool(c.get("is_table", False)),
                    "word_count": int(c.get("word_count", 0)),
                }
                for c in batch
            ],
        )

    print(f"\nCollection '{CHROMA_COLLECTION_NAME}' built with {collection.count()} chunks.")
    print(f"\nNext step: update HybridDocumentRetriever to use collection_name='{CHROMA_COLLECTION_NAME}'")
    print("and prepend the BGE query prefix when embedding queries.")


if __name__ == "__main__":
    build_embeddings_v2()
