# src/offline_pipeline/semantic_chunker.py
"""
Semantic Chunking with Parent-Child Metadata.

Strategy: semantic chunking + sliding window fallback, linked via parent-child.
- Semantic chunks control semantic completeness (topic boundaries)
- RecursiveCharacterTextSplitter fallback controls retrieval accuracy (size guard)
- Parent-child links allow pulling full page context when a child chunk matches

Design rationale:
- Fixed-size chunking breaks mid-sentence/topic, hurting retrieval precision
- Pure semantic chunking may produce very large chunks, hurting embedding quality
- Hybrid approach: semantic first, then size-guard fallback
"""

import json
import os
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

DATA_PATH = "data/Database B"
OUTPUT_PATH = "data/chunks"
MAX_CHUNK_SIZE = 800
FALLBACK_CHUNK_SIZE = 400
FALLBACK_OVERLAP = 100

os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_pdfs():
    pdf_files = []
    for root, _dirs, files in os.walk(DATA_PATH):
        for file in sorted(files):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def semantic_chunk_documents():
    """Chunk documents using semantic boundaries with parent-child metadata."""

    # Embedding model for semantic splitting
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cuda"},
    )

    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85,
    )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=FALLBACK_CHUNK_SIZE,
        chunk_overlap=FALLBACK_OVERLAP,
    )

    chunks = []
    pdf_paths = load_pdfs()
    print(f"\nTotal PDFs: {len(pdf_paths)}\n")

    for doc_id, path in enumerate(tqdm(pdf_paths, desc="Semantic Chunking")):
        file = os.path.basename(path)

        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
        except Exception as e:
            print(f"SKIP {file}: {e}")
            continue

        for page in pages:
            page_num = page.metadata.get("page_number", page.metadata.get("page", 0))
            page_text = page.page_content
            parent_id = f"{doc_id}_page_{page_num}"

            if not page_text.strip():
                continue

            # Try semantic chunking
            try:
                semantic_texts = semantic_splitter.split_text(page_text)
            except Exception:
                semantic_texts = [page_text]

            for i, text in enumerate(semantic_texts):
                method = "semantic"

                # Size guard: if chunk too large, fall back to recursive splitting
                if len(text) > MAX_CHUNK_SIZE:
                    sub_texts = fallback_splitter.split_text(text)
                    for j, sub_text in enumerate(sub_texts):
                        chunk = {
                            "chunk_id": f"{doc_id}_{page_num}_{i}_{j}",
                            "doc_id": doc_id,
                            "source_file": file,
                            "page": page_num + 1,
                            "text": sub_text,
                            "parent_id": parent_id,
                            "chunk_method": "recursive_fallback",
                        }
                        chunks.append(chunk)
                else:
                    chunk = {
                        "chunk_id": f"{doc_id}_{page_num}_{i}",
                        "doc_id": doc_id,
                        "source_file": file,
                        "page": page_num + 1,
                        "text": text,
                        "parent_id": parent_id,
                        "chunk_method": method,
                    }
                    chunks.append(chunk)

        if (doc_id + 1) % 50 == 0:
            print(f"  Processed {doc_id + 1}/{len(pdf_paths)}, chunks so far: {len(chunks)}")

    output_file = f"{OUTPUT_PATH}/chunks_v2_semantic.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    # Stats
    semantic_count = sum(1 for c in chunks if c["chunk_method"] == "semantic")
    fallback_count = sum(1 for c in chunks if c["chunk_method"] == "recursive_fallback")
    avg_len = sum(len(c["text"]) for c in chunks) / max(len(chunks), 1)

    print(f"\nSemantic Chunking completed.")
    print(f"Total chunks: {len(chunks)}")
    print(f"  Semantic: {semantic_count}")
    print(f"  Fallback: {fallback_count}")
    print(f"  Avg chunk length: {avg_len:.0f} chars")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    semantic_chunk_documents()
