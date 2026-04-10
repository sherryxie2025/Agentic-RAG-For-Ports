import sys
import os

# Ensure the offline_pipeline directory is on sys.path for sibling imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chunk_documents import chunk_documents
from build_embeddings import build_embeddings
from build_vector_db import build_vector_db


def run():

    print("=" * 60)
    print("Step 1: Chunking documents from Database B")
    print("=" * 60)
    chunk_documents()

    print("\n" + "=" * 60)
    print("Step 2: Building embeddings (BAAI/bge-small-en)")
    print("=" * 60)
    build_embeddings()

    print("\n" + "=" * 60)
    print("Step 3: Building ChromaDB vector store")
    print("=" * 60)
    build_vector_db()

    print("\n" + "=" * 60)
    print("Offline pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    run()