import os
import json
from pathlib import Path
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_PATH = "data/Database B"
OUTPUT_PATH = "data/chunks"

os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_pdfs():
    """Recursively discover all PDFs under DATA_PATH."""
    pdf_files = []
    for root, _dirs, files in os.walk(DATA_PATH):
        for file in sorted(files):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def chunk_documents():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = []

    pdf_paths = load_pdfs()

    print(f"\nTotal PDFs: {len(pdf_paths)}\n")

    for doc_id, path in enumerate(tqdm(pdf_paths, desc="Processing PDFs")):

        file = os.path.basename(path)

        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
        except Exception as e:
            print(f"SKIP {file}: {e}")
            continue

        for page in pages:

            page_num = page.metadata.get("page_number", page.metadata.get("page", 0))

            texts = splitter.split_text(page.page_content)

            for i, text in enumerate(texts):

                chunk = {
                    "chunk_id": f"{doc_id}_{page_num}_{i}",
                    "doc_id": doc_id,
                    "source_file": file,
                    "page": page_num + 1,
                    "text": text
                }

                chunks.append(chunk)

        print(f"{file} → pages: {len(pages)} → chunks so far: {len(chunks)}")

    with open(f"{OUTPUT_PATH}/chunks_v1.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("\nChunking completed.")
    print(f"Total chunks: {len(chunks)}")


if __name__ == "__main__":
    chunk_documents()