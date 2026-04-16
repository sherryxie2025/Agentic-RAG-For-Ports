import json
import chromadb
from tqdm import tqdm

CHUNK_PATH = "data/chunks/chunks_with_embeddings_v1.json"

CHROMA_PATH = "storage/chroma"


def build_vector_db():

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Drop old collection if exists, then recreate
    try:
        client.delete_collection("port_documents")
        print("Deleted old collection.")
    except Exception:
        pass
    collection = client.create_collection("port_documents")

    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    batch_size = 1000

    for i in tqdm(range(0, len(chunks), batch_size), desc="Building Vector DB"):

        batch = chunks[i:i+batch_size]

        documents = []
        embeddings = []
        ids = []
        metadatas = []

        for chunk in batch:

            documents.append(chunk["text"])
            embeddings.append(chunk["embedding"])
            ids.append(str(chunk["chunk_id"]))

            metadata = {
                "source_file": chunk["source_file"],
                "page": chunk["page"]
            }

            metadatas.append(metadata)

        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    print(f"Vector DB built successfully. Total: {collection.count()} chunks.")


if __name__ == "__main__":
    build_vector_db()