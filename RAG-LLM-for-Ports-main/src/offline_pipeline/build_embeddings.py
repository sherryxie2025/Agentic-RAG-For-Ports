import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

CHUNK_PATH = "data/chunks/chunks_v1.json"
OUTPUT_PATH = "data/chunks/chunks_with_embeddings_v1.json"

model = SentenceTransformer("BAAI/bge-small-en", device="cuda")


def build_embeddings():

    with open(CHUNK_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]

    embeddings = model.encode(texts, show_progress_bar=True)

    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    print("Embeddings generated.")


if __name__ == "__main__":
    build_embeddings()