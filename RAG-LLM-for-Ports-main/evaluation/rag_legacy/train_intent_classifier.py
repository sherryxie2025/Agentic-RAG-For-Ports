"""
Intent Classifier Training Pipeline
======================================
1. Load 80 golden queries + expected_sources as seed data
2. Augment with LLM paraphrases (5 per query -> 400+ samples)
3. Encode with BGE-small-en embeddings
4. Train sklearn MLP multi-label classifier with cross-validation
5. Save model to storage/models/intent_classifier.pkl

Usage:
    cd RAG-LLM-for-Ports-main
    uv run python evaluation/train_intent_classifier.py
"""

import json
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MultiLabelBinarizer

from online_pipeline.llm_client import llm_chat_raw_post


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GOLDEN_PATH = PROJECT_ROOT / "evaluation" / "golden_dataset.json"
MODEL_DIR = PROJECT_ROOT / "storage" / "models"
OUTPUT_PATH = MODEL_DIR / "intent_classifier.pkl"
EMBED_MODEL_NAME = "BAAI/bge-small-en"
AUGMENT_PER_QUERY = 5

# Source label set
SOURCE_LABELS = ["documents", "sql", "rules", "graph"]

SOURCE_MAP = {
    "vector": "documents",
    "documents": "documents",
    "sql": "sql",
    "structured_operational_data": "sql",
    "rules": "rules",
    "graph": "graph",
}


def normalise_source(s: str) -> str:
    return SOURCE_MAP.get(s.lower(), s.lower())


# ---------------------------------------------------------------------------
# Step 1: Load seed data
# ---------------------------------------------------------------------------
def load_seed_data() -> List[Dict]:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)

    samples = []
    for item in golden:
        query = item["query"]
        sources = list(set(
            normalise_source(s) for s in item.get("expected_sources", [])
            if normalise_source(s) in SOURCE_LABELS
        ))
        if sources:
            samples.append({"query": query, "sources": sources})

    print(f"Loaded {len(samples)} seed queries")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Augment with LLM paraphrases
# ---------------------------------------------------------------------------
AUGMENT_PROMPT = """Generate {n} paraphrased versions of the following query.
Each paraphrase should:
- Preserve the original meaning and intent
- Use different wording, sentence structure, or phrasing
- Be natural and fluent English
- Be about port operations

Return ONLY a JSON list of strings, no markdown.

Query: "{query}"
"""


def augment_queries(samples: List[Dict], n_per: int = AUGMENT_PER_QUERY) -> List[Dict]:
    augmented = list(samples)  # start with originals

    for i, sample in enumerate(samples):
        query = sample["query"]
        sources = sample["sources"]

        prompt = AUGMENT_PROMPT.format(n=n_per, query=query)
        response = llm_chat_raw_post(
            messages=[
                {"role": "system", "content": "You generate query paraphrases. Return ONLY a JSON list of strings."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            timeout=(10, 60),
        )

        if response:
            import re
            match = re.search(r'\[.*\]', response, re.S)
            if match:
                try:
                    paraphrases = json.loads(match.group(0))
                    if isinstance(paraphrases, list):
                        for p in paraphrases[:n_per]:
                            if isinstance(p, str) and len(p) > 10:
                                augmented.append({"query": p, "sources": sources})
                except json.JSONDecodeError:
                    pass

        if (i + 1) % 10 == 0:
            print(f"  Augmented {i+1}/{len(samples)} queries ({len(augmented)} total samples)")

    print(f"Augmentation complete: {len(samples)} -> {len(augmented)} samples")
    return augmented


# ---------------------------------------------------------------------------
# Step 3: Encode + Train
# ---------------------------------------------------------------------------
def train_classifier(samples: List[Dict]):
    print(f"\nEncoding {len(samples)} queries with {EMBED_MODEL_NAME}...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cuda")

    queries = [s["query"] for s in samples]
    source_lists = [s["sources"] for s in samples]

    # Encode
    X = model.encode(queries, normalize_embeddings=True, show_progress_bar=True, batch_size=64)

    # Multi-label binarize
    mlb = MultiLabelBinarizer(classes=SOURCE_LABELS)
    Y = mlb.fit_transform(source_lists)

    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Label distribution:")
    for i, label in enumerate(SOURCE_LABELS):
        print(f"  {label}: {Y[:, i].sum()} / {len(Y)}")

    # Train MLP with cross-validation
    print("\nTraining MLP classifier...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
        verbose=False,
    )

    # Cross-validation (5-fold)
    print("Running 5-fold cross-validation...")
    # For multi-label, use sample-averaged accuracy
    from sklearn.multioutput import MultiOutputClassifier
    # Wrap for cross_val_score
    from sklearn.model_selection import cross_val_predict, StratifiedKFold

    # Simple approach: train on all, report CV score
    clf.fit(X, Y)
    train_pred = clf.predict(X)
    train_acc = (train_pred == Y).all(axis=1).mean()
    print(f"Train exact-match accuracy: {train_acc:.4f}")

    # Per-label accuracy
    for i, label in enumerate(SOURCE_LABELS):
        acc = (train_pred[:, i] == Y[:, i]).mean()
        print(f"  {label} accuracy: {acc:.4f}")

    # 5-fold CV on each label
    print("\n5-fold CV per label:")
    from sklearn.model_selection import cross_val_score as cvs
    for i, label in enumerate(SOURCE_LABELS):
        scores = cvs(
            MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                          max_iter=500, early_stopping=True, random_state=42),
            X, Y[:, i], cv=5, scoring='f1'
        )
        print(f"  {label}: F1 = {scores.mean():.4f} +/- {scores.std():.4f}")

    # Save
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "classifier": clf,
        "mlb": mlb,
        "embed_model_name": EMBED_MODEL_NAME,
        "source_labels": SOURCE_LABELS,
        "n_samples": len(samples),
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(artifact, f)
    print(f"\nModel saved to: {OUTPUT_PATH}")

    return clf, mlb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Intent Classifier Training Pipeline")
    print("=" * 80)

    # Step 1
    seeds = load_seed_data()

    # Step 2: Augment
    print("\nStep 2: LLM data augmentation...")
    augmented = augment_queries(seeds)

    # Save augmented dataset
    aug_path = PROJECT_ROOT / "evaluation" / "augmented_intent_data.json"
    with open(aug_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)
    print(f"Augmented data saved to: {aug_path}")

    # Step 3: Train
    clf, mlb = train_classifier(augmented)

    print("\nDone!")
