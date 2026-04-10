import json
import os
import re
from tqdm import tqdm

INPUT_PATH = "data/chunks/chunks_v1.json"
OUTPUT_PATH = "data/rules/rule_candidate_chunks_v1.json"

os.makedirs("data/rules", exist_ok=True)

"""
RULE_PATTERNS = [
    "must",
    "shall",
    "required",
    "prohibited",
    "not allowed",
    "only when",
    "if ",
    "maximum",
    "minimum",
    "limit",
    "cannot",
    "may not",
]
"""


RULE_PATTERNS = [
    r"\bmust\b",
    r"\bshall\b",
    r"\bshould\b",
    r"\bprohibited\b",
    r"\bnot permitted\b",
    r"\bmust not\b",
    r"\brequired\b",
    r"\bat least\b",
    r"\bmaximum\b",
    r"\bminimum\b",
    r"\bwithin\b",
    r"\bbefore\b",
    r"\bafter\b"
    r"\bon condition\b",
    r"\bif\b",
    r"\bonly when\b",
    r"\bno vessel\b",
    r"\blimit\b",
    r"\bnot allowed\b",
    r"\bmay not\b",
    r"\bcannot\b"
]

def is_rule_candidate(text: str) -> bool:
    text_lower = text.lower()
    for p in RULE_PATTERNS:
        if re.search(p, text_lower):
            return True
    return False

def detect_patterns():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    candidates = []

    for chunk in tqdm(chunks, desc="Pattern detection"):

        if is_rule_candidate(chunk["text"]):
            candidates.append(chunk)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=2)

    print(f"Rule candidate chunks: {len(candidates)}")


if __name__ == "__main__":
    detect_patterns()