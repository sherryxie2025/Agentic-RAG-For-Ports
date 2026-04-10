import json
import os
import re
import sys
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Allow import from src/ when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from online_pipeline.llm_client import llm_chat_raw_post, get_model_name

INPUT_PATH = "data/rules/rule_candidate_chunks_v1.json"
OUTPUT_PATH = "data/rules/raw_rules.json"

PROMPT = """
Extract operational rules from the following texts.

Return ONLY JSON.

Each rule must follow this schema:

[
{
"rule_text": "...",
"condition": "...",
"action": "...",
"variable": "...",
"operator": "...",
"value": "...",
"unit": "...",
"chunk_id": 123
}
]

Rules:
- variable must represent measurable quantity when possible
- operator should be one of > < >= <= = !=
- chunk_id MUST be copied from the chunk header

If no rule exists return [].

Texts:
"""


def call_llm(text):
    result = llm_chat_raw_post(
        messages=[
            {"role": "system", "content": "You extract structured rules and return JSON only."},
            {"role": "user", "content": PROMPT + text},
        ],
        temperature=0,
        timeout=(10, 180),
    )
    return result or "[]"


def extract_json_from_text(text):
    """
    Extract JSON list from LLM response.
    """
    match = re.search(r"\[.*\]", text, re.S)

    if match:
        try:
            return json.loads(match.group(0))
        except:
            return []

    return []


def build_batch_prompt(batch):

    texts = []

    for chunk in batch:

        texts.append(
            f"ChunkID: {chunk['chunk_id']}\n"
            f"{chunk['text']}"
        )

    joined_text = "\n\n".join(texts)

    return joined_text


def extract_rules():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    rules = []

    BATCH_SIZE = 5

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="LLM rule extraction"):

        batch = chunks[i:i+BATCH_SIZE]

        batch_map = {c["chunk_id"]: c for c in batch}

        try:

            prompt = build_batch_prompt(batch)

            response = call_llm(prompt)

            print("\nLLM RAW RESPONSE:")
            print(response)

            rule_json = extract_json_from_text(response)

            for r in rule_json:

                chunk_id = int(r.get("chunk_id"))

                source_chunk = batch_map.get(chunk_id)

                if source_chunk:

                    r["source_file"] = source_chunk["source_file"]
                    r["page"] = source_chunk["page"]

                else:

                    r["source_file"] = None
                    r["page"] = None

                for c in batch:
                    if c["chunk_id"] == chunk_id:
                        source_chunk = c
                        break

            rules.extend(rule_json)

        except Exception as e:
            print("LLM error:", e)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(rules, f, indent=2)

    print(f"Extracted rules: {len(rules)}")


if __name__ == "__main__":
    extract_rules()