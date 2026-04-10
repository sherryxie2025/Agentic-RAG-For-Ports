import json
import os

RAW_RULES = "data/rules/raw_rules.json"
SCHEMA_PATH = "data/sql_data/schema_metadata.json"

GROUNDED_OUT = "data/rules/grounded_rules.json"
POLICY_OUT = "data/rules/policy_rules.json"


def load_schema_variables():
    """Load all column names from schema_metadata.json as grounding targets."""

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    variables = set()

    # schema_metadata.json has structure: {"tables": [...], ...}
    for table in schema.get("tables", []):
        for col in table.get("columns", []):
            col_name = col.get("name", "").lower()
            variables.add(col_name)
            # Also add synonyms
            for syn in col.get("synonyms", []):
                variables.add(syn.lower())

    return variables


def normalize():

    with open(RAW_RULES, "r") as f:
        rules = json.load(f)

    schema_vars = load_schema_variables()

    grounded = []
    policy = []

    for r in rules:

        var = r.get("variable", "").lower().strip()

        # Check if variable matches any schema column or synonym
        if var in schema_vars:
            r["sql_variable"] = var
            grounded.append(r)
        else:
            policy.append(r)

    with open(GROUNDED_OUT, "w") as f:
        json.dump(grounded, f, indent=2)

    with open(POLICY_OUT, "w") as f:
        json.dump(policy, f, indent=2)

    print("Grounded rules:", len(grounded))
    print("Policy rules:", len(policy))


if __name__ == "__main__":
    normalize()