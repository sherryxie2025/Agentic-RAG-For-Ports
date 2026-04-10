import json
import re
from taxonomy import RULE_VARIABLE_TAXONOMY, VARIABLE_SYNONYMS


INPUT_PATH = "data/rules/raw_rules.json"

OUTPUT_GROUNDED = "data/rules/grounded_rules.json"
OUTPUT_POLICY = "data/rules/policy_rules.json"


def normalize(text):

    if text is None:
        return ""

    return text.lower().replace("_", " ").strip()


def flatten_taxonomy():

    variables = []

    for group in RULE_VARIABLE_TAXONOMY.values():
        variables.extend(group)

    return variables


SQL_VARIABLES = flatten_taxonomy()


def build_synonym_map():

    mapping = {}

    for var in SQL_VARIABLES:

        mapping[normalize(var)] = var

    for var, synonyms in VARIABLE_SYNONYMS.items():

        for s in synonyms:

            mapping[normalize(s)] = var

    return mapping


SYNONYM_MAP = build_synonym_map()


def ground_variable(variable):

    if variable is None:
        return None

    v = normalize(variable)

    if v in SYNONYM_MAP:
        return SYNONYM_MAP[v]

    for key in SYNONYM_MAP:

        if key in v:
            return SYNONYM_MAP[key]

    return None


def normalize_operator(op):

    if op is None:
        return None

    op = op.strip()

    valid = [">", "<", ">=", "<=", "=", "!="]

    if op in valid:
        return op

    return None


def normalize_value(value):

    if value is None:
        return None

    try:
        return float(value)
    except:
        return value


def run_grounding():

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)

    grounded = []
    policy = []

    for r in rules:

        variable = r.get("variable")

        sql_variable = ground_variable(variable)

        r["operator"] = normalize_operator(r.get("operator"))
        r["value"] = normalize_value(r.get("value"))

        if sql_variable:

            r["sql_variable"] = sql_variable

            grounded.append(r)

        else:

            policy.append(r)

    with open(OUTPUT_GROUNDED, "w", encoding="utf-8") as f:
        json.dump(grounded, f, indent=2)

    with open(OUTPUT_POLICY, "w", encoding="utf-8") as f:
        json.dump(policy, f, indent=2)

    print("Grounded rules:", len(grounded))
    print("Policy rules:", len(policy))


if __name__ == "__main__":

    run_grounding()