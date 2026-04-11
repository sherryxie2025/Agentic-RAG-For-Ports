# src/offline_pipeline/taxonomy_generator.py
"""
Auto-generate rule taxonomy from DuckDB schema.

Previously, `taxonomy.py` was a hand-maintained Python dict with 57 variables
and a partial synonym map. This file replaces the manual maintenance with an
automated pipeline that reads the SQL schema and produces the taxonomy at
runtime.

Key advantages:
- Zero maintenance when schema changes (new columns auto-join taxonomy)
- Units inferred from column name suffixes (_ms, _m, _hours, _pct, ...)
- Category derived from table name
- Human-readable synonyms generated from column basename

Output:
    data/rules/taxonomy_auto.json  — generated taxonomy + synonym map

Usage:
    python -m src.offline_pipeline.taxonomy_generator
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import duckdb

logger = logging.getLogger("offline_pipeline.taxonomy_generator")

DEFAULT_DB_PATH = "storage/sql/port_ops.duckdb"
OUTPUT_PATH = "data/rules/taxonomy_auto.json"

# Tables to include (operational data). Views and derived tables excluded.
_INCLUDED_TABLES = {
    "environment",
    "berth_operations",
    "crane_operations",
    "yard_operations",
    "gate_operations",
    "vessel_calls",
}

# Columns to exclude — IDs, keys, raw timestamps
_EXCLUDED_COLUMNS = {
    "call_id", "mmsi", "visit_id", "vessel_name", "shipping_line",
    "terminal_code", "berth_id", "crane_id", "crane_operation_id",
    "timestamp", "created_at", "updated_at", "yard_operation_id",
    "gate_operation_id", "direction",
}

# Unit suffix → human-readable unit
_UNIT_SUFFIXES = {
    "_ms": "m/s",
    "_mph": "moves per hour",
    "_m": "meters",
    "_meters": "meters",
    "_ft": "feet",
    "_feet": "feet",
    "_hours": "hours",
    "_hrs": "hours",
    "_hr": "hours",
    "_minutes": "minutes",
    "_mins": "minutes",
    "_seconds": "seconds",
    "_s": "seconds",
    "_pct": "percent",
    "_percent": "percent",
    "_deg": "degrees",
    "_c": "celsius",
    "_hpa": "hectopascals",
    "_teu": "TEU",
    "_days": "days",
    "_day": "days",
    "_kg": "kilograms",
    "_tons": "tons",
    "_tonnes": "tonnes",
    "_knots": "knots",
}

# Table → category mapping (keeps compat with existing code)
_TABLE_TO_CATEGORY = {
    "environment": "weather_ocean_conditions",
    "berth_operations": "berth_operations",
    "crane_operations": "crane_operations",
    "yard_operations": "yard_operations",
    "gate_operations": "gate_operations",
    "vessel_calls": "vessel_characteristics",
}

# Manual seed synonyms for common port terms. Users may extend; auto-generator
# will add more based on the column basename split.
_SEED_SYNONYMS = {
    "wind": "wind_speed_ms",
    "wave": "wave_height_m",
    "tide": "tide_ft",
    "pressure": "pressure_hpa",
    "temperature": "air_temp_c",
    "berth_productivity": "berth_productivity_mph",
    "crane_productivity": "crane_productivity_mph",
    "turn_time": "average_turn_time_minutes",
    "dwell": "average_dwell_days",
    "delay": "arrival_delay_hours",
    "berth_delay": "berth_delay_hours",
}


# ---------------------------------------------------------------------------
# Unit + basename parsing
# ---------------------------------------------------------------------------

def _extract_unit(column_name: str) -> Optional[str]:
    """Infer unit from column name suffix."""
    name = column_name.lower()
    # Try longer suffixes first (avoid "_m" matching when "_mph" is present)
    for suffix in sorted(_UNIT_SUFFIXES.keys(), key=len, reverse=True):
        if name.endswith(suffix):
            return _UNIT_SUFFIXES[suffix]
    return None


def _basename(column_name: str) -> str:
    """Strip unit suffix from column name to get the base concept."""
    name = column_name.lower()
    for suffix in sorted(_UNIT_SUFFIXES.keys(), key=len, reverse=True):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _humanize(name: str) -> str:
    """Turn snake_case to human-readable words."""
    return name.replace("_", " ").strip()


def _generate_synonyms(column_name: str, basename: str) -> List[str]:
    """
    Generate plausible synonyms for a column.

    Strategy:
    - The basename in spaces form ("wave height")
    - The column name humanized ("wave height m")
    - Common abbreviations (if basename has multiple words, try acronym)
    - The basename as a single word for pattern matching
    """
    synonyms: Set[str] = set()
    synonyms.add(_humanize(basename))           # "wave height"
    synonyms.add(_humanize(column_name))        # "wave height m"
    synonyms.add(basename)                       # "wave_height"

    # Single-word queries ("wind" should match wind_speed_ms, "wave" → wave_height_m)
    parts = basename.split("_")
    if len(parts) >= 2 and len(parts[0]) > 2:
        synonyms.add(parts[0])

    # Remove empty
    return sorted(s for s in synonyms if s)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_taxonomy(
    db_path: str = DEFAULT_DB_PATH,
    output_path: str = OUTPUT_PATH,
) -> Dict[str, Any]:
    """
    Connect to DuckDB, read schema, and produce:
        {
            "taxonomy": {category -> [var1, var2, ...]},
            "variable_meta": {var -> {unit, category, source_table, sample_type}},
            "synonym_map": {synonym -> canonical_var},
            "stats": {tables_scanned, columns_included, columns_skipped}
        }
    """
    db_file = Path(db_path)
    if not db_file.exists():
        raise FileNotFoundError(f"DuckDB file not found: {db_file}")

    con = duckdb.connect(str(db_file), read_only=True)

    taxonomy: Dict[str, List[str]] = {}
    variable_meta: Dict[str, Dict[str, Any]] = {}
    synonym_map: Dict[str, str] = {}
    skipped = 0
    included = 0

    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='main' ORDER BY table_name"
        ).fetchall()

        for (table_name,) in tables:
            if table_name not in _INCLUDED_TABLES:
                continue

            category = _TABLE_TO_CATEGORY.get(table_name, table_name)
            taxonomy.setdefault(category, [])

            cols = con.execute(f'DESCRIBE "{table_name}"').fetchall()
            for col_info in cols:
                col_name = col_info[0]
                col_type = col_info[1]

                if col_name in _EXCLUDED_COLUMNS:
                    skipped += 1
                    continue

                # Only include numeric types (rules typically refer to metrics)
                if not any(
                    t in col_type.upper()
                    for t in ("INT", "DOUBLE", "FLOAT", "DECIMAL", "NUMERIC", "BIGINT")
                ):
                    skipped += 1
                    continue

                unit = _extract_unit(col_name)
                basename = _basename(col_name)
                synonyms = _generate_synonyms(col_name, basename)

                taxonomy[category].append(col_name)
                variable_meta[col_name] = {
                    "unit": unit,
                    "category": category,
                    "source_table": table_name,
                    "sql_type": col_type,
                    "basename": basename,
                    "synonyms": synonyms,
                }
                for syn in synonyms:
                    # Only set if not already claimed (first-come wins)
                    synonym_map.setdefault(syn.lower(), col_name)
                included += 1

        # Add seed synonyms (human-provided; override auto ones)
        for syn, canonical in _SEED_SYNONYMS.items():
            if canonical in variable_meta:
                synonym_map[syn] = canonical
    finally:
        con.close()

    result = {
        "taxonomy": taxonomy,
        "variable_meta": variable_meta,
        "synonym_map": synonym_map,
        "stats": {
            "tables_scanned": len([t for (t,) in tables if t in _INCLUDED_TABLES]),
            "columns_included": included,
            "columns_skipped": skipped,
            "unique_variables": len(variable_meta),
            "synonym_entries": len(synonym_map),
        },
    }

    # Save to JSON
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Taxonomy generated: %d variables across %d categories "
        "(%d synonyms, %d columns skipped)",
        len(variable_meta), len(taxonomy),
        len(synonym_map), skipped,
    )
    return result


def load_auto_taxonomy(
    path: str = OUTPUT_PATH, regenerate_if_missing: bool = True
) -> Dict[str, Any]:
    """
    Load the auto-generated taxonomy, regenerating from DuckDB if the file
    doesn't exist and regenerate_if_missing=True.
    """
    p = Path(path)
    if not p.exists():
        if regenerate_if_missing:
            return generate_taxonomy(output_path=path)
        raise FileNotFoundError(f"Auto-taxonomy file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = generate_taxonomy()

    print("\n" + "=" * 60)
    print("Auto-generated Taxonomy")
    print("=" * 60)
    stats = result["stats"]
    print(f"Tables scanned:    {stats['tables_scanned']}")
    print(f"Columns included:  {stats['columns_included']}")
    print(f"Columns skipped:   {stats['columns_skipped']}")
    print(f"Unique variables:  {stats['unique_variables']}")
    print(f"Synonym entries:   {stats['synonym_entries']}")

    print("\nCategories:")
    for cat, vars_list in result["taxonomy"].items():
        print(f"  {cat:<30} ({len(vars_list)} vars)")

    print("\nSample variables (first 5 per category):")
    for cat, vars_list in result["taxonomy"].items():
        print(f"\n  [{cat}]")
        for v in vars_list[:5]:
            meta = result["variable_meta"][v]
            unit = meta.get("unit") or "?"
            print(f"    {v:<35} unit={unit}")
