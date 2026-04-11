# src/offline_pipeline/semantic_chunker_v2.py
"""
Semantic Chunker v2: optimized for port domain PDFs with Small-to-Big retrieval.

Key improvements over v1 (chunk_documents.py) and v1.5 (semantic_chunker.py):

1. **Small-to-Big (Parent-Child) architecture**:
   - Child chunks (~250 words): precise retrieval units → goes into vector DB
   - Parent chunks (~1500 words): rich context → used for answer generation
   - Each child links to its parent via parent_id

2. **Structural chunking**: splits by numbered section headers (e.g., "2.1.4 Tides")
   rather than blind fixed-size splits. Preserves logical boundaries.

3. **Cross-page aggregation**: concatenates pages per document FIRST, then splits,
   so content flowing across pages is not broken.

4. **Enriched metadata**:
   - Publish year (extracted from filename)
   - Category (from directory structure: operations/environment/etc.)
   - Document type (handbook/policy/sustainability_report/...)
   - Section number, section title
   - is_table flag
   - parent_id for small-to-big lookup

5. **Table extraction**: uses pdfplumber to extract tables as markdown and
   stores them as dedicated chunks with is_table=True.

6. **Text cleaning**: fixes common PDF extraction artifacts:
   - Broken words ("differ ent" -> "different")
   - `?` substitution for spaces in some fonts
   - Page headers/footers (repeated across pages)

7. **Noise filtering**: drops "intentionally left blank", page numbers, etc.

Output:
    data/chunks/chunks_v2_children.json  (small chunks, go into vector DB)
    data/chunks/chunks_v2_parents.json   (big chunks, fetched for generation)

Usage:
    python -m src.offline_pipeline.semantic_chunker_v2

Then re-build embeddings:
    python -m src.offline_pipeline.build_embeddings_v2
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

logger = logging.getLogger("offline_pipeline.semantic_chunker_v2")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = "data/raw_documents"
OUTPUT_PATH = "data/chunks"

# --- Parent chunk (big, for generation context) ---
PARENT_TARGET_WORDS = 1500  # ~3 paragraphs / 1 full section
PARENT_MIN_WORDS = 400
PARENT_MAX_WORDS = 2500
PARENT_OVERLAP = 200

# --- Child chunk (small, for precise retrieval) ---
CHILD_TARGET_WORDS = 250    # 1-2 paragraphs / single idea
CHILD_MIN_WORDS = 60
CHILD_MAX_WORDS = 400
CHILD_OVERLAP = 50

# --- Performance guards ---
MAX_PDF_SIZE_MB = 15        # skip PDFs larger than this (pdfplumber slow on huge files)
DISABLE_TABLE_EXTRACTION = True  # skip per-page table extraction (main slowdown)

# Section header regex (matches "2.1", "2.1.4", "3.2.1.1", etc.)
_SECTION_HEADER_RE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+){0,3})\s+(?P<title>[A-Z][A-Za-z0-9 ,/\-()'\"&]{3,80})\s*$",
    re.MULTILINE,
)

# Heading candidates (ALL CAPS, sentence case)
_ALL_CAPS_HEADING_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9 &,\-/()'\"]{8,80})\s*$",
    re.MULTILINE,
)

# Noise patterns (case-insensitive; MULTILINE so ^$ match per line)
_NOISE_PATTERNS = [
    r"this page (is )?intentionally left blank",
    r"^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
    r"^\s*\d+\s*$",                           # just a page number
    r"^\s*(?:\u00a9|\(c\)).*all rights reserved.*$",
    r"^\s*printed\s+(on|by).*$",
]
_NOISE_RE = re.compile(
    "|".join(_NOISE_PATTERNS),
    re.MULTILINE | re.IGNORECASE,
)

# Broken word pattern: lower + space + lower (e.g., "differ ent", "L owest")
# Only fix when the second half is short (< 5 chars) to avoid false positives
_BROKEN_WORD_RE = re.compile(r"([a-z]{3,})\s+([a-z]{1,4})(?=\s)")

# Character substitution: `?` for space in known contexts
_QMARK_FIX_RE = re.compile(r"([a-zA-Z])\?([a-zA-Z])")

# Multiple whitespace → single
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str, known_broken: Optional[set] = None) -> str:
    """Apply cleaning pipeline to raw PDF text."""
    if not text:
        return ""

    # 1. Replace `?` between letters with space (font artifact)
    text = _QMARK_FIX_RE.sub(lambda m: m.group(1) + " " + m.group(2), text)

    # 2. Drop noise lines
    text = _NOISE_RE.sub("", text)

    # 3. Normalize whitespace (but preserve paragraph breaks)
    # First collapse single newlines (line wraps) but keep double newlines (paragraphs)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # single \n → space
    text = re.sub(r"\n{3,}", "\n\n", text)         # 3+ newlines → 2

    # 4. Fix broken-word patterns only for known_broken words if provided
    if known_broken:
        def _fix(m):
            joined = m.group(1) + m.group(2)
            if joined.lower() in known_broken:
                return joined
            return m.group(0)
        text = _BROKEN_WORD_RE.sub(_fix, text)

    # 5. Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def detect_repeated_headers_footers(pages_text: List[str]) -> List[str]:
    """
    Detect text that appears on many pages (likely header/footer).
    Returns list of strings to filter out.
    """
    if len(pages_text) < 5:
        return []

    # Extract first/last 3 lines of each page
    candidates = Counter()
    for text in pages_text:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            if 10 <= len(line) <= 120:
                candidates[line] += 1

    # Keep lines that appear on > 30% of pages
    threshold = max(3, int(len(pages_text) * 0.3))
    return [line for line, cnt in candidates.items() if cnt >= threshold]


# ---------------------------------------------------------------------------
# Section detection & splitting
# ---------------------------------------------------------------------------

def detect_sections(text: str) -> List[Dict[str, Any]]:
    """
    Detect logical sections in text using numbered headers.
    Returns list of {start, end, number, title, text}.
    """
    sections = []
    matches = list(_SECTION_HEADER_RE.finditer(text))

    if not matches:
        return [{"start": 0, "end": len(text), "number": "", "title": "", "text": text}]

    # First section: from start to first match
    if matches[0].start() > 100:
        sections.append({
            "start": 0,
            "end": matches[0].start(),
            "number": "",
            "title": "preamble",
            "text": text[:matches[0].start()].strip(),
        })

    # Numbered sections
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "start": start,
            "end": end,
            "number": m.group("num"),
            "title": m.group("title").strip(),
            "text": text[start:end].strip(),
        })

    return sections


def split_long_section(
    section: Dict[str, Any],
    target_words: int,
    max_words: int,
    overlap: int,
) -> List[Dict[str, Any]]:
    """
    Split an oversized section into sub-chunks with overlap, preserving metadata.
    """
    text = section["text"]
    words = text.split()

    if len(words) <= max_words:
        return [section]

    sub_chunks = []
    start = 0
    part = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        sub_text = " ".join(words[start:end])
        sub_chunks.append({
            **section,
            "text": sub_text,
            "sub_part": part,
            "word_count": end - start,
        })
        if end >= len(words):
            break
        start = end - overlap
        part += 1

    return sub_chunks


def split_parent_into_children(
    parent_text: str,
    target_words: int = CHILD_TARGET_WORDS,
    overlap: int = CHILD_OVERLAP,
    min_words: int = CHILD_MIN_WORDS,
) -> List[str]:
    """
    Split a parent chunk into child chunks for precise retrieval.

    Uses sentence-boundary-aware sliding window: targets `target_words` per
    child with `overlap` word overlap, but tries to cut at sentence boundaries
    when possible.
    """
    words = parent_text.split()
    if len(words) <= min_words:
        return [parent_text] if parent_text.strip() else []

    # Simple approach: sliding window on words, adjust to nearest sentence end
    children = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))

        # Try to extend to next sentence boundary (up to +30 words)
        if end < len(words):
            for look in range(min(30, len(words) - end)):
                if words[end + look].endswith((".", "!", "?", ":")):
                    end = end + look + 1
                    break

        sub = " ".join(words[start:end]).strip()
        if len(sub.split()) >= min_words or not children:
            children.append(sub)

        if end >= len(words):
            break
        start = end - overlap

    return children


def merge_small_sections(
    sections: List[Dict[str, Any]],
    min_words: int = PARENT_MIN_WORDS,
) -> List[Dict[str, Any]]:
    """
    Merge tiny sections into their neighbors.
    """
    if not sections:
        return sections

    merged = []
    buffer = None

    for sec in sections:
        wc = len(sec["text"].split())
        if wc < min_words:
            if buffer is None:
                buffer = dict(sec)
            else:
                buffer["text"] += "\n\n" + sec["text"]
                buffer["title"] = buffer.get("title", "") + " / " + sec.get("title", "")
        else:
            if buffer is not None:
                # Merge buffer into this section's start
                sec = dict(sec)
                sec["text"] = buffer["text"] + "\n\n" + sec["text"]
                buffer = None
            merged.append(sec)

    if buffer is not None:
        if merged:
            merged[-1]["text"] += "\n\n" + buffer["text"]
        else:
            merged.append(buffer)

    return merged


# ---------------------------------------------------------------------------
# PDF extraction (prefer pdfplumber for tables + PyMuPDF for text)
# ---------------------------------------------------------------------------

def extract_pdf_text_and_tables(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Extract per-page text + tables from a PDF.

    Strategy:
    - Fast text extraction via PyMuPDF (fitz) if available — 10-20x faster
      than pdfplumber on large PDFs.
    - Optional table extraction via pdfplumber (disabled by default — very slow
      on large documents; toggle via DISABLE_TABLE_EXTRACTION).
    - Fall back to PyPDFLoader if neither is available.

    Returns:
        pages_text: list of cleaned page texts
        tables: list of {page, markdown, row_count, col_count}
    """
    pages_text: List[str] = []
    tables: List[Dict[str, Any]] = []

    # --- Fast text extraction (PyMuPDF preferred) ---
    extracted = False
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        for page in doc:
            pages_text.append(page.get_text() or "")
        doc.close()
        extracted = True
    except ImportError:
        pass
    except Exception as e:
        logger.warning("PyMuPDF failed on %s: %s; will try pdfplumber", path, e)

    if not extracted:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    pages_text.append(page.extract_text() or "")
            extracted = True
        except ImportError:
            pass

    if not extracted:
        # Last resort
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        for page in loader.load():
            pages_text.append(page.page_content or "")

    # --- Optional table extraction (slow on large PDFs) ---
    if not DISABLE_TABLE_EXTRACTION:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_tables = page.extract_tables() or []
                    except Exception:
                        continue
                    for t in page_tables:
                        if not t or len(t) < 2:
                            continue
                        md = _table_to_markdown(t)
                        if md:
                            tables.append({
                                "page": page_num + 1,
                                "markdown": md,
                                "row_count": len(t),
                                "col_count": max(len(r) for r in t),
                            })
        except Exception:
            pass

    return pages_text, tables


def _table_to_markdown(rows: List[List[Any]]) -> str:
    """Convert table rows (list of lists) to markdown format."""
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]

    def cell(s):
        if s is None:
            return ""
        s = str(s).strip().replace("\n", " ")
        return s

    md_lines = []
    md_lines.append("| " + " | ".join(cell(c) for c in header) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in body:
        # pad row if shorter than header
        row = list(row) + [""] * (len(header) - len(row))
        md_lines.append("| " + " | ".join(cell(c) for c in row[:len(header)]) + " |")
    return "\n".join(md_lines)


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

# Year patterns — prefer 4-digit year in filename
_FILENAME_YEAR_RE = re.compile(r"(?<!\d)(19\d{2}|20[0-3]\d)(?!\d)")

# Month pattern in filename
_FILENAME_MONTH_RE = re.compile(
    r"(?:^|[-_])(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|"
    r"january|february|march|april|june|july|august|september|october|november|december)(?:[-_]|$)",
    re.IGNORECASE,
)

# Category mapping from the raw_documents directory structure
_DIR_TO_CATEGORY = {
    "operations": "operations",
    "environment_infrastructure": "environment",
    "management_governance": "management",
    "high_tech": "technology",
    "_duplicates": "unknown",
}


def extract_publish_year(filename: str, first_page_text: str = "") -> Optional[int]:
    """
    Extract publication year: prefer filename, fall back to first-page text.
    Ignores years in the future.
    """
    import datetime
    current_year = datetime.datetime.now().year

    # 1. Try filename first (most reliable)
    candidates = [int(m.group(1)) for m in _FILENAME_YEAR_RE.finditer(filename)]
    candidates = [y for y in candidates if 1990 <= y <= current_year]

    if candidates:
        return max(candidates)  # most recent year in filename

    # 2. Fall back to first-page text (look for "2023", "Report 2023", etc.)
    text_candidates = [int(m.group(1)) for m in _FILENAME_YEAR_RE.finditer(first_page_text[:1500])]
    text_candidates = [y for y in text_candidates if 1990 <= y <= current_year]

    if text_candidates:
        # Most common year in first page (Counter max)
        c = Counter(text_candidates).most_common(1)
        return c[0][0]

    return None


def extract_category(path: str) -> str:
    """Extract category from directory structure."""
    path_lower = path.replace("\\", "/").lower()
    for dir_name, category in _DIR_TO_CATEGORY.items():
        if f"/{dir_name}/" in path_lower or path_lower.endswith(f"/{dir_name}"):
            return category
    return "unknown"


def detect_doc_type(filename: str, first_page_text: str) -> str:
    """Heuristic document type classification from filename + first page."""
    fn = filename.lower()
    txt = first_page_text.lower()[:500]

    if "handbook" in fn or "handbook" in txt:
        return "handbook"
    if "policy" in fn or "white paper" in txt or "white-paper" in fn:
        return "policy"
    if "sustainab" in fn or "environment" in fn or "ghg" in fn:
        return "sustainability_report"
    if "annual report" in txt or "annual-report" in fn:
        return "annual_report"
    if "masterplan" in fn or "master-plan" in fn or "master plan" in txt:
        return "master_plan"
    if "facts" in fn or "statistics" in fn:
        return "facts_figures"
    if "guideline" in fn:
        return "guideline"
    return "document"


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def load_pdf_paths(root: str) -> Tuple[List[str], List[str]]:
    """
    Return (kept_paths, skipped_oversized_paths).
    PDFs larger than MAX_PDF_SIZE_MB are skipped for performance reasons.
    """
    kept = []
    skipped = []
    size_limit = MAX_PDF_SIZE_MB * 1024 * 1024
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if not f.lower().endswith(".pdf"):
                continue
            full = os.path.join(dirpath, f)
            try:
                sz = os.path.getsize(full)
            except OSError:
                continue
            if sz > size_limit:
                skipped.append(full)
            else:
                kept.append(full)
    return kept, skipped


def chunk_one_document(
    doc_id: int, path: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single PDF into parent and child chunks.

    Returns:
        (parents, children) — two lists of chunk dicts.
        Each child has parent_id pointing to its parent.
    """
    filename = os.path.basename(path)
    pages_text, tables = extract_pdf_text_and_tables(path)

    if not pages_text:
        return [], []

    # Detect and remove repeated headers/footers
    repeated = detect_repeated_headers_footers(pages_text)
    for line in repeated:
        pages_text = [t.replace(line, "") for t in pages_text]

    # Clean each page
    cleaned_pages = [clean_text(t) for t in pages_text]

    # Metadata extraction
    first_text = next((t for t in cleaned_pages if t), "")
    doc_type = detect_doc_type(filename, first_text)
    publish_year = extract_publish_year(filename, first_text)
    category = extract_category(path)

    base_metadata = {
        "doc_id": doc_id,
        "source_file": filename,
        "doc_type": doc_type,
        "publish_year": publish_year,
        "category": category,
    }

    # Concatenate all pages with page markers (for traceability)
    full_text_parts = []
    page_offsets = []  # (char_offset, page_num)
    offset = 0
    for page_idx, t in enumerate(cleaned_pages):
        page_offsets.append((offset, page_idx + 1))
        full_text_parts.append(t)
        offset += len(t) + 2  # +2 for "\n\n"

    full_text = "\n\n".join(full_text_parts)

    # Section-based splitting
    sections = detect_sections(full_text)
    sections = merge_small_sections(sections, min_words=PARENT_MIN_WORDS)

    # Split oversized sections into parent-sized chunks
    parent_sections: List[Dict[str, Any]] = []
    for sec in sections:
        parent_sections.extend(
            split_long_section(
                sec,
                target_words=PARENT_TARGET_WORDS,
                max_words=PARENT_MAX_WORDS,
                overlap=PARENT_OVERLAP,
            )
        )

    parents: List[Dict[str, Any]] = []
    children: List[Dict[str, Any]] = []

    for p_idx, sec in enumerate(parent_sections):
        text = sec["text"].strip()
        if not text or len(text) < 100:
            continue

        # Find which page this section starts on
        start_char = sec.get("start", 0)
        page_num = 1
        for off, pn in page_offsets:
            if off <= start_char:
                page_num = pn
            else:
                break

        section_num = sec.get("number", "") or f"p{page_num}"
        parent_id = f"{doc_id}__p__{section_num.replace('.', '-')}__{p_idx}"

        parent_chunk = {
            "chunk_id": parent_id,
            "parent_id": None,
            "chunk_type": "parent",
            "page": page_num,
            "text": text,
            "section_number": sec.get("number", ""),
            "section_title": sec.get("title", ""),
            "is_table": False,
            "word_count": len(text.split()),
            "char_count": len(text),
            **base_metadata,
        }
        parents.append(parent_chunk)

        # Split parent into child chunks
        child_texts = split_parent_into_children(text)
        for c_idx, child_text in enumerate(child_texts):
            if len(child_text.split()) < CHILD_MIN_WORDS and len(child_texts) > 1:
                continue

            child_id = f"{doc_id}__c__{section_num.replace('.', '-')}__{p_idx}__{c_idx}"
            children.append({
                "chunk_id": child_id,
                "parent_id": parent_id,
                "chunk_type": "child",
                "page": page_num,
                "text": child_text,
                "section_number": sec.get("number", ""),
                "section_title": sec.get("title", ""),
                "is_table": False,
                "word_count": len(child_text.split()),
                "char_count": len(child_text),
                **base_metadata,
            })

    # Add table chunks (tables are atomic — stored as BOTH parent and child)
    for t_idx, tbl in enumerate(tables):
        table_id = f"{doc_id}__t__{tbl['page']}__{t_idx}"
        table_chunk = {
            "chunk_id": table_id,
            "parent_id": None,
            "chunk_type": "parent",
            "page": tbl["page"],
            "text": tbl["markdown"],
            "section_number": "",
            "section_title": f"Table (page {tbl['page']})",
            "is_table": True,
            "word_count": len(tbl["markdown"].split()),
            "char_count": len(tbl["markdown"]),
            **base_metadata,
        }
        parents.append(table_chunk)
        # Also store as child (for retrieval) since tables are indivisible
        child_table = dict(table_chunk)
        child_table["chunk_id"] = f"{doc_id}__tc__{tbl['page']}__{t_idx}"
        child_table["parent_id"] = table_id
        child_table["chunk_type"] = "child"
        children.append(child_table)

    return parents, children


def run(data_path: str = DATA_PATH, output_path: str = OUTPUT_PATH) -> None:
    """Main entry point: produces parent and child chunk files."""
    os.makedirs(output_path, exist_ok=True)

    pdf_paths, skipped = load_pdf_paths(data_path)
    print(f"\nFound {len(pdf_paths)} PDFs under {data_path}")
    if skipped:
        print(f"  (skipped {len(skipped)} PDFs > {MAX_PDF_SIZE_MB}MB for performance)")
        for s in skipped[:5]:
            sz_mb = os.path.getsize(s) / 1024 / 1024
            print(f"    - {os.path.basename(s)} ({sz_mb:.1f} MB)")

    all_parents: List[Dict[str, Any]] = []
    all_children: List[Dict[str, Any]] = []
    doc_type_counts: Counter = Counter()
    category_counts: Counter = Counter()
    year_counts: Counter = Counter()
    failed: List[str] = []

    for doc_id, path in enumerate(tqdm(pdf_paths, desc="Chunking PDFs")):
        try:
            parents, children = chunk_one_document(doc_id, path)
            if parents:
                all_parents.extend(parents)
                all_children.extend(children)
                doc_type_counts[parents[0]["doc_type"]] += 1
                category_counts[parents[0].get("category", "unknown")] += 1
                yr = parents[0].get("publish_year")
                if yr:
                    year_counts[yr] += 1
        except Exception as e:
            logger.warning(f"FAILED {os.path.basename(path)}: {e}")
            failed.append(os.path.basename(path))

    parents_file = Path(output_path) / "chunks_v2_parents.json"
    children_file = Path(output_path) / "chunks_v2_children.json"

    with open(parents_file, "w", encoding="utf-8") as f:
        json.dump(all_parents, f, indent=2, ensure_ascii=False)
    with open(children_file, "w", encoding="utf-8") as f:
        json.dump(all_children, f, indent=2, ensure_ascii=False)

    # Back-compat: also write chunks_v2.json with the children (for BM25)
    # so existing code reading chunks_v2.json still works
    backcompat_file = Path(output_path) / "chunks_v2.json"
    with open(backcompat_file, "w", encoding="utf-8") as f:
        json.dump(all_children, f, indent=2, ensure_ascii=False)

    # Statistics
    pt = len(all_parents)
    ct = len(all_children)
    parent_wcs = [c["word_count"] for c in all_parents]
    child_wcs = [c["word_count"] for c in all_children]
    parent_tables = sum(1 for c in all_parents if c.get("is_table"))

    print("\n" + "=" * 60)
    print("Chunking v2 (Small-to-Big) — Complete")
    print("=" * 60)
    print(f"Parent chunks:     {pt}  (for generation context)")
    print(f"Child chunks:      {ct}  (for precise retrieval)")
    print(f"Table chunks:      {parent_tables}")
    print(f"Parent avg words:  {sum(parent_wcs) / max(pt, 1):.0f}")
    print(f"Parent median:     {sorted(parent_wcs)[len(parent_wcs) // 2] if parent_wcs else 0}")
    print(f"Child avg words:   {sum(child_wcs) / max(ct, 1):.0f}")
    print(f"Child median:      {sorted(child_wcs)[len(child_wcs) // 2] if child_wcs else 0}")
    print(f"Failed PDFs:       {len(failed)}")
    if failed[:5]:
        print(f"  First failures: {failed[:5]}")

    print(f"\nDocument types:")
    for dtype, cnt in doc_type_counts.most_common():
        print(f"  {dtype:<25} {cnt}")

    print(f"\nCategories (from directory):")
    for cat, cnt in category_counts.most_common():
        print(f"  {cat:<25} {cnt}")

    print(f"\nPublish years (top 10):")
    for yr, cnt in year_counts.most_common(10):
        print(f"  {yr}: {cnt}")

    print(f"\nOutputs:")
    print(f"  parents:  {parents_file}")
    print(f"  children: {children_file}")
    print(f"  (compat): {backcompat_file}")


if __name__ == "__main__":
    run()
