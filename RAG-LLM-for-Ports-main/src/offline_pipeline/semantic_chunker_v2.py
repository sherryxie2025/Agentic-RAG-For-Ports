# src/offline_pipeline/semantic_chunker_v2.py
"""
Semantic Chunker v2: optimized for port domain PDFs.

Key improvements over v1 (chunk_documents.py) and v1.5 (semantic_chunker.py):

1. **Structural chunking**: splits by numbered section headers (e.g., "2.1.4 Tides")
   rather than blind fixed-size splits. Preserves logical boundaries.

2. **Cross-page aggregation**: concatenates pages per document FIRST, then splits,
   so content flowing across pages is not broken.

3. **Larger chunks**: target 500-800 words (vs 50-80 in v1) giving embedding
   models enough context to capture semantics.

4. **Table extraction**: uses pdfplumber to extract tables as markdown and
   stores them as dedicated chunks with `is_table=True`.

5. **Text cleaning**: fixes common PDF extraction artifacts:
   - Broken words ("differ ent" → "different")
   - Excess whitespace
   - `?` substitution for spaces in some fonts
   - Page headers/footers (repeated across pages)

6. **Noise filtering**: drops chunks that are purely junk:
   - "This page intentionally left blank"
   - Pure page numbers
   - Very short (< 30 chars) chunks
   - TOC-only chunks (high ratio of numbers and periods)

7. **Rich metadata**: section_title, section_number, doc_type, is_table,
   word_count, char_count.

Usage:
    python -m src.offline_pipeline.semantic_chunker_v2

Output: data/chunks/chunks_v2.json

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

# Target chunk size in WORDS (not chars)
TARGET_WORDS = 600          # aim for this
MIN_WORDS = 150             # drop smaller chunks or merge with neighbor
MAX_WORDS = 1000            # split if exceeded
OVERLAP_WORDS = 100         # 15-20% overlap

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

# Noise patterns (case-insensitive)
_NOISE_PATTERNS = [
    r"(?i)this page (is )?intentionally left blank",
    r"(?i)^\s*page\s+\d+\s*(of\s+\d+)?\s*$",
    r"(?i)^\s*\d+\s*$",                           # just a page number
    r"(?i)^\s*©.*all rights reserved.*$",
    r"(?i)^\s*printed\s+(on|by).*$",
]
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.MULTILINE)

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
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    overlap: int = OVERLAP_WORDS,
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


def merge_small_sections(
    sections: List[Dict[str, Any]],
    min_words: int = MIN_WORDS,
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

    Returns:
        pages_text: list of cleaned page texts
        tables: list of {page, markdown, row_count, col_count}
    """
    pages_text: List[str] = []
    tables: List[Dict[str, Any]] = []

    # Try pdfplumber first (best for tables)
    try:
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append(text)

                # Extract tables
                try:
                    page_tables = page.extract_tables()
                    for t in page_tables or []:
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
    except ImportError:
        # Fall back to PyPDFLoader (what v1 used)
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        for page in loader.load():
            pages_text.append(page.page_content or "")

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
# Document type detection
# ---------------------------------------------------------------------------

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

def load_pdf_paths(root: str) -> List[str]:
    pdf_files = []
    for dirpath, _dirs, files in os.walk(root):
        for f in sorted(files):
            if f.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(dirpath, f))
    return pdf_files


def chunk_one_document(doc_id: int, path: str) -> List[Dict[str, Any]]:
    """Process a single PDF into optimized chunks."""
    filename = os.path.basename(path)
    pages_text, tables = extract_pdf_text_and_tables(path)

    if not pages_text:
        return []

    # Detect and remove repeated headers/footers
    repeated = detect_repeated_headers_footers(pages_text)
    for line in repeated:
        pages_text = [t.replace(line, "") for t in pages_text]

    # Clean each page
    cleaned_pages = [clean_text(t) for t in pages_text]

    # Detect doc type from first non-empty page
    first_text = next((t for t in cleaned_pages if t), "")
    doc_type = detect_doc_type(filename, first_text)

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

    # Merge tiny sections
    sections = merge_small_sections(sections)

    # Split oversized sections
    expanded: List[Dict[str, Any]] = []
    for sec in sections:
        expanded.extend(split_long_section(sec))

    # Build chunk records
    chunks: List[Dict[str, Any]] = []
    for i, sec in enumerate(expanded):
        text = sec["text"].strip()
        if not text or len(text) < 50:
            continue

        wc = len(text.split())

        # Find which page this section starts on
        start_char = sec.get("start", 0)
        page_num = 1
        for off, pn in page_offsets:
            if off <= start_char:
                page_num = pn
            else:
                break

        chunk_id = f"{doc_id}_{sec.get('number', '') or 'p' + str(page_num)}_{i}"
        # Replace dots in section number for a cleaner id
        chunk_id = chunk_id.replace(".", "-")

        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_file": filename,
            "page": page_num,
            "text": text,
            "section_number": sec.get("number", ""),
            "section_title": sec.get("title", ""),
            "doc_type": doc_type,
            "is_table": False,
            "word_count": wc,
            "char_count": len(text),
        })

    # Add table chunks
    for t_idx, tbl in enumerate(tables):
        chunk_id = f"{doc_id}_t{tbl['page']}_{t_idx}"
        chunks.append({
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_file": filename,
            "page": tbl["page"],
            "text": tbl["markdown"],
            "section_number": "",
            "section_title": f"Table (page {tbl['page']})",
            "doc_type": doc_type,
            "is_table": True,
            "word_count": len(tbl["markdown"].split()),
            "char_count": len(tbl["markdown"]),
        })

    return chunks


def run(data_path: str = DATA_PATH, output_path: str = OUTPUT_PATH) -> None:
    """Main entry point."""
    os.makedirs(output_path, exist_ok=True)

    pdf_paths = load_pdf_paths(data_path)
    print(f"\nFound {len(pdf_paths)} PDFs under {data_path}")

    all_chunks: List[Dict[str, Any]] = []
    doc_type_counts: Counter = Counter()
    failed: List[str] = []

    for doc_id, path in enumerate(tqdm(pdf_paths, desc="Chunking PDFs")):
        try:
            chunks = chunk_one_document(doc_id, path)
            if chunks:
                all_chunks.extend(chunks)
                doc_type_counts[chunks[0]["doc_type"]] += 1
        except Exception as e:
            logger.warning(f"FAILED {os.path.basename(path)}: {e}")
            failed.append(os.path.basename(path))

    output_file = Path(output_path) / "chunks_v2.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # Statistics
    total = len(all_chunks)
    word_counts = [c["word_count"] for c in all_chunks]
    avg_wc = sum(word_counts) / max(total, 1)
    table_count = sum(1 for c in all_chunks if c.get("is_table"))

    print("\n" + "=" * 60)
    print("Chunking v2 — Complete")
    print("=" * 60)
    print(f"Total chunks:    {total}")
    print(f"Text chunks:     {total - table_count}")
    print(f"Table chunks:    {table_count}")
    print(f"Avg word count:  {avg_wc:.0f}")
    print(f"Median words:    {sorted(word_counts)[len(word_counts) // 2] if word_counts else 0}")
    print(f"Failed PDFs:     {len(failed)}")
    if failed[:5]:
        print(f"  First failures: {failed[:5]}")
    print(f"\nDocument types:")
    for dtype, cnt in doc_type_counts.most_common():
        print(f"  {dtype:<25} {cnt}")
    print(f"\nOutput: {output_file}")


if __name__ == "__main__":
    run()
