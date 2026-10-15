"""
Technical identifier extraction and canonical normalization for Servy RAG.

Extracts fault codes, error codes, part numbers (IPN), model IDs, and
other technical identifiers from queries and document text. Provides
deterministic exact-match priority for identifier queries and abstention
when an identifier is not found in authorized sources.
"""

import re
from typing import Set

# Unicode dashes: ASCII hyphen, figure dash, en-dash, em-dash, horizontal bar, minus sign
DASH_CHARS = r"[\-\u2010\u2011\u2012\u2013\u2014\u2015\u2212]"

_IDENTIFIER_PATTERNS = [
    # IPN part numbers: IPN-442098, IPN 558812, IPN–442098
    (rf"\bIPN\s*{DASH_CHARS}?\s*(\d{{4,9}})\b", "IPN-{0}"),
    # Explicit error codes: E12, E-17, E 04, ERR-04, ERROR-04, E0042, ERR 12
    (rf"\b(?:ERR(?:OR)?|E)\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", "E{0}"),
    # Fault codes: F12, F-2, F 07
    (rf"\bF\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", "F{0}"),
    # Alarm codes: ALM-01, ALM 01, ALARM-02
    (rf"\b(?:ALM|ALARM)\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", "ALM-{0}"),
    # General alphanumeric codes: BA3K-REV2, PX7-A, BA-001 (1-4 letters + separator + alphanum)
    (rf"\b([A-Z]{{1,4}}{DASH_CHARS}[A-Z0-9]{{1,10}}(?:{DASH_CHARS}[A-Z0-9]{{1,8}})*)\b", "{0}"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), fmt) for p, fmt in _IDENTIFIER_PATTERNS]


def extract_identifiers(text: str) -> Set[str]:
    """Extract all technical identifiers from text with canonical normalization.

    Returns a set of normalized identifier strings, e.g. {"E04", "IPN-442098", "BA3K-REV2"}.
    Both query and document texts are processed through this exact canonical parser.
    """
    text = (text or "").strip()
    if not text:
        return set()

    identifiers = set()
    for pattern, fmt in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if groups:
                raw_group = groups[0]
                cleaned_group = re.sub(DASH_CHARS, "-", raw_group)
                normalized = fmt.format(cleaned_group).upper()
            else:
                normalized = re.sub(DASH_CHARS, "-", match.group(0)).upper()
            identifiers.add(normalized)
    return identifiers


def extract_error_codes(text: str) -> Set[str]:
    """Extract only error/fault codes (E-series, F-series, ERR-series, ALM-series)."""
    text = (text or "").strip()
    if not text:
        return set()

    codes = set()
    # E-series: E04, E-04, E 04, E004, ERR-04, ERROR-04
    for m in re.finditer(rf"\b(?:ERR(?:OR)?|E)\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", text, re.IGNORECASE):
        codes.add(f"E{m.group(1).upper()}")
    # F-series: F2, F-07, F 07
    for m in re.finditer(rf"\bF\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", text, re.IGNORECASE):
        codes.add(f"F{m.group(1).upper()}")
    # ALM-series: ALM-01, ALARM-02
    for m in re.finditer(rf"\b(?:ALM|ALARM)\s*{DASH_CHARS}?\s*(\d{{1,5}})\b", text, re.IGNORECASE):
        codes.add(f"ALM-{m.group(1).upper()}")
    return codes


def extract_part_numbers(text: str) -> Set[str]:
    """Extract IPN part numbers."""
    text = (text or "").strip()
    if not text:
        return set()

    parts = set()
    for m in re.finditer(rf"\bIPN\s*{DASH_CHARS}?\s*(\d{{4,9}})\b", text, re.IGNORECASE):
        parts.add(f"IPN-{m.group(1)}")
    return parts


def extract_document_ids(text: str) -> Set[int]:
    """Extract explicit document ID references from text, e.g. 'document 12', 'doc #4', 'doc 42'."""
    text = (text or "").strip()
    if not text:
        return set()
    doc_ids = set()
    for m in re.finditer(r"\b(?:doc(?:ument)?|doc_id|manual)\s*(?:#|id|no\.?)?\s*(\d+)\b", text, re.IGNORECASE):
        try:
            doc_ids.add(int(m.group(1)))
        except ValueError:
            pass
    return doc_ids


def query_identifiers_not_in_sources(query: str, source_texts: list) -> Set[str]:
    """Find identifiers in the query that do not appear in any source text.

    Uses exact canonical set difference instead of naive substring matching,
    preventing prefix collisions and Unicode dash mismatches.
    """
    query_ids = extract_error_codes(query)
    if not query_ids:
        return set()

    source_codes = set()
    for text in source_texts:
        if text:
            source_codes.update(extract_error_codes(text))
    return query_ids - source_codes
