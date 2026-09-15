"""
Technical identifier extraction for Servy RAG.

Extracts fault codes, error codes, part numbers (IPN), model IDs, and
other technical identifiers from queries and document text.  Provides
deterministic exact-match priority for identifier queries and abstention
when an identifier is not found in authorized sources.
"""

import re
from typing import Set

# ---------------------------------------------------------------------------
# Identifier patterns (order matters — more specific first)
# ---------------------------------------------------------------------------

_IDENTIFIER_PATTERNS = [
    # IPN part numbers: IPN-442098, IPN 558812
    (r"\bIPN\s*[-–]?\s*(\d{4,9})\b", "IPN-{0}"),
    # Explicit error codes: E12, E-17, E0042, ERR-12
    (r"\b(?:ERR(?:OR)?|E)\s*[-–]?\s*(\d{2,4})\b", "E{0}"),
    # Fault codes: F12, F-2
    (r"\bF\s*[-–]?\s*(\d{1,4})\b", "F{0}"),
    # General alphanumeric codes: BA3K-REV2, PX7-A (1-3 uppercase + optional separator + digits + optional letter)
    (r"\b([A-Z]{1,3}[-–]?\d{2,6}[A-Z]?(?:[-–][A-Z0-9]{1,6})?)\b", "{0}"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), fmt) for p, fmt in _IDENTIFIER_PATTERNS]


def extract_identifiers(text: str) -> Set[str]:
    """Extract all technical identifiers from text.

    Returns a set of normalized identifier strings, e.g. {"E12", "IPN-442098"}.
    """
    text = (text or "").strip()
    if not text:
        return set()

    identifiers = set()
    for pattern, fmt in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            groups = match.groups()
            if groups:
                normalized = fmt.format(*groups).upper()
            else:
                normalized = match.group(0).upper()
            identifiers.add(normalized)
    return identifiers


def extract_error_codes(text: str) -> Set[str]:
    """Extract only error/fault codes (E-series, F-series, ERR-series)."""
    text = (text or "").strip()
    if not text:
        return set()

    codes = set()
    # E-series: E12, E-17, E0042, ERR-12
    for m in re.finditer(r"\b(?:ERR(?:OR)?|E)\s*[-–]?\s*(\d{2,4})\b", text, re.IGNORECASE):
        codes.add(f"E{m.group(1)}")
    # F-series: F2, F-12
    for m in re.finditer(r"\bF\s*[-–]?\s*(\d{1,4})\b", text, re.IGNORECASE):
        codes.add(f"F{m.group(1)}")
    return codes


def extract_part_numbers(text: str) -> Set[str]:
    """Extract IPN part numbers."""
    text = (text or "").strip()
    if not text:
        return set()

    parts = set()
    for m in re.finditer(r"\bIPN\s*[-–]?\s*(\d{4,9})\b", text, re.IGNORECASE):
        parts.add(f"IPN-{m.group(1)}")
    return parts


def query_identifiers_not_in_sources(query: str, source_texts: list) -> Set[str]:
    """Find identifiers in the query that do not appear in any source text.

    Used to detect when a user asks about an identifier (e.g. error code E99)
    that is not documented in any retrieved source, triggering abstention.
    """
    query_ids = extract_error_codes(query)
    if not query_ids:
        return set()

    combined_sources = " ".join(source_texts).upper()
    missing = set()
    for qid in query_ids:
        if qid.upper() not in combined_sources:
            missing.add(qid)
    return missing
