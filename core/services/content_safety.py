"""
Content safety module for Servy RAG.

Performs Unicode normalization, zero-width character stripping, structural
delimiter neutralization, and expanded prompt-injection / query-abuse detection.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Unicode and structural pre-processing
# ---------------------------------------------------------------------------

# Zero-width and invisible characters that can be used to bypass regex
_ZERO_WIDTH_CHARS = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff\u00ad]"
)

# Structural delimiters used in the RAG prompt that could be spoofed
_STRUCTURAL_PATTERNS = [
    r"</?retrieved_source\b[^>]*>",
    r"UNTRUSTED_CONTENT_START",
    r"UNTRUSTED_CONTENT_END",
    r"\bSYSTEM\s*:",
    r"\bASSISTANT\s*:",
    r"\bHUMAN\s*:",
    r"\bUSER\s*:",
]

_STRUCTURAL_RE = re.compile(
    "|".join(_STRUCTURAL_PATTERNS), re.IGNORECASE
)


def _normalize_text(text: str) -> str:
    """Canonicalize text for safety scanning: NFKC normalize, strip zero-width
    characters, collapse whitespace."""
    text = text or ""
    # NFKC normalizes full-width characters, ligatures, etc.
    text = unicodedata.normalize("NFKC", text)
    # Strip zero-width characters
    text = _ZERO_WIDTH_CHARS.sub("", text)
    # Collapse multiple whitespace but preserve newlines
    text = re.sub(r"[^\S\n]+", " ", text)
    return text


# ---------------------------------------------------------------------------
# Prompt injection detection patterns (for document content)
# ---------------------------------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(the\s+)?system\s+prompt",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(me\s+)?(your\s+)?developer\s+message",
    r"you\s+are\s+chatgpt",
    r"jailbreak",
    r"follow\s+these\s+instructions\s+instead",
    r"exfiltrat(e|ion)",
    r"reveal\s+secrets?",
    # Expanded patterns
    r"disregard\s+(all\s+)?prior\s+(instructions|context)",
    r"override\s+(system|safety)\s+(instructions|prompt|controls)",
    r"pretend\s+you\s+are\s+(not\s+)?a",
    r"act\s+as\s+(if\s+)?(you\s+)?(are|were)\s+",
    r"new\s+instructions?\s*:",
    r"forget\s+(everything|all|your)\s+(above|previous|prior)",
    r"do\s+not\s+follow\s+(the\s+)?(system|safety|original)",
    r"bypass\s+(the\s+)?(safety|content|security|filter)",
    r"output\s+(the\s+)?(system|initial|hidden)\s+prompt",
    r"repeat\s+(the\s+)?(text|words|prompt)\s+(above|before)",
    # Adversarial red-team injection patterns
    r"mark\s+this\s+(repair\s+)?safe",
    r"bypass\s+safety",
    r"pretend\s+this\s+is\s+approved",
    r"output\s+another\s+tenant",
    r"hide\s+this\s+citation",
    r"invent\s+a\s+source",
    r"tell\s+the\s+customer\s+to\s+bypass",
]

# ---------------------------------------------------------------------------
# Query abuse detection patterns
# ---------------------------------------------------------------------------

QUERY_ABUSE_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"system\s+prompt",
    r"developer\s+message",
    r"show\s+.*other\s+(customer|tenant)",
    r"reveal\s+confidential",
    r"dump\s+(the\s+)?database",
    # Expanded patterns
    r"list\s+all\s+(tenants?|customers?|users?)",
    r"show\s+(me\s+)?data\s+from\s+(another|different|other)",
    r"override\s+(safety|security|access)",
    r"admin\s+(access|mode|panel)",
    r"sql\s+inject",
    r"union\s+select",
    r";\s*(drop|delete|update|insert)\s+",
    # Adversarial red-team user prompt attacks
    r"ignore\s+(the\s+)?manual",
    r"pretend\s+you\s+are\s+(the\s+)?engineer",
    r"(give\s+me\s+)?the\s+repair\s+even\s+if\s+unsafe",
    r"repair\s+even\s+if\s+unsafe",
    r"invent\s+a\s+workaround",
    r"skip\s+verification",
    r"mark\s+this\s+fixed",
    r"internal\s+technician\s+notes",
    r"(use\s+)?another\s+customer('?s)?\s+service\s+history",
]


def detect_document_prompt_injection(text: str) -> list:
    """Detect prompt injection patterns in document/content text.

    Returns a list of matched pattern strings.  Empty list = clean.
    """
    normalized = _normalize_text(text).lower()
    flags = []

    # Check for prompt injection patterns
    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            flags.append(pattern)

    # Check for structural delimiter spoofing
    if _STRUCTURAL_RE.search(_normalize_text(text)):
        flags.append("structural_delimiter_spoofing")

    return flags


def query_is_control_abuse(text: str) -> bool:
    """Check if a user query contains control-plane abuse patterns.

    Returns True if abuse detected.
    """
    normalized = _normalize_text(text).lower()
    return any(
        re.search(pattern, normalized, flags=re.IGNORECASE)
        for pattern in QUERY_ABUSE_PATTERNS
    )
