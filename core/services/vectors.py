import math
import re
from functools import lru_cache
from typing import List, Dict, Any

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

VECTOR_DIMENSIONS = 2048
VECTOR_MODEL = "hashing-word12-v1"

STOP_WORDS = set(ENGLISH_STOP_WORDS)

TECH_SYNONYM_MAP = {
    "escalation": ("escalate", "stop"),
    "escalate": ("escalation", "stop"),
    "precautions": ("precaution", "safety", "stop", "hazard"),
    "precaution": ("precautions", "safety", "stop"),
    "intervals": ("interval", "schedule", "monthly", "quarterly", "annual", "daily", "weekly", "preventive"),
    "interval": ("intervals", "schedule", "monthly", "quarterly", "annual", "preventive"),
    "schedule": ("intervals", "interval", "monthly", "quarterly", "annual", "preventive"),
    "monthly": ("intervals", "schedule", "preventive"),
    "quarterly": ("intervals", "schedule", "preventive"),
    "annual": ("intervals", "schedule", "preventive"),
    "preventive": ("intervals", "schedule", "maintenance"),
}

_UNIT_PATTERN = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(v|vac|vdc|kv|mv|a|ma|hz|khz|w|kw|va|kva|ohm|psi|bar|kpa|mpa|c|f|nm|mm|cm|t|ton)\b"
)
_VOLTAGE_SPEC_PATTERN = re.compile(
    r"\b(?:operating\s+v|voltage)\s+(?:specification\s+)?([0-9/\.\s]+)"
)
_ERROR_CODE_TOKEN_PATTERN = re.compile(
    r"\b([ef]|err|alm)[\s\-]?(\d{1,4})\b"
)



@lru_cache(maxsize=1)
def _vectorizer():
    return HashingVectorizer(
        n_features=VECTOR_DIMENSIONS,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
        ngram_range=(1, 2),
        lowercase=True,
    )


def embed_text(text):
    text = (text or "").strip()
    if not text:
        return {}
    row = _vectorizer().transform([text]).tocsr()[0]
    return {str(int(i)): round(float(v), 8) for i, v in zip(row.indices, row.data)}


def cosine_sparse(a, b):
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(float(v) * float(b.get(k, 0.0)) for k, v in a.items()))


def tokenize_words(text: str, filter_stopwords: bool = True) -> List[str]:
    """Tokenize technical text into words, retaining error codes, numeric ratings,
    voltage/unit tokens (e.g. 5V, 12V, 19V <-> 'operating V specification 5/12/19'),
    and bounded morphological variants while filtering English stop words."""
    raw_text = (text or "").lower()
    expanded: List[str] = []

    # 1. Fused or space-separated number + engineering unit patterns: 5v, 12v, 19v, 2t, 4c, etc.
    for m in _UNIT_PATTERN.finditer(raw_text):
        num, unit = m.group(1), m.group(2)
        unit_norm = "v" if unit in ("v", "vac", "vdc") else unit
        expanded.append(f"{num}{unit_norm}")
        expanded.append(num)
        if unit_norm == "v":
            expanded.append("voltage")
            expanded.append("volt")

    # 2. Operating V / voltage specification patterns: "operating v specification 5/12/19"
    for m in _VOLTAGE_SPEC_PATTERN.finditer(raw_text):
        for n in re.findall(r"\d+(?:\.\d+)?", m.group(1)):
            expanded.append(f"{n}v")
            expanded.append(n)
            expanded.append("voltage")

    # 3. Error / fault code patterns: E-12 -> e12, e-12
    for m in _ERROR_CODE_TOKEN_PATTERN.finditer(raw_text):
        prefix, num = m.group(1), m.group(2)
        expanded.append(f"{prefix}{num}")
        expanded.append(f"{prefix}-{num}")

    # 4. Standard word & numeric tokens (preserving single-digit numbers)
    words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-]+\b", raw_text)]
    base_tokens: List[str] = []
    for w in words:
        if filter_stopwords and w in STOP_WORDS:
            continue
        if len(w) > 1 or w.isdigit():
            base_tokens.append(w)
            if "-" in w:
                for sub in w.split("-"):
                    if sub and (len(sub) > 1 or sub.isdigit()) and (not filter_stopwords or sub not in STOP_WORDS):
                        base_tokens.append(sub)
            if w in TECH_SYNONYM_MAP:
                base_tokens.extend(TECH_SYNONYM_MAP[w])

    return base_tokens + expanded


@lru_cache(maxsize=8192)
def _get_doc_tokens(text: str) -> tuple:
    """Precompute and cache tokenization and term counts for a chunk text."""
    tokens = tokenize_words(text, filter_stopwords=False)
    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    return tuple(tokens), counts, len(tokens)


@lru_cache(maxsize=8192)
def _get_char_ngrams(text: str, n_min: int = 3, n_max: int = 5) -> tuple:
    """Precompute and cache character n-grams with word-boundary padding and L2 norm."""
    text = (text or "").strip().lower()
    if not text:
        return {}, 0.0

    words = re.findall(r"\b[A-Za-z0-9_\-]+\b", text)
    counts: Dict[str, int] = {}
    for w in words:
        padded = f" {w} "
        L = len(padded)
        for n in range(n_min, min(n_max + 1, L + 1)):
            for i in range(L - n + 1):
                ng = padded[i:i+n]
                counts[ng] = counts.get(ng, 0) + 1

    norm = math.sqrt(sum(v * v for v in counts.values()))
    return counts, norm


def compute_bm25_scores(query: str, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """True BM25 implementation (Lucene/BM25Plus smoothed) over authorized corpus texts.

    Returns normalized scores scaled by query term coverage in range [0, 1].
    Uses precomputed cached token statistics to eliminate per-query tokenization overhead.
    """
    if not query or not corpus_texts:
        return [0.0] * len(corpus_texts)

    tokenized_query = tokenize_words(query, filter_stopwords=True)
    if not tokenized_query:
        return [0.0] * len(corpus_texts)

    n_docs = len(corpus_texts)
    doc_data = [_get_doc_tokens(doc) for doc in corpus_texts]
    doc_lens = [d[2] for d in doc_data]
    avg_len = sum(doc_lens) / max(1, n_docs)

    # Document frequencies across candidate corpus
    df: Dict[str, int] = {}
    query_term_set = set(tokenized_query)
    for _, counts, _ in doc_data:
        for term in query_term_set:
            if term in counts:
                df[term] = df.get(term, 0) + 1

    scores = []
    n_query_terms = len(query_term_set)
    for _, counts, doc_len in doc_data:
        matched_query_terms = 0
        raw_score = 0.0
        for q_term in query_term_set:
            if q_term not in counts:
                continue
            matched_query_terms += 1
            tf = counts[q_term]
            term_df = df.get(q_term, 0)
            idf = math.log(1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            num = tf * (k1 + 1.0)
            den = tf + k1 * (1.0 - b + b * (doc_len / max(1.0, avg_len)))
            raw_score += idf * (num / den)

        term_coverage = matched_query_terms / max(1, n_query_terms)
        norm_score = (raw_score / (raw_score + 2.0)) * term_coverage
        scores.append(round(norm_score, 6))

    return scores


def compute_char_ngram_scores(query: str, corpus_texts: List[str], ngram_range=(3, 5)) -> List[float]:
    """Character n-gram cosine similarity for typo and misspelling recovery.

    Computes cosine similarity in range [0, 1] using cached character n-gram counts
    without the expensive TfidfVectorizer fit_transform bottleneck.
    Filters stop-words from query so common functional words don't create false overlap.
    """
    if not query or not corpus_texts:
        return [0.0] * len(corpus_texts)

    clean_query = " ".join(tokenize_words(query, filter_stopwords=True))
    if not clean_query:
        return [0.0] * len(corpus_texts)

    q_counts, q_norm = _get_char_ngrams(clean_query, ngram_range[0], ngram_range[1])
    if q_norm == 0.0:
        return [0.0] * len(corpus_texts)

    scores = []
    for doc in corpus_texts:
        d_counts, d_norm = _get_char_ngrams(doc, ngram_range[0], ngram_range[1])
        if d_norm == 0.0:
            scores.append(0.0)
            continue
        dot = sum(c * d_counts.get(ng, 0) for ng, c in q_counts.items())
        sim = dot / (q_norm * d_norm)
        scores.append(round(float(sim), 6))

    return scores
