import math
import re
from functools import lru_cache
from typing import List, Dict, Any

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

VECTOR_DIMENSIONS = 2048
VECTOR_MODEL = "hashing-word12-v1"

STOP_WORDS = set(ENGLISH_STOP_WORDS)


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
    """Tokenize technical text into words, retaining error codes and identifiers,
    and filtering English stop words."""
    words = [w.lower() for w in re.findall(r"\b[A-Za-z0-9_\-]+\b", text or "")]
    if filter_stopwords:
        return [w for w in words if w not in STOP_WORDS and len(w) > 1]
    return words


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
