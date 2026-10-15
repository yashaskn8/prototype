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


def compute_bm25_scores(query: str, corpus_texts: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """True BM25 implementation (Lucene/BM25Plus smoothed) over authorized corpus texts.

    Returns normalized scores scaled by query term coverage in range [0, 1].
    """
    if not query or not corpus_texts:
        return [0.0] * len(corpus_texts)

    tokenized_query = tokenize_words(query, filter_stopwords=True)
    if not tokenized_query:
        return [0.0] * len(corpus_texts)

    tokenized_corpus = [tokenize_words(doc, filter_stopwords=False) for doc in corpus_texts]
    n_docs = len(corpus_texts)
    doc_lens = [len(doc) for doc in tokenized_corpus]
    avg_len = sum(doc_lens) / max(1, n_docs)

    # Document frequencies
    df: Dict[str, int] = {}
    for doc in tokenized_corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    scores = []
    n_query_terms = len(set(tokenized_query))
    for doc, doc_len in zip(tokenized_corpus, doc_lens):
        doc_term_counts: Dict[str, int] = {}
        for term in doc:
            doc_term_counts[term] = doc_term_counts.get(term, 0) + 1

        matched_query_terms = 0
        raw_score = 0.0
        for q_term in set(tokenized_query):
            if q_term not in doc_term_counts:
                continue
            matched_query_terms += 1
            tf = doc_term_counts[q_term]
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
    """Character n-gram TF-IDF cosine similarity for typo and misspelling recovery.

    Returns similarity scores in range [0, 1].
    Filters stop-words from query so common functional words don't create false overlap.
    """
    if not query or not corpus_texts:
        return [0.0] * len(corpus_texts)

    clean_query = " ".join(tokenize_words(query, filter_stopwords=True))
    if not clean_query:
        return [0.0] * len(corpus_texts)

    try:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, lowercase=True)
        tfidf_matrix = vec.fit_transform(corpus_texts)
        q_vec = vec.transform([clean_query])
        sims = (tfidf_matrix * q_vec.T).toarray().flatten()
        return [round(float(s), 6) for s in sims]
    except Exception:
        return [0.0] * len(corpus_texts)
