import math
from functools import lru_cache

from sklearn.feature_extraction.text import HashingVectorizer

VECTOR_DIMENSIONS = 2048
VECTOR_MODEL = "hashing-word12-v1"


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
