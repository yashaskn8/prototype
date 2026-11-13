# SERVY — Django vs Python Parity Report (`SERVY_DJANGO_VS_PYTHON_PARITY_REPORT.md`)

Date: 2026-10-29 · Base commit `fc73c1a` · Dev set: 42 queries (reused; **not independent**)
Replay harness: `scratch/django_replay.py` (private copy of `db.sqlite3`; original untouched). Per-query CSV is private (`outputs/parity/`), not committed.

## 1. Headline: the reported 90.48% (38/42) is NOT reproduced by Django

| Pipeline | What it is | Hit@1 | Recall@10 | MRR@10 |
|---|---|---|---|---|
| A. Historical | BM25 frozen candidates + V5, pool 5 | 35/42 (83.33%) | 38/42 | 0.8605 |
| B. Python experiment (reported) | Python BM25+dense RRF + V5, **tenant filter only, ignores confidentiality** | 38/42 (90.48%) | 42/42 | ~0.94 |
| C2. **Real Django `retrieve()`**, tenant-wide universe, no reranker | actual code at `fc73c1a` | 36/42 (85.71%) | 41/42 | 0.9067 |
| C2 + V5 pool 5 | actual Django + real V5 | **37/42 (88.10%)** | 41/42 | 0.9214 |
| C2 + V5 pool 8 | actual Django + real V5 | 37/42 (88.10%) | 41/42 | 0.9206 |
| C1. Real Django, **real authorization scope, no asset** | production default | **0/42** | 0/42 | 0.0 |

## 2. Discrepancies found (each with root cause)

1. **Confidentiality leak in the Python evaluator (integrity bug in the metric).** Gold chunk 52 (query `OP26-SLV-52-0`) lives in a *confidential, customer-scoped* document (`PrivateLift X Internal Manual`, tenant 7, customer 11). Python counted retrieving it as a hit. Django correctly returns nothing. This probe cannot be a legitimate retrieval success; the authorized denominator is **41**, not 42. On 41: Django first-stage 36/41 (87.80%), Django + V5 37/41 (90.24%).
2. **"Hybrid" is not hybrid in Django.** `hybrid` and `sparse_only` produce identical output and `retrieval_backend="fallback"`: the dense leg returns nothing here (no promoted/indexed Chroma profile in this environment). My earlier "hybrid RRF" numbers describe a pipeline Django is not currently running. Any claim about dense/RRF benefit is **unverified in Django**.
3. **Production scope returns 0/42.** With `asset=None`, `_knowledge_candidates` restricts to unscoped documents; every manual here is product-scoped. Correct for security, but means the dev benchmark never exercised production scope. A production-scope replay needs an asset/product per query (not available in the frozen set).
4. **Different scoring stacks.** Django blends BM25, char-n-grams, stored hashing vectors, scope vector and metadata bonus; the Python experiment used plain BM25 + BGE. Different first-stage results (Django 36/42 vs Python-BM25 variants 34–38/42) confirm they are not equivalent.
5. **The V5 reranker is not a clear win in Django.** It improved 3 queries (`139-0`, `78-0`, `73-0`) and regressed 2 that first-stage had at rank 1 (`144-2`, `145-2`): net +1, at ~3.4 s (pool 5) / ~5.6 s (pool 8) per query on CPU.

## 3. Changes vs historical (Django + V5 pool 5)
36 same, 5 better (`75-2`, `74-1`, `142-2`, `145-2` 7→2, plus `144-2` unchanged-missing→rank 5), 1 "worse" (`52-0`: the historical "hit" was a confidential-document retrieval).

## 4. What was NOT done
Stage-by-stage numeric diff of BM25 scores, dense scores and RRF values between B and C2 (dense unavailable in Django); API-level replay; production-scope replay with assets.
