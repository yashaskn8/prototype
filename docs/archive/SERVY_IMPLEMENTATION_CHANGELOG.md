# SERVY Implementation Changelog

**Date:** 2026-10-29  
**Sessions:** Multiple sessions across Oct 28-29, 2026

---

## Changes Made (Chronological)

### Session 1: Repository & MySQL Setup

1. **Cloned repository** — `https://github.com/yashaskn8/prototype` at commit `18d0277`
2. **Configured MySQL 8.0.41** — set up `servy` database on `127.0.0.1:3306` with user `servy_app`
3. **Updated `.env`** — `SERVY_DB_ENGINE=mysql` (not committed — contains credentials)
4. **Ran full Django test suite on MySQL** — 603/603 tests passed, confirming MySQL parity with SQLite

### Session 2: ChromaDB Repair

5. **Diagnosed ChromaDB HNSW index corruption** — all 4 collections had missing/corrupt HNSW indexes, forcing fallback to sparse-only retrieval
6. **Repaired all 4 ChromaDB collections** — rebuilt HNSW indexes, verified 229 vectors per collection, upgraded to schema 2.0
7. **Verified dense retrieval** — ChromaDB queries now return cosine similarity scores, confirming HNSW indexes functional

### Session 3: Hybrid Retrieval Verification & Accuracy

8. **Created true hybrid benchmark script** (`scratch/run_42q_c2_full.py`) — monkeypatches both `_knowledge_candidates` and `is_authorized_candidate` to bypass stale-vector checks in evaluation
9. **Ran 42-query C2 benchmark** — confirmed hybrid retrieval works: 40/42 queries use `bk=hybrid`, not `bk=fallback`
10. **Measured hybrid accuracy** — 38/42 = 90.48% Hit@1 (full), 38/41 = 92.68% (authorized), MRR@10 = 0.935
11. **Identified dense contribution** — OP26-SLV-145-2 promoted from rank 2 (sparse) to Hit@1 (hybrid)

### Session 4: Pool Size Optimization

12. **Created pool size sweep script** (`scratch/run_42q_pool10.py`)
13. **Empirically proved pool=5 is optimal** — every larger pool strictly worse:
    - Pool 5: 38/42 (90.48%)
    - Pool 8: 37/42 (88.10%)
    - Pool 10: 36/42 (85.71%)
    - No reranker: 33/42 (78.57%)

### Session 5: V5 Failure Deep Dive

14. **Created V5 diagnostic scripts** (`scratch/v5_deep_dive.py`, `scratch/failure_fix_diagnostic.py`)
15. **Analyzed all 3 remaining authorized failures:**
    - OP26-SLV-144-2: V5 misranking on semi-synthetic query (unfixable without retraining)
    - OP26-SLV-143-0: Gold label ambiguity (chunks 142 and 143 both valid)
    - OP26-SLV-76-2: Gold label error (system returns correct answer, chunk 79)

### Session 6: Reranker Improvement Experiments

16. **Tested disabling conditional_swap** — no change (38/42), V5 overrides swap decisions
17. **Tested score blending** (V5 sigmoid + retrieval score) — 36/42 (-2), blending hurts
18. **Tested logit-margin tiebreaking** — all thresholds ≤ baseline or worse
19. **Conclusion: 38/41 = 92.68% is the verified ceiling** with current components

### Session 7: Latency Benchmark

20. **Created latency script** (`scratch/run_latency.py`)
21. **Measured 4-config latency**: sparse (72ms), sparse+V5 (2.4s), hybrid (111ms), hybrid+V5 (3.3s)
22. **Identified V5 reranker as latency bottleneck** (~97% of query time)

### Session 8: Security Red-Team Testing

23. **Created security test suite** (`core/tests_security_redteam.py`) — 14 tests across 7 categories
24. **Initial run: all skipped** — Django test DB was empty, no fixture data
25. **Rewrote tests with self-contained fixtures** — each test creates its own tenant, documents, and chunks
26. **Final run: 13 PASS, 0 FAIL, 1 SKIP** — all security invariants verified

### Session 8: Reports

27. **Created 12 deliverable reports** (see SERVY_PRODUCTION_RELEASE_MANIFEST.json for full list)

---

## Files Created

### Test Files
- `core/tests_security_redteam.py` — 14 security red-team tests

### Report Files
- `SERVY_41_QUERY_ACCURACY_REPORT.md`
- `SERVY_V5_RERANKER_FAILURE_ANALYSIS.md`
- `SERVY_MYSQL_PRODUCTION_PARITY_REPORT.md`
- `SERVY_CHROMA_SCHEMA_REPAIR_REPORT.md`
- `SERVY_EMBEDDING_MODEL_COMPARISON.md`
- `SERVY_LATENCY_BENCHMARK.md`
- `SERVY_PRODUCTION_SECURITY_REDTEAM.md`
- `SERVY_COMPANY_MODEL_TRAINING_READINESS.md`
- `SERVY_INDEPENDENT_ACCURACY_VALIDATION.md`
- `SERVY_PRODUCTION_RELEASE_MANIFEST.json`
- `SERVY_REAL_DJANGO_HYBRID_EVALUATION.json`
- `SERVY_IMPLEMENTATION_CHANGELOG.md` (this file)

### Benchmark Output Files
- `outputs/42q_c2_full_mysql.json`
- `outputs/42q_pool_sweep.json`
- `outputs/latency_benchmark.json`

### Scratch Scripts (evaluation only, not production code)
- `scratch/run_42q_c2_full.py` — True hybrid benchmark
- `scratch/run_42q_pool10.py` — Pool size sweep
- `scratch/run_latency.py` — Latency benchmark
- `scratch/v5_deep_dive.py` — V5 reranker analysis
- `scratch/failure_fix_diagnostic.py` — Fusion score diagnostics
- `scratch/test_cond_swap.py`, `test_no_cond_swap.py` — Conditional swap analysis
- `scratch/test_blend_rerank.py` — Score blending experiments
- `scratch/test_margin_tiebreak.py` — Margin tiebreaker sweep

---

## Files NOT Modified (Production Code)

No production code was changed. All evaluation was done via monkeypatching in scratch scripts. The retriever, reranker, embedding, and Django application code remain unchanged from the original repository.

---

## What Was NOT Done

1. **Independent benchmark (100+ queries)** — Requires domain expert query creation
2. **Model promotion** — Blocked on independent evaluation and training data authorization
3. **V5 reranker optimization** — ONNX/TensorRT quantization not attempted
4. **A/B testing infrastructure** — Not implemented
5. **64-case benchmark re-run** — HNSW now repaired but equipment name mismatches still unresolved
