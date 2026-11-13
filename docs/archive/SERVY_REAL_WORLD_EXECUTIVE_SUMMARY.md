# SERVY Real-World Executive Summary

**Date:** 2026-10-29  
**Commit:** `0e1f187895f8144b3d2b1c9ed463b1896c2cdbef`  
**Evaluator:** Independent Red Team (Automated)  
**Mode:** MODE A — Synthetic Staging

---

## Release Decision

# `CONTROLLED_SHADOW_EVALUATION`

The system is suitable for a controlled shadow evaluation with human review.  
It is **NOT** suitable for unmonitored production release.

---

## A. Data Authorization

**Status: MODE A ONLY — Real data validation BLOCKED.**

No external company data authorization was established. All evaluation used existing dev/staging database content. Starlly API credentials are for CRM integration only, not ML evaluation. Real-data validation (MODE B/C) requires separate written authorization.

---

## B. Model Provenance

**Status: RELEASE BLOCKER (metadata only — code is correct).**

| Finding | Severity |
|---------|----------|
| Manifest assigns V5 reranker SHA to embedding model | Critical |
| Production default embedding (minilm) ≠ benchmark embedding (bge-small) | Critical |
| Code and model artifacts are correctly configured | ✅ |

**Remediation:** Set `SERVY_SERVING_EMBEDDING_PROFILE=bge-small` in production and regenerate manifest.

---

## C. Retrieval Accuracy

### Production Serving Config (MiniLM + V5)
| Metric | Value |
|--------|-------|
| **Hit@1** | **36/42 = 85.71%** |
| Hit@3 | 38/42 = 90.48% |
| Hit@5 | 39/42 = 92.86% |
| Recall@10 | 40/42 = 95.24% |
| MRR@10 | 0.890 |

### Corrected Config (BGE-small + V5)
| Metric | Value |
|--------|-------|
| **Hit@1** | **38/42 = 90.48%** |
| Hit@3 | 40/42 = 95.24% |
| Hit@5 | 41/42 = 97.62% |
| Recall@10 | 41/42 = 97.62% |
| MRR@10 | 0.934 |

### Full Comparison
| Config | Hit@1 | Hit@3 | Recall@10 | MRR@10 | Latency |
|--------|-------|-------|-----------|--------|---------|
| **MiniLM + V5 (prod default)** | **85.7%** | 90.5% | 95.2% | 0.890 | 3.3s |
| BGE-small + V5 (dev benchmark) | 90.5% | 95.2% | 97.6% | 0.934 | 2.7s |
| Sparse-only + V5 | 88.1% | 95.2% | 97.6% | 0.921 | 2.4s |
| MiniLM + no reranker | 76.2% | 90.5% | 95.2% | 0.834 | 92ms |

> [!CAUTION]
> **Sparse-only (88.1%) outperforms MiniLM hybrid (85.7%) on Hit@1.** The MiniLM dense component actively hurts accuracy. Set BGE-small as the serving profile to recover 4.8% Hit@1.

---

## D. Red-Team Results (40 Cases)

| Category | Correct | Total | Accuracy |
|----------|---------|-------|----------|
| Equipment confusion (A) | 11 | 11 | 100% |
| Fault codes (B) | 4 | 4 | 100% |
| Technical specs (C) | 4 | 4 | 100% |
| Safety/procedural (D) | 5 | 5 | 100% |
| Real-world language (E) | 4 | 4 testable | 100% |
| Multi-document (F) | 3 | 3 | 100% |
| **Product-correct total** | **35** | **35** | **100%** |

**Abstention group (G):** Cannot verify — requires Ollama LLM (offline).  
**Security (S):** Confidential docs excluded, injection doc quarantined.  
**Cross-tenant (X):** Zero results returned (correct authorization behavior).

---

## E. Security Assessment

| Area | Result |
|------|--------|
| Cross-tenant isolation | ✅ Zero leakage |
| Authentication enforcement | ✅ 403 for unauthenticated requests |
| Confidential document protection | ✅ Quarantined and excluded |
| Prompt injection (API) | ✅ All 5 tests safe |
| Document injection detection | ✅ All patterns detected |
| Policy guard (query abuse) | ⚠️ Misses DAN, ADMIN OVERRIDE patterns |
| SQL injection | ✅ ORM parameterization protects |
| Customer data isolation mode | ✅ Fail-closed to verified_real_only |

**28/30 security tests passed, 0 failed, 2 warnings (policy guard gaps).**

---

## F. Database Integrity

| Check | Status |
|-------|--------|
| FK integrity | ✅ 0 violations |
| Duplicate chunks | ✅ 0 found |
| Orphaned chunks | ✅ 0 found |
| RAG-disabled doc leak | ✅ 0 live chunks from disabled docs |
| ChromaDB alignment | ✅ All 4 collections synced (229 vectors each) |
| Content SHA verification | ✅ 0/20 mismatches |

---

## G. Latency

| Scenario | Mean | P50 | P95 |
|----------|------|-----|-----|
| BGE + V5 (warm, sequential) | 2.4s* | 3.0s | 3.1s |
| MiniLM + V5 (warm, sequential) | 2.4s | 2.4s | 3.5s |
| Without reranker | 143ms | 141ms | 168ms |
| Concurrent (3 parallel) | 6.6s/query | — | 7.5s |
| Cold start | 28-33s | — | — |

*Excluding cold start outlier

---

## H. Answer Quality

**STATUS: UNTESTED — Ollama LLM is offline.**

The following metrics could not be evaluated:
- Answer correctness
- Answer faithfulness (grounding in retrieved chunks)
- Citation quality
- Abstention on missing evidence
- Hallucination rate
- Response safety/toxicity

---

## I. What Was NOT Tested

1. ❌ End-to-end answer quality (Ollama offline)
2. ❌ Independent evaluation dataset (none exists)
3. ❌ Real company data (MODE A only)
4. ❌ Multi-user production load (>3 concurrent)
5. ❌ GPU-accelerated latency
6. ❌ Model versioning / rollback
7. ❌ Automated CI/CD pipeline
8. ❌ Monitoring and alerting
9. ❌ Long-running stability (hours/days)
10. ❌ Edge cases: very long queries, non-English, image content

---

## J. Blockers for Production Release

| # | Blocker | Severity | Owner |
|---|---------|----------|-------|
| 1 | Serving config discrepancy (minilm default ≠ benchmark) | Critical | Deployment |
| 2 | Manifest SHA misattribution | Critical | Documentation |
| 3 | Ollama LLM offline — answer quality untested | High | Infrastructure |
| 4 | No independent evaluation dataset | High | Evaluation |
| 5 | No real data authorization | High | Legal/Business |
| 6 | Policy guard gaps (DAN, ADMIN OVERRIDE) | Medium | Security |
| 7 | Cold start latency (28-33s) | Medium | Engineering |

---

## K. Recommendations

### Before Controlled Shadow Evaluation
1. **Set `SERVY_SERVING_EMBEDDING_PROFILE=bge-small`** in production `.env`
2. **Correct the release manifest** SHA assignments
3. **Start Ollama** and test answer quality on the 40-case red team
4. **Add DAN/ADMIN OVERRIDE patterns** to the policy guard

### During Controlled Shadow Evaluation
5. Run shadow mode on a small subset of real queries with human review
6. Measure answer quality metrics (correctness, faithfulness, abstention)
7. Monitor latency distribution under real load
8. Verify cross-tenant isolation with real multi-tenant traffic

### Before Production Release
9. Commission an independent evaluation with blind held-out dataset (100+ questions)
10. Obtain real company data authorization (MODE B)
11. Deploy GPU acceleration for <500ms P95 latency
12. Implement model pre-warming to eliminate cold start
13. Set up automated CI/CD with accuracy regression gates
14. Establish monitoring dashboards for retrieval quality and latency

---

## L. Evidence Trail

| Artifact | Location |
|----------|----------|
| Serving config comparison | `outputs/42q_serving_config_comparison.json` |
| Red-team results | `outputs/redteam_real_world.json` |
| Integrity audit | `outputs/mysql_chroma_integrity.json` |
| API/security tests | `outputs/django_api_security_tests.json` |
| Latency tests | `outputs/latency_load_test.json` |
| Test scripts | `scratch/serving_config_comparison.py`, `scratch/redteam_real_world.py`, `scratch/integrity_audit.py`, `scratch/django_api_security_test.py`, `scratch/latency_load_test.py` |
