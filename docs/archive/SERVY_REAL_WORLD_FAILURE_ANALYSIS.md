# SERVY Real-World Failure Analysis

**Date:** 2026-10-29  
**Commit:** `0e1f187`  
**Mode:** MODE A — Synthetic Staging  
**Evaluator:** Independent Red Team

---

## 1. Executive Summary

Three independent failure populations were analyzed:

| Population | Total | Failures | Accuracy |
|-----------|-------|----------|----------|
| 42-query dev benchmark (BGE-small + V5) | 42 | 4 | 90.48% Hit@1 |
| 42-query dev benchmark (**MiniLM + V5**) | 42 | 6 | **85.71% Hit@1** |
| 40-case red-team (BGE-small + V5) | 35 testable | 0 | 100% product-correct |

> [!CAUTION]
> The production serving configuration (MiniLM) achieves only **85.71%** Hit@1, not the reported 90.48%. The 4.8 percentage-point gap is caused by the serving config discrepancy (see SERVY_MODEL_PROVENANCE_CORRECTION.md).

---

## 2. Serving Config Comparison — Full Results

| Config | Hit@1 | Hit@3 | Hit@5 | R@10 | MRR@10 | Avg Latency |
|--------|-------|-------|-------|------|--------|-------------|
| **PROD: MiniLM hybrid + V5** | **36/42 = 85.7%** | 38/42 = 90.5% | 39/42 = 92.9% | 40/42 = 95.2% | 0.890 | 3,280ms |
| DEV: BGE-small hybrid + V5 | 38/42 = 90.5% | 40/42 = 95.2% | 41/42 = 97.6% | 41/42 = 97.6% | 0.934 | 2,749ms |
| Sparse-only + V5 | 37/42 = 88.1% | 40/42 = 95.2% | 41/42 = 97.6% | 41/42 = 97.6% | 0.921 | 2,436ms |
| MiniLM hybrid (no reranker) | 32/42 = 76.2% | 38/42 = 90.5% | 39/42 = 92.9% | 40/42 = 95.2% | 0.834 | 92ms |

### Critical Observations

1. **Sparse-only (88.1%) OUTPERFORMS MiniLM hybrid (85.7%)** on Hit@1. The MiniLM dense component is net-negative: it introduces 1 additional failure that sparse-only gets correct.
2. **The V5 reranker rescues 4 queries** that MiniLM hybrid would otherwise miss (76.2% → 85.7%), but this still falls short of BGE-small performance.
3. **BGE-small adds 2 Hit@1 successes** over MiniLM (85.7% → 90.5%), but both share the same Recall@10 ceiling (95.2% vs 97.6%).

---

## 3. Per-Query Failure Analysis (MiniLM + V5 Production Config)

### Failure 1: OP26-SLV-144-2 (rank=5)
- **Gold chunk:** 144-2
- **Backend:** hybrid
- **Root cause:** MiniLM cosine similarity insufficient to rank the correct chunk above 4 competing candidates. BGE-small ranks this at position 4 (also a miss).

### Failure 2: OP26-SLV-143-0 (rank=2)
- **Gold chunk:** 143-0
- **Backend:** hybrid
- **Root cause:** Persistent near-miss. Both MiniLM and BGE-small rank at position 2. The reranker cannot rescue this because the gold chunk's reranker score is below the top candidate.

### Failure 3: OP26-SLV-76-2 (rank=2)
- **Gold chunk:** 76-2
- **Backend:** hybrid
- **Root cause:** Another persistent rank-2 near-miss. Consistent across both embedding profiles.

### Failure 4: OP26-SLV-75-2 (MISS — not in top 10)
- **Gold chunk:** 75-2
- **Backend:** hybrid
- **Root cause:** MiniLM fails to retrieve this chunk entirely. BGE-small successfully retrieves and ranks at position 1 (HIT@1). This is the highest-impact MiniLM-specific failure.

### Failure 5: OP26-SLV-52-0 (MISS — fallback backend, 4ms)
- **Gold chunk:** 52-0
- **Backend:** fallback
- **Root cause:** Both sparse and dense fail to match. Backend falls to "fallback" with near-zero latency. This query likely uses terminology not present in the chunk index. **Consistent failure across ALL configs.**

### Failure 6: OP26-SLV-145-2 (rank=6)
- **Gold chunk:** 145-2
- **Backend:** hybrid
- **Root cause:** MiniLM ranks at position 6; BGE-small ranks at position 1 (HIT@1). Another MiniLM-specific regression.

---

## 4. Red-Team Failure Analysis (40 cases)

### Group G — Missing Evidence (Expected Abstention)

| Case | Query | Expected | Actual | Score | Assessment |
|------|-------|----------|--------|-------|------------|
| G01 | "How much does the Voltcore X3 cost?" | No price data → abstain | Returned Voltcore chunks | 0.503 | Retrieval correct (equipment matched), LLM needed to abstain |
| G02 | "Can I use goat milk as coolant?" | Absurd → abstain | Returned Voltcore chunks | 0.751 | High score on irrelevant query — LLM must reject |
| G03 | "Honda generator spark plug replacement" | Out-of-scope → empty | No results (score=0.088) | 0.088 | ✅ Correctly returned no relevant results |
| G04 | "What is the warranty period?" | No warranty data → abstain | Low-relevance result | 0.133 | Low score suggests system would abstain with LLM |

> [!WARNING]
> **Abstention quality cannot be verified.** The Ollama LLM is offline. Whether the system correctly abstains on G01/G02 depends entirely on the answer generation step, which was not tested.

### Group E02 — Vague Query
- Query: "the thing keeps beeping and I don't know what's wrong"
- Result: No product match (score=0.049)
- Assessment: Correct — query is too vague for equipment identification without asset context.

### Groups S/X — Security
- S01: Confidential Aerolift PM doc correctly excluded
- S02: Prompt injection test doc (Doc 24) quarantined, not retrieved
- X01/X02: Cross-tenant queries returned 0 results (correct authorization behavior)

---

## 5. Systematic Failure Patterns

### Pattern 1: Persistent Rank-2 Near-Misses
Queries OP26-SLV-143-0 and OP26-SLV-76-2 consistently rank at position 2 across ALL configurations. The reranker cannot resolve these because the competing chunk has a genuinely higher relevance signal.

### Pattern 2: MiniLM-Specific Dense Failures
Two queries (OP26-SLV-75-2, OP26-SLV-145-2) succeed with BGE-small but fail with MiniLM. These represent cases where BGE-small's content-first 384-dimensional embeddings capture semantic nuance that MiniLM's general-purpose embeddings miss.

### Pattern 3: Universal Fallback Failure (OP26-SLV-52-0)
This query fails across all configurations with near-zero latency and "fallback" backend. The sparse index does not match the query terms, and the dense index returns no relevant candidates. This is likely a query/document terminology mismatch.

---

## 6. Recommendations

1. **Set `SERVY_SERVING_EMBEDDING_PROFILE=bge-small`** in production to recover the 4.8% accuracy gap.
2. **Investigate OP26-SLV-52-0** — universal failure suggests missing or mis-indexed content.
3. **Deploy Ollama** to enable answer-quality testing (abstention, citation, faithfulness).
4. **Do not claim 90.48% production accuracy** — the actual production config achieves 85.71%.
5. **Audit the 2 MiniLM-specific failures** (75-2, 145-2) to understand embedding quality differences.
