# SERVY Real-World Latency and Load Report

**Date:** 2026-10-29  
**Commit:** `0e1f187`  
**Hardware:** CPU-only (no GPU), Windows, Python 3.12  
**Database:** MySQL 8.0.41  
**Vector Store:** ChromaDB (229 vectors per collection)

---

## 1. Executive Summary

| Config | Mean Latency | P50 | Notes |
|--------|-------------|-----|-------|
| BGE-small hybrid + V5 reranker | **4,970ms** | 3,003ms | Cold start inflates mean |
| MiniLM hybrid + V5 reranker | **2,402ms** | 2,441ms | Production default |
| Sparse-only + V5 reranker | **2,583ms** | 3,124ms | |
| BGE-small hybrid (no reranker) | **143ms** | 141ms | 30x faster without reranker |
| Concurrent (3 parallel, BGE + V5) | **6,641ms** per query | — | 14.2s wall time for 6 queries |

> [!IMPORTANT]
> The V5 reranker (XLM-RoBERTa-large, 2.27 GB) dominates latency. On CPU, it adds ~2,500-3,200ms per query. Without it, retrieval completes in ~140ms.

---

## 2. Warm Sequential Latency (BGE-small + V5)

10 diverse queries, pre-warmed model:

| Query | Latency | Results | Backend |
|-------|---------|---------|---------|
| Voltcore X3 operating voltage | 28,454ms ⚠️ | 10 | hybrid |
| CMC1000 fault code E-12 | 2,498ms | 10 | hybrid |
| Aerolift 5000 safety shutdown | 1,038ms | 10 | hybrid |
| MA-100 filter replacement | 1,503ms | 10 | hybrid |
| Calibrate pressure sensor | 3,003ms | 10 | hybrid |
| Refrigerant type | 2,992ms | 10 | hybrid |
| KTM engine oil change | 3,079ms | 10 | hybrid |
| Autopick AP-3X error codes | 3,127ms | 10 | hybrid |
| Monthly PM checklist | 968ms | 10 | hybrid |
| 5V DC motor wiring specs | 3,035ms | 10 | hybrid |

**Statistics:**
- Mean: 4,970ms (inflated by 28.5s first-query cold start)
- P50: 3,003ms
- P95: 28,454ms
- Min: 968ms
- Max: 28,454ms

> [!NOTE]
> The first query (28.5s) includes model loading time. Excluding cold start, mean drops to ~2,360ms.

---

## 3. MiniLM Hybrid + V5 (Production Config)

| Query # | Latency |
|---------|---------|
| 1 | 1,344ms |
| 2 | 3,002ms |
| 3 | 1,269ms |
| 4 | 1,820ms |
| 5 | 3,439ms |
| 6 | 3,432ms |
| 7 | 2,004ms |
| 8 | 3,487ms |
| 9 | 1,347ms |
| 10 | 2,878ms |

**Mean: 2,402ms** — 50% faster than BGE-small (models already warm).

---

## 4. Sparse-Only + V5

**Mean: 2,583ms** — Similar to MiniLM hybrid because the reranker dominates.

---

## 5. Without Reranker (BGE-small hybrid)

| Query # | Latency |
|---------|---------|
| 1 | 161ms |
| 2 | 168ms |
| 3 | 133ms |
| 4 | 156ms |
| 5 | 137ms |
| 6 | 144ms |
| 7 | 122ms |
| 8 | 144ms |
| 9 | 126ms |
| 10 | 141ms |

**Mean: 143ms** — Retrieval alone is fast. The reranker accounts for >95% of total latency.

---

## 6. Concurrent Load (3 Parallel Workers)

6 queries dispatched to 3 parallel workers:

| Completion Order | Latency |
|-----------------|---------|
| 1st | 5,444ms |
| 2nd | 5,933ms |
| 3rd | 6,739ms |
| 4th | 7,000ms |
| 5th | 7,240ms |
| 6th | 7,490ms |

- **Wall time:** 14,241ms (for 6 queries)
- **Mean per-query:** 6,641ms
- **Throughput:** ~0.42 queries/second

> [!WARNING]
> Concurrent load approximately doubles per-query latency (3,003ms → 6,641ms). The V5 reranker's CPU-bound inference is the bottleneck. The Python GIL and single-threaded model inference severely limit concurrency.

---

## 7. Cold Start

| Scenario | Latency |
|----------|---------|
| First query (model loading) | **28,454ms** (BGE) / **33,811ms** (MiniLM from 42q benchmark) |
| Previous session measurement | 32,287ms |
| Warm query | 2,400-3,100ms |

Cold start is ~10x warm-query latency. This is expected for loading a 2.27 GB reranker model into CPU memory.

---

## 8. Latency Budget Breakdown (estimated)

| Component | Estimated Time | % of Total |
|-----------|---------------|------------|
| BM25 sparse scoring | ~10-20ms | <1% |
| Dense embedding (encode query) | ~20-40ms | 1-2% |
| ChromaDB ANN search | ~10-30ms | 1% |
| MySQL candidate loading | ~30-60ms | 2% |
| RRF fusion | ~5ms | <1% |
| **V5 Reranker inference** | **~2,400-3,200ms** | **>95%** |
| Authorization filtering | ~5-10ms | <1% |

---

## 9. Recommendations

### Production Deployment
1. **GPU acceleration** for V5 reranker would reduce inference from ~3s to ~50-100ms
2. **Reranker batching** — process multiple queries in a single forward pass
3. **Model quantization** (INT8/INT4) for CPU deployment, trading ~1% accuracy for 2-4x speedup
4. **Async reranking** — return initial results immediately, rerank in background

### SLA Feasibility

| Target SLA | Feasible? | Config Required |
|-----------|-----------|-----------------|
| <500ms P95 | ❌ No (CPU) / ✅ Yes (GPU) | GPU or disable reranker |
| <3s P95 | ⚠️ Marginal (CPU, single user) | CPU, no concurrent load |
| <5s P95 | ✅ Yes (CPU, low concurrency) | Current config |
| <10s P95 | ✅ Yes (CPU, moderate load) | Current config |
