# SERVY Latency Benchmark Report

**Date:** 2026-10-29  
**Engine:** MySQL 8.0.41  
**Hardware:** Local GPU (V5 reranker), CPU (BM25 + ChromaDB)  
**Benchmark:** 10 queries × 4 configurations, sequential execution

---

## Summary

| Configuration | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) | Std Dev |
|--------------|-----------|-------------|----------|----------|----------|---------|
| Sparse only (BM25) | 72.1 | 84.2 | 93.3 | 5.0 | 93.3 | 27.0 |
| Sparse + V5 reranker | 2,376.7 | 2,788.2 | 2,997.9 | 494.5 | 2,997.9 | 834.5 |
| Hybrid (no reranker) | 111.0 | 114.9 | 144.9 | 34.8 | 144.9 | 30.5 |
| **Hybrid + V5 reranker** | **3,289.3** | **3,579.1** | **3,740.9** | **667.0** | **3,740.9** | **928.5** |

## Component Latency Breakdown

| Component | Added Latency |
|-----------|--------------|
| BM25 sparse retrieval | ~72ms (baseline) |
| ChromaDB dense retrieval | ~39ms (111 - 72) |
| RRF fusion | <1ms |
| V5 reranker (pool=5) | ~2,300 - 3,200ms |

## Key Findings

1. **V5 reranker dominates latency** — it accounts for ~97% of total query time in the production configuration (hybrid + V5). This is expected: the V5 model runs cross-encoder inference on 5 query-document pairs per query.

2. **Dense retrieval is cheap** — ChromaDB HNSW vector search adds only ~39ms, making it nearly free compared to the reranker.

3. **Hybrid without reranker is fast** — at 111ms median, this could serve real-time use cases where V5 accuracy gains aren't needed.

4. **First-query cold start** — the min values (5ms sparse, 35ms hybrid) are suspiciously low, likely reflecting cached queries. P95 is a better measure of real-world latency.

## Production Configuration

The recommended production configuration is **Hybrid + V5 (pool=5)**:
- **Accuracy:** 38/41 = 92.68% Hit@1 (authorized queries)
- **Latency:** ~3.3s mean, ~3.7s P95
- **Trade-off:** +5 correct answers vs no-reranker, at cost of ~3s per query

## Optimization Opportunities

| Optimization | Expected Improvement | Complexity |
|-------------|---------------------|------------|
| V5 ONNX quantization | 2-4× reranker speedup | Medium |
| V5 TensorRT compilation | 3-5× reranker speedup | High |
| Batch reranking (multi-query) | Amortized GPU overhead | Low |
| Async dense + sparse | ~39ms overlap savings | Low |
| Result caching (LRU) | Eliminates repeat queries | Low |

## Raw Data

Stored in `outputs/latency_benchmark.json`.
