# SERVY Embedding Model Comparison

> [!IMPORTANT]
> This report compares embedding models available in the Servy RAG system
> based on empirical retrieval accuracy on the 42-query C2 benchmark.

## Models Evaluated

| Profile | Model | Dims | Status | Collection |
|---------|-------|------|--------|------------|
| `minilm` | `all-MiniLM-L6-v2` | 384 | Baseline | `servy_dense_kb` |
| `company-minilm` | `all-MiniLM-L6-v2` | 384 | Baseline | `servy_company_dense_kb` |
| `bge-small` | `BAAI/bge-small-en-v1.5` | 384 | **Active** | `servy_dense_bge_small_v1` |
| `company-bge-small` | `BAAI/bge-small-en-v1.5` | 384 | **Active** | `servy_company_bge_small_v1` |
| `company-bge-ft` | Local fine-tuned BGE | 384 | Experimental | Not promoted |
| `qwen3` | `Qwen/Qwen3-Embedding-0.6B` | 1024 | Unavailable | Weights not installed |

## Accuracy Results

### BGE-small (Active Serving Model)

| Mode | Reranker | Hit@1 (42q) | Hit@1 (41q) | MRR@10 |
|------|----------|-------------|-------------|--------|
| Sparse only | V5 pool=5 | 37/42 (88.10%) | 37/41 (90.24%) | 0.9214 |
| **Hybrid** | **V5 pool=5** | **38/42 (90.48%)** | **38/41 (92.68%)** | **0.9345** |
| Hybrid | None | 33/42 (78.57%) | 33/41 (80.49%) | 0.873 |

### MiniLM (Legacy Baseline)

MiniLM collections contain legacy schema 1.0 metadata. The Django retriever rejects
these vectors as stale, so MiniLM dense retrieval is non-functional in the current
system. MiniLM was the original embedding model before the BGE migration.

No direct accuracy comparison is available because:
1. MiniLM vectors have schema 1.0 metadata (missing required fields)
2. The retriever's `_matches_current_bge_vector()` validation rejects them
3. Reindexing with MiniLM would require a separate experiment

### Qwen3 (Not Available)

Qwen3 embedding weights (`Qwen/Qwen3-Embedding-0.6B`) are not installed locally.
The model requires 1024-dimensional collections (vs 384 for BGE/MiniLM).
No evaluation data is available.

### Company BGE Fine-Tuned (Experimental)

The `company-bge-ft` checkpoint exists in the repository but:
1. Training data provenance has not been independently verified
2. Organizational authorization for fine-tuning has not been confirmed
3. The model is not promotion-eligible (`promotion_eligible=False`)
4. No independent evaluation has been conducted

> [!WARNING]
> Do not promote `company-bge-ft` to production serving without:
> - Verified training data authorization
> - Independent accuracy evaluation on held-out queries
> - Organizational sign-off

## Dense vs Sparse Contribution

The BGE-small dense embeddings provide measurable value:

| Evidence | Detail |
|----------|--------|
| Unique improvement | OP26-SLV-145-2: rank=2 (sparse) → HIT@1 (hybrid) |
| MRR improvement | 0.9214 → 0.9345 (+0.0131) |
| Latency | 2,892ms → 2,691ms (-201ms, faster due to better fusion) |
| Backend coverage | 40/42 queries use `hybrid` backend |

## Recommendations

1. **Keep BGE-small as the active serving model** — it provides the best verified results
2. **Do not revert to MiniLM** — schema 1.0 vectors are non-functional
3. **Do not install Qwen3 without evaluation** — 1024d model requires new collections
4. **Do not promote company-bge-ft** — requires authorization and independent evaluation
5. **Keep hybrid retrieval enabled** — measurable +1 Hit@1 over sparse-only

---

*Generated: 2026-10-29 | DB: MySQL 8.0.41 | Commit: 18d0277*
