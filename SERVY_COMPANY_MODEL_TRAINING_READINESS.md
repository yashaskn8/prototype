# SERVY Company Model Training Readiness Assessment

**Date:** 2026-10-29  
**Model Under Review:** `company-bge-ft` (BGE fine-tuned embedding model)  
**Reranker Under Review:** V5 cross-encoder reranker

---

## Executive Summary

Servy's evaluated retrieval configuration uses pretrained BGE-small embeddings with an optional V5 cross-encoder reranker. The experimental company fine-tuned BGE embedding checkpoint is separate and unpromoted. **Neither model is ready for unsupervised production promotion.** Both require independent evaluation on held-out data before deployment decisions.

---

## Model Inventory

### 1. BGE Fine-Tuned Embedding Model (`company-bge-ft`)

| Attribute | Status |
|-----------|--------|
| Base model | BGE (BAAI/bge-small-en-v1.5 or similar) |
| Fine-tuning data | Company service manual chunks |
| Training data provenance | ⚠️ **UNVERIFIED** — no audit trail for training data authorization |
| Checkpoint location | `ml_models/checkpoints/servy_company_bge_v1/final/` |
| Checkpoint SHA-256 | `7d394413398be69d2d1093538e5eec6cafa58cbfeae858cac4abf608b5db0d33` (embedding registry fingerprint) |
| ChromaDB collections | 4 collections × 229 vectors = 916 total |
| HNSW index status | ✅ Repaired (schema 2.0) |
| `promotion_eligible` | **False** |

### 2. V5 Cross-Encoder Reranker

| Attribute | Status |
|-----------|--------|
| Architecture | Cross-encoder (query, document) → relevance score |
| Pool size | 5 (empirically optimal) |
| Training data | ⚠️ **UNVERIFIED** — may overlap with evaluation queries |
| Accuracy contribution | +5 Hit@1 (33→38 out of 42 queries) |
| Latency cost | ~2.3-3.4 seconds per query |

---

## Readiness Checklist

| Requirement | BGE-FT | V5 Reranker |
|-------------|--------|-------------|
| Training data authorized by organization | ❌ Unverified | ❌ Unverified |
| Training/eval data separation verified | ❌ Not audited | ❌ Not audited |
| Independent held-out evaluation | ❌ Does not exist | ❌ Does not exist |
| Performance baseline documented | ✅ 40/42 hybrid queries | ✅ +5 Hit@1 vs no reranker |
| Model checkpoint integrity (SHA) | ✅ Verified | ✅ Verified |
| Rollback procedure defined | ⚠️ Can fall back to base BGE | ⚠️ Can disable reranker |
| A/B test framework ready | ❌ Not implemented | ❌ Not implemented |
| Bias/fairness audit | ❌ Not performed | ❌ Not performed |
| Latency within SLA | ✅ 39ms added | ⚠️ 3.3s — needs SLA review |

---

## Blockers for Production Promotion

### Blocker 1: Training Data Authorization
No record exists confirming that the data used to fine-tune either model was authorized for ML training by the data owners. Before promotion:
- Identify all source documents used in training
- Obtain explicit authorization from data owners
- Document the authorization chain

### Blocker 2: Train/Eval Leakage Risk
The 42-query evaluation set may overlap with training data. The 3 "failures" at the current ceiling (38/41) include gold label ambiguities, suggesting the evaluation set itself was used during model development. Before promotion:
- Create a **completely independent** evaluation set (100+ queries from source families excluded from all training/model selection)
- Re-evaluate both models on the independent set
- Document the separation methodology

### Blocker 3: No Independent Evaluation
The current accuracy numbers (92.68% Hit@1) were measured on the same query set used during iterative model development. This is selection bias, not independent validation. The true accuracy on unseen queries is unknown.

---

## Recommendations

1. **Do NOT promote** either model to production until Blockers 1-3 are resolved
2. **Create an independent benchmark** of 100+ queries from held-out equipment families
3. **Establish a model governance process**: training data registry, approval workflow, independent eval gate
4. **Implement A/B testing** infrastructure to measure real-world impact before full rollout
5. **Set a latency SLA** — if 3.3s per query is unacceptable, consider V5 quantization or a lighter reranker

---

## Current Deployment Recommendation

| Component | Action |
|-----------|--------|
| BGE-FT embeddings | ⚠️ **Use with caution** — keep `promotion_eligible=False` |
| V5 reranker (pool=5) | ⚠️ **Use with caution** — keep as optional enhancement |
| Sparse BM25 retrieval | ✅ **Production ready** — battle-tested, no ML training concerns |
| Hybrid retrieval pipeline | ✅ **Architecture ready** — but model weights need governance review |
