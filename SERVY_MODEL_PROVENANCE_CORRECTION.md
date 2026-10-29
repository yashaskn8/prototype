# SERVY Model Provenance Correction

**Date:** 2026-10-29  
**Commit:** `0e1f187`  
**Status:** RELEASE BLOCKER — Metadata errors in prior release manifest

---

## 1. Problem Statement

The prior release manifest (`SERVY_PRODUCTION_RELEASE_MANIFEST.json`) incorrectly associates the V5 reranker's SHA-256 hash with the embedding model identity. Additionally, the production serving configuration defaults to a different embedding model than what was used to measure the reported 90.48% accuracy.

---

## 2. Verified Model Inventory

### V5 Reranker (Cross-Encoder)
| Field | Value |
|-------|-------|
| **Location** | `best_model/` |
| **Architecture** | XLMRobertaForSequenceClassification |
| **Base model** | XLM-RoBERTa-large |
| **Parameters** | ~560M |
| **Size on disk** | 2.27 GB |
| **SHA-256** | `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda` |
| **Purpose** | Re-rank candidate chunks by query-passage relevance |
| **Status** | ✅ Verified, operational |

### Company BGE Fine-Tuned Embedding
| Field | Value |
|-------|-------|
| **Location** | `ml_models/checkpoints/servy_company_bge_v1/final/` |
| **Architecture** | BertModel (BGE-small fine-tuned) |
| **Dimensions** | 384 |
| **SHA-256** | `7d394413...` |
| **Purpose** | Experimental company-specific dense embeddings |
| **Status** | ⚠️ Available but NOT the production serving default |

### BGE-small Base (Production Benchmark Model)
| Field | Value |
|-------|-------|
| **Source** | HuggingFace `BAAI/bge-small-en-v1.5` rev `5c38ec7c...` |
| **Architecture** | BERT, 384-d |
| **Purpose** | Dense retrieval embeddings (used in dev benchmark) |
| **Status** | ✅ Used when `SERVY_SERVING_EMBEDDING_PROFILE=bge-small` |

### MiniLM (Actual Production Default)
| Field | Value |
|-------|-------|
| **Source** | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| **Architecture** | BERT, 384-d |
| **Purpose** | General-purpose sentence embeddings |
| **Status** | ✅ DEFAULT serving model when no profile is set |

### Ollama LLM
| Field | Value |
|-------|-------|
| **Model** | qwen2.5:3b |
| **Status** | ❌ OFFLINE — answer generation untested |

---

## 3. Manifest Error Details

### Error 1: SHA Misattribution
The prior manifest assigned SHA `b1cf23ba...` (the V5 reranker checkpoint) to the `company-bge-ft` embedding model. The actual embedding model SHA is `7d394413...`.

**Impact:** Anyone verifying the embedding model against the manifest would match the wrong artifact. This is a metadata-only error — the code and serving infrastructure use the correct models.

### Error 2: Serving Config Discrepancy
The manifest implies the system was evaluated with the serving default configuration. In reality:

| Item | Manifest/Report Claim | Actual |
|------|----------------------|--------|
| Benchmark embedding | `bge-small` (via monkeypatch) | `minilm` (production default) |
| `SERVY_USE_EXPERIMENTAL_RAG` | Set to `1` during benchmark | Default `0` in production |
| `SERVY_SERVING_EMBEDDING_PROFILE` | Not set → `auto` | Resolves to `minilm` |
| Reported accuracy | 38/42 = 90.48% (on bge-small) | 36/42 = 85.71% (on minilm) |

**Impact:** The reported 90.48% Hit@1 accuracy does not represent the actual production serving configuration. The true production accuracy is 85.71%.

---

## 4. Measured Accuracy by Configuration

| Config | Hit@1 | Hit@3 | MRR@10 |
|--------|-------|-------|--------|
| BGE-small hybrid + V5 (dev benchmark) | 38/42 = 90.5% | 40/42 = 95.2% | 0.934 |
| **MiniLM hybrid + V5 (prod default)** | **36/42 = 85.7%** | **38/42 = 90.5%** | **0.890** |
| Sparse-only + V5 | 37/42 = 88.1% | 40/42 = 95.2% | 0.921 |
| MiniLM hybrid (no reranker) | 32/42 = 76.2% | 38/42 = 90.5% | 0.834 |

---

## 5. Corrections Required

### Immediate (Before Any Release)
1. **Set `SERVY_SERVING_EMBEDDING_PROFILE=bge-small`** in production `.env` to match the evaluated configuration
2. **Update all accuracy claims** to specify which embedding profile was used
3. **Regenerate the release manifest** with correct model SHA assignments

### Documentation
4. **Clearly separate** the V5 reranker identity from the embedding model identity
5. **Add `SERVY_SERVING_EMBEDDING_PROFILE` to the deployment checklist** as a mandatory configuration item
6. **Document the profile resolution logic** for operators

---

## 6. Code Integrity Assessment

| Component | Status |
|-----------|--------|
| `.env` configuration | ✅ Correct (models load from right paths) |
| `core/ml/embeddings.py` profile registry | ✅ Correct |
| `core/ml/reranker.py` model loading | ✅ Correct (SHA verified on load) |
| `core/services/retriever.py` profile resolution | ✅ Correct (deterministic) |
| Release manifest metadata | ❌ INCORRECT (SHA misattribution) |
| Accuracy reports | ❌ MISLEADING (profile not specified) |

The code and model artifacts are correct. Only the documentation/metadata is wrong.
