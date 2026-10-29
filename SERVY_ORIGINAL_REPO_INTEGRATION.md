# SERVY â€” Original Repository Integration & Deployment Guide (`SERVY_ORIGINAL_REPO_INTEGRATION.md`)

**Report Date:** 2026-10-29  
**Development Repository:** `https://github.com/yashaskn8/prototype` (`C:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype`)  
**Pre-GitHub Local Snapshot Inspected:** `C:\Users\prana\Downloads\servy_rag_hardened_final\servy_rag_hardened` (commit `be0e247`, remote `https://github.com/yashaskn8/prototype.git`)

---

## 1. Repository & Upstream Connectivity Audit

We inspected all Git remotes and local directories across the workstation:
1. `C:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype` has a single remote: `origin -> https://github.com/yashaskn8/prototype.git`.
2. The original pre-GitHub local archive at `C:\Users\prana\Downloads\servy_rag_hardened_final\servy_rag_hardened` is at commit `be0e247` (the root commit of `prototype` PR #1) and points to the same `https://github.com/yashaskn8/prototype.git` remote.
3. No separate external company GitHub remote is configured on this machine. Therefore, this integration package provides exact drop-in file mappings, checkpoint installation commands, environment configurations, index rebuild steps, and rollback procedures to port the verified V5 reranker and hybrid retrieval improvements into any upstream Servy production deployment.

---

## 2. Files to Copy / Synchronize into Upstream Servy Repository

| Upstream Target Path | Purpose & Changes |
|---|---|
| [`core/services/vectors.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/services/vectors.py) | Technical voltage/unit tokenization (`5V`/`12V`/`19V` <-> `"operating V specification 5/12/19"`), single-digit number preservation, fault code normalization (`E-12` <-> `e12`), and bounded morphological normalization (`escalation` <-> `escalate`/`stop`, `intervals` <-> `monthly`/`quarterly`/`annual`/`preventive`). |
| [`core/services/retriever.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/services/retriever.py) | Equipment-aware lexical indexing (`_sparse_index_text`), query-to-document equipment/title matching in `_calculate_metadata_bonus`, configurable `ServingConfig` (`SERVY_RRF_K`, `SERVY_RRF_WEIGHT_DENSE`, `SERVY_RRF_WEIGHT_SPARSE`, `SERVY_RETRIEVAL_MODE`), deterministic RRF tie-breaking, and bounded V5 reranker invocation. |
| [`core/ml/reranker.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/ml/reranker.py) | Thread-safe offline `CrossEncoder` loader with strict SHA-256 weight verification (`SERVY_V5_WEIGHTS_SHA256`), raw logit identity scoring (`torch.nn.Identity()`), `require_verified_v5_reranker()` startup gate, and fail-closed/fallback handling. |
| [`core/ml/embeddings.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/ml/embeddings.py) | Versioned Chroma collection profiles (`minilm`, `bge-small`, `company-bge-small`, `qwen3`, `company-qwen3`, `company-bge-ft`) with manifest/fingerprint enforcement. |
| [`core/ml/indexer.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/ml/indexer.py) | Parallel idempotent dense index builder (`build_index`, `reconcile_index`, `verify_index`) with company zero-deletion protection. |
| [`core/management/commands/verify_v5_reranker.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/management/commands/verify_v5_reranker.py) | Django CLI verification command (`python manage.py verify_v5_reranker`) that validates weight SHA-256 and runs a live serving-path probe. |
| [`tools/install_servy_v5_checkpoint.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/install_servy_v5_checkpoint.py) | Fail-closed local checkpoint installer that verifies `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda` before copying into `best_model/`. |
| [`core/tests_v5_activation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/tests_v5_activation.py), [`core/tests_v5_runtime.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/tests_v5_runtime.py), [`core/tests_production_rag_accuracy.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/tests_production_rag_accuracy.py) | Complete activation, runtime HTTP/RAG, technical tokenization, RRF, and security test suites. |

---

## 3. Step-by-Step Production Deployment & Verification

### Step 1: Install & Verify the Private V5 Checkpoint Outside Git
Never commit `best_model/model.safetensors` (2.27 GB) to Git. Provision it onto the production server from secure artifact storage and run:

```powershell
python tools/install_servy_v5_checkpoint.py `
  --source "/secure/storage/servy_v5_best_model" `
  --target "./best_model"
```

Verify the installed weights SHA-256 matches `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`.

### Step 2: Configure Production Environment Variables (`.env`)

```ini
# Retrieval & RRF Configuration
SERVY_RETRIEVAL_MODE=hybrid
SERVY_RRF_K=60
SERVY_RRF_WEIGHT_DENSE=0.55
SERVY_RRF_WEIGHT_SPARSE=0.45
SERVY_DENSE_RELEVANCE_GATE=0.38
SERVY_SPARSE_RELEVANCE_GATE=0.055
SERVY_RAG_TOP_K=6

# Verified V5 CrossEncoder Reranker Configuration
SERVY_RERANKER_ENABLED=1
SERVY_RERANKER_REQUIRED=1
SERVY_RERANKER_MODEL_PATH=best_model
SERVY_RERANKER_EXPECTED_SHA256=b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda
SERVY_RERANKER_POOL_SIZE=8
SERVY_RERANKER_DEVICE=cpu
```

> **Note on `SERVY_RERANKER_POOL_SIZE=8`:** Increasing the reranker candidate pool from `5` to `8` recovers candidate-depth failures (such as `OP26-SLV-145-2` at rank 7 in BM25) while keeping CPU reranking latency bounded.

### Step 3: Build & Verify Versioned Dense Indexes (Optional BGE Upgrade)
If enabling `bge-small` / `company-bge-small` (`SERVY_USE_EXPERIMENTAL_RAG=1`):

```powershell
python manage.py shell -c "from core.ml.indexer import build_index, verify_index; print(build_index('company-bge-small')); print(verify_index('company-bge-small'))"
```

### Step 4: Run Startup & Runtime Verification Gates
Before routing live traffic:

```powershell
python manage.py verify_v5_reranker
python manage.py test core.tests_v5_activation core.tests_production_rag_accuracy -v 2
```

---

## 4. Instant Rollback Plan

1. **Reranker-Only Rollback (Zero Downtime, No Re-indexing):**
   Set `SERVY_RERANKER_REQUIRED=0` and `SERVY_RERANKER_ENABLED=0`. `retrieve()` immediately returns the first-stage Hybrid RRF ranking (`first-stage measurements are in SERVY_HYBRID_RETRIEVAL_EVALUATION.json; Recall@10 = 100%` on the dev set).
2. **Dense Profile Rollback:**
   Set `SERVY_USE_EXPERIMENTAL_RAG=0` and `SERVY_SERVING_EMBEDDING_PROFILE=minilm`. Because `core/ml/indexer.py` never modifies the legacy `servy_dense_kb` or `servy_company_dense_kb` collections, rollback to MiniLM is instantaneous.

