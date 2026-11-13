# SERVY — 83.33% Accuracy Recovery, Verification & Evaluation Audit Report

**Audit Date:** October 29, 2026  
**Auditor Roles:** Principal Machine Learning Engineer, Information Retrieval Researcher, RAG Systems Engineer, ML Evaluation Auditor  
**Repository:** https://github.com/yashaskn8/prototype  
**Current Commit (HEAD):** `358c48c` (Fast-forwarded with `origin/main`)  
**Target Evaluation:** Historical 35/42 = 83.33% Hit@1 frozen development benchmark  

---

## Executive Summary

1. **Recovery Status:** **YES — 100% RECOVERED AND REPRODUCED.**  
   The historical **35/42 = 83.33% Hit@1** result was located across the project's local authorized recovery archives, exact model weights, and frozen datasets. The evaluation was replayed deterministically using `tools/servy_checkpoint_audit.py` on the verified local V5 checkpoint on Windows CPU.
2. **Exact Reproduction Numbers:**
   * **Hit@1:** **35/42 = 83.33%** (Exact match with historical figure, $\Delta = 0.00\%$)
   * **Hit@5:** **37/42 = 88.10%** (Exact match)
   * **Hit@10:** **38/42 = 90.48%** (Exact match)
   * **Candidate Recall@10:** **38/42 = 90.48%** (Exact match)
   * **MRR@10:** **0.8605**
   * **Head-5 Top-One Oracle Ceiling:** **37/42 = 88.10%**
   * **Queries Missing Gold in Top-10 Candidates:** **4/42 = 9.52%**
3. **Model Weights Fingerprint:**  
   The local checkpoint at `prototype/best_model/model.safetensors` (size: 2,271,071,852 bytes) matches the repository's approved SHA-256 fingerprint:  
   `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`
4. **Runtime Serving Verification:**  
   `python manage.py verify_v5_reranker` **PASSED** (model loaded, SHA verified, CPU inference finite).  
   `python manage.py test core.tests_v5_runtime -v 2` **PASSED** (Ran 1 test in 33.722s, OK) using synthetic records, zero confidentiality leaks, zero tenant leaks, and verified response headers `reranker_applied=True` and `reranker_model="servy_v5_bge_verified"`.
5. **Why V6 Was Not Promoted:**  
   V6 achieved identical aggregate accuracy (35/42 = 83.33% Hit@1 and 37/42 = 88.10% Hit@5) on the reused dev set. It was bounded by the same candidate retrieval ceiling, had zero human sign-off (`PENDING_AUTHORIZED_SIGN_OFF`), and lacked an independent holdout.

---

## 1. Recovered Artifacts and File Provenance

All primary files required to execute, audit, and reproduce the 83.33% evaluation were located in the local workspace:

| Artifact Description | Local File Location | File Size (Bytes) | SHA-256 Checksum | Provenance / Role |
|---|---|---|---|---|
| **V5 Model Weights** | `prototype/best_model/model.safetensors` | 2,271,071,852 | `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda` | Verified V5 CrossEncoder weights |
| **V5 Model Config** | `prototype/best_model/config.json` | 873 | `91e8eaecfca4819bafe4dfc9497d95d1033235b2e564a5113d5e23c04efab7e4` | XLMRoberta architecture config |
| **V5 Tokenizer** | `prototype/best_model/tokenizer.json` | 17,083,153 | `141b7fcf76ebec3932824cf96517a9689f4b0d876bc7551066c8fb7bc1caee78` | BAAI BGE reranker tokenizer |
| **Frozen Candidates** | `prototype/private_training_data/recovery_v4/research_artifacts_resplit_20261019/frozen_candidates.jsonl` | 4,247 | `8b2b3f54a597655e432ee63bcec86d8607085cf72c56e818776e0de960910e8d` | 42 dev queries $\times$ 10 BM25 candidates |
| **Dev Split Cases** | `prototype/private_training_data/recovery_v4/family_resplit_20261018_234238/dev_unapproved_candidate.jsonl` | 73,315 | `c8f391b9591da9e3a7e4a97aedc117254a4c90169b382ffab550183620a4559c` | 42 development evaluation cases |
| **Recovered Corpus** | `prototype/private_training_data/recovery_v4/servy_corpus_recovered_20261018T150933Z.jsonl` | 147,569 | `50792737f0b297d2deb63e6a4222fd75fa569fd1ea7d104395517d60b39bd2fe` | 182 recovered knowledge chunks |
| **Source Families** | `prototype/private_training_data/recovery_v4/ai_review_evidence_resplit_20261018_234537/ai_source_family_recommendations.jsonl` | 23,860 | `598553900e22fccb8af64aaa71f0481344f8b32279f0b1c251411a90eee499e5` | 57 canonical document source families |
| **Colab Package Zip** | `prototype/private_training_data/servy_v6_colab_inputs.zip` | 2,126,620,011 | `df130a05a109a250325dff6696b996766468cbbe7c87c328e18df0cf6238b91c` | Complete archive uploaded to `/content/servy_v6_inputs/` |

### Reconciliation with Historical Paths
Historical documentation in `SERVY_ACCURACY_EVIDENCE_GATES.md` and `V5_PRIVATE_DATASET_AUDIT.md` referenced paths under:
- `/content/servy_v6_inputs/recovery_v4/...`
- `/content/servy_v6_inputs/best_model/...`

Audit confirms that the local files in `prototype/private_training_data/recovery_v4/` and `prototype/best_model/` are the exact underlying files that were packed into `servy_v6_colab_inputs.zip` and unpacked in Google Colab under `/content/servy_v6_inputs/`. All SHA-256 hashes match identically.

---

## 2. Model Verification and Checkpoint Identity

1. **Base Model Architecture:** `BAAI/bge-reranker-v2-m3` (`XLMRobertaForSequenceClassification`), loaded through `sentence_transformers.CrossEncoder`.
2. **Fine-Tuning Configuration:**
   - Loss Function: `BinaryCrossEntropyLoss`
   - Maximum Sequence Length: 256 tokens
   - Batch Size: 8
   - Activation Function: Raw identity logit scoring during evaluation and ranking; monotonic calibrated sigmoid for UI metadata exposure.
3. **Approved Checkpoint Fingerprint:**
   - Single-file weights: `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`
   - Complete directory digest (7 files): `47cd54fc552c1cf5327885365dbe443bbe8f3dbc955bb6b0ef1a9c35bd8b0192`
4. **Input Preprocessing and Format:**
   - Pairs formatted as: `[query_text, passage_text]`
   - Query format: `raw-question`
   - Passage format: `heading-body`
5. **Candidate Generation Method:**
   - Frozen candidate set generated via BM25 lexical retrieval filtered strictly by `tenant_id`.
   - Candidate pool: 10 candidates per query.

---

## 3. Evaluation Protocol and Replay Execution

### Command Executed
```powershell
python tools/servy_checkpoint_audit.py `
  --cases private_training_data/recovery_v4/family_resplit_20261018_234238/dev_unapproved_candidate.jsonl `
  --corpus private_training_data/recovery_v4/servy_corpus_recovered_20261018T150933Z.jsonl `
  --candidates private_training_data/recovery_v4/research_artifacts_resplit_20261019/frozen_candidates.jsonl `
  --model v5=best_model `
  --pool-size 5 `
  --top-k 10 `
  --device cpu `
  --out-dir outputs/v5_replay_recovery_20261029
```

### Actual Output and Verification Log
```text
Loading weights: 100%|##########| 393/393 [00:00<00:00, 6166.06it/s]
cases_sha256: c8f391b9591da9e3a7e4a97aedc117254a4c90169b382ffab550183620a4559c
corpus_sha256: 50792737f0b297d2deb63e6a4222fd75fa569fd1ea7d104395517d60b39bd2fe
candidates_sha256: 8b2b3f54a597655e432ee63bcec86d8607085cf72c56e818776e0de960910e8d
pool_size: 5, top_k: 10
Results:
  queries: 42
  hit_at_1: 83.33% (35/42)
  hit_at_5: 88.10% (37/42)
  hit_at_10: 90.48% (38/42)
  candidate_recall_at_10: 90.48% (38/42)
  document_hit_at_1_unverified: 90.48% (38/42)
  document_hit_at_5_unverified: 92.86% (39/42)
  mrr_at_10: 0.8605
```

---

## 4. Metric Comparison and Historical Reconciliation

### Reconciling the 88.10% vs 83.33% Discrepancy
The historical documentation noted two different V5 measurements: 37/42 (88.10%) Hit@1 vs 35/42 (83.33%) Hit@1.
Audit of `artifact_manifest.json` and `V5_LOCAL_RERANKER_SETUP.md` reveals the exact technical reason:
1. **The 88.10% (37/42) Run:** Used **Qwen3-0.6B dense embedding retrieval** (`qwen3_candidates_top10.jsonl`). Candidate Recall@10 was **40/42 (95.24%)**. Only 2 queries lacked gold candidates. Under head-5 reranking, V5 scored 37/42 = 88.10%.
2. **The 83.33% (35/42) Run:** Used **BM25 lexical retrieval** (`frozen_candidates.jsonl`). Candidate Recall@10 dropped to **38/42 (90.48%)**. 4 queries had zero gold candidates in the top 10. Under head-5 reranking, V5 scored 35/42 = 83.33%.

The two runs evaluated two **different candidate retrieval frontends** against the same 42 development questions. The reranker weights were identical.

### Counterfactual Pool Sweep Analysis
Reusing the exact replayed logits through `simulate_pool_sweep` (tools/servy_accuracy_diagnostics.py):

| Pool Size | Hit@1 | Hit@5 | Hit@10 | MRR@10 | Oracle Head Ceiling | Note |
|---|---|---|---|---|---|---|
| **Pool = 5 (Serving Baseline)** | **83.33% (35/42)** | 88.10% (37/42) | 90.48% (38/42) | 0.8605 | 88.10% (37/42) | Official baseline configuration |
| **Pool = 8** | **83.33% (35/42)** | 90.48% (38/42) | 90.48% (38/42) | 0.8619 | 90.48% (38/42) | Hits 145 at rank 5, but Hit@1 stays 35/42 |
| **Pool = 10** | **83.33% (35/42)** | 88.10% (37/42) | 90.48% (38/42) | 0.8566 | 90.48% (38/42) | Full candidate rerank; no Hit@1 gain |

**Key Finding:** Increasing reranking pool size from 5 to 10 on the existing BM25 candidates produces **0% gain in Hit@1**. The bottleneck is candidate retrieval quality, token mismatch, and label noise, not pool truncation.

---

## 5. Confidence Classification of Conclusions

| Finding / Conclusion | Empirical Status | Confidence Level | Evidence Supporting |
|---|---|---|---|
| Historical 35/42 = 83.33% was reached on BM25 candidates | Verified | **HIGH (100%)** | Exact reproduction using verified hashes and code |
| Approved V5 weights SHA-256 is `b1cf23ba...` | Verified | **HIGH (100%)** | Byte-for-byte SHA-256 match on `prototype/best_model/model.safetensors` |
| Real-weight Django runtime passes zero-leakage security gates | Verified | **HIGH (100%)** | `core.tests_v5_runtime` ran and passed in 33.7s |
| Head-5 Oracle Ceiling on this benchmark is 88.10% | Verified | **HIGH (100%)** | 5 queries have gold rank $>5$ or absent |
| Candidate Recall@10 is 90.48% | Verified | **HIGH (100%)** | Exactly 4 queries have zero gold in top 10 |
| PR #16 contains serving verification hardening | Verified | **HIGH (100%)** | Inspected PR #16 commits and diffs |
| PR #16 CI failure is due to regex assertion mismatch | Verified | **HIGH (100%)** | `tests_v5_activation.py` line 129 expected string |
| Independent holdout is invalid due to data leakage | Verified | **HIGH (100%)** | `servy_v6_holdout_audit.py` proves 10/10 family overlap |
| V6 should NOT be promoted to production | Verified | **HIGH (100%)** | Zero accuracy advantage over V5; pending approvals |
