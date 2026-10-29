# SERVY — Company Model Fine-Tuning & Training Report (`SERVY_COMPANY_MODEL_TRAINING_REPORT.md`)

**Report Date:** 2026-10-29  
**Training Execution Status:** **NOT EXECUTED (`PENDING_AUTHORIZED_SIGN_OFF` + `CUDA_NOT_AVAILABLE_LOCALLY` + `RETRIEVAL_ALREADY_SOLVED_BOTTLENECK`)**  
**Active Verified Reranker Checkpoint:** `best_model/model.safetensors` (`b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`)

---

## 1. Executive Determination: Why New Weight Fine-Tuning Was Not Executed

Per Section 7 ("Fine-Tuning with Company Data (Only When Ready)") and the repository's fail-closed governance rules in [`SERVY_ACCURACY_EVIDENCE_GATES.md`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/SERVY_ACCURACY_EVIDENCE_GATES.md) and [`tools/servy_v6_continuation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_continuation.py):

1. **Governance & Authorization Gate (`PENDING_AUTHORIZED_SIGN_OFF`)**:
   - [`tools/servy_v6_continuation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_continuation.py#L87-L116) enforces `check_permission()`, which requires a genuine, externally signed organizational authorization file (`status: "APPROVED_FOR_RESEARCH_TRAINING"`, `approved_by`, `approval_reference`, `selected_train_sha256: "4282d2166149e7f3979524e3919f58d23803ca0aa7266d9d2f597e971ea39012"`).
   - No such organizational authorization file exists in the workspace, and fabricating one is strictly prohibited.
2. **Hardware Gate (`CUDA_REQUIRED_FOR_V6_FINETUNING`)**:
   - Local hardware is a Windows CPU-only workstation (`torch.cuda.is_available() == False`). [`tools/servy_v6_continuation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_continuation.py#L139-L140) explicitly refuses CPU fine-tuning (`raise ValueError("CUDA_REQUIRED_FOR_V6_FINETUNING")`).
3. **Technical Justification Gate (Reranker vs. Retrieval Bottleneck)**:
   - Our controlled experiments (Experiments A–E) proved that the **existing verified V5 CrossEncoder (`b1cf23ba...`) was not the bottleneck**.
   - Once candidate retrieval was fixed via technical voltage/unit tokenization (`5V`/`12V`/`19V` <-> `"operating V specification 5/12/19"`), equipment-metadata-aware indexing, and hybrid RRF (`pool_size=8`), the **existing unmodified V5 checkpoint rose from 35/42 (83.33%) to 38/42 (90.48%) Hit@1** (pool 5 or 8) on the reused 42-query development set, with candidate Recall@10 rising 90.48% → 100%. Measured recoveries: `OP26-SLV-75-2`, `OP26-SLV-74-1`, `OP26-SLV-142-2`; no regressions. Still failing: `OP26-SLV-144-2` (now in the candidate pool, but V5 ranks it below rank 1), `OP26-SLV-145-2`, `OP26-SLV-143-0`, `OP26-SLV-76-2`. So the reranker is now part of the remaining error, and new training stays blocked by the gates above.
   - Furthermore, the historical V6 continuation run (`v6_research_first_run`) already demonstrated that further fine-tuning V5 on the 111-query subset yielded **0 net Hit@1 gain** (35/42 vs 35/42) on frozen candidates.

---

## 2. Historical & Local Fine-Tuned Checkpoints Audited

| Model / Checkpoint | Base Architecture | Local Path | Weights / Directory SHA-256 | Training Dataset | Measured Outcome | Promotion Status |
|---|---|---|---|---|---|---|
| **Servy V5 CrossEncoder Reranker** (Verified Baseline) | `BAAI/bge-reranker-v2-m3` (`XLMRobertaForSequenceClassification`, 256 max seq len) | `prototype/best_model/model.safetensors` | Weights: `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`<br>Dir: `47cd54fc552c1cf5327885365dbe443bbe8f3dbc955bb6b0ef1a9c35bd8b0192` | `v5_selected_111/train_keep_ai_provisional.jsonl` (111 queries, 111 pos, 161 neg, BCE loss) | **35/42 (83.33%) Hit@1** on frozen BM25 (`pool=5`); **38/42 (90.48%) Hit@1** with Hybrid RRF + EquipMeta + TechToken (`pool=8`) | Verified for guarded local serving (`SERVY_RERANKER_ENABLED=1`); not promoted as 95% independent claim |
| **Servy Company BGE Dense Embedder v1** | `BAAI/bge-small-en-v1.5` (384-D, normalized) | `prototype/ml_models/checkpoints/servy_company_bge_v1/final` | Dir: `7d394413398be69d2d1093538e5eec6cafa58cbfeae858cac4abf608b5db0d33` | Weakly supervised company pairs (`COMPANY_DATA_WEAKLY_SUPERVISED_EXPERIMENT`) | measured in hybrid + V5: **38/42 (90.48%)** at pool 8, identical to pretrained `bge-small-en-v1.5` (no gain over pretrained) | `promotion_eligible=False` (retained as experimental checkpoint; pretrained `bge-small-en-v1.5` matches it without overfitting risk) |
| **Servy V6 Continuation Reranker** (Historical Colab Run) | Continued from V5 (`b1cf23ba...`) | Historical Colab `/content/drive/MyDrive/Servy_V6_Authorized/outputs/v6_research_first_run/best_model` | Weights: `3fb4a7074bf6838d166865e811484bc8ec0218bb793e535205212a87846b803f` | `v5_selected_111` (1 epoch, `lr=3e-7`, `batch_size=1`, seed 42) | **35/42 (83.33%) Hit@1** on frozen BM25 (`pool=5`), MRR@10 `0.8599` vs V5 `0.8605` (0 rank changes in top-5) | Not promoted (`V6_EXPLORATORY_COMPLETED_NOT_PROMOTED`) |

---

## 3. Guarded Fine-Tuning Protocol When New SME-Approved Data Arrives

Once (a) human SMEs review and sign off on the training labels (`review_status="approved_by_sme"`), (b) a new unseen source-family-disjoint independent test benchmark is collected and verified via [`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py), and (c) a scoped organizational permission JSON record is issued, execute V6 continuation on an authorized GPU workstation:

```powershell
python tools/servy_v6_continuation.py `
  --split-dir "private_training_data/recovery_v4/family_resplit_20261018_234238" `
  --corpus "private_training_data/recovery_v4/servy_corpus_recovered_20261018T150933Z.jsonl" `
  --families "private_training_data/recovery_v4/ai_review_evidence_resplit_20261018_234537/ai_source_family_recommendations.jsonl" `
  --candidates "private_training_data/recovery_v4/research_artifacts_resplit_20261019/frozen_candidates.jsonl" `
  --candidate-train "private_training_data/recovery_v4/v5_selected_111/train_keep_ai_provisional.jsonl" `
  --manifest "private_training_data/recovery_v4/v5_selected_111/selection_integrity_manifest.json" `
  --v5-model "best_model" `
  --out-dir "outputs/v6_authorized_run" `
  --train `
  --authorization-file "private_training_data/org_training_permission.json" `
  --execution-environment "local_secure" `
  --epochs 1 `
  --batch-size 1 `
  --learning-rate 3e-7
```
