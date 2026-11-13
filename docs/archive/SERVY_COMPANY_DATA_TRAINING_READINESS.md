# SERVY — Company Data Training Readiness Report (`SERVY_COMPANY_DATA_TRAINING_READINESS.md`)

**Report Date:** 2026-10-29  
**Pipeline Implementation:** [`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py)  
**Preflight & Continuation Gates:** [`tools/servy_v5_training_preflight.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v5_training_preflight.py), [`tools/servy_v6_continuation.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_continuation.py), [`tools/servy_v6_holdout_audit.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_holdout_audit.py)

---

## 1. Executive Summary

We audited all locally available company data assets in `prototype/private_training_data/` and implemented the 10-step company-data curation and leakage-verification pipeline in [`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py) (validated by unit tests in [`tools/test_servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/test_servy_company_data_pipeline.py)).

| Readiness Gate | Measured Status | Verified Value / Output |
|---|---|---|
| **V5 Training Preflight Gate** (`servy_v5_training_preflight.py`) | **`READY_FOR_EXPLORATORY_FINE_TUNING`** | 111 selected train queries, 111 positive pairs, 161 active hard-negative pairs, 42 dev queries, 109 quarantined negative references |
| **V6 Continuation Readiness Gate** (`servy_v6_continuation.py`) | **`V6_PLAN_READY_TRAINING_NOT_AUTHORIZED`** | Base V5 SHA-256 `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda` verified; `external_training_authorized=false` |
| **V6 Holdout Leakage Gate** (`servy_v6_holdout_audit.py`) | **`BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE`** | 10/10 holdout source families overlap training families; 0 unused families remain in the 18-family recovered corpus |
| **Human SME Review Gate** | **`PENDING_AUTHORIZED_SIGN_OFF`** | `expert_labels_approved=false`; no authenticated human SME sign-off or organizational training authorization file exists on disk |
| **Production Promotion Gate** | **`BLOCKED`** (`production_promotion_allowed=false`) | Requires SME-approved labels + unseen source-family-disjoint independent benchmark |

---

## 2. Inventory & Cryptographic Fingerprints of Local Company Datasets

All private files were inspected read-only on the local workstation (`prototype/private_training_data/`) and remain excluded from Git via `.gitignore`:

| Dataset Artifact | Local Relative Path | SHA-256 Digest | Records / Scope |
|---|---|---|---|
| **Recovered Company Corpus** | `private_training_data/recovery_v4/servy_corpus_recovered_20261018T150933Z.jsonl` | `50792737f0b297d2deb63e6a4222fd75fa569fd1ea7d104395517d60b39bd2fe` | 182 chunks across 57 documents (18 canonical source families) |
| **Canonical Source Family Map** | `private_training_data/recovery_v4/ai_review_evidence_resplit_20261018_234537/ai_source_family_recommendations.jsonl` | `598553900e22fccb8af64aaa71f0481344f8b32279f0b1c251411a90eee499e5` | 57 document-to-family mappings (13 train families, 5 dev families) |
| **Original V4 Train Split** | `private_training_data/recovery_v4/family_resplit_20261018_234238/train_unapproved_candidate.jsonl` | `6343b62fb6dec8fef2966503c25259436e3466806b89475e2ea053ed9979bbb1` | 353 provisional queries (13 source families) |
| **Selected V5 Train Subset** | `private_training_data/recovery_v4/v5_selected_111/train_keep_ai_provisional.jsonl` | `4282d2166149e7f3979524e3919f58d23803ca0aa7266d9d2f597e971ea39012` | 111 machine-triaged queries (111 positive pairs, 161 active hard negatives) |
| **V5 Selection Manifest** | `private_training_data/recovery_v4/v5_selected_111/selection_integrity_manifest.json` | `e52b2859369f21f99a3a5b62b3056269666b026675800e266cf89fb83f623721` | `expert_approved=false`, `production_promotion_allowed=false` |
| **Frozen 42-Query Dev Split** | `private_training_data/recovery_v4/family_resplit_20261018_234238/dev_unapproved_candidate.jsonl` | `c8f391b9591da9e3a7e4a97aedc117254a4c90169b382ffab550183620a4559c` | 42 queries (5 source families disjoint from train) |
| **Frozen BM25 Candidates** | `private_training_data/recovery_v4/research_artifacts_resplit_20261019/frozen_candidates.jsonl` | `8b2b3f54a597655e432ee63bcec86d8607085cf72c56e818776e0de960910e8d` | 42 x top-10 BM25 candidates |
| **Historical 64-Query Holdout File** | `private_training_data/frozen_independent_holdout_benchmark.jsonl` | `f516a183d510a275de5856cb9d26eefc055181223d28cc8630db914613508dda` | 64 queries (43 relevance, 21 abstention/security; 10/10 families overlap train) |

---

## 3. Ten-Step Company Data Curation Pipeline Implementation

[`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py) implements all 10 mandatory steps from Section 6:

1. **Document Ingestion & Validation (`validate_and_quarantine_documents`)**: Verifies `document_id`, `tenant_id`, `title`, and non-empty technical body text; quarantines incomplete records with explicit reason codes.
2. **Section-Aware Segmentation (`segment_document_sections`)**: Splits manuals on markdown headings (`#`, `##`, `###`), preserving the parent heading inside each chunk (`## <Heading>\n<Body>`) and computing per-chunk `content_sha256`.
3. **Version & Checksum Tracking**: Computes normalized SHA-256 content digests (`checksum_sha256`) and binds chunks to `version`.
4. **Equipment & Model Metadata Mapping (`canonical_source_family`)**: Normalizes `equipment_family`, `model_number`, and derives canonical `source_family` identifiers (stripping revision/copy suffixes so duplicate manual versions map to the same family).
5. **Duplicate & Near-Duplicate Detection**: Detects identical normalized content within a tenant and quarantines duplicate manuals (`DUPLICATE_CONTENT_OF_DOC_<id>`).
6. **Confidential Content Minimization (`minimize_confidential_text`)**: Excludes documents marked `is_confidential=True` (unless explicitly authorized) and redacts incidental email addresses (`[REDACTED_EMAIL]`) and phone numbers (`[REDACTED_PHONE]`).
7. **Revoked Document Quarantine**: Quarantines any document with `revoked_at != None` or `is_revoked=True` (`REVOKED_DOCUMENT_QUARANTINED`).
8. **Query, Positive & Hard-Negative Schema Validation (`validate_training_example_schema`)**: Enforces the exact 14-field JSON schema (`query_id`, `tenant_id`, `equipment_family`, `model_number`, `query_type`, `query_text`, `positive_chunk_ids`, `positive_doc_ids`, `hard_negative_chunk_ids`, `source_family`, `expected_abstain`, `review_status`, `reviewer`, `split`).
9. **Human SME Review Gate**: Enforces that `review_status == "approved_by_sme"` requires a non-empty `reviewer` identifier and never auto-populates SME approval.
10. **Family-Disjoint Split & Historical Leakage Verification (`verify_family_disjoint_splits`)**: Rejects any split where `source_family`, chunk IDs, or normalized query texts overlap across `train`, `dev`, and `independent_test`, or where `independent_test` overlaps any historical V4/V5/V6 family or chunk ID.

---

## 4. Exact Preflight & Continuation Gate Outputs (Executed 2026-10-29)

### 4.1 V5 Training Preflight (`tools/servy_v5_training_preflight.py`)

```json
{
  "status": "READY_FOR_EXPLORATORY_FINE_TUNING",
  "production_promotion_allowed": false,
  "independent_holdout_verified": false,
  "expert_labels_approved": false,
  "train_queries": 111,
  "exact_original_subset_verified": true,
  "active_negative_per_query_verified": true,
  "positive_pairs": 111,
  "negative_pairs": 161,
  "dev_queries": 42,
  "quarantined_negative_references": 109,
  "original_train_sha256": "6343b62fb6dec8fef2966503c25259436e3466806b89475e2ea053ed9979bbb1",
  "candidate_train_sha256": "4282d2166149e7f3979524e3919f58d23803ca0aa7266d9d2f597e971ea39012"
}
```

### 4.2 V6 Continuation Plan (`tools/servy_v6_continuation.py`)

```json
{
  "status": "V6_PLAN_READY_TRAINING_NOT_AUTHORIZED",
  "base_checkpoint": {
    "weights_sha256": "b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda",
    "model_path": "C:\\Users\\prana\\OneDrive\\Desktop\\SURVY-RAG\\prototype\\best_model"
  },
  "selected_train_sha256": "4282d2166149e7f3979524e3919f58d23803ca0aa7266d9d2f597e971ea39012",
  "original_train_sha256": "6343b62fb6dec8fef2966503c25259436e3466806b89475e2ea053ed9979bbb1",
  "train_queries": 111,
  "positive_pairs": 111,
  "negative_pairs": 161,
  "dev_queries": 42,
  "quarantined_negative_references": 109,
  "starts_from_v5_weights": true,
  "uses_selected_subset_only": true,
  "expert_labels_approved": false,
  "independent_holdout_verified": false,
  "production_promotion_allowed": false,
  "external_training_authorized": false,
  "training_performed": false
}
```
