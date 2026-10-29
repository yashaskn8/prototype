# SERVY — Independent Benchmark & Evaluation Report (`SERVY_INDEPENDENT_BENCHMARK_REPORT.md`)

**Report Date:** 2026-10-29  
**Leakage Auditor:** [`tools/servy_v6_holdout_audit.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_v6_holdout_audit.py) & [`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py)  
**Independent Benchmark Status:** **`BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE` (True unseen independent company benchmark does NOT yet exist in the local 18-family corpus)**

---

## 1. Honest Scientific Distinction Between Available Benchmarks

Per Section 8 ("Construct a Valid Independent Benchmark"), we strictly separate and report the three evaluation populations in the Servy project:

| Benchmark Population | File / Digest | Query Count | Source Family Independence | Valid for 95%+ Independent Production Claim? |
|---|---|---|---|---|
| **1. Provisional 42-Query Development Split** | `dev_unapproved_candidate.jsonl`<br>(`c8f391b9591da9e3a7e4a97aedc117254a4c90169b382ffab550183620a4559c`) | **42 queries** (5 source families) | Disjoint from the 13 V4/V5 training families (`0` family overlap with train), **but reused across V4, V5, V6, and hybrid retrieval development** | **NO** — Reused development benchmark; shows **38/42 = 90.48% Hit@1**, **40/42 = 95.24% document Hit@1** and **100% candidate Recall@10** with hybrid + equipment-aware + technical-tokenization retrieval plus V5 (pool 5 or 8), but cannot be claimed as an unseen independent test set. |
| **2. Historical 64-Query Holdout File** | `frozen_independent_holdout_benchmark.jsonl`<br>(`f516a183d510a275de5856cb9d26eefc055181223d28cc8630db914613508dda`) | **64 queries** (43 positive-evidence + 21 empty-gold security/abstention probes) | **CONTAMINATED** — **10/10** mapped source families overlap the 13 V5 training families; **40/64** questions directly share chunk IDs (`31` distinct chunk IDs) with `v5_selected_111`. | **NO** — Explicitly blocked by `servy_v6_holdout_audit.py` (`BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE`, `evaluation_role: "HISTORICAL_DIAGNOSTIC_ONLY"`). |
| **3. Public CI Software Regression Fixture** | `sample_data/frozen_independent_holdout_benchmark.jsonl`<br>(`0c3d1cbc7ac332aa58597cdf212d6225d5fd4a4b6c188a0734cc82d8c437df14`) | **64 synthetic/sanitized queries** | Synthetic CI fixture (`ROLE_SOFTWARE_REGRESSION_FIXTURE`) | **NO** — Software regression test only, never a company accuracy benchmark. |

---

## 2. Exact Read-Only Holdout Audit Output (`servy_v6_holdout_audit.py`)

Executed on 2026-10-29 against `private_training_data/frozen_independent_holdout_benchmark.jsonl`:

- **Status:** `BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE`
- **Evaluation Role:** `HISTORICAL_DIAGNOSTIC_ONLY`
- **Independent Holdout Verified:** `false`
- **Valid for Independent Accuracy Claims:** `false`
- **Source Family Accounting:**
  - Total canonical source families in recovered corpus: **18**
  - Original V4/V5 train families: **13**
  - Provisional 42-query dev families: **5**
  - **Unused source families remaining in recovered corpus:** **`0`** (`18 - (13 + 5) = 0`)
  - Holdout families shared with V5 selected train: **10 / 10 (100% family overlap)**
- **Chunk Reference Overlap Accounting:**
  - Holdout positive vs. V5 train positive: **24 chunks**
  - Holdout positive vs. V5 train negative: **10 chunks**
  - Holdout negative vs. V5 train positive: **21 chunks**
  - Holdout negative vs. V5 train negative: **9 chunks**
  - Distinct shared chunk IDs with V5 selected train: **31 chunks** across **40 / 64 holdout questions**
- **Evidence Integrity Accounting:**
  - Structurally resolvable positive-evidence questions: **42**
  - Unresolved positive chunk reference in frozen corpus: **1**
  - Questions with no positive references (deliberate security/abstention probes): **21**
    - `unsupported_question`: 7
    - `tenant_isolation_attack`: 4
    - `doc_id_attack`: 3
    - `out_of_domain`: 3
    - `boundary_injection`: 2
    - `revoked_doc_query`: 2

---

## 3. Framework & Requirements for True 95%+ Independent Verification

Because `13 train families + 5 dev families = 18 total families` in `servy_corpus_recovered_20261018T150933Z.jsonl`, **no reshuffling or filtering of the existing 182 chunks can produce a family-disjoint independent benchmark**.

To achieve a scientifically valid, independently verified **95%+ production accuracy** claim, we implemented `verify_family_disjoint_splits()` in [`tools/servy_company_data_pipeline.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/tools/servy_company_data_pipeline.py#L228-L315), which enforces the following mandatory gates before unlocking `independent_holdout_verified=True`:

1. **New Unseen Equipment Manuals**: Ingest at least **6–10 new authorized company equipment manuals** belonging to **new canonical source families** not present in the 18 historical families (`ai_source_family_recommendations.jsonl`).
2. **Minimum Question Volume & Category Stratification**: Construct at least **100+ human-SME-verified questions** (`split="independent_test"`, `review_status="approved_by_sme"`) covering all 7 mandatory categories from Section 8:
   - Exact fault codes (`exact_error_code`)
   - Voltage & electrical specifications (`voltage_electrical_spec`)
   - Preventive maintenance schedules (`preventive_maintenance_schedule`)
   - Safety & escalation procedures (`safety_escalation`)
   - Multi-step troubleshooting (`general_symptom`, `multi_symptom`)
   - Unsupported / abstention queries (`unsupported_question`, `out_of_domain`)
   - Cross-tenant / unauthorized access refusals (`tenant_isolation_attack`, `doc_id_attack`, `revoked_doc_query`, `boundary_injection`)
3. **Zero Historical Overlap Verification**: Run `verify_family_disjoint_splits()` passing all 18 historical families and 182 historical chunk IDs as `historical_families` and `historical_chunk_ids`. Any overlap raises `CompanyDataPipelineError`.
4. **Full End-to-End Evaluation**: Evaluate both retrieval metrics (`Hit@1`, `Hit@3`, `Hit@5`, `Recall@10`, `MRR@10`) and RAG answer-level metrics (Answer Accuracy, Grounded Citation Accuracy, Abstention Accuracy, 0.0% Tenant/Customer/Equipment Leakage) via [`core/ml/evaluate_benchmark.py`](file:///C:/Users/prana/OneDrive/Desktop/SURVY-RAG/prototype/core/ml/evaluate_benchmark.py).
