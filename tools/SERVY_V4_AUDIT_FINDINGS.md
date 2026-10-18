# Servy V4 — evidence-backed source audit, 2026-10-18

**Scope:** public `yashaskn8/prototype` audit branch, compared to historical
`7243578fd2aedbd89ee6ad9cf55ddaacc21e48ed`. All line numbers refer to
the current audit-branch revision except where identified. No company
documents, private train/dev data, Colab checkpoints, or production database
were accessed in this review. Not every branch in every file has received
dynamic runtime testing.

**Verdict: TRAINING BLOCKED — VALIDATION.** The structural gate can pass with
fully synthetic rows, but no private approved dataset, verified human labels,
unseen answer correctness holdout, restored V2/V3 weights, or successful
real-GPU V4 training verification is available. Standalone model-free safety
checks are **not** expert ground truth.

## Actual execution trace (verified code references)

1. Company document intake: `core/services/starlly_ingest.py`,
   `ingest_starlly_real_data:377` / `split_document_into_sections:272`
   -> `core/models.py` KnowledgeDocument. Attachment validation is
   `sanitize_attachment_filename:56` / `validate_attachment_url:143` /
   `fetch_attachment_content:166`. Company-data runtime content **not verified**.
2. Manual ingestion: `core/services/indexing.py:index_document:24` ->
   `core/services/document_loader.py:extract_document_text:39` ->
   `core/services/chunking.py:chunk_markdownish:26` ->
   local `embed_text` and KnowledgeChunk creation, then MySQL atomic swap
   (`indexing.py:68-85`) -> `chroma_store.py:queue_chroma_sync:295` ->
   `sync_document_to_chroma:174` -> versioned Chroma `upsert:259`.
   The separate profile index is `core/ml/indexer.py:build_index:80`.
3. Query entry: `core/services/rag.py:ask_rag:8` ->
   `core/services/retriever.py:retrieve:884`. Candidate authority:
   `_knowledge_candidates:157` / `is_authorized_candidate:66` /
   `core/services/knowledge_scope.py:get_applicable_chunks_queryset:93`;
   sparse `_sparse_score_candidates:476`; dense
   `_dense_score_candidates:600`; `_reciprocal_rank_fusion:818`;
   `core/ml/conditional_rerank.py:conditional_swap_rerank:623`;
   `core/ml/reranker.py:rerank_candidates:55` (optional MiniLM,
   disabled by default), source tiering and bounded selection
   `retriever.py:1095-1152`.
4. Answer authority: `core/services/llm.py:generate_answer:817`
   -> `check_evidence_sufficiency:684`, equipment/source/content checks ->
   `_extractive_answer:434` or guarded LLM path ->
   `core/services/rag.py:78-98` exposes *claim-supporting references*, not
   every retrieved candidate. This distinction is essential for evaluating CI
   failures; a retrieved document is not automatically an approved answer.
5. Label generation: `tools/op26_build_training_dataset.py:202` ->
   `find_hard_negatives:141` -> `verify_negative_is_safe:175` (heuristic,
   NOT expert proof); reviewer-aware path in `core/ml/labels.py:
   resolve_provenance_verified_examples:183`. General
   `core/ml/training.py` and Op27 training in
   `tools/op27_train_and_evaluate.py:train_single_model` are separate from
   proposed BGE reranker V4.
6. Benchmark: `tools/servy_checkpoint_audit.py:normalize_inputs:58`,
   `validate_candidates:193`, `build_qwen_candidates:221`,
   `rerank_fixed:250`, `evaluate_one:293`. It freezes research
   candidates but does **not** reproduce customer/site/asset ACLs. Final
   independent answer safety was **not** evaluated.

## High priority / directly supported findings

| Severity | Evidence & mechanism | Repair / verification |
|---|---|---|
| HIGH fixed in audit tools | `servy_finetune_preflight.py:inspect_split:133-211` accepted omitted label-row group despite corpus having canonical family; this could miss guide-family overlap. | Source-family ID now gathered from positive AND negative corpus chunks; synthetic split-family collision test in `test_servy_finetune_preflight.py`. |
| HIGH fixed in audit tools | `servy_finetune_preflight.py:training_collisions:235` originally warned about repeated positives but did not block training where one row's negative was another row's positive. | Cross-row candidate conflict search now checks chunk ID and normalized corpus text across Stage 1 and 2. Four synthetic collision/leakage tests added. It does not detect **semantic** equivalence without expert labels. |
| HIGH fixed in audit tools | Original optional holdout accepted absent gold IDs without explicit unanswerable status, which could dilute or bias measured correctness. | Empty gold is now blocked unless explicitly labeled unanswerable; regression added. |
| HIGH fixed in strict gate | No single strict model-free command binding input hashes, corpus versions, reviewer status, model/tokenizer revision, split isolation and checkpoint overwrite refusal. | Added `tools/servy_training_preflight.py` and `test_servy_training_preflight.py`; statuses are machine readable. Approval and model revisions are unverified until supplied from authorized external evidence. |
| HIGH fixed in provenance labels | `tools/op26_build_training_dataset.py:141-199,325-369` names lexical/heading screens verified negatives despite absence of human review. | Reclassified these examples as `provisional_unverified`, with generation contract tests in `test_op26_negative_governance.py`. **Still unsuitable as production gold labels.** |
| HIGH fixed in admission | Strict label admission formerly omitted singular `negative_chunk_id` and `candidate_negative_chunk_id` when requiring negative review approval. | Both now trigger negative approval verification; two synthetic bypass regression tests. |
| HIGH partial | Missing Colab checkpoint persistence caused historical V2/V3 artifacts to be potentially lost; comparison weights cannot be recreated from logs. | Existing `servy_checkpoint_backup.py` verifies copy hashes and mounted persistent destinations; new runner requires verified epoch and best-model copies. **Actual backup to approved corporate storage not tested here.** |
| HIGH open | Original latest full MySQL CI executed 540 Django tests, 5 failures, 9 skipped. Failures were `RagIsolationTests.test_tenant_isolation` and four `BgeRedTeamScenariosTests`, all missing `references`. `rag.py:82-88` intentionally suppresses citations absent verified claim support. | Root cause of the five failures is **not independently determined**: the failing assertions may conflate retrieved documents with claim-approved evidence. Do not bypass evidence checks to force green tests. Inspect `retrieved_candidates` vs `claim_supporting_passages` in isolated MySQL regression tests first. |
| HIGH fixed in source (GPU execution unverified) | `core/ml/training.py:276-293` previously saved final-epoch weights without explicit best-validation reload. Op27 already used `load_best_model_at_end`. | General embedding trainer now enables best-dev-loss restoration, fails absent/nonfinite best checkpoint, and blocks real-data training without a dev set. Six synthetic guard tests were added in `test_servy_best_checkpoint.py`. Runtime with real model is **not verified**. |
| HIGH fixed in source (runtime unverified) | Research script accepted a model/tokenizer config that could differ from the tokenizer actually used by `CrossEncoder`, creating a false reproducibility claim. | V4 runner now requires equal declared model/tokenizer ID and revision before loading GPU weights; synthetic regressions added. |
| HIGH open | Research Qwen3 tenant-filtered dense+BGE reranker differs from production Django-authorized hybrid+MiniLM path. | Production promotion blocked pending identical scoped candidate sets, fallback, score sign, document/asset ACL and answer-level tests. |
| HIGH unverified | The 27 exact Stage 2 triplets, 32 weak dev labels, guide family lineage, and missing original V2/V3 checkpoints have not been independently inspected. | Cannot attribute V3 regression to Stage 2 or claim 95% answer accuracy. |

## Document quality / safety hypotheses, NOT measured failures

- `core/services/document_loader.py:58-79`: scanned PDFs with no
  extractable text lead to OCR-required operator status; actual company-guide
  incidence unknown. No new OCR or external upload introduced.
- `document_loader.py:84-115`: DOCX paragraphs and tables are extracted
  in separate passes; original table position relative to paragraphs may be
  lost. Requires a synthetic multi-section DOCX fixture and approved review
  before ingestion change.
- `core/services/chunking.py:26-80`: 1100-character chunks/140 overlap may
  split a procedure, error code, or engineering unit. Actual answer corruption
  is **not verified**. Avoid broad retokenization without an index migration
  and provenance strategy.
- `core/services/chroma_store.py:259`: versioned upsert retains stale
  vector IDs from prior versions unless filtered/reconciled. The dense
  retriever's per-request authority checks must be evaluated with a
  simulated stale-index fixture before asserting an exploit; no destructive
  vector cleanup has been performed.
- `core/services/llm.py`: retrieved manuals are untrusted data. Prompt
  injection, unsafe commands, numerical grounding, confidentiality, and
  unsourced technical advice require production-equivalent red-team traces.
  Passing synthetic safety tests does not prove immunity.

## Research training policy

1. No V4 run until label review, source-guide/version isolation and full
   data preflight pass. Model compatibility must be checked against the
   *installed* Sentence Transformers version on an approved environment.
2. Compare frozen Qwen3 candidates (query/document encoders separate,
   normalized, config hashed) using the pretrained BGE reranker and only
   actually recovered V2/V3 checkpoints. Do not substitute lost weights.
3. Once approved, a **small exploratory** matrix is baseline vs binary
   verified positive/negative BCE; defaults: `BAAI/bge-reranker-v2-m3`,
   learning rate `7e-7`, batch size `4`, at most `2` epochs, seeds `42`,
   max length `256` subject to actual tokenizer truncation audit.
   These are conservative **starting settings**, not optimized hyperparams.
4. Select by dev `eval_loss` and separately report fixed-candidate Hit@1/5/10
   and MRR; holdout must be independently reviewed and never used for
   checkpoint selection. Report per-equipment, document/chunk vs factual
   answer accuracy, refusal and safe action separately. Denominators,
   confidence intervals and provenance must accompany any performance claim.
5. Never call a 32-query repeatedly used development Hit@1 rate a
   95% production accuracy outcome. Fix candidate recall and answer evidence
   first; a reranker cannot recover a gold chunk not in its candidate set.

## CI and delivery limits

The dedicated workflow runs **offline synthetic regressions** without Docker,
a GPU or company data. The full application CI is a separate MySQL test suite,
and earlier SHA `dd226411...` failed 5/540. Verify authoritative final-head
workflow status before reviewing PR #2; a cancelled intermediate CI run is not
a pass. All changes stay on the draft audit PR. `main`, production models,
company databases and external documents remain untouched.

See `tools/SERVY_V4_COLAB_RUNBOOK.md` for exact commands and backup gates.
