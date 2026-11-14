# MiniLM retirement, SQLite migration and Qwen validation — final report

Date: 2026-11-14. Local development only. Nothing committed, pushed or deployed.
Phase 1 inventory: `docs/SERVY_MINILM_SQLITE_INVENTORY.md`.

## Final model architecture

| Model | Responsibility | Evidence |
|---|---|---|
| BGE-small `BAAI/bge-small-en-v1.5` @ `5c38ec7c…` | All document and query embeddings | Ingestion writes stamped BGE vectors only (`core/tests_bge_ingestion.py`, real model) |
| V5 (`servy_v5_bge_verified`, XLM-RoBERTa) | Reranking | 67/67 evaluation turns reranked by V5; real-V5 request test passes |
| Qwen `qwen3.5:2b-q4_K_M` (local Ollama) | Customer wording, source-validated | 42/67 evaluation turns use validated Qwen wording |

## Acceptance results

**Three-asset evaluation** (real HTTP, servy_dev gateway, BGE + isolated Chroma + V5 + Qwen;
`tools/servy_three_asset_gateway_eval.py`): **67/67 turns**.

| Asset | Documented answers | Unsupported refusals | Safety turns |
|---|---:|---:|---:|
| 59 | 17/17 | 5/5 | – |
| 60 | 12/12 | 6/6 | 4/4 |
| 61 | 17/17 | 4/4 | 2/2 |

Cross-asset leaks 0 · claims not verbatim in the cited original 0 · citation open failures 0 ·
restricted content 0.

Progression on the official gateway path: 61/67 → 64/67 (ordinal follow-up resolution) → 67/67
(fact and value questions).

**Qwen source validation**, measured on the same captured Qwen outputs for every documented-answer
question (97 claims):

| Asset | Accepted before | Accepted after |
|---|---:|---:|
| 59 | 7/45 | 38/45 |
| 60 | 1/18 | 14/18 |
| 61 | 4/34 | 32/34 |
| Total | 12/97 (12%) | 84/97 (87%) |

At the turn level, Qwen wording was used in 42/67 evaluation turns, against 12/67 before. Unverifiable
wording falls back to the exact cited source text.

- **Root cause:** in 54 of the 85 rejected claims, Qwen correctly restated a listed cause but without
  the word "possible". The rest came from strict in-order token matching (reordering, inflection,
  "6"/"six") and from rejecting the manual's own imperatives.
- **Asset 60 was not better.** The original observation (60 accepted, 59 and 61 rejected) was a
  per-question artefact; at the claim level asset 60 had the lowest acceptance (1/18).
- **The 13 remaining rejections are correct.** For example, Qwen tied one cause to two symptoms, added
  "of this unit" or "installation work", or changed "over-temperature" to "overheating".

**Live workflow** (both UIs' proxies): 26/26. Persistence after a backend restart: 5/5. After a full
cold restart of MySQL and all services, the technician inbox showed the same ServiceCalls.

**Tests:**

| Suite | Result |
|---|---|
| Backend, MySQL `test_servy_dev` | 878 OK, 3 skipped |
| Backend, SQLite in-memory | 878 OK, 3 skipped |
| CI tool tests | 319 OK, 1 skipped |
| Frontend | 59/59; lint 0; build OK |

**Skips** (each verified genuine):
- the test_servy_dev-only and MySQL row-locking tests run in the MySQL run;
- the SQLite-only checks run in the SQLite run;
- real V5 is opt-in, and it passes when enabled;
- symlink creation fails because Windows refuses it with error 1314.

The former "Celeron" skip was a hidden fixture defect (`index_status="completed"` is not a real
status). It was fixed, and the test now runs and passes.

**Production:**
- Production `chroma_db` is byte-identical to its state after the two approved test-vector removals
  (0 mismatches).
- Production MySQL `servy`: no new writes (latest Oct 2026), and migration 0008 is absent.
- The repository's `media/` is unchanged.

## Changes in this phase

### MiniLM
- **Ingestion** (`chroma_store.sync_document_to_chroma`) writes the approved BGE profile only. The
  MiniLM loader, its download fallback and the MiniLM collection helpers were removed.
- **Profiles and settings:**
  - the `minilm` and `company-minilm` profiles were removed from `core/ml/embeddings.py`;
  - the `SERVY_EMBEDDING_MODEL` default was removed;
  - `SERVY_SERVING_EMBEDDING_PROFILE=*minilm*` now refuses to start;
  - the retriever's auto, unpromoted and Qwen3-not-ready fallbacks all resolve to BGE.
- **Reranker:** the `ms-marco-MiniLM` network fallback was removed, so V5 requires its local
  checkpoint.
- **Evaluation harness:** the MiniLM track was removed from `core/ml/evaluate_benchmark.py`. Its
  historical results remain in the JSON reports.
- **Removed files:**
  - `tools/verify_chroma_and_rag.py`, an obsolete MiniLM verifier that loaded production credentials;
  - 2.46 GB of model files: `op26_minilm_ft`, `op27_minilm_ft` and three Hugging Face MiniLM caches.
  - The manifest with every path, size and SHA-256 is at
    `%LOCALAPPDATA%\Servy\recovery\minilm_deletion_manifest.json`.

### SQLite
- **Normal application:** the RAG Ticket Workspace tab, page, client, API routes and
  `core/local_ticket_store.py` / `core/local_ticket_workspace.py` were removed. Old
  `/technician/rag-tickets` links now open the official MySQL Technician Inbox.
- **Legacy demo:** `LOCAL_SQLITE` mode and its SQLite ticket code were removed from
  `tools/servy_real_demo.py` and `tools/servy_real_demo_assignment.py`.
  `tools/servy_technician_demo.py`, its test and `tools/rollback_to_sqlite.py` were deleted, as were
  the CI step and the `SERVY_LOCAL_TICKET_DB_PATH` setting.
- **SQLite engine:** allowed only for `manage.py test`, using an in-memory database. `runserver` or
  `migrate` on SQLite is refused, and `db.sqlite3` is never opened as a runtime database.
- **`transfer_sqlite_to_mysql`** now opens its source read-only.
- **Ticket workspace file:** archived with SHA-256 checksums (2 old demo tickets, not migrated) to
  `%LOCALAPPDATA%\Servy\recovery\`, then removed.

### Qwen and the answer pipeline
- **Validator** (`customer_answer_generation.py`):
  - the server adds a fixed "Possible cause:" hedge;
  - word order is enforced only for sequences;
  - inflections are folded;
  - a number word is accepted only for the same number in the source;
  - an imperative is allowed only when it repeats the source's own imperative;
  - the customer is never addressed directly.
- **Unchanged:** the certainty, number, negation and action checks, length limits, the exact quote
  check and the fallback.
- **Ordinal follow-ups:** "Explain the second cause" now resolves against the previous answer's listed
  causes (`conversation_context.resolve_cause_reference`), and the other causes are removed from that
  answer.
- **Fact and value questions:** timing and other fact questions are answered with the documented facts
  instead of a word-matched cause. Value or specification questions are never answered with a cause; a
  documented number is required, otherwise the answer is a refusal.
- **Streaming bug fixed:** the SSE result frame failed on datetimes. It now uses `DjangoJSONEncoder`.

### Protected test files
`tools/test_servy_real_demo.py` and `tools/test_servy_real_demo_exhaustive.py` were ported to the
MySQL gateway contract, as you approved. Your earlier uncommitted edits are preserved as `.patch` files
in `%LOCALAPPDATA%\Servy\recovery\`. `test_servy_real_demo_exhaustive.py` needed no change. Both
stashes are intact.

## Remaining references (corrected, classified per file)

Re-audited after the seven production MiniLM collections and their seven orphaned
index folders were removed. Test files are counted with the real pattern `core/tests_*.py`.

Classes:
- **active-guard**: running code that *refuses* MiniLM or SQLite (keep).
- **protected**: running code that must stay as it is, such as a read-only migration reader or test-only config.
- **test-only**: tests asserting the retirement holds.
- **historical**: non-runtime research code or evidence (keep).

### Verified results

**Zero active MiniLM dependencies:**
- `get_profile("minilm")` and `get_profile("company-minilm")` raise.
- No `PROFILES` entry uses a MiniLM model.
- The legacy loader and the reranker's MiniLM fallback no longer exist.
- `SERVY_SERVING_EMBEDDING_PROFILE=minilm` stops startup.
- Nothing calls `load_model(..., allow_download=True)`.
- No MiniLM model ID appears in active code, launchers or `.env`.
- The production collections and the model files are gone.

**Zero persistent application-owned SQLite writes:**
- The SQLite engine is allowed only for `manage.py test`, using an in-memory database.
- `runserver` and `migrate` on SQLite are refused.
- No `LOCAL_SQLITE` or workspace code remains.
- Every SQLite connection in running application code is read-only.

**No production fallback:**
- Gateways refuse any database other than `servy_dev`.
- An env file naming `servy` or `test_servy` is refused.
- `servy_dev_app` and `servy_dev_test` have privileges only on their own database, and production `servy` returns error 1142 for both.
- Tests use a temporary Chroma directory by default and refuse the production `chroma_db`.
- Both launchers pin the isolated `servy_dev_chroma`; the technician launcher was corrected in this phase.

### MiniLM

| File | Refs | Class | Purpose |
|---|---:|---|---|
| `servy_rag/settings.py` | 3 | active-guard | Refuses a MiniLM serving profile; retirement notice |
| `core/ml/embeddings.py` | 2 | active-guard | Docstring: MiniLM retired, no profile |
| `core/ml/indexer.py` | 4 | active-guard | Refuses to write the old `servy_dense_kb` |
| `core/ml/evaluate_benchmark.py` | 1 | historical | Note that the MiniLM track was removed |
| `core/ml/op23_private_retrieval.py` | 2 | historical | Research baseline; takes its encoder as an argument and loads no model; used only by `tools/op23_evaluate_reviewed.py` |
| `core/management/commands/seed_servy_dev.py` | 2 | active-guard | Comments: "MiniLM is never loaded" |
| `core/tests_embeddings.py` | 6 | test-only | Asserts MiniLM profiles are refused and absent |
| `core/tests_serving_truth.py` | 3 | test-only | Asserts startup refuses a MiniLM profile |
| `core/tests_bge_ingestion.py` | 3 | test-only | Asserts ingestion creates no MiniLM collection |
| `core/tests_breakpoint4_adversarial.py` | 2 | test-only | Asserts readiness has no `minilm_vectors_ready` |
| `core/tests_operation_18_readiness_gate.py` | 2 | test-only | Generic stamped-collection guard test (comment) |
| `core/tests_breakthrough_redteam.py` | 1 | test-only | Asserts the approved model is BGE |
| `core/tests_serving_configuration.py` | 1 | test-only | Asserts no legacy loader exists |
| `core/tests_op23_private_retrieval.py` | 5 | test-only | Tests the research baseline with a fake encoder |
| `docs/ARCHITECTURE.md`, `docs/V5_LOCAL_RERANKER_SETUP.md` | 1 each | documentation | **Corrected** in this phase; they previously described MiniLM as active |
| `docs/SERVY_MINILM_SQLITE_*.md` | many | documentation | These reports |

**Historical, kept and non-runtime:**
- `tools/` (15 files): `op21/op21_accuracy_audit.py`, `op21/op21_latency_benchmark.py`, `op21/op21_rerank_audit.py`, `op22_company_diagnostic.py`, `op22_latency_benchmark.py`, `op22_rank_experiment.py`, `op23_evaluate_reviewed.py`, `op25/op25_diagnose_candidates.py`, `op25/op25_reconcile_and_diagnose.py`, `op26_build_training_dataset.py`, `op26_dual_model_training.py`, `op27_phase0_audit.py`, `op27_phase0_evaluate.py`, `op27_phase0_eval_hybrid.py`, `op27_train_and_evaluate.py`. Their MiniLM paths can no longer run, because the checkpoints and caches are removed.
- `scratch/` (18 files).
- `docs/archive/` (30 files).
- Evidence JSON: `ml_models/` 3, `ml_artifacts/` 4, `outputs/` 3, `private_training_data/` 21.
- `.github/`: 0.

### SQLite and `LOCAL_SQLITE`

| File | Class | Purpose |
|---|---|---|
| `servy_rag/settings.py` | protected | SQLite engine for `manage.py test` only, `NAME=":memory:"`; refuses any other command |
| `core/test_database_guard.py` | protected | Refuses tests against a non-test database file such as `db.sqlite3` |
| `core/management/commands/transfer_sqlite_to_mysql.py` | protected | Migration reader for the legacy `db.sqlite3`; **read-only** (`mode=ro`) |
| `core/tests_config_isolation.py` (22 refs), `tests_test_database_guard.py` (6), `tests_serving_configuration.py`, `tests_serving_truth.py`, `tests_operation_16_*`, `tests_operation_18_*`, `tests_conversation_ticket_workflow.py` (1 each) | test-only | In-memory test database, refusal checks and SQLite-vs-MySQL skips |
| `tools/test_servy_real_demo.py` (protected file, ported) | test-only | The only `LOCAL_SQLITE` mention: asserts the mode and the store are gone |
| `tools/migrate_sqlite_to_mysql.py`, `verify_all_records.py`, `inspect_framework_records.py`, `inspect_source_sqlite.py`, `benchmark_performance.py`, `inspect_chroma.py` | protected tooling | Open SQLite read-only (`mode=ro`) |
| `tools/backup_all.py` | protected tooling | Reads `db.sqlite3` read-only; writes only its own backup copy |
| `tools/servy_v4_ai_evidence_auditor.py` (+ test) | historical | Research auditor. Issues `SELECT` only and never commits, but opens without `mode=ro`. **Optional hardening, not done:** add `mode=ro` |
| `tools/op21–op25*`, `op22*`, `op23*`, `print_inventory.py`, `topological_sorter.py`, `check_content_type_refs.py`, `security/red_team_static.py`, `run_customer_answer_checks.py`, `run_local_tests.ps1` (in-memory test runner), `servy_real_demo_index_review.py` (Chroma file names), test fixtures | historical / test-only | Research, inspection and the local test runner |
| `scratch/` (30 files) | historical | Research scratch |
| `docs/archive/` (8 files), `docs/V5_LOCAL_RERANKER_SETUP.md` | historical | Documentation |
| `chroma_db/chroma.sqlite3` | protected | Chroma internal storage (permitted) |

**Preserved database files** (not runtime, untouched):
- `prototype/db.sqlite3` (SHA-256 `ac3d4988…cf25251`, last modified 2026-10-19): 7,706 legacy ServiceCalls against 182 in production MySQL. Its provenance needs a separate data-reconciliation investigation and explicit approval; it is **not** a cleanup success.
- `backups_servy_migration/…/db.sqlite3` and `outputs/parity/db_copy.sqlite3`: copies.
- `private_training_data/local_disposable_demo/demo.sqlite3` and `v5_live.sqlite3`: research fixtures.

### Production Chroma

- **10 collections, 1,560 vectors, all BGE.**
- **7 MiniLM collections deleted** (1,327 vectors). The recovery export is bit-exact and restore-verified; it is in `%LOCALAPPDATA%\Servy\recovery\minilm_collections_20261114\`.
- **7 newly orphaned index folders deleted.** They were empty placeholders; a copy and a 28-file SHA-256 manifest are in `orphan_dirs_new7\`.
- **14 pre-existing orphan folders kept untouched.** One of them, `5f05d72e…`, holds 451 old MiniLM vectors (proven by exact vector match), with no text.
- **Current checksum baseline:** `chroma_baseline_after_orphan_cleanup.sha256` (94 files).
