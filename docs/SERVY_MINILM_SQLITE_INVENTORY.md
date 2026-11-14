# MiniLM and application SQLite inventory (Phase 1, read-only)

Date: 2026-11-14. Nothing was modified to produce this report. Chroma figures come
from throwaway copies of `prototype/chroma_db` (production) and
`%LOCALAPPDATA%\Servy\servy_dev_chroma` (development). SQLite files were opened
`mode=ro&immutable=1`.

## 1. Model roles actually in use

| Role | Model | Evidence |
|---|---|---|
| Query and document embeddings (serving) | BGE-small `BAAI/bge-small-en-v1.5` @ `5c38ec7c…` | `SERVY_SERVING_EMBEDDING_PROFILE` defaults to `bge-small` |
| Reranking | V5 (`best_model`, XLM-RoBERTa cross-encoder, 24 layers, SHA-pinned) | `best_model/config.json`; **V5 is not MiniLM** |
| Generation | Qwen `qwen3.5:2b-q4_K_M` via Ollama | gateway launcher |

## 2. MiniLM findings, classified

### Active application or ingestion dependencies (must migrate)

| Location | What it does | Class |
|---|---|---|
| `core/services/chroma_store.py` `get_sentence_transformer()` | Loads `all-MiniLM-L6-v2`, **falls back to a network download** | Active ingestion |
| `chroma_store.sync_document_to_chroma()` | Every uploaded/re-indexed document is embedded with **MiniLM only** into `servy_dense_kb` / `servy_company_dense_kb`; BGE indexing runs only if `SERVY_USE_EXPERIMENTAL_RAG=1` | Active ingestion |
| `chroma_store.delete_document_vectors()` | Deletes from the MiniLM collection | Active ingestion |
| `chroma_store.get_document_retrieval_readiness()` | Reports `minilm_vectors_ready` | Active application |
| `core/services/retriever.py` `_dense_score_candidates` | Falls back to `minilm` / `company-minilm` when the profile is `auto`, unpromoted, or Qwen3 is not ready | Active fallback |
| `retriever.py` `_qwen3_profile_if_ready`, `_matches_current_minilm_vector`, `model_version="minilm"` default | MiniLM fallback and labels | Active fallback |
| `core/ml/embeddings.py` profiles `minilm`, `company-minilm` and the MiniLM name rewrite in `load_model` | MiniLM profiles | Active configuration |
| `servy_rag/settings.py` `SERVY_EMBEDDING_MODEL="minilm"` default | MiniLM default | Active configuration |
| `core/ml/reranker.py` `RERANKER_MODEL_ID="cross-encoder/ms-marco-MiniLM-L6-v2"` | Legacy fallback: if no V5 path and no SHA pin are set, **downloads** a MiniLM cross-encoder | Active fallback |
| `core/apps.py` comment | Mentions the on-demand MiniLM path | Legacy comment |
| `core/services/rag.py` `model_version` default `minilm` | Label | Active label |

### Legacy, research and test-only references

| Location | Class |
|---|---|
| `core/ml/evaluate_benchmark.py`, `core/ml/op23_private_retrieval.py`, `core/ml/technical_retrieval.py` (docstring) | Research/experimental code inside `core/ml` |
| `tools/op21*`–`op27*` (16 scripts), `tools/verify_chroma_and_rag.py` | Research/experimental tools |
| `scratch/` (19 scripts) | Legacy/research scratch |
| 13 test modules (`tests_embeddings`, `tests_operation_14_truth_recovery` with 13 refs, `tests_v5_reranker`, `tests_serving_truth`, …) | Test-only |
| `docs/` (~22 files), `ml_models/*.json`, `ml_artifacts/*.json`, `outputs/*.json`, `private_training_data/*` | Historical evidence (non-runtime, keep) |

### Chroma collections, production `prototype/chroma_db` (protected)

All are 384-dimensional.

| Collection | Vectors | Model | Tenants | Class |
|---|---:|---|---|---|
| `servy_dense_kb` | 229 | MiniLM (all chunks stamped `all-MiniLM-L6-v2`) | 9:69, 6:131, others | Protected company data, MiniLM |
| `servy_company_dense_kb` | 229 | MiniLM | same as above | Protected company data, MiniLM |
| `servy_dense_kb_v1_backup` | 179 | 138 unstamped + 41 MiniLM | includes old test tenants 571–589 | Protected backup, MiniLM |
| `servy_company_dense_kb_v1_backup` | 69 | unstamped (MiniLM era) | 9:69 | Protected backup, MiniLM |
| `servy_op26_minilm_base_v1` | 207 | unstamped | 9:69, 6:131, … | Research, MiniLM |
| `servy_op26_minilm_ft_v1` | 207 | `op26_minilm_ft` checkpoint | same | Research, MiniLM |
| `servy_op27_minilm_ft_v1` | 207 | `op27_minilm_ft` checkpoint | same | Research, MiniLM |
| `servy_dense_bge_small_v1` | 227 | BGE (stamped) | BGE twin of `servy_dense_kb` (−2 approved test vectors, −3 tenant-6 chunks from the earlier approved cleanup) | **Existing BGE replacement** |
| `servy_company_bge_small_v1` | 229 | BGE (stamped) | BGE twin of `servy_company_dense_kb` | **Existing BGE replacement** |
| `servy_company_bge_small_v1_v1_backup`, `servy_dense_bge_small_v1_v1_backup` | 69 / 138 | BGE | backups | Protected backup |
| `servy_company_bge_ft_v1`, `servy_dense_op20_bge_ft_v1`, `servy_op26_bge_*`, `servy_op27_bge_ft_v1` | 69–207 | BGE fine-tune experiments | | Research, BGE |
| `servy_sparse_fallback` | 0 | — | | Empty |

Development `servy_dev_chroma` has `servy_dense_bge_small_v1` with 69 BGE vectors for tenant 1 (9 manuals, assets 1–3) and no MiniLM.

**Conclusion:** BGE replacements for both active MiniLM collections already exist in production. The migration is mainly code (ingestion, fallbacks, profiles), not building new production collections.

### Model files

| Path | Size | Class |
|---|---:|---|
| `ml_models/checkpoints/op26_minilm_ft` | 1.1 GB | Research artifact (git-ignored) |
| `ml_models/checkpoints/op27_minilm_ft` | 867 MB | Research artifact (git-ignored) |
| `~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2` | 175 MB | Cache used by active ingestion today |
| `~/.cache/huggingface/hub/models--cross-encoder--ms-marco-MiniLM-L-6-v2` | 88 MB | Cache for the legacy reranker fallback |
| `~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-MiniLM-L6-v2` | 88 MB | Unused cache |

## 3. SQLite findings

### Database files

| File | Contents | Class |
|---|---|---|
| `%LOCALAPPDATA%\Servy\local-ticket-workspace.sqlite3` | 2 tickets (`LOCAL-20261113-74065EF7`, `-3BF5DC8F`), created 13 Nov by the old demo, **tenant 9 Skydream, real customer name, asset 59** | Demo records referencing real company identifiers |
| `prototype/db.sqlite3` (Oct 19) | 55 tables; **7,706 ServiceCalls**, 166 documents, 67 assets, tenants incl. `skydream`. Production MySQL has 182 ServiceCalls | **Possibly unique genuine company data — preserve** |
| `backups_servy_migration/backup_20261017_134931/db.sqlite3` | Same counts as `db.sqlite3` | Backup — preserve |
| `outputs/parity/db_copy.sqlite3` | Same counts as `db.sqlite3` | Parity copy — preserve |
| `private_training_data/local_disposable_demo/demo.sqlite3` | 7,665 calls, demo tenants only | Research/demo |
| `private_training_data/v5_live.sqlite3` | 1 tenant `v5-browser-test` | Test fixture |
| `chroma_db/chroma.sqlite3` | Chroma internals | Permitted exception |

### Code

| Location | Class |
|---|---|
| `core/local_ticket_store.py`, `core/local_ticket_workspace.py`, routes `api/local-ticket-workspace/…` in `core/api_urls.py` | Active application (normal app "RAG Ticket Workspace") |
| `frontend/src/pages/localTicketStore.js`, `TicketWorkspacePage.jsx`, nav entry `rag-ticket-workspace` | Active application frontend |
| `servy_rag/settings.py` `SERVY_LOCAL_TICKET_DB_PATH`; `SERVY_DB_ENGINE=sqlite3` branch | Active configuration (runtime engine is `mysql` via `.env`) |
| `tools/servy_real_demo.py` (1,937 lines, `LOCAL_SQLITE` mode), `tools/servy_technician_demo.py`, `servy_real_demo_assignment.py`, `servy_real_demo_index_review.py` | Legacy demo backends, replaced by the servy_dev gateways but still imported by tooling |
| `tools/test_servy_real_demo.py`, `tools/test_servy_real_demo_exhaustive.py` | **Protected** tests of the legacy backend |
| `tools/test_servy_technician_demo.py`, `tools/test_servy_real_demo_assignment.py` | Tests of the legacy backends |
| `tools/servy_three_asset_rag_audit.py` | Three-asset evaluation, imports `tools.servy_real_demo` and the production read-only identity |
| `core/test_database_guard.py`, `tools/migrate_sqlite_to_mysql.py`, `transfer_sqlite_to_mysql.py`, `rollback_to_sqlite.py`, `backup_all.py`, `inspect_*` | Migration/test infrastructure |
| `scratch/`, `tools/op2x*`, `private_training_data/*.py` | Research |
