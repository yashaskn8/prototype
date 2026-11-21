# Servy RAG — production readiness, operations and recovery

Status: **prototype; production deployment not authorized.** This document records how the
single-user local prototype is operated and recovered today, and what must exist before any
real deployment. Nothing here has been deployed.

## 1. Current serving profile (single user, local)

| Item | Setting / behaviour | Where |
|---|---|---|
| Request admission | One active RAG request per process; others get HTTP 503 `rag_busy` immediately (never queued) | `core/services/rag_admission.py` |
| Request deadline | 5 s monotonic end-to-end; stage caps dense 1.0 s, V5 2.6 s, Qwen 3.5 s; 0.4 s reserve | `core/services/request_deadline.py`, `SERVY_RAG_DEADLINE_SECONDS` |
| Fallbacks | Dense timeout → authorized sparse only; V5 timeout → authorized pre-rerank order (`reranker_applied=false`); Qwen timeout → exact cited source wording | same + `contract.outcome` |
| Models | BGE-small (CPU), V5 cross-encoder SHA-pinned (CPU; ONNX Runtime fp32 export or PyTorch), Qwen `qwen3.5:2b-q4_K_M` via local Ollama (GPU, keep-alive 30 min) | `core/ml/*`, launcher |
| Warm-up | Models preloaded at server start; launcher waits for `models_ready` | `core/ml/preload.py`, `/demo-api/health/` |
| V5 runtime rollback | `SERVY_RERANKER_BACKEND=torch` (or remove the ONNX directory); unverified exports fall back to PyTorch automatically | `core/ml/reranker.py` |
| Metrics | One JSON line per request (no customer text), 3 × 2 MB rotation; `manage.py rag_metrics_report` | `core/services/rag_metrics.py` |

## 2. Backup and restore (MySQL era)

`tools/backup_all.py` is the retired SQLite-era script and does **not** back up MySQL. Use
`tools/servy_mysql_backup.py`:

```powershell
# Backup (writes outside the repository; no .env or secrets are copied)
.\.venv\Scripts\python.exe tools\servy_mysql_backup.py backup --env-file .env.mysql.servy_dev `
  --chroma '<dev chroma dir>' --media '<dev media dir>' --out '<backup root outside repo>'

# Verify every file hash in a backup
.\.venv\Scripts\python.exe tools\servy_mysql_backup.py verify --backup '<backup dir>'

# Restore into an isolated database only (refuses servy/test_servy and the source database;
# the target name must be typed again)
.\.venv\Scripts\python.exe tools\servy_mysql_backup.py restore --backup '<backup dir>' `
  --env-file .env.mysql.servy_dev_test --confirm-target test_servy_dev `
  --chroma-to '<empty dir>' --media-to '<empty dir>'
```

A backup contains a `--single-transaction` dump (one INSERT per row, so row counts in the
manifest describe exactly the restored snapshot), the vector index, source documents and
attachments, model identities and the git commit, with SHA-256 for every file.

### Recovery procedures

| Failure | Procedure |
|---|---|
| Database corruption / bad migration | Stop the launcher. Restore the latest verified backup into a new empty database, point `SERVY_MYSQL_ENV_FILE` at it, run `manage.py check_dev_database`, start, run the gateway evaluation. Only then retire the damaged database. |
| Missing or damaged vector index | Restore `chroma/` from the backup into a new directory and point `SERVY_CHROMA_PERSIST_DIR` at it; or re-index from the approved documents. Dense hits are verified per chunk against current authorized chunks, so stale vectors are never served. |
| Failed document re-index | `index_document` swaps chunks atomically; a failure leaves the previous chunks. Re-run indexing after fixing the source. |
| V5 ONNX export damaged | Serving detects the hash mismatch and uses the PyTorch checkpoint. Re-run `manage.py export_v5_onnx --out <dir>`. |
| Startup does not reach READY | Check `/demo-api/health/` (`models`, `v5_runtime`) and the private `startup.err`; Ollama not running is started by the launcher; a model failing to load leaves requests on their bounded fallbacks. |

### Clearing the demo chat history (one-time, manual)

The customer chat list is cleared with the official soft-delete endpoint
(`DELETE /demo-api/customer-support/conversations/{uuid}/`), called for every listed
conversation while authenticated as the demo customer. Soft delete sets
`customer_deleted_at`/`customer_deleted_by`: the conversation disappears from the
customer's list and cannot be reopened by them, while its immutable messages, feedback
and any linked service call stay in the database for technicians and audit. Take a
verified backup first. This is never run automatically; permanent deletion of rows is
not part of this procedure and would need separate approval.

## 3. Before any production deployment (P2 — not implemented)

- **Server**: a production WSGI server (e.g. waitress on Windows or gunicorn on Linux) with
  exactly one worker process (models are ~2.2 GB resident per process), behind HTTPS.
- **Configuration**: `DJANGO_DEBUG=0` (enables secure cookies and HSTS in settings), a unique
  `SECRET_KEY`, secrets from the host secret store, never committed.
- **Pinning**: release record with commit, V5 weights SHA-256, ONNX manifest, BGE revision,
  Qwen tag, Chroma collection count and corpus document checksums.
- **Health**: `/demo-api/health/` readiness (`models_ready`) for the load balancer; restart on failure.
- **Migrations**: backup → `migrate` on a restored copy → verify → migrate production.
- **Rollback**: previous commit + previous backup; V5 `SERVY_RERANKER_BACKEND=torch`.
- **Monitoring**: `rag_metrics_report` p95 and fallback rates; alert thresholds to be agreed.
- **Privacy/retention**: conversation retention period and deletion policy to be agreed.
- **Capacity**: one request at a time per process; more users need a separate inference host
  (the RAG entry point is `process_customer_diagnostic_turn`; model access is isolated in
  `core/ml`), not more Django workers on this laptop.

Deployment remains blocked by release contract R13–R15.
