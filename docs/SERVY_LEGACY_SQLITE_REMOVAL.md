# Legacy SQLite removal — evidence and record

Date: 2026-11-23. Local development. Servy's own legacy SQLite databases were investigated
read-only, backed up with verification, and removed. **ChromaDB still uses SQLite internally**
(`chroma.sqlite3` in each Chroma persist directory) for its own collection metadata; that file,
its collections, embeddings, documents, indexes and metadata were not touched, and Python's
`sqlite3` module (which ChromaDB needs) was not removed. Servy is therefore free of its own
SQLite database, not of SQLite as such.

## What SQLite was used for

| Use | Status before | Now |
|---|---|---|
| Runtime database | Already refused by settings | MySQL only |
| Automated tests | Optional in-memory SQLite (`SERVY_DB_ENGINE=sqlite3`, `manage.py test` only) | Removed; tests run on MySQL (`test_servy_dev` locally, the CI test database in CI) |
| Legacy `prototype/db.sqlite3` and two copies | Preserved pending a data-reconciliation decision | Verified as synthetic/test data, backed up, removed |
| Research files `local_disposable_demo/demo.sqlite3`, `v5_live.sqlite3` | Kept | Backed up, removed |
| Migration/inspection tooling for `db.sqlite3` | Read-only tools | Removed (recoverable from Git history) |
| ChromaDB internal storage | In use | **Unchanged** |

## Evidence that the removed records were synthetic or already in MySQL

All checks opened SQLite files read-only (`mode=ro`) and MySQL inside `START TRANSACTION READ ONLY`.

**ServiceCalls (7,706 in the legacy file, 182 in the company MySQL database).**

- All 7,706 belong to one tester tenant (`starlly-tester`); none belong to the real-company tenant.
- The company database's 182 calls are a strict subset of the legacy file: all 182 `servy_id`s,
  complaint texts, customers and asset codes are present, with identical `created_at`.
- The 7,524 calls that exist only in SQLite all come from one generator run:
  - every one of the 7,665 rows with row id 1–7,665 has the same minute:second;
  - they use exactly 720 distinct timestamps (30 days × 24 hourly batches of about ten);
  - complaint types and workflow statuses are near-uniform round-robin (about 770–800 and
    about 1,240 per value), including a type named "Generic Test 1";
  - each customer has a single contact name and phone number, and contact addresses use
    reserved example domains or two internal staff test addresses;
  - the same generator output (7,665 calls, the same 510 complaint texts, a single other
    minute:second) is reproduced in the separate demo database.
- The 41 calls not produced by the generator (row ids 7,666–7,709) are all already in MySQL.

**Every other table (55 tables).**

- Rows present in both stores are identical once JSON key order and timestamp formatting are
  normalized (customer interactions, diagnostic sessions/events/commands, recovery passports).
- No row of the real-company tenant exists only in SQLite.
- Rows only in SQLite belong to the tester, demo and test tenants: generated knowledge-document
  stubs (a fixed list of 34 titles cycled with sequence numbers, 61-character bodies, RAG
  disabled), test fixtures (a prompt-injection test, a random-name PDF test, a private-tenant
  test), earlier drafts of the repository's own `sample_kb/` demo manuals, 526 generated
  "Demo Spare" inventory items, and small sets of tester workflow records (projects, part
  requests, claims, settings).
- The older backup copy contained no row absent from the legacy file (one user row differed
  only by an older version).
- Nothing in the application, tests, CI or running processes read these files.

**Ambiguous record, preserved.** One tester-tenant knowledge document (id 23, a 272-character
start-of-shift procedure) has no source in the repository. It is very likely demo content, but its
origin could not be proven, so it was exported separately and kept with the backup.

## Backup

`%LOCALAPPDATA%\Servy\backups\legacy_sqlite_retired_20261123\` (outside the repository):

- the five removed SQLite files, each verified byte-for-byte (SHA-256) and with
  `PRAGMA integrity_check = ok` on the copy;
- `preserved_ambiguous_records.json` (document id 23);
- `tool_outputs/` (inventory and benchmark outputs of the removed migration tooling);
- `manifest.json` (original paths, sizes, SHA-256, row counts, verification results).

The earlier recovery archive `%LOCALAPPDATA%\Servy\recovery\local-ticket-workspace.sqlite3`
(two old demo tickets) was left as it is.

## Code, tests and CI

- `servy_rag/settings.py`: the `sqlite3` engine branch is gone; any engine other than MySQL raises
  `Unsupported database engine`, for every process including `manage.py test`, so Django can never
  create a SQLite database file.
- `core/test_database_guard.py`: only MySQL test databases (`test_*`) are accepted.
- Tests converted to MySQL: `tests_config_isolation`, `tests_test_database_guard`,
  `tests_serving_configuration`, `tests_serving_truth`, `tests_api_security` (MySQL lock-wait
  error), `tests_conversation_ticket_workflow`, `tests_operation_16_*`, `tests_operation_18_*`.
- Removed: `transfer_sqlite_to_mysql` command; `tools/` migration, inspection, backup, benchmark and
  in-memory runner scripts; the SQLite-registry research auditor, its test and its CI step.
- `tools/run_local_tests.ps1` now runs tests on the isolated MySQL test database.
- Research scripts `tools/op21`–`op25` no longer default to the SQLite engine.
- Kept on purpose: absence guards (tests asserting that no SQLite store or `db.sqlite3` exists),
  `db.sqlite3` in `.gitignore`, and ChromaDB-internal references (`chroma.sqlite3`).
- Historical reports under `docs/archive/` and dated result tables are unchanged records.
