# Final red-team report — Servy RAG hardened prototype

## Scope

This review targets the local Django prototype, its RAG layer, multi-tenant data model, customer/staff authorization boundaries, Knowledge Base ingestion, Call Register import/export, local LLM boundary, dummy database, startup scripts and offline UI assets.

## Fixes completed

- Django authentication and role-based route guards.
- Explicit tenant memberships; tenant switching is limited to memberships owned by the logged-in user.
- Customer identity is server-derived from the authenticated customer membership rather than accepted from POST data.
- Tenant/customer/product/asset constraints on RAG retrieval.
- Confidential Knowledge Base exclusion for customer mode and staff permission checks for protected downloads.
- Cross-tenant/cross-customer foreign-key validation plus Project↔Asset many-to-many scope validation.
- Product-level manuals can support multiple assets; asset-specific guides can rank above shared product docs.
- Closed Call Register resolutions are indexed as reusable RAG memory after PII redaction.
- Customer mode restricts past-resolution memory to the same customer; staff mode can use sanitized same-product cases inside the same tenant.
- Prompt-injection document quarantine plus control-abuse query refusal.
- Model-produced citation labels are removed; references shown in the UI come from server-verified retrieval metadata.
- Knowledge uploads have size, extension, basic signature and archive-expansion limits.
- XLSX import has file-size, ZIP-expansion, internal-file, sheet, column and row limits.
- XLSX export neutralises spreadsheet-formula injection prefixes.
- Local LLM endpoint defaults to loopback-only, ignores proxy environment variables and blocks HTTP redirects.
- AI requests have per-user/per-tenant rate limiting.
- Audit events are recorded for high-value actions.
- Direct public `/media/` serving is not enabled; uploaded KB files use an authenticated tenant-scoped download endpoint.
- UI runtime assets are local; there is no CDN dependency.
- Docker is not required.
- Startup scripts no longer rebuild/reset demo data during normal server startup.
- `seed_demo` is idempotent in normal mode; destructive rebuild requires explicit `--reset`.
- Standalone review SQLite DB was rebuilt as a 36-table relational, tenant-aware schema with foreign keys and integrity checks.

## Verification completed in this environment

- Python source compilation: **PASS**.
- Dependency-free static red-team suite: **54 passed / 0 failed**.
- Offline behavioural RAG/security suite: **23 passed / 0 failed**.
- Standalone SQLite `PRAGMA foreign_key_check`: **PASS**.
- Local UI Bootstrap CSS/JS presence: **PASS**.
- ZIP packaging/integrity: verify after packaging.

## Dynamic Django tests included

The project contains **22 Django tests** (`core/tests.py` + `core/tests_security.py`) covering tenant isolation, customer IDOR attempts, cross-tenant object access, confidential KB retrieval, prompt injection quarantine, product-level manuals, previous-resolution PII redaction, same-customer resolution scope, staff sanitized cross-customer memory, protected KB downloads, rate limiting, Excel formula injection and remote-LLM rejection.

These dynamic tests could not be executed in this build environment because Django itself is not installed here and outbound package installation is unavailable. They are intended to be run immediately after `setup_windows.bat` installs dependencies on the user's IDE/machine.

## Commands to run locally

```bat
setup_windows.bat
.venv\Scripts\activate
python manage.py check
python manage.py test core -v 2
python red_team_static.py
python red_team_offline.py
python manage.py runserver
```

## Remaining limitations

This is a hardened proof-of-concept, not a production security certification. Before real customer/Servy data is used, run the full Django test suite and browser/API penetration tests in the target environment, add SSO/MFA as appropriate, production secret management, malware scanning/CDR for uploaded documents, centralised audit logs, shared rate limiting, backups, monitoring and a production relational database. The supplied schema is based on the screenshots/workflow provided; exact Starlly production integration still requires repository/API/database-schema access.
