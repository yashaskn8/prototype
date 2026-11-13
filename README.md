# Servy RAG

Servy is a source-grounded customer RAG application with a separate technician workflow. The supported local demonstration uses approved company manuals and asset-scoped data; it is not the earlier screenshot-driven dashboard with synthetic business records.

## Run the current customer and technician applications

Run both launchers from the repository root in separate Windows PowerShell windows.

**Customer RAG**

```powershell
.\tools\Start-ServyRealDemo.ps1
```

Open `http://127.0.0.1:5175/`. The launcher starts the customer demo backend on port 8002 and the Vite customer UI on port 5175. Use the existing private customer access code. Keep the terminal open to see the live request and RAG pipeline logs.

**Technician portal**

```powershell
.\tools\Start-ServyTechnicianDemo.ps1
```

Open `http://127.0.0.1:5176/`. The launcher starts the technician backend on port 8003 and the Vite technician UI on port 5176. Use the separate private technician access code printed by the launcher.

Do not put access codes, credentials, customer details, signed URLs or private log contents into GitHub.

## Current demo data and ticket handoff

The customer RAG demo reads approved company data in read-only mode. Its local demonstration handoff uses a dedicated SQLite workspace at:

```text
%LOCALAPPDATA%\Servy\local-ticket-workspace.sqlite3
```

The technician portal reads the same workspace. This is part of the **current customer-to-technician demo**; it is not the retired dummy review database. Do not delete or reset this workspace as a cleanup shortcut.

The local demo handoff is intentionally separate from an official Servy service call. It does not write company MySQL `ServiceCall` records.

## Normal authenticated application (MySQL)

```powershell
.\tools\Start-ServyApp.ps1
```

This starts the authenticated Django/DRF backend on port 8000 against the configured MySQL database (read-only readiness check, no data initialization) and the React application on port 5173 (`frontend/src/App.jsx`, without the demo `VITE_*` flags). Customers sign in and use Customer AI Support, which stores conversations and creates official service calls through the API below; technicians use `http://127.0.0.1:5173/technician/inbox`. Its other screens (calls, knowledge base, assets, projects, inventory, staff) read live application data through the DRF API. It is a separate application from the two demos above and is retained.

## Application-layer conversation-to-service-call API

The Django/DRF application also exposes the official database-backed workflow:

```http
POST /api/customer-support/conversations/{conversation_uuid}/ticket/
GET  /api/technician/inbox/{service_call_id}/
```

The first endpoint requires an authenticated customer and explicit `confirm_ticket_creation: true`; it creates or returns the Servy service call linked to the stored conversation. The technician detail endpoint returns the authorized call and conversation context.

Route registration is in `core/api_urls.py`. Implementations are in `core/api_views.py` and `core/technician_inbox.py`. These routes are distinct from the local demo proxies `/demo-api/*` and `/tech-api/*`.

## RAG implementation map

| Area | Main files |
|---|---|
| Customer question, conversation and ticket API | `core/api_views.py`, `core/api_urls.py` |
| Technician service-call inbox API | `core/technician_inbox.py` |
| Retrieval and asset/customer scope | `core/services/retriever.py` |
| Embeddings and dense indexing | `core/ml/embeddings.py`, `core/ml/indexer.py` |
| V5 reranking and checkpoint verification | `core/ml/reranker.py` |
| Answer generation and citation validation | `core/services/customer_answer.py`, `core/services/llm.py` |
| Django configuration | `servy_rag/settings.py` |
| Customer and technician React views | `frontend/src/pages/RealDemoPage.jsx`, `frontend/src/pages/TechnicianSecretPage.jsx` |
| Current launchers | `tools/Start-ServyRealDemo.ps1`, `tools/Start-ServyTechnicianDemo.ps1` |

The three-asset evaluation scope is Split AC Unit 2T (asset 59), VRF Outdoor Module (asset 60), and Ductable AC 5T (asset 61), with approved manuals bound to their respective assets. A plausible HVAC answer is not accepted as correct unless the approved source material supports it.

## Verification

Run the strict serving diagnostic in the configured Python environment:

```powershell
python manage.py check_rag_serving --strict
```

The GitHub Actions workflow at `.github/workflows/ci.yml` runs the backend security/regression suite, three-asset checks, customer RAG tests, technician isolation tests, frontend contract tests, lint and production build. Use the result for the exact commit being reviewed; historical reports are not proof that the current commit passes.

## Repository layout

| Path | Contents |
|---|---|
| `core/`, `servy_rag/`, `manage.py` | Django project: models, DRF APIs, RAG services, management commands, tests |
| `frontend/` | React/Vite apps: customer RAG (`VITE_SERVY_REAL_DEMO=1`), technician portal (`VITE_SERVY_TECH_DEMO=1`) and the normal authenticated app |
| `tools/Start-Servy*.ps1` | Supported launchers for the three applications |
| `tools/security/` | `red_team_static.py` and `red_team_offline.py` (run by CI from the repository root) |
| `tools/diagnostics/` | Directly executed diagnostics, e.g. `test_rag_handoff_regression.py` |
| `tools/launchers/` | Environment setup (`setup_windows.bat`, `setup_linux.sh`) and the Linux runner (`run_linux.sh`); these run `migrate`. On Windows start the app with `tools/Start-ServyApp.ps1` |
| `tools/` (other) | Evaluation, research and maintenance utilities; most are run directly |
| `docs/` | Architecture, decision records and design notes; `docs/archive/` holds historical reports |

## Repository cleanup boundary

The retired server-rendered Django dashboard, its project templates and Bootstrap assets, the high-volume synthetic `seed_demo` command, and the standalone dummy SQLite/Excel review dataset have been removed. The current React customer RAG and technician portals, DRF APIs, RAG services, approved knowledge assets, launchers, isolated test fixtures and current local technician handoff remain.

The ignored local `db.sqlite3` file is not tracked by GitHub. Do not delete it until you have checked whether your local environment still uses it and made any required backup. The active technician demo workspace above is a different file and must be preserved.
