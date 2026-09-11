# Servy RAG Upgrade — Local Django Prototype

A standalone proof-of-concept that mirrors the supplied Servy modules and implements the requested **Call Register + Knowledge Base RAG layer** before touching the private production repository.

## What is included

- Pure Python/Django backend with Django REST Framework (DRF) JSON API layer.
- Modern **React + Vite frontend** (`frontend/`) with Plus Jakarta Sans typography, sleek dark/light interfaces, and role-based workflows.
- Dual AI Workflows:
  - **Workflow 1 (Customer Pre-Ticket Self-Service)**: Facility & equipment diagnostics with verified technical references, past customer resolutions, "Issue Resolved" feedback, or direct escalation to a service call.
  - **Workflow 2 (Engineer Field Copilot)**: Active ticket diagnostics, confidential schematics, and cross-customer fleet resolution memory for technicians.
- Local vector database using **ChromaDB** with `SentenceTransformer('all-MiniLM-L6-v2')` embeddings and sparse hashing fallback.
- SQLite local database with strict Django security & tenant boundary revalidation.
- Multi-tenant data model with hard tenant filters in every RAG query.
- Product hierarchy: **Domain → Category → Brand → Product → Asset**.
- Servy-like screens for Projects, Assets, Inventory, Part Requests, Knowledge Base, and Call Register.
- Knowledge Base upload flow with tags, document type, customer, product, asset, confidentiality and RAG enable/disable.
- Local document parsing for text/Markdown, PDF, DOCX and CSV.
- Local chunk indexing and vector synchronization.
- Optional **local Ollama LLM**; when Ollama is unavailable the app falls back to a grounded extractive answer so the demo still works without paid APIs.
- Call Register XLSX export and XLSX import with safety bounds.
- Demo seed volume matching the screenshots: **51 assets, 26 projects, 118 KB items, 532 inventory rows, 7,665 calls**.
- Two high-quality initial RAG domains: **Milk Analyzer** and **Aerolift**.
- Standalone `sample_data/servy_dummy_schema.sqlite3` and `sample_data/servy_dummy_data.xlsx` for database/data-table review before running Django.

## Quick start (Windows)

### 1. Backend (Django)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

### 2. Frontend (React + Vite)
```bat
cd frontend
npm install
npm run dev
```
Open React App at `http://127.0.0.1:5173/` (or compiled SPA at `http://127.0.0.1:8000/app/`).

### Demo Login Personas
- **Admin**: `admin` / `adminpass123`
- **Manager**: `manager` / `managerpass123`
- **Technician**: `tech1` / `techpass123`
- **Customer**: `acme_user` / `acmepass123`

## Local LLM (optional, zero API cost)

The default setting is `SERVY_LLM_PROVIDER=auto`. The app tries a local Ollama endpoint and falls back to an extractive grounded answer if Ollama is not running.

Example:

```bash
ollama pull qwen2.5:3b
ollama serve
```

Then run Django normally. You can change the local model in `.env.example` / environment variables.

**No OpenAI/Anthropic/Gemini API key is required.**

## Best demo flow

1. Run the seed command.
2. Open **Customer AI Support**.
3. Choose `FreshDairy Labs` → `Milk Analyzer Lab-01` and ask: `The milk analyzer gives unstable readings after cleaning. What should I check?`
4. Review the grounded answer and exact source references.
5. Click **No — create service call** to show automatic escalation into Call Register with attempted RAG guidance attached.
6. Open **Engineer Copilot** to show staff-mode retrieval, including confidential internal maintenance documents.
7. Open **Call Register** and use **Download Excel** / **Import Excel**.
8. Logout and login as `aero_admin / AeroAdmin!2026` to demonstrate a completely separate tenant. The normal tenant selector only shows organisations for which that authenticated user has an explicit membership.

## Knowledge Base design

The RAG system does not blindly search all 118 items. It first filters by:

- tenant,
- customer-specific vs shared documents,
- confidentiality,
- selected asset,
- same product,
- tags/document type,

and then ranks the permitted chunks. This is important because the supplied Knowledge Base screenshot contains test data, duplicates and unrelated content.

## Included sample data

- `sample_data/servy_dummy_data.xlsx` — reviewable multi-sheet representation of the supplied screen schema.
- `sample_data/call_register_import_template.xlsx` — directly usable import example.
- `sample_data/servy_dummy_schema.sqlite3` — compact 36-table relational review database aligned with the hardened domain model.
- `sample_data/build_review_sqlite.py` — reproducibly rebuilds and integrity-checks that review database without Django.
- `python manage.py seed_demo --reset` — creates the full high-volume Django dummy dataset.

## Production notes

This is intentionally a **prototype**, not a production auth/security implementation. The tenant switcher is a demo convenience. In production, tenant identity must come from the authenticated user's organisation membership, not a user-selectable session value. Add production authentication, object-level permissions, audit logging, malware scanning for uploads, background indexing, rate limits, backup/restore and full test coverage before deployment.

See `PROJECT_SCOPE.md`, `ARCHITECTURE.md` and `UI_LICENSE.md` for the design rationale.
