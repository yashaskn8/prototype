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

## Quick Start & Running Commands

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **Node.js**: v18+ and npm
- **Git**

---

### Step 1: Backend Setup & Run (Terminal 1)

#### Windows (PowerShell)
```powershell
# 1. Navigate to prototype directory (if not already there)
cd prototype

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Apply migrations
python manage.py migrate

# 6. Seed demo data and diagnostic playbooks (first-time setup)
python manage.py seed_demo --reset
python manage.py seed_diagnostic_playbook

# 7. Start the Django API backend server
python manage.py runserver 127.0.0.1:8000
```

#### Windows (Command Prompt / CMD)
```cmd
cd prototype
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo --reset
python manage.py seed_diagnostic_playbook
python manage.py runserver 127.0.0.1:8000
```

#### macOS / Linux (Bash / Zsh)
```bash
cd prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py migrate
python manage.py seed_demo --reset
python manage.py seed_diagnostic_playbook
python manage.py runserver 127.0.0.1:8000
```

Backend will be available at:
- **API Base**: `http://127.0.0.1:8000/api/`
- **Django Admin**: `http://127.0.0.1:8000/admin/`

---

### Step 2: Frontend Setup & Run (Terminal 2)

In a separate terminal window:

```bash
# 1. Navigate to the frontend directory
cd prototype/frontend

# 2. Install npm dependencies
npm install

# 3. Start the Vite development server
npm run dev
```

Frontend will be available at:
- **Web App (Vite Dev)**: `http://localhost:5173/`
- **Compiled SPA (via Django)**: `http://127.0.0.1:8000/app/` (after running `npm run build`)

---

### Demo Login Personas

| Role | Username | Password | Access Scope |
| :--- | :--- | :--- | :--- |
| **Admin** | `admin` | `adminpass123` | Full system access, all tenants, admin panel |
| **Manager** | `manager` | `managerpass123` | Service calls, dispatch, analytics, KB |
| **Technician** | `tech1` | `techpass123` | Assigned calls, Engineer Copilot, confidential docs |
| **Customer** | `acme_user` | `acmepass123` | Self-service diagnostics, assets, customer tickets |
| **AeroLift Admin** | `aero_admin` | `AeroAdmin!2026` | Isolated second tenant demonstration |

---

### Useful Commands

#### Stop / Kill Running Processes

If ports `8000` (Django) or `5173` (Vite) are stuck or already in use:

**Windows PowerShell:**
```powershell
# Stop process on port 8000 (Django)
Get-Process -Id (Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force

# Stop process on port 5173 (Vite)
Get-Process -Id (Get-NetTCPConnection -LocalPort 5173 -ErrorAction SilentlyContinue).OwningProcess -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Windows CMD:**
```cmd
netstat -ano | findstr :8000
taskkill /F /PID <PID>

netstat -ano | findstr :5173
taskkill /F /PID <PID>
```

**macOS / Linux:**
```bash
lsof -ti :8000 | xargs kill -9
lsof -ti :5173 | xargs kill -9
```

#### Run Automated Test Suites
```bash
# Run all 131 tests and 10,000-session adversarial simulation
python manage.py test -v 1

# Run diagnostic recovery & hardening tests only
python manage.py test core.tests_diagnostics core.tests_diagnostics_hardening -v 2
```

#### Build Frontend for Production Deployment
```bash
cd prototype/frontend
npm run build
```

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

## Deterministic Diagnostic Recovery & Zero-Repeat Architecture

Servy includes a mathematically deterministic diagnostic execution engine and **Zero-Repeat** field dispatch system that eliminates customer troubleshooting loops and technician guesswork:

1. **Append-Only Immutable Event Ledger (`DiagnosticEvent`)**:
   - Every observation, action presentation, action confirmation, clarification, and contradiction is persisted as an append-only cryptographic sequence with strict monotonic `seq_num` ordering.
   - Reduced state is derived dynamically through pure functional reduction (`reduce_session_events`).

2. **Durable Evidence Anchoring & Tamper Detection**:
   - Every troubleshooting node and safe action is bound to an `EvidenceAnchor` pointing to an approved `KnowledgeDocument` with SHA-256 checksum and version pinning.
   - Requires exact source locators (`chunk_id`, verified `heading`, or normalized `excerpt`).
   - If a document is updated or its checksum drifts, any dependent actions or sessions are automatically invalidated.
   - Quarantined or prompt-injected chunks/documents are strictly blocked from authorizing any actions.

3. **Cryptographic Presentation Tokens & Causality Verification**:
   - A customer cannot confirm an action (`SAFE_ACTION_CONFIRMED`) without a strictly prior presentation event (`SAFE_ACTION_PRESENTED`) verified via a cryptographic HMAC token.
   - Prevents replay attacks, out-of-order execution, and skipped troubleshooting steps.

4. **Deterministic Recovery Passport (`RecoveryPassport`)**:
   - **Zero LLM inside authoritative passport data**: facts and completed actions are derived 100% deterministically from the immutable event ledger.
   - Real-time document freshness tracking: flags whether evidence remains verified or has drifted (`DOCUMENT_MODIFIED_SINCE_CONFIRMATION`, `DOCUMENT_REMOVED`).
   - Automatically populates `do_not_repeat_items` attached to the resulting `ServiceCall` so dispatch technicians never ask the customer to repeat steps already completed.
   - Strictly separates authoritative `system_escalation_reason` from non-authoritative customer comments.

5. **Safe Resolution & Contradiction Guards**:
   - Sessions can only be marked `RESOLVED` if the active deterministic path reaches an explicit terminal resolution node.
   - Sessions with unresolved contradictions or incomplete evidence paths cannot be marked resolved.

6. **Safety Policy & Prohibited Keyword Gating**:
   - Hazardous procedural repairs (disassembly, high-voltage panel access, firmware resets, custom calibration, soldering) cannot be presented to customers and require professional technician dispatch.
   - Clarification intake questions are schema-constrained (`BOOLEAN`, `SINGLE_CHOICE`, `SHORT_TEXT`, etc.) and automatically stripped of procedural repair suggestions.

---

## Ultra-Aggressive Hallucination-Prevention Guardrails

The RAG and LLM pipeline includes structural defenses against AI hallucinations:

- **Server-Side Verified Citations Only**: The model is prohibited from emitting citations. All citations and verified references are stripped from model output and attached server-side exclusively from verified database retrieval metadata.
- **Structural Delimiter Scrubbing**: Strips prompt delimiters (`<retrieved_source>`, `UNTRUSTED_CONTENT_START/END`, etc.) if echoed back by the model.
- **Speculative Repair Phrasing Rejection**: Automatically detects and rejects speculative repair suggestions (`"you can try"`, `"might want to try"`, `"perhaps you should"`, `"a quick workaround"`) without approved grounding, returning a safe insufficient-evidence fallback.
- **Header Spoofing Neutralization**: Sanitizes customer notes and clarification texts to prevent injection of fake escalation headers (`System Diagnostic Escalation Reason:`, `ALREADY COMPLETED — DO NOT REPEAT WITH CUSTOMER:`).
- **Anti-Prompt Injection & Query Abuse Detection**: Real-time scanning for adversarial prompts (`"ignore previous instructions"`, `"bypass safety"`, `"mark this fixed"`, `"invent a workaround"`).

---

## Running the Automated Test & Simulation Suite

The project includes unit tests, regression tests, and a 10,000-session adversarial Monte Carlo simulation harness:

```bat
# Run complete test suite (131 tests + 10,000 adversarial trajectories)
python manage.py test -v 1

# Run diagnostic recovery & hardening tests specifically
python manage.py test core.tests_diagnostics core.tests_diagnostics_hardening -v 2
```

The 10,000-session adversarial harness stress-tests 15 zero-tolerance invariants:
- Zero cross-tenant data leakage
- Zero ungrounded or unsafe actions presented to customers
- Zero stale playbook procedures authorized
- Zero duplicate service calls created
- Zero unverified recovery passport claims
- Zero customer mutations on terminal sessions

---

## Included sample data

- `sample_data/servy_dummy_data.xlsx` — reviewable multi-sheet representation of the supplied screen schema.
- `sample_data/call_register_import_template.xlsx` — directly usable import example.
- `sample_data/servy_dummy_schema.sqlite3` — compact 36-table relational review database aligned with the hardened domain model.
- `sample_data/build_review_sqlite.py` — reproducibly rebuilds and integrity-checks that review database without Django.
- `python manage.py seed_demo --reset` — creates the full high-volume Django dummy dataset.

## Production notes

This is intentionally a **prototype**, not a production auth/security implementation. The tenant switcher is a demo convenience. In production, tenant identity must come from the authenticated user's organisation membership, not a user-selectable session value. Add production authentication, object-level permissions, audit logging, malware scanning for uploads, background indexing, rate limits, backup/restore and full test coverage before deployment.

See `PROJECT_SCOPE.md`, `ARCHITECTURE.md` and `UI_LICENSE.md` for the design rationale.
