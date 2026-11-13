# Servy RAG Upgrade — Local Django Prototype

A standalone proof-of-concept that mirrors the supplied Servy modules and implements the requested **Call Register + Knowledge Base RAG layer** before touching the private production repository.

## What is included

- Pure Python/Django backend with Django REST Framework (DRF) JSON API layer.
- Modern **React + Vite frontend** (`frontend/`) with Plus Jakarta Sans typography, sleek dark/light interfaces, and role-based workflows.
- Dual AI Workflows:
  - **Workflow 1 (Customer Pre-Ticket Self-Service)**: Facility & equipment diagnostics with verified technical references, past customer resolutions, "Issue Resolved" feedback, or direct escalation to a service call.
  - **Workflow 2 (Engineer Field Copilot)**: Active ticket diagnostics, confidential schematics, and cross-customer fleet resolution memory for technicians.
- Local vector database using **ChromaDB** with `SentenceTransformer('all-MiniLM-L6-v2')` embeddings and sparse hashing fallback.
- **Authoritative MySQL relational database** (InnoDB, utf8mb4, strict SQL mode) with strict Django security & tenant boundary revalidation.
- Multi-tenant data model with hard tenant filters in every RAG query.
- Product hierarchy: **Domain → Category → Brand → Product → Asset**.
- Servy-like screens for Projects, Assets, Inventory, Part Requests, Knowledge Base, and Call Register.
- Knowledge Base upload flow with tags, document type, customer, product, asset, confidentiality and RAG enable/disable.
- Local document parsing for text/Markdown, PDF, DOCX and CSV.
- Local chunk indexing and vector synchronization.
- Optional local text generation: legacy Ollama support and a loopback llama.cpp Qwen provider for staff-mode RAG. Extractive answers remain the fallback; the authorized customer demo stays source-structured and does not require a chat server.
- Call Register XLSX export and XLSX import with safety bounds.
- Demo seed volume matching the screenshots: **51 assets, 26 projects, 118 KB items, 532 inventory rows, 7,665 calls**.
- Two high-quality initial RAG domains: **Milk Analyzer** and **Aerolift**.
- Standalone review artifacts: `sample_data/servy_dummy_schema.sqlite3` and `sample_data/servy_dummy_data.xlsx` for independent data review.

## Windows PowerShell: Start the Authorized Customer and Technician RAG Instances

**Important folder rule:** the Git checkout is the prototype subfolder. If your PowerShell prompt is
C:\Users\prana\OneDrive\Desktop\SURVY-RAG>, you are one directory above the repository.
From that exact prompt, first run:

~~~powershell
Set-Location .\prototype
git switch main
git fetch origin
git pull --ff-only origin main
~~~

Your prompt must now end in SURVY-RAG\prototype>. The launchers are under
prototype\tools\. Running .\tools\... from the parent SURVY-RAG folder will fail because that
relative path does not exist there.

**Start the customer authorized RAG first:**

~~~powershell
.\tools\Start-ServyRealDemo.ps1
~~~

Keep this PowerShell window open: it streams customer backend HTTP statuses and Vite activity.
Examples include `HTTP 200 OK | POST /demo-api/query/` and
`HTTP 201 Created | POST /demo-api/tickets/create-local/`. The log stream masks ticket IDs,
question text, access codes, signed source tokens and response bodies. Press Ctrl+C to close only
the log viewer; the launched servers remain running. Use `-NoWatch` to return to the prompt instead.

Open http://127.0.0.1:5175/ and enter the existing customer RAG access code. The launcher now
automatically uses a guarded, disposable BGE rebuild from verified approved documents only when
the frozen Chroma bytes differ. It does not repair/write production Chroma or write company MySQL
records. The first guarded rebuild can take up to four minutes.

Create/confirm a ticket in the customer RAG to create the dedicated local SQLite ticket workspace.
This does not create an official MySQL service call. If backend startup fails, the launcher reports
the exact private `startup.out` and `startup.err` paths. Inspect the error with:

~~~powershell
Get-Content -LiteralPath '<startup.err path printed by the launcher>' -Tail 80
~~~

Remove or redact access codes, credentials, passwords and customer details before sharing log output.

### Three-asset / nine-manual quality audit

The authorized real-data scope currently contains three assets and nine approved manuals:
- Split AC Unit 2T: asset 59, docs 168–170
- VRF Outdoor Module: asset 60, docs 171–173
- Ductable AC 5T: asset 61, docs 174–176

The offline CI suite checks all nine source-to-asset bindings, document types, exact-asset applicability, and the 28 relevant regression cases in the frozen three-asset fixture. The fixture is a software regression set, **not** a production accuracy certification.

To run the real read-only retrieval audit against the existing company MySQL data and the current serving/index on this Windows machine, open a separate PowerShell window and run:

~~~powershell
cd "C:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype"

$env:LOCALAPPDATA = "C:\Users\prana\AppData\Local\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local"

.\.venv\Scripts\python.exe -B tools\servy_three_asset_rag_audit.py
~~~

This requires the existing MySQL service and the current user's DPAPI-protected SELECT-only evaluation credential. It verifies the nine source attachments and records current retrieval hit@1/hit@5/hit@10, citation coverage, cross-asset leakage, and per-document coverage in a private folder under `%LOCALAPPDATA%\Servy\private_evaluations`. Manuals with no matching query in the frozen fixture are reported as **NOT_EVALUATED**, not as passes or failures. It does not update MySQL records, production Chroma, application indexes, or model checkpoints.

### Prepare source-grounded SME training review

To prepare a private review workspace from the nine verified manuals, run:

~~~powershell
.\.venv\Scripts\python.exe -B tools\servy_three_asset_training_export.py
~~~

The exporter creates:
- `approved_source_chunks.jsonl` — exact verified chunk text and provenance for the nine manuals.
- `sme_review_template.csv` — candidate positive passages, with the query, hard negatives, reviewer, and split left for review.
- `manifest.json` — asset/document/file hashes and unchanged operational record counts.

Every row is marked `pending_sme_review`. **Do not train from this template as-is.** Each question/positive/negative relationship must be reviewed by an authorized HVAC subject-matter reviewer, and train/dev/independent-test splits must be source-family disjoint. The earlier BGE fine-tune was weakly supervised and measured below the fair MiniLM baseline, so this workflow intentionally does not auto-promote or start another blind fine-tune.

**Start the separate technician code-gated instance (it can start before the first ticket):**

~~~powershell
.\tools\Start-ServyTechnicianDemo.ps1
~~~

Keep this second PowerShell window open: it streams technician API HTTP statuses and Vite activity,
for example `HTTP 200 OK | GET /tech-api/tickets/` and
`HTTP 200 OK | PATCH /tech-api/tickets/{ticket_id}/`. Ticket IDs and ticket contents are masked.
Press Ctrl+C to close only the log viewer; the technician services remain running. Use
`-NoWatch` to return to the prompt instead.

Open http://127.0.0.1:5176/ and use the code in the local technician_access.txt file printed by
the launcher. If no customer ticket exists yet, the portal opens with an empty queue and does not
create or initialize the SQLite database. Create/confirm a ticket in the customer RAG; the queue
refreshes automatically. Both launchers resolve the same Windows LocalAppData ticket database path.

You can also invoke either launcher directly while remaining in the parent SURVY-RAG folder by
including the repository directory in the path:

~~~powershell
.\prototype\tools\Start-ServyRealDemo.ps1
.\prototype\tools\Start-ServyTechnicianDemo.ps1
~~~

If a launcher gets past the path check but fails during startup, use the private log path printed by
that launcher; do not delete the ticket database, reset the company database, or start another MySQL
instance as a troubleshooting shortcut.

---

## Quick Start & Running Commands

### Prerequisites
- **MySQL**: 8.0+ server running locally or accessible remotely
- **Python**: 3.10, 3.11, or 3.12 (with `mysqlclient` / `pymysql`)
- **Node.js**: v18+ and npm
- **Git**

### Database Setup (Authoritative MySQL)
1. Create the database and dedicated application user:
   ```sql
   CREATE DATABASE servy CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
   CREATE USER 'servy_app'@'localhost' IDENTIFIED BY 'your_secure_password';
   GRANT ALL PRIVILEGES ON servy.* TO 'servy_app'@'localhost';
   FLUSH PRIVILEGES;
   ```
2. Configure credentials in `.env`:
   ```ini
   SERVY_DB_ENGINE=mysql
   MYSQL_HOST=127.0.0.1
   MYSQL_PORT=3306
   MYSQL_DATABASE=servy
   MYSQL_USER=servy_app
   MYSQL_PASSWORD=your_secure_password
   ```

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

# 5. Verify MySQL Readiness and Apply migrations
python manage.py check_mysql_ready
python manage.py migrate

# 6. Seed demo data and diagnostic playbooks (safe and idempotent; omit --reset on protected databases)
python manage.py seed_demo
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
python manage.py check_mysql_ready
python manage.py migrate
python manage.py seed_demo
python manage.py seed_diagnostic_playbook
python manage.py runserver 127.0.0.1:8000
```

#### macOS / Linux (Bash / Zsh)
```bash
cd prototype
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python manage.py check_mysql_ready
python manage.py migrate
python manage.py seed_demo
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

## Local model and generation configuration

The authorized read-only real-data demo launched with `tools/Start-ServyRealDemo.ps1` uses the local, source-grounded path:

- **BGE-small** creates dense query/document embeddings for the approved company collection.
- **Hybrid retrieval** combines dense retrieval with lexical/sparse matching.
- **The verified Servy V5 reranker** reorders the retrieved candidate passages.
- **The deterministic customer answer builder** selects only source-supported claims, validates exact sentence spans, applies customer-action safety rules and creates signed citations.
- **Qwen3.5-2B through local Ollama** can rephrase only already-approved informational and possible-cause claims after the final customer answer has passed dialogue focus. Source quotes, claim kinds and citation bindings remain unchanged. Observations and customer-approved procedures stay in source wording. Unsupported terms, quantities, codes, certainty or instructions are rejected; exact source wording is the fallback.

BGE and V5 are retrieval components, not text-generation models. The real-data demo keeps retrieval, evidence sufficiency, action authorization, answer selection and citation creation deterministic. Its launcher enables the constrained local Qwen wording step using the `qwen3.5:2b-q4_K_M` Ollama model; `-NoLocalChat` disables it. No cloud LLM or remote endpoint is used.

Download the model once before the demo:
```powershell
ollama pull qwen3.5:2b-q4_K_M
ollama list
```
If Ollama or the model is unavailable, the launcher continues with exact approved source wording.

For staff-mode RAG, the repository also supports an optional local Qwen GGUF generator through llama.cpp. Install llama.cpp on Windows with `winget install llama.cpp`, then serve the Q4_K_M model. The first Hugging Face launch downloads the model if it is not already cached. Configure the backend process:

```powershell
$env:SERVY_LLM_PROVIDER = "llama_cpp"
$env:SERVY_LLAMA_CPP_URL = "http://127.0.0.1:9931/v1/chat/completions"
$env:SERVY_LLAMA_CPP_MODEL = "gguf/Qwen3.5-2B-GGUF:Q4_K_M"
$env:SERVY_LLM_REQUIRE_LLAMA_CPP = "0"
```

Start the local server separately with:

```powershell
llama serve -hf gguf/Qwen3.5-2B-GGUF:Q4_K_M --alias gguf/Qwen3.5-2B-GGUF:Q4_K_M -c 4096 -np 1
```

The backend calls llama.cpp's OpenAI-compatible chat endpoint, disables proxy inheritance and redirects, and rejects non-loopback URLs. If the server is unavailable, the default behavior falls back to the existing extractive answer; set `SERVY_LLM_REQUIRE_LLAMA_CPP=1` only when an explicit abstention is preferred over fallback. Generation is permitted only when `ask_rag(..., staff_mode=True)`; customer-mode RAG deliberately does not call the model. The customer answer contract independently rebuilds its display from authorized source sentences and citations.

The persistent customer-conversation API uses the same guarded answer builder. To enable Qwen wording outside the demo launcher, configure `SERVY_CUSTOMER_ANSWER_GENERATION=ollama`, `SERVY_CUSTOMER_GENERATION_MODEL=qwen3.5:2b-q4_K_M`, and the loopback `SERVY_CUSTOMER_GENERATION_URL=http://127.0.0.1:11434/api/chat`. The feature is disabled by default elsewhere. Staff-mode RAG retains the explicit Ollama and llama.cpp generation providers; customer mode does not use unchecked free-form output.

**Conversational RAG smoke tests** after updating `main`:

```bash
python manage.py test core.tests_customer_conversation_multiturn core.tests_customer_answer -v 2
cd frontend
node --test test/customerAnswerContract.test.mjs
npm run lint
npm run build
```

Use the approved Knowledge Base documents for the selected asset. Missing, unapproved, or unrelated documents must produce an insufficient-evidence response instead of an invented answer.

**No OpenAI/Anthropic/Gemini API key or Ollama installation is required for the real-data demo.**

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
- `python manage.py seed_demo` — creates the initial high-volume Django dummy dataset idempotently without destroying existing data.

## Production notes

This is intentionally a **prototype**, not a production auth/security implementation. The tenant switcher is a demo convenience. In production, tenant identity must come from the authenticated user's organisation membership, not a user-selectable session value. Add production authentication, object-level permissions, audit logging, malware scanning for uploads, background indexing, rate limits, backup/restore and full test coverage before deployment.

See `PROJECT_SCOPE.md`, `ARCHITECTURE.md` and `UI_LICENSE.md` for the design rationale.
