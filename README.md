# Servy RAG

## Architecture

Servy answers customer equipment questions from the approved manuals of one selected
asset and hands unresolved issues to technicians as official MySQL ServiceCalls.
Django/DRF backend (`core/`, `servy_rag/`), React/Vite frontend (`frontend/`).

**Models**

| Stage | Implementation |
|---|---|
| Embeddings | BGE-small (`BAAI/bge-small-en-v1.5`, pinned revision) for documents and queries — `core/ml/embeddings.py`, `core/ml/indexer.py`; vectors in a stamped ChromaDB collection |
| Retrieval | Hybrid: BGE dense + lexical retrieval, merged by rank fusion — `core/services/retriever.py` |
| Reranking | V5 local cross-encoder, weights SHA-256-pinned (`SERVY_RERANKER_MODEL_PATH`, `SERVY_RERANKER_EXPECTED_SHA256`) — `core/ml/reranker.py` |
| Generation | Qwen `qwen3.5:2b-q4_K_M` through local Ollama, used only to reword already-selected claims — `core/services/customer_answer_generation.py` |

**Answer pipeline** (`core/services/customer_conversation.py`)

1. *Conversation context* (`core/services/conversation_context.py`): the stored conversation
   is read from MySQL; follow-ups are rewritten into a retrieval query using the earlier
   complaint and symptoms; references such as "the second cause" resolve against the
   previous answer's cited list; corrections and explicit topic changes replace the earlier
   subject. The customer's message is stored unchanged.
2. *Retrieval and isolation*: candidates are restricted to the authorized tenant, customer
   and selected asset before ranking, then reranked by V5.
3. *Claim selection and citations* (`core/services/customer_answer.py`): each technical claim
   is an exact sentence from a retrieved manual passage, bound to one citation (document,
   section, signed source link re-authorized when opened). Unsupported questions are refused.
4. *Qwen validation*: a Qwen paraphrase is accepted only if it keeps every source term,
   number, negation and uncertainty and adds no new term, instruction or certainty;
   otherwise the exact source sentence is used.
5. *Composed reply* (`core/services/customer_response_composer.py`): the validated, cited claims
   are arranged by intent (cause list, direct fact, single selected cause, clarifying
   question for an ambiguous message). Nothing is added to the claims.

**API 1 — Conversational RAG** (`core/api_views.py`)

| Route | Purpose |
|---|---|
| `POST /api/customer-support/conversations/` | Start a conversation for an authorized asset |
| `POST …/conversations/{uuid}/messages/` | Ask a question (JSON); optional `X-Idempotency-Key` |
| `POST …/conversations/{uuid}/messages-stream/` | Same, as server-sent events (`log` stages, then `result`) |
| `GET …/conversations/`, `GET …/conversations/{uuid}/` | List and reopen stored conversations |

**API 2 — Conversation to ServiceCall** (`core/services/conversation_ticket.py`, `core/services/ticketing.py`)

| Route | Purpose |
|---|---|
| `GET …/conversations/{uuid}/ticket/` | Read-only preview: nine-section description built from the full conversation, message count, missing contact fields, `can_create` |
| `POST …/conversations/{uuid}/ticket/` | Create the ServiceCall with `confirm_ticket_creation: true`; row-locked and idempotent (`created: true/false`); the stored description equals the preview |

**Technician workflow** (`core/technician_inbox.py`): technicians list authorized open
ServiceCalls, read the description, conversation and citations, download attachments and
record status changes and notes as append-only `CallUpdate` records on the same MySQL
ServiceCall. The generated description is never overwritten.

**Customer and Technician applications** (`core/demo_gateway.py`, `servy_rag/demo_*_urls.py`):
two access-code gateways serve the official APIs above under `/demo-api/` (Customer) and
`/tech-api/` (Technician). Each accepts loopback requests only, checks its access code in
constant time and maps it to one synthetic user. Both run only on the isolated `servy_dev`
MySQL database (account `servy_dev_app`) and refuse any other database. Customer
attachments (0–10 files, ≤ 25 MB each and in total; JPEG, PNG, GIF, WebP, PDF, text;
content-validated) are stored as MySQL metadata plus files in an isolated media directory.
Automated tests use `test_servy_dev` (account `servy_dev_test`) or in-memory SQLite, and a
temporary ChromaDB directory.

## Commands

Prerequisites: Windows PowerShell, Python virtualenv at `.venv` (`pip install -r requirements.txt`),
Node 20 with `frontend/node_modules` installed, local MySQL 8 on port 3306, Ollama on port
11434 with `qwen3.5:2b-q4_K_M`, the local BGE-small and V5 checkpoints, and the private
Customer and Technician access-code files (outside the repository).

**Configure the development database (one time)**

```powershell
.\tools\Provision-ServyDevMySQL.ps1          # creates servy_dev / servy_dev_app and test_servy_dev / servy_dev_test
$env:SERVY_DB_ENGINE = 'mysql'
$env:SERVY_MYSQL_ENV_FILE = '.env.mysql.servy_dev'
$env:SERVY_MEDIA_ROOT = '<private media directory>'
$env:SERVY_CHROMA_PERSIST_DIR = '<private development ChromaDB directory>'
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py seed_servy_dev        # synthetic records + nine approved manuals (BGE vectors)
.\.venv\Scripts\python.exe manage.py check_dev_database    # verifies servy_dev isolation
```

**Launch the applications** (each in its own PowerShell window, from the repository root)

```powershell
.\tools\Start-ServyRealDemo.ps1          # Customer RAG: backend 127.0.0.1:8002, UI http://127.0.0.1:5175/
.\tools\Start-ServyTechnicianDemo.ps1    # Technician:   backend 127.0.0.1:8003, UI http://127.0.0.1:5176/
```

Options: `-NoWatch` (return after startup), `-Restart` (Customer only: replace a running
instance), `-NoLocalChat` (Customer only: no Qwen, source wording only).

**Start backend and frontend manually** (what the launchers run)

```powershell
# Customer backend (8002); the Technician backend uses SERVY_DEMO_GATEWAY='technician',
# SERVY_TECHNICIAN_ACCESS_FILE and port 8003
$env:SERVY_DB_ENGINE = 'mysql'
$env:SERVY_MYSQL_ENV_FILE = '.env.mysql.servy_dev'
$env:SERVY_MEDIA_ROOT = '<private media directory>'
$env:SERVY_CHROMA_PERSIST_DIR = '<private development ChromaDB directory>'
$env:SERVY_SERVING_EMBEDDING_PROFILE = 'bge-small'
$env:SERVY_DEMO_GATEWAY = 'customer'
$env:SERVY_DEMO_ACCESS_FILE = '<private customer access-code file>'
$env:SERVY_CUSTOMER_DATA_MODE = 'demo_only'
$env:SERVY_CUSTOMER_ANSWER_GENERATION = 'ollama'
$env:SERVY_CUSTOMER_GENERATION_URL = 'http://127.0.0.1:11434/api/chat'
$env:SERVY_CUSTOMER_GENERATION_MODEL = 'qwen3.5:2b-q4_K_M'
$env:HF_HUB_OFFLINE = '1'
.\.venv\Scripts\python.exe manage.py check_dev_database
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8002 --noreload

# Customer UI (5175); the Technician UI uses VITE_SERVY_TECH_DEMO='1' and port 5176
cd frontend
$env:VITE_SERVY_REAL_DEMO = '1'
npm run dev -- --host 127.0.0.1 --port 5175 --strictPort
```

**Test**

```powershell
# Backend, isolated MySQL test database
$env:SERVY_DB_ENGINE = 'mysql'
$env:SERVY_MYSQL_ENV_FILE = '.env.mysql.servy_dev_test'
.\.venv\Scripts\python.exe manage.py test core --keepdb

# Relevant backend suites only
.\.venv\Scripts\python.exe manage.py test core.tests_demo_gateway core.tests_conversation_ticket_workflow core.tests_customer_response_composer core.tests_customer_answer_generation core.tests_call_attachments --keepdb

# Frontend tests
cd frontend
node --test test/*.test.mjs test/*.test.js

# Live checks against the running gateways (real models over HTTP)
.\.venv\Scripts\python.exe tools\servy_three_asset_gateway_eval.py --access-file '<customer access-code file>' --report-dir '<private report directory>'
.\.venv\Scripts\python.exe tools\servy_multiturn_eval.py --access-file '<customer access-code file>' --report-dir '<private report directory>'
.\.venv\Scripts\python.exe tools\servy_api_acceptance.py --private-root '<folder containing both access-code folders>'
```

**Lint and build**

```powershell
cd frontend
npm run lint
npm run build
```
