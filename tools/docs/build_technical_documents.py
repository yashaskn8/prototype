"""Build the Servy Technical Functional Document and API Integration Guide (DOCX).

    .venv/Scripts/python.exe tools/docs/build_technical_documents.py [--examples examples.json]

Every statement names the module/function it describes so it can be re-checked
against the code. API examples are taken from a real test run
(SERVY_CAPTURE_API_EXAMPLES, see core/tests_portal_api.py) when --examples is given.
"""
import argparse
import json
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs"


def _doc(title, subtitle):
    d = Document()
    style = d.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)
    t = d.add_heading(title, 0)
    t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p = d.add_paragraph(subtitle)
    p.runs[0].font.color.rgb = RGBColor(0x55, 0x55, 0x55)
    return d


def bullets(d, items):
    for item in items:
        d.add_paragraph(item, style="List Bullet")


def code(d, text):
    p = d.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(8.5)


def table(d, header, rows):
    t = d.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = h
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = str(value)
    d.add_paragraph()


def technical_document(path):
    d = _doc("Servy RAG Backend - Technical Functional Document",
             f"Implementation-specific description of the Django backend, AI models and APIs. Generated {date.today():%d %B %Y} "
             "from repository yashaskn8/prototype (branch main).")

    d.add_heading("1. System purpose and functional scope", 1)
    d.add_paragraph(
        "Servy is a customer-support and equipment-troubleshooting backend. A customer (or a portal acting for "
        "the customer) opens a conversation about one of the customer's assets, describes a symptom, and receives a "
        "conversational answer grounded only in approved, authorized equipment manuals. Every technical statement "
        "shown to the customer is a claim validated against an exact source span and carries a citation. When the "
        "customer asks for it explicitly, the backend creates a service ticket linked to the conversation.")
    bullets(d, [
        "In scope: conversations, AI troubleshooting messages (sync and streaming), ticket preview/creation/status, "
        "source citation opening, health/readiness, API-key authentication for external portals.",
        "Out of scope for the portal integration: the React client in frontend/ (kept for local demonstration only), "
        "the tools/servy_real_demo*.py utilities and exhaustive-search evaluation scripts (not the production path).",
        "The assistant does not diagnose: causes are presented as documented possibilities, never as a confirmed fault.",
    ])

    d.add_heading("2. Backend-only integration architecture", 1)
    d.add_paragraph(
        "The external portal calls the Django REST API over HTTPS with an API key. Django serves only JSON (and "
        "text/event-stream for the streaming endpoint). The React client is optional at runtime: with "
        "SERVY_SERVE_FRONTEND=0 the production URLconf (servy_rag/urls.py) publishes only /api/ and /admin/.")
    code(d, "Portal UI -> HTTPS (nginx, TLS) -> gunicorn (gthread) -> Django/DRF (core.api_views)\n"
            "  -> MySQL 8.0 (conversations, messages, tickets, assets, users, documents, chunks)\n"
            "  -> ChromaDB PersistentClient (BGE vectors, local directory)\n"
            "  -> BGE-small (in-process, sentence-transformers) / V5 cross-encoder (in-process, ONNX Runtime or torch)\n"
            "  -> Qwen via Ollama on 127.0.0.1:11434 (same VM, loopback only)")

    d.add_heading("3. Django module responsibilities", 1)
    table(d, ["Module", "Responsibility"], [
        ["servy_rag/settings.py", "Environment-driven settings; MySQL-only runtime guard; DRF auth, throttles; model settings"],
        ["servy_rag/urls.py, core/api_urls.py", "Production URL routing (/api/...)"],
        ["core/api_keys.py", "Server-to-server API keys (Bearer), hashed at rest, expiry and revocation"],
        ["core/security.py", "require_api_context: tenant/customer/role binding; customer_visible_documents"],
        ["core/api_views.py", "Conversation, message, ticket, knowledge endpoints"],
        ["core/health.py", "Unauthenticated liveness/readiness probes"],
        ["core/services/rag_event_stream.py", "SSE framing for messages-stream"],
        ["core/services/customer_conversation.py", "One customer turn: idempotency, routing, retrieval, contract, persistence"],
        ["core/services/conversation_context.py", "Conversation state: active issue, pending question, replies, negations, ordinals, query rewriting"],
        ["core/services/rag.py (ask_rag)", "Retrieval entry point for a customer turn"],
        ["core/services/retriever.py (retrieve)", "Authorized candidates, sparse + dense hybrid scoring, RRF fusion, V5 rerank"],
        ["core/services/customer_diagnostic.py", "build_structured_diagnostic_contract: response type, actions, safety"],
        ["core/services/customer_answer.py", "Claim extraction/validation against source spans, citations (source_chunk, open_citation)"],
        ["core/services/customer_answer_generation.py", "Optional Qwen paraphrase of validated claims with source validator"],
        ["core/services/customer_response_composer.py", "Natural reply composition from validated claims; displayed-claims record"],
        ["core/services/cause_explanations.py", "Per-cause explanation retrieval for 'explain each/the Nth cause'"],
        ["core/services/conversation_ticket.py, ticketing.py", "Ticket preview and creation, de-duplication, assignment"],
        ["core/services/rag_admission.py, request_deadline.py", "Single in-flight AI turn per process; per-stage deadlines"],
        ["core/ml/embeddings.py, reranker.py, preload.py", "Model loading, verification and background warm-up"],
    ])

    d.add_heading("4. Model inventory", 1)
    table(d, ["Model", "Role", "Identity / artifact", "Runs where"], [
        ["BGE-small-en-v1.5", "Dense embeddings for semantic retrieval",
         "BAAI/bge-small-en-v1.5, pinned revision (SERVY_BGE_SMALL_REVISION), 384 dims, loaded with local_files_only",
         "In the Django process (CPU), sentence-transformers"],
        ["V5 cross-encoder", "Reranks retrieved candidate passages",
         "Local fine-tuned checkpoint (SERVY_RERANKER_MODEL_PATH, SHA-256 pinned via SERVY_RERANKER_EXPECTED_SHA256); "
         "optional verified ONNX export (SERVY_RERANKER_ONNX_DIR)", "In the Django process (CPU), ONNX Runtime or torch CrossEncoder"],
        ["Qwen (qwen3.5:2b-q4_K_M in the demo launcher)", "Optional rewording of already-validated claims",
         "Ollama model name (SERVY_CUSTOMER_GENERATION_MODEL)", "Ollama server on the same machine, loopback 127.0.0.1:11434"],
    ])
    d.add_paragraph("These are three model components inside one deterministic pipeline; none of them is an autonomous agent. "
                    "Sparse retrieval (BM25 / hashing vectors, core/services/vectors.py) needs no model.")

    d.add_heading("5. Model initialization (actual functions)", 1)
    bullets(d, [
        "BGE: core.ml.embeddings.get_profile('bge-small') -> load_model(profile) (SentenceTransformer, local_files_only=True; "
        "missing weights raise EmbeddingConfigError). get_collection(profile) opens the stamped Chroma collection and refuses "
        "unstamped or mismatched collections.",
        "V5: core.ml.reranker.get_reranker() loads OnnxV5 (onnxruntime, manifest + hash verified) when SERVY_RERANKER_BACKEND=onnx, "
        "else a sentence-transformers CrossEncoder (max_length 256). require_verified_v5_reranker() runs at Django start when "
        "SERVY_RERANKER_REQUIRED=1 and stops startup if the pinned checkpoint cannot score.",
        "Qwen: no Python load; core.services.customer_answer_generation calls the Ollama /api/chat endpoint, which must be a "
        "loopback URL (_endpoint).",
        "Warm-up: core.ml.preload.start_preload() (started from core/apps.py when SERVY_PRELOAD_MODELS=1 in a serving "
        "process - runserver or the WSGI app, which sets SERVY_SERVING_PROCESS=1) warms BGE (+ one Chroma query), Qwen "
        "(empty generate with keep_alive) and V5, and records per-model status: not_started, pending, ready, failed, disabled.",
        "Readiness: GET /api/health/ready/ returns 200 only when the database answers, no migrations are pending, every "
        "model is ready or disabled and the vector index is present and non-empty; otherwise 503 with coarse states "
        "(database, schema, models, index). The demo launcher also refuses to start with pending migrations.",
    ])

    d.add_heading("6. BGE retrieval workflow", 1)
    bullets(d, [
        "Indexing: approved document text is split into sections/chunks (core/services/indexing.py, starlly_ingest.py); "
        "chunk rows live in MySQL (KnowledgeChunk) and BGE passage vectors in ChromaDB.",
        "Query: retrieve() encodes the retrieval query with the BGE query prefix (encode_queries) and searches Chroma only "
        "within the chunk IDs already authorized for this request (_dense_score_candidates).",
        "Dense search runs under a stage deadline (dense 1.0 s); on timeout or unavailability the turn continues sparse-only "
        "and diagnostics record retrieval_status (sparse_only_dense_timeout / sparse_only_dense_unavailable).",
        "One deadline covers the whole turn (SERVY_RAG_DEADLINE_SECONDS, default 5 s; caps: dense 1.0 s, V5 2.6 s, "
        "Qwen 3.5 s; 0.4 s reserve). Once the dense budget is spent, later retrieval passes in the same turn skip vector "
        "search up front (error_type DeadlineExhausted) instead of retrying; at most one warning is logged per request.",
        "'Explain each cause' turns retrieve once per manual section (not once per cause), and every cause still starts "
        "from its own cited chunk. Turns where some passes were hybrid and later ones sparse-only are reported as "
        "hybrid_partial_dense_timeout with dense_passes counts, and the outcome is marked degraded, never as full hybrid.",
        "HTTP 200 and 'models READY' mean the request completed and models are loaded; per-turn model execution is "
        "reported only in contract.outcome (retrieval_status, reranker_status, generation_status).",
    ])

    d.add_heading("7. Hybrid sparse and dense search", 1)
    d.add_paragraph("retriever._sparse_score_candidates scores BM25/character n-gram vectors; _dense_score_candidates scores BGE "
                    "similarity; _reciprocal_rank_fusion merges them (weights 0.55 dense / 0.45 sparse) with metadata and "
                    "identifier boosts. Candidate scope is computed first by _knowledge_candidates and every candidate passes "
                    "is_authorized_candidate, so ranking never sees unauthorized text.")

    d.add_heading("8. V5 reranking", 1)
    d.add_paragraph("core.ml.reranker.rerank_candidates(query, candidates) scores (query, passage) pairs with the V5 cross-encoder "
                    "under the V5 stage deadline (2.6 s) and reorders the candidate pool. If V5 is disabled, missing or "
                    "times out, the authorized original order is kept and the outcome is recorded (reranker_status); no "
                    "reranked result is ever claimed when V5 did not run. A high similarity score is never treated as a diagnosis.")

    d.add_heading("9. Qwen generation workflow", 1)
    d.add_paragraph("Qwen does not answer questions. After claims are extracted and validated against source spans, "
                    "generate_customer_claim_texts_with_status() may ask Qwen to reword eligible claims within the remaining "
                    "time budget. Each paraphrase must pass _validate_paraphrase (numbers, key terms, no added facts) or the exact "
                    "source wording is kept. Every outcome is recorded as generation_status (applied, disabled, deadline_skipped, "
                    "ollama_unreachable, timeout, rejected_by_source_validator, request_failed). The fallback reason is kept "
                    "in contract.outcome/diagnostics for monitoring and is not shown as customer text.")

    d.add_heading("10. Evidence grounding and citation validation", 1)
    bullets(d, [
        "core.services.customer_answer.source_chunk() is the gate for any chunk used in a customer answer: tenant, customer "
        "ownership of the asset, serving policy, disable_sharing, quarantine, is_rag_enabled, index status, confidentiality, "
        "asset hierarchy, provenance revocation, verified-real provenance (verified_real_only mode) and prompt-injection checks.",
        "Claims are kinds possible_cause, information/document_fact, observation, customer_procedure; observations and "
        "procedures are role-separated (only customer-approved procedures are presented as checks).",
        "Citations are signed tokens (django.core.signing, 1 hour) bound to user, tenant, customer, asset, document, chunk, "
        "document version, chunk SHA-256 and span; open_citation() re-runs source_chunk on every open.",
        "The composed reply (customer_response_composer.compose_response) contains only validated claims plus fixed connective "
        "wording, and records displayed_claims - the exact ordered list the customer saw.",
    ])

    d.add_heading("11. Conversation state and follow-up logic", 1)
    bullets(d, [
        "extract_conversation_context() rebuilds state from MySQL messages on every turn (bounded window), so state survives "
        "restarts and reopening.",
        "State: original complaint, active issue (the latest newly reported problem or explicit topic change), active topics "
        "(manual sections last shown), pending question (clarification, closing question, or a 'record whether ...' "
        "observation shown), resolved answers, reported symptoms, negated observations ('no water leak'), completed checks.",
        "A short reply to the pending question ('Only at startup.', 'Yes, the fan is running.', 'No error code.') is resolved "
        "against that question and the active issue; the retrieval query is built from the active topic and issue, never the "
        "first complaint. Questions, topic changes, explain requests and newly reported symptoms are not treated as replies.",
        "Replies only show evidence from the sections being discussed; if nothing new is documented the assistant acknowledges "
        "the answer and says the manuals add nothing more, instead of switching to an unrelated section.",
        "Ordinal references ('the first', 'the seventh', 'the last', 'cause number 12') resolve against displayed_claims.",
        "'Should I contact a technician?' is answered directly without retrieval and never claims a ticket exists.",
        "Durable state is rebuilt from the whole saved conversation (up to 400 messages): original and current "
        "problem, corrections, answers to questions, negative observations and completed checks survive long "
        "conversations; only the wording context uses the last 10 messages.",
        "'Summarize what I told you' and similar requests are answered from saved customer statements only "
        "(response_type CONVERSATION_SUMMARY), with no retrieval and no technical claims.",
        "Ticket help ('create a ticket'), ticket status ('did you create the ticket?'), ticket contents and "
        "certainty questions ('is this definitely X?') are answered from conversation state and the linked ticket "
        "only (response_type CONVERSATION_REPLY, reason_code TICKET_HELP/TICKET_STATUS/TICKET_CONTENTS/CERTAINTY). "
        "No ticket is ever created by chat; safety-guarded messages are never routed here.",
        "Unconfirmed answers use plain wording ('I can't confirm that from the information I have...') that "
        "repeats what the customer reported and offers a technician once, not after every message.",
    ])

    d.add_heading("12. Ticket workflow and ownership", 1)
    bullets(d, [
        "GET /conversations/{uuid}/ticket/ returns the linked ticket or a read-only preview built from the conversation "
        "(build_conversation_ticket_description) with missing contact fields.",
        "POST requires confirm_ticket_creation=true. The conversation row is locked (select_for_update); an existing linked "
        "ticket is returned with created=false (no duplicate). create_call_from_interaction() allocates servy_id under a "
        "tenant row lock and assigns the least-loaded technician in zone/branch.",
        "The AI never creates a ticket because it recommends a technician; creation is always an explicit API call.",
        "Ownership: Servy owns ticket creation in its MySQL database (ServiceCall). If the evaluator's portal is the system of "
        "record for tickets, the portal should create the ticket on its side and either (a) call the Servy ticket endpoint "
        "to keep the conversation linkage, or (b) store the Servy conversation_uuid on its ticket. This boundary was not "
        "settled in the evaluator meeting and must be agreed before go-live.",
    ])

    d.add_heading("13. Database responsibilities: MySQL 8.0 vs ChromaDB", 1)
    table(d, ["Store", "Holds", "Notes"], [
        ["MySQL 8.0 (authoritative)", "Tenants, users, memberships, API keys (hashes), customers, sites, assets, documents, chunks "
         "(text, headings), provenance, conversations, messages + structured contracts, idempotency records, tickets",
         "The only relational database, for runtime and tests (test_servy_dev / CI test database); "
         "settings refuse any other engine."],
        ["ChromaDB (local PersistentClient)", "BGE vectors per chunk with metadata stamps", "SERVY_CHROMA_PERSIST_DIR; derived "
         "from MySQL chunks and rebuildable; never authoritative for authorization"],
    ])

    d.add_heading("14. API endpoints and authentication", 1)
    d.add_paragraph("Authentication: Authorization: Bearer servy_<prefix>_<secret>. Keys are issued per integration account "
                    "with manage.py servy_api_key issue --username <account>; only the SHA-256 hash is stored; keys can expire "
                    "(--days) and are revoked with manage.py servy_api_key revoke --prefix <prefix>. The key acts as that "
                    "account, so the existing customer/tenant/asset rules apply. Bearer requests need no CSRF token. "
                    "Session cookies remain for the local React client.")
    table(d, ["Method", "Path (/api prefix)", "Purpose"], [
        ["GET/POST", "/customer-support/conversations/", "List / start a conversation for an asset"],
        ["GET/DELETE", "/customer-support/conversations/{uuid}/", "History and ticket / hide"],
        ["GET/POST", "/customer-support/conversations/{uuid}/messages/", "History / send message (synchronous)"],
        ["POST", "/customer-support/conversations/{uuid}/messages-stream/", "Send message (SSE stream)"],
        ["GET/PUT", "/customer-support/conversations/{uuid}/messages/{id}/feedback/", "Answer feedback"],
        ["GET/POST", "/customer-support/conversations/{uuid}/ticket/", "Ticket preview / explicit creation"],
        ["GET", "/customer-support/tickets/, /customer-support/tickets/{id}/", "Ticket list / detail"],
        ["GET", "/customer-support/source/?citation=...", "Open a cited source excerpt"],
        ["GET", "/health/live/, /health/ready/", "Probes (no authentication)"],
    ])

    d.add_heading("15. Request and response examples", 1)
    d.add_paragraph("See the API Integration Guide (docs/Servy_API_Integration_Guide.docx), whose examples are captured from a "
                    "real test run, and the OpenAPI file docs/api/servy-portal-api.openapi.yaml.")

    d.add_heading("16. Streaming and error handling", 1)
    bullets(d, [
        "messages-stream returns text/event-stream: a ': servy-rag-stream' comment, 'event: log' progress frames, "
        "': keep-alive' every 12 s and exactly one terminal 'event: result' (same JSON as the synchronous response) or "
        "'event: error'. Validation errors are plain JSON before the stream starts.",
        "Status codes: 400 invalid payload/closed conversation/idempotency collision/message over 4000 characters; 401 "
        "invalid, expired or revoked key; 403 no credentials or role not allowed; 404 unknown or other customer's resource (indistinguishable); 429 "
        "rate limit (per account: AI messages 20/minute, ticket creation 10/hour, conversation create/resolve 60/hour; "
        "reading history is not counted); 503 rag_busy or stream capacity.",
        "Idempotency: X-Idempotency-Key per message; a replay returns the stored turn (idempotent_replay=true).",
        "Client disconnect during a stream: the turn still completes and is saved (bounded by the request deadline); the "
        "single AI slot is held until it finishes. This is intended bounded-worker behavior; the portal can fetch the result "
        "with GET messages/ or retry with the same idempotency key.",
        "Unexpected errors return {'detail': 'Support request failed. Please retry.'} (500) without exception text.",
    ])

    d.add_heading("17. Security and tenant isolation", 1)
    bullets(d, [
        "Every query is scoped by tenant; customers see only their own assets, conversations and tickets; other customers' "
        "resources return 404.",
        "Original-document gate (verified-real mode): when the approved original file is missing, altered or revoked, "
        "the refusal is authoritative for the whole turn; the generic cause scan, document-fact lookup and "
        "explanations of previously listed causes cannot reintroduce technical claims.",
        "All retrieval paths (main answer, generic cause overview, explanations, follow-ups, clarification options) use the "
        "same candidate scope and per-chunk gate; confidential, sharing-disabled, provenance-revoked, quarantined and "
        "rag-disabled documents are excluded for customers.",
        "Customers never receive quarantined chunk text or original files containing quarantined content.",
        "Production settings: DEBUG off, secret key required, HTTPS redirect, HSTS, secure cookies, environment-driven "
        "ALLOWED_HOSTS / CORS / CSRF origins, TLS proxy header only when DJANGO_BEHIND_TLS_PROXY=1.",
        "API keys: hashed, expiring, revocable, uniform 401 message; never logged (gunicorn access log format excludes headers). "
        "Keys are issued only to customer-role accounts linked to one customer and stop working if that membership "
        "changes, so a portal key can never read more than one customer's data.",
        "Uploads: store roles cannot upload; documents a technician stages through the asset upload stay invisible to customers until approved "
        "(is_rag_enabled); customers also see their own pending uploads.",
        "Streaming disconnects: model stages not yet started are skipped (ClientDisconnected); the turn is still saved.",
    ])

    d.add_heading("Data visibility contract", 2)
    table(d, ["Data", "Customer (own only)", "Store roles", "Technician / manager / admin"], [
        ["Conversation messages and answers", "Yes", "No", "Assigned/authorized calls"],
        ["Ticket reference, status, priority, due dates", "Yes", "Reference/status only", "Yes"],
        ["Generated technician handover report (preview description/sections, technician_notes)",
         "No", "No", "Yes"],
        ["Recovery passport (internal diagnostic history)", "No", "No", "Yes"],
        ["Complaint text, contact details, customer name on calls", "Own calls", "No", "Yes"],
        ["Part requests on a call", "No", "Yes", "Yes"],
        ["Documents staged for approval by staff", "No (own uploads only)", "Cannot upload", "Yes"],
        ["Technician identity on calls; dashboard customer names and questions", "No", "No", "Yes"],
    ])
    d.add_paragraph("These rules are enforced in the Django views (not the UI) and covered by "
                    "core/tests_data_visibility.py. Whether store roles need any call data beyond reference, "
                    "status and equipment is a company policy decision; the default is deny.")

    d.add_heading("18. Production VM architecture", 1)
    d.add_paragraph("The company has specified a VM but has not yet provided its operating system, CPU/RAM, GPU, "
                    "network or hosting details, so nothing below assumes them. The backend needs one long-running WSGI "
                    "process (models are loaded once per process), a TLS reverse proxy in front of it, MySQL 8.0, a local "
                    "Ollama server bound to 127.0.0.1, and persistent directories for ChromaDB, media and model artifacts. "
                    "The repository provides Linux examples (deploy/gunicorn.conf.py, deploy/servy-api.service, "
                    "deploy/nginx-servy-api.conf); on Windows Server the same roles are filled by a WSGI host and IIS/ARR or "
                    "another TLS proxy, which must not buffer text/event-stream responses. No Docker or React is required.")
    d.add_heading("Items awaiting company confirmation", 2)
    bullets(d, [
        "Operating system and version of the VM; whether a Python 3.12 runtime may be installed.",
        "vCPU count, RAM and disk; whether a GPU is available (the code runs CPU-only; a GPU only affects Qwen speed in Ollama).",
        "MySQL 8.0 location (same VM or managed) and TLS requirements; backup policy.",
        "Public host name, TLS certificate source, and the portal's network path to the API (private network or internet).",
        "Whether the portal calls the API server-to-server (no CORS) or from browsers (CORS origins needed).",
        "How portal users map to Servy customer accounts (one API key per Servy customer account is implemented).",
        "Ticket ownership: Servy ServiceCall vs the portal's own ticket system (section 12).",
        "Outbound internet policy (runtime needs none once models are copied; HF_HUB_OFFLINE=1).",
        "Real approved manuals, V5 checkpoint and pinned hashes to install on the VM.",
    ])

    d.add_heading("19. Hardware and software prerequisites", 1)
    bullets(d, [
        "Software: Python 3.12, MySQL 8.0, Ollama, a TLS reverse proxy, and the packages in requirements.txt "
        "(gunicorn is installed on Linux only; onnxruntime for the V5 ONNX runtime).",
        "Model artifacts copied to the VM: BGE-small at the pinned revision, the V5 checkpoint (and verified ONNX export), "
        "and the configured Qwen model pulled into Ollama.",
        "Sizing must be measured on the company VM. Development measurements (i7-10870H CPU; RTX 3050 Ti used by Ollama): "
        "vector search ~0.05 s, V5 over 5 pairs 1.35-2.05 s, Qwen wording of 6 claims ~2.7-3.3 s. These are not VM figures.",
        "If the VM is CPU-only, expect slower Qwen wording; the per-stage deadline then falls back to exact source wording "
        "(reported as generation_status=timeout) rather than delaying the answer.",
    ])

    d.add_heading("20. Deployment procedure (runbook)", 1)
    for i, step in enumerate([
        "Confirm the items listed under 'Items awaiting company confirmation'.",
        "Create a dedicated service account and directories for code, virtual environment, data (chroma, media) and models; "
        "restrict the secrets file to that account.",
        "Check out the release commit; create a Python 3.12 virtual environment; pip install -r requirements.txt.",
        "Create the MySQL 8.0 database (utf8mb4) and a least-privilege user; fill the environment file from "
        "deploy/servy-api.env.example (DJANGO_DEBUG=0, real secret key, ALLOWED_HOSTS, SERVY_SERVE_FRONTEND=0).",
        "Copy model artifacts and the approved Chroma/media data to the data and model directories.",
        "Back up the database, run manage.py migrate (additive migrations 0011-0012 in this release) and "
        "manage.py check --deploy.",
        "Start Ollama bound to 127.0.0.1 and pull the configured Qwen model.",
        "Start the WSGI service and the TLS proxy (Linux: the provided systemd unit and nginx site).",
        "Wait for GET /api/health/ready/ = 200; then issue one key per Servy customer account with "
        "manage.py servy_api_key issue --username <customer account>.",
        "Smoke test with the key: create a conversation, send a message, preview a ticket; check contract.outcome reports "
        "retrieval_status=hybrid and reranker/generation statuses for real-model execution.",
        "Rollback: stop the service, check out the previous release, restore the database backup only if a migration was "
        "applied, restart.",
    ], 1):
        d.add_paragraph(f"{i}. {step}")

    d.add_heading("21. Testing and verification", 1)
    bullets(d, [
        "Backend: manage.py test core on MySQL 8.0 (CI service mysql:8.0; locally the isolated test_servy_dev database with "
        "--keepdb). Includes core.tests_portal_api (API keys, isolation, idempotency, tickets, health), "
        "core.tests_conversation_state (reply resolution, Split AC five-turn conversation through the real gateway, "
        "negations, ordinals, post-refusal authorization) and the existing VRF/red-team suites.",
        "Frontend (local demo only): node --test, eslint, vite build.",
        "Real-model and real-manual verification requires the private models and approved manuals on the target machine; "
        "synthetic fixtures do not prove real equipment accuracy.",
    ])

    d.add_heading("22. Known limitations and deployment blockers", 1)
    bullets(d, [
        "Deployment to the company VM has not been performed (no VM access or VM details yet).",
        "Store roles can read the tenant call register (/api/calls/, asset calls) for the existing UI; whether that is "
        "intended is a policy decision awaiting confirmation.",
        "Capacity: one AI turn runs at a time per process for all tenants (others get 503 rag_busy). Concurrency, "
        "latency (p95) and memory have not been measured on the company VM; sizing and any multi-process design must "
        "be decided once the VM resources and expected concurrent users are known.",
        "Compound cause bullets ('A, B, or C') are kept as one documented cause with its exact wording; they are not "
        "split into separate causes.",
        "One AI turn at a time per process (others get 503 rag_busy); scale out with more VMs.",
        "Ticket ownership between Servy and the evaluator's portal must be agreed (section 12).",
        "Citations expire after one hour; reopening an old conversation shows the stored answer, and a fresh citation needs a "
        "new answer.",
        "API keys are issued only to customer-role accounts and act for exactly one customer; a portal serving many "
        "customers holds one key per Servy customer account (a shared multi-customer key is deliberately not supported).",
        "Real equipment accuracy has been tested in depth for the Split AC and VRF demo assets only.",
    ])
    d.save(path)


def api_guide(path, examples):
    d = _doc("Servy API Integration Guide", "For integrating an existing customer portal with the Servy Django backend.")
    d.add_heading("1. Authentication", 1)
    d.add_paragraph("Send the API key on every request (placeholders shown, never real keys):")
    code(d, "Authorization: Bearer servy_<prefix>_<secret>\nContent-Type: application/json")
    bullets(d, ["Keys are issued by the Servy operator (manage.py servy_api_key issue) and shown once.",
                "Rotate by issuing a new key, switching the portal, then revoking the old prefix.",
                "An invalid/expired/revoked key returns 401 {\"detail\": \"Invalid or inactive API key.\"}; a request with no credentials returns 403."])
    d.add_heading("2. Typical flow", 1)
    for i, step in enumerate([
        "POST /api/customer-support/conversations/ {\"asset_id\": <id>} -> conversation_uuid.",
        "POST /api/customer-support/conversations/{uuid}/messages/ {\"message\": \"...\"} with X-Idempotency-Key "
        "-> render contract.presentation.response.segments (or assistant_message.message_text).",
        "Repeat step 2 for each customer message; Servy keeps the context server-side.",
        "Optional: GET .../ticket/ for a preview; POST .../ticket/ {\"confirm_ticket_creation\": true} only when the customer "
        "asks for a technician visit.",
        "GET .../conversations/{uuid}/ to reload history; GET /api/customer-support/tickets/{id}/ for ticket status.",
    ], 1):
        d.add_paragraph(f"{i}. {step}")
    d.add_heading("3. Rendering the answer", 1)
    bullets(d, [
        "contract.response_type tells the UI what happened (GREETING, CLARIFICATION, DIAGNOSTIC_GUIDANCE, "
        "INSUFFICIENT_EVIDENCE, TECHNICIAN_HANDOFF, ESCALATION_REQUIRED, POLICY_VIOLATION).",
        "presentation.response.segments: kind text/question (no citation) or list_item/claim (one citation id). Show "
        "[n] links from presentation.citations; open them with GET /api/customer-support/source/?citation=... within 1 hour.",
        "contract.outcome and contract.diagnostics are for engineers and monitoring, not customer text.",
        "A TECHNICIAN_HANDOFF answer is advice only: no ticket exists until the ticket POST succeeds.",
    ])
    d.add_heading("4. Errors and retries", 1)
    table(d, ["Status", "Meaning", "Portal action"], [
        ["400", "Invalid input, closed conversation, idempotency collision, message > 4000 chars", "Fix the request"],
        ["401", "Key invalid/expired/revoked", "Check key configuration"],
        ["403", "No credentials sent, or account role cannot do this", "Send the key; use a customer integration account"],
        ["404", "Unknown or not this customer's resource", "Do not retry"],
        ["429", "Rate limit", "Back off and retry"],
        ["503 rag_busy", "Another AI turn is running", "Retry after a few seconds with the same X-Idempotency-Key"],
        ["500", "Unexpected server error (no internal details)", "Retry with the same idempotency key; alert operator"],
    ])
    d.add_heading("5. Streaming (optional)", 1)
    d.add_paragraph("POST .../messages-stream/ with the same body returns text/event-stream. Read frames until the single "
                    "'event: result' (JSON identical to the synchronous response) or 'event: error'. Ignore comment lines "
                    "(': keep-alive'). If the connection drops, fetch GET .../messages/ - the turn is still saved.")
    d.add_heading("6. Verified examples", 1)
    if not examples:
        d.add_paragraph("No captured examples were supplied to this build.")
    for name, ex in (examples or {}).items():
        d.add_heading(name.replace("_", " ").capitalize(), 2)
        code(d, f"{ex['method']} {ex['path']}\nAuthorization: Bearer servy_<prefix>_<secret>")
        if ex.get("request") is not None:
            code(d, json.dumps(ex["request"], indent=2))
        body = json.dumps(ex.get("response"), indent=2, default=str)
        if len(body) > 3500:
            body = body[:3500] + "\n  ... (truncated)"
        code(d, f"HTTP {ex['status']}\n{body}")
    d.add_paragraph("Examples come from the automated integration test run on synthetic data (core/tests_portal_api.py).")
    d.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples")
    args = parser.parse_args()
    examples = json.loads(Path(args.examples).read_text(encoding="utf-8")) if args.examples else None
    OUT.mkdir(exist_ok=True)
    technical_document(OUT / "Servy_Technical_Functional_Document.docx")
    api_guide(OUT / "Servy_API_Integration_Guide.docx", examples)
    print("written", OUT / "Servy_Technical_Functional_Document.docx", OUT / "Servy_API_Integration_Guide.docx")
