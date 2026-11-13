# Servy RAG — Current Architecture

This document distinguishes the standard Django/DRF application from the two isolated local demo services. They are related workflows, but they are not the same API or ticket database.

## Runtime entry points

```text
Browser
  +-- Standard React/Vite application
  |      /api/*       --> Django REST Framework (normally 127.0.0.1:8000)
  |                       core/api_urls.py
  +-- Authorized customer RAG demo
  |      /demo-api/*  --> tools/servy_real_demo.py (127.0.0.1:8002)
  |                       read-only company-data access; local ticket workspace
  +-- Technician demo
         /tech-api/* --> tools/servy_technician_demo.py (127.0.0.1:8003)
                          dedicated local ticket workspace
```

Vite routing is configured in `frontend/vite.config.js`. The two demo launchers are `tools/Start-ServyRealDemo.ps1` and `tools/Start-ServyTechnicianDemo.ps1`.

## Standard Django/DRF application

- URL registration: `core/api_urls.py`
- Main API views: `core/api_views.py`
- Technician inbox API: `core/technician_inbox.py`
- Data models: `core/models.py`
- Django settings: `servy_rag/settings.py`

### Conversation-to-service-call handoff

The standard DRF route is:

```http
POST /api/customer-support/conversations/{conversation_uuid}/ticket/
```

It is implemented by `CustomerConversationTicketView` in `core/api_views.py`. It requires an authenticated customer and explicit `confirm_ticket_creation: true`, validates conversation/customer/asset scope, returns an already-linked call idempotently, and creates a Servy `ServiceCall` linked to that conversation.

The standard technician detail route is:

```http
GET /api/technician/inbox/{service_call_id}/
```

It is implemented in `core/technician_inbox.py`. The response includes authorized ticket details, the linked customer/assistant transcript, supporting document references and outstanding clarification information.

These application-layer routes are distinct from the isolated local demo routes below.

## Customer RAG pipeline

1. Establish authenticated tenant, customer and asset context.
2. Apply authorization and asset/customer scope before evidence is used.
3. Retrieve candidate passages using the sparse and configured dense retrieval paths.
4. Apply the configured reranking stage. Strict/required V5 serving verifies the configured local checkpoint against its expected SHA-256.
5. Build the customer answer and validate source support/citations.
6. Use the generation integration only within the guarded evidence-grounded answer path; invalid or unavailable generated wording must retain the source-grounded fallback.

Key files:
- Retrieval: `core/services/retriever.py`
- Embeddings/profiles: `core/ml/embeddings.py`
- Dense indexing: `core/ml/indexer.py`
- V5 reranking: `core/ml/reranker.py`
- Answer construction/evidence validation: `core/services/customer_answer.py`
- Generation integration: `core/services/llm.py`
- Strict serving diagnostic: `core/management/commands/check_rag_serving.py`

Legacy MiniLM/Chroma functionality remains for compatibility and ingestion paths. It is not interchangeable with the configured BGE serving profile; do not remove it without verifying all ingestion and indexing dependencies.

## Isolated local demonstration workflow

- Customer: `tools/Start-ServyRealDemo.ps1` starts the customer demo on port 8002 and Vite on port 5175.
- Technician: `tools/Start-ServyTechnicianDemo.ps1` starts the technician demo on port 8003 and Vite on port 5176.

The customer real-data demo uses read-only company-data access. Its local ticket handoff is stored in a dedicated local SQLite workspace. The technician demo reads that workspace and does not connect to MySQL or write official Servy `ServiceCall` records.

**Do not describe the 8002/8003 local demonstration handoff as an official MySQL service-call creation.** Use the standard Django/DRF conversation-ticket endpoint above for the Servy application-layer service-call flow.

## Verification

- Strict serving diagnostics: `python manage.py check_rag_serving --strict`
- Backend tests: current CI workflow at `.github/workflows/ci.yml`
- Three-asset regressions: current tests/evaluation tools under `core/` and `tools/`
- Demo readiness: use the two PowerShell launchers

Archived reports are historical records. Their old test counts and metrics do not prove the current commit passes; verify current CI and local strict serving results.
