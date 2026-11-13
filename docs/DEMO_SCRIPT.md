# Servy — Mentor Demo Walkthrough

## Start the two local demo instances

Run each command from the repository root in a separate PowerShell window.

**Customer RAG**

```powershell
.\tools\Start-ServyRealDemo.ps1
```

Open `http://127.0.0.1:5175/` and use the private access code produced by the launcher. Keep the terminal open to show live request and pipeline logs.

**Technician portal**

```powershell
.\tools\Start-ServyTechnicianDemo.ps1
```

Open `http://127.0.0.1:5176/` and use the separate private technician access code. Keep this terminal open to show technician API logs.

Do not put access codes, credentials or private log contents into GitHub or presentation screenshots.

## Demonstrate the customer RAG

1. Select the intended asset, such as VRF Outdoor Module (asset 60).
2. Ask a short fault question.
3. Show the answer, exact approved-manual citations and evidence limitations.
4. Continue with a follow-up observation to demonstrate conversation context.
5. Do not present an unsupported possibility as a confirmed diagnosis.

## Demonstrate the technician handoff

Create/confirm a ticket through the customer demo UI. Open the technician inbox and verify that the asset, complaint description and conversation context match the customer session.

The isolated customer/technician demo uses a dedicated local SQLite ticket workspace. The customer demo's company-data access is read-only; this local demo handoff does **not** write an official Servy MySQL `ServiceCall`.

## Show the Network tab

In Chrome, press `F12`, open **Network**, then select **Fetch/XHR**.

- Customer browser requests use the `/demo-api/` proxy to port 8002.
- Technician browser requests use the `/tech-api/` proxy to port 8003.
- Open a request and show its Request URL, method, status, response and Initiator.

Vite proxy routing is configured in `frontend/vite.config.js`.

## Show the application-layer API in GitHub

For the standard Django/DRF service-call workflow, open:

- `core/api_urls.py` — route registration.
- `core/api_views.py` — `CustomerConversationTicketView`, which creates a service call from an authorized conversation after explicit customer confirmation.
- `core/technician_inbox.py` — technician ticket detail and conversation transcript response.
- `core/models.py` — conversation and service-call relationships.

The application-layer route is:

```http
POST /api/customer-support/conversations/{conversation_uuid}/ticket/
```

The technician detail route is:

```http
GET /api/technician/inbox/{service_call_id}/
```

These standard DRF routes are separate from the `/demo-api/` and `/tech-api/` local demonstration proxies. Explain that distinction accurately.

## RAG implementation files

- `core/services/retriever.py`: retrieval and scope filtering.
- `core/ml/embeddings.py`: embedding profiles.
- `core/ml/reranker.py`: V5 reranking and checkpoint verification.
- `core/services/customer_answer.py`: evidence-grounded answer and citation validation.
- `core/services/llm.py`: generation integration and guarded fallback.
- `core/management/commands/check_rag_serving.py`: serving readiness checks.

Use the current CI result and current strict serving-check output as evidence. Archived reports are historical records, not substitutes for current verification.
