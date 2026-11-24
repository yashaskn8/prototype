# Servy — backend integration and handover

**Scope:** Conversational RAG, conversation-to-ServiceCall conversion, and backend-only deployment into the existing company portal. The React client and access-code gateways are development demonstrations, not production integration dependencies.

**Repository:** [prototype](https://github.com/yashaskn8/prototype/tree/main). Runtime reviewed at `3c1dc3b01b2c`. Deploy only a revision approved by the release owner after the acceptance checks below.

## 1. Conversational RAG APIs

Base URL: `https://<company-api-host>/api/`. The existing portal's server calls Servy using a scoped integration-user API key over HTTPS. Keep the bearer key on the portal server, not in browser JavaScript. Each request is authorized against the user's tenant, customer, and asset; the portal must preserve that identity mapping.

| Method | Path (relative to /api/) | Purpose |
|---|---|---|
| GET | customer-support/conversations/ | List conversations; optional asset filter |
| POST | customer-support/conversations/ | Create conversation; optional first message |
| GET | customer-support/conversations/{uuid}/ | Retrieve conversation and history |
| GET / POST | customer-support/conversations/{uuid}/messages/ | Load history / send a turn |
| POST | customer-support/conversations/{uuid}/messages-stream/ | Stream progress and final result |
| DELETE | customer-support/conversations/{uuid}/ | Soft-delete customer conversation |
| GET | customer-support/source/ | Open signed citation under the caller's scope |

Create: `{"asset_id": 123, "message": "My AC is not cooling."}` (123 is a placeholder for an authorized asset ID). Store the returned `conversation_uuid`. Send subsequent turns as `{"message": "The airflow is also weak."}`. Use a unique `X-Idempotency-Key` per logical turn and reuse it for retries of that same payload. Render `contract.presentation.response.segments` with matching citation IDs; for conversation-only replies use the contract message/answer. Do not display raw execution logs, unused passages or internal diagnostics to customers.

The SSE endpoint sends progress events and a terminal result or error. On interruption, reload history before retrying. Handle 400 (invalid request), 401/403 (authentication/permission), 404 (unavailable resource), 429 (throttling), and 503 (busy/not ready) without creating another logical turn.

**Workflow:** Django/DRF authorization → conversation context and technical turn plan → asset-scoped BGE/Chroma dense and lexical retrieval → rank fusion → V5 reranking → authorized source-span claim selection → optional validated local Qwen wording → response composition and final structural/intent validation for the technical serving path → MySQL message persistence.

The approved documents are the technical authority. Qwen cannot introduce technical facts; rejected or unavailable model output falls back to cited source wording. If a cause has no supporting explanation, report that limitation rather than inventing details. The experimental semantic intent selector remains disabled by default.

## 2. Conversation-to-ServiceCall

| Method | Path (relative to /api/) | Purpose |
|---|---|---|
| GET | customer-support/conversations/{uuid}/ticket/ | Read-only preview or linked ticket |
| POST | customer-support/conversations/{uuid}/ticket/ | Confirm and create ServiceCall |
| GET | customer-support/tickets/ | Customer ticket list |
| GET | customer-support/tickets/{id}/ | Ticket detail |
| GET | technician/inbox/ | Authorized technician's inbox |
| GET | technician/inbox/{id}/ | Technician handover and context |
| POST | technician/inbox/{id}/acknowledge/ | Acknowledge assigned ticket |

POST body: `{"confirm_ticket_creation": true}` (optional `user_observation`). Preview exposes complaint, message count, missing contact fields and creation eligibility to the customer; detailed description/sections are staff-only. A ServiceCall is linked to the conversation under a database lock; retries return the existing call rather than creating another. The handover distinguishes customer reports, absent/withdrawn symptoms, assistant guidance, cited findings and unconfirmed work. A question is not a confirmed fault. Contact details come from the authorized customer record. No diagnosis is implied.

## 3. VM deployment

Linux VM assumption; confirm OS, available CPU/RAM/GPU, hostname, TLS certificates, database ownership, service user and portal identity mapping with the company's operations team. Do not run local demo seeding or deploy against production before authorization.

1. Check out the approved commit in `/opt/servy/prototype`; create the Python 3.11 environment at `/opt/servy/venv`; install `requirements.txt`.
2. Provision MySQL 8.0, persistent media and Chroma storage, approved source documents and provenance, pinned BGE/V5 artifacts and the local Ollama Qwen model. Verify original-file checksums and index compatibility before serving.
3. Populate `/etc/servy/servy-api.env` using [the environment template](../deploy/servy-api.env.example); keep secrets outside Git, restrict file access, set `DJANGO_DEBUG=0`, `SERVY_SERVE_FRONTEND=0`, `SERVY_CUSTOMER_DATA_MODE=verified_real_only` and disable remote models. Replace every placeholder. Use the company's authorized database and credentials, not the demo database.
4. With the same provisioned environment, execute the following on a staging copy before production. Take a verified backup before migrations.

```bash
cd /opt/servy/prototype
/opt/servy/venv/bin/python manage.py check --deploy
/opt/servy/venv/bin/python manage.py migrate --plan
/opt/servy/venv/bin/python manage.py migrate
/opt/servy/venv/bin/python manage.py check_rag_serving --strict
```

5. Install [systemd](../deploy/servy-api.service), [Gunicorn](../deploy/gunicorn.conf.py) and [Nginx](../deploy/nginx-servy-api.conf) configurations after adapting paths, TLS and permissions. Run one Gunicorn worker with threads; only one active RAG turn per process. Keep Ollama on loopback. Start the service; require HTTP 200 from `/api/health/live/` and `/api/health/ready/` before routing traffic.
6. Issue an API key to an existing authorized non-superuser integration account using `manage.py servy_api_key issue --username <integration-user> --name "Company portal"`. Transfer the key through the company's secret-management channel. Integrate the portal using the [OpenAPI contract](api/servy-portal-api.openapi.yaml).
7. Monitor readiness, latency, failures and fallback rates; retain the previous approved release plus verified MySQL/media/Chroma backup and model identities. Rehearse rollback on staging; coordinate any database restore with the DBA to avoid losing later writes.

## 4. Acceptance before release

- Full CI green for the deployed revision; no skipped release-critical tests.
- Verify all three assets with real models and originals: accurate cited answers, follow-ups, corrections, honest gaps, no unauthorized sources or unsupported claims.
- Verify conversation creation, streaming, history, deletion and retry; verify ticket preview, confirmation, duplicate prevention and technician visibility with isolated accounts.
- Verify VM readiness, actual portal identity mapping, end-to-end latency/capacity, backup and rollback. Obtain owner sign-off.

At preparation time, the [fix run](https://github.com/yashaskn8/prototype/actions/runs/45360046371) had passed the focused red-team, isolation and RAG end-to-end stages; the full MySQL suite was still pending. [Historical real-model acceptance](SERVY_ACCEPTANCE_ROUND_20261124.md) did not meet its accuracy gates. These are not production deployment or final acceptance evidence.

**Additional reference:** [OpenAPI](api/servy-portal-api.openapi.yaml) · [Historical API examples](api/verified-examples.json) · [Operations and recovery](PRODUCTION_READINESS.md).
