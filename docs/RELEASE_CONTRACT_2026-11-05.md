# Servy customer support release contract (frozen baseline: 2026-11-05)

Scope: existing `yashaskn8/prototype` customer support app, including the SELECT-only Windows demo. This is an **initial mandatory contract**, not evidence of completion. PASS requires recorded execution evidence; `NOT RUN` is not PASS. No live-company write is authorized by this file.

| ID | Customer outcome | Review owner | Affected components | Acceptance test / evidence required | Evidence | Status |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | Direct answers fit the question, not four canned sections | Conversation | dialogue, customer-answer contract, UI | Real Split AC causes browser trace + independent citation review | Historical audit only; new release not tested | NOT RUN |
| R02 | Vague reports receive one useful clarification | Conversation / Evaluation | dialogue, demo API | Automated first-turn and multi-turn tests, then browser trace | Not run for this release | NOT RUN |
| R03 | Corrections, prior checks and active asset persist without contaminating new issues | Conversation / Evaluation | signed continuation, retrieval, UI | Multi-turn state, asset change, correction and topic-switch cases | Not run for this release | NOT RUN |
| R04 | Each technical claim is supported by an accessible approved passage | Knowledge / Security | ingestion, retrieval, citations | Held-out claim-to-page review with real approved manuals | Requires approved Windows corpus | BLOCKED |
| R05 | Identity-derived tenant and document permissions prevent disclosure, including after revocation | Security | auth, source viewer, retrieval, cache | Authenticated cross-tenant and revoked grant adversarial suites | Not run for this release | NOT RUN |
| R06 | Approved source lifecycle is versioned, revocable and recoverable | Knowledge | parsing, derived index, jobs | Idempotence, corrupt extraction, rollback and revocation tests | Not run for this release | NOT RUN |
| R07 | Exact codes, table units, variants and retrieval quality pass frozen benchmarks | Knowledge / Evaluation | hybrid ranking, reranking | Frozen per-slice dataset, recall and citation metrics | Not run for this release | NOT RUN |
| R08 | Safe response validation precedes display and failures are distinguished | Backend / Security | answer contract, streaming | Injection, unsafe procedure, provider outage, parser failure and citation mismatch tests | Not run for this release | NOT RUN |
| R09 | Ticket review and explicit confirmation bind actor, asset and fields | Backend / Security | authorized ticket API and UI | Editable preview and denial of unconfirmed writes in isolated environment | Read-only demo cannot submit | BLOCKED |
| R10 | Submitted tickets are idempotent and uncertain outcomes reconciled | Backend / Evaluation | ticket API, storage | Fault injection after commit, refresh/retry/reconnect duplicate tests | Authorized nonproduction write API not provided | BLOCKED |
| R11 | Customer UI remains accessible, responsive and provides safe retry/cancel | Frontend / Evaluation | React app | Keyboard/mobile/browser/a11y and cancellation suite | Not run for this release | NOT RUN |
| R12 | Backend protects requests and behaves under load and restart | Backend / Security | API, concurrency, configuration | Auth, CSRF/CORS, rate/size, load, recovery, logs tests | Not run for this release | NOT RUN |
| R13 | Reproducible independent signoff, documentation and deploy/rollback exist | Evaluation / Architecture | CI, runbooks | CI artifact, independent review, load, recovery and migration evidence | Not run for this release | NOT RUN |
| R14 | Real authorized frontend and backend verified together | Evaluation | Windows demo / staging | Browser + API traces with original source opening | Local Windows environment unavailable here | BLOCKED |
| R15 | Production deployment and rollback verified after authorization | Operations | deployment | Deployment record, health, rollback rehearsal | No production deployment authorization | BLOCKED |

## Gate policy

Mandatory total: **15**. No requirement passes from code existence, screenshot, mock, or self-assessment alone. Release status is **BLOCKED** until all 15 have independent evidence-backed PASS results. The corpus, model, index, configuration and commit must be pinned in each release candidate record. No fabricated facts or benchmark-label ingestion. The user requested direct `main` changes; branch protections and unrelated edits remain authoritative.

## Review separation

A: research and architecture (design decisions, official sources); B: retrieval and source lifecycle; C: conversation and UI; D: backend/actions; E: independent security red team; F: independent evaluation/reliability. These are **review roles**, not claims that six separate sub-agents were available. An implementer's own tests cannot constitute independent signoff.

## Baseline limitations

The repository's README identifies itself as a standalone prototype, not the private production Servy repository. The Windows demo launcher explicitly uses a local SELECT-only database identity; the real-data React demo confirms ticket details **locally only**. The provided screenshot URL `127.0.0.1:5175` is only reachable on its host. Browser and real-manual tests cannot be inferred from historical reports.

## Freeze / amendments

This initial matrix is the minimum release contract. Any later added discovery requirement remains mandatory; do not weaken thresholds, change denominator, or label blocked dependencies PASS. Thresholds for probabilistic scoring must be declared before executing the held-out suite. Zero unauthorized disclosure, unsafe customer instructions, unauthorized writes, unconfirmed ticket writes, or duplicate committed tickets in the declared security/action suite.
