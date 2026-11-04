# Customer answer redesign — implementation and validation

## Scope and flow

The normal customer answer is now a structured contract, rather than a second
interpretation of the extractive generator's prose. Existing authorization,
retrieval thresholds, BGE-small hybrid retrieval and V5 weights/configuration
were preserved. No migration, dependency, production setting or company record
was changed. Earlier uncommitted research-tool changes were preserved.

Flow: authenticated request → tenant/customer/asset policy → existing retrieval
and reranking → existing evidence sufficiency → persisted document/chunk access
revalidation → bounded source-backed claims → customer action classification
and exact source-bound procedure approval → four-section contract → shared
React renderer. Hazard, clarification and insufficient-evidence responses are
exceptions to the ordinary four sections.

Source opening is a separate authenticated GET. Its signed, one-hour citation
binds the issuing user, tenant, customer, equipment, document, chunk, document
index version, full chunk hash and exact character span. Every open rechecks
current scope, sharing, confidentiality, quarantine, document availability,
equipment applicability and provenance. Procedure citations additionally
recheck approval. Errors disclose no inaccessible excerpt. Responses use
`Cache-Control: no-store, private`; the renderer always refetches.

## A. Actual files changed for this task

| File | Purpose |
| --- | --- |
| `core/services/customer_answer.py` | New source-backed contract, bounded claims, exact source spans, authorization and signed citations. |
| `core/customer_sources.py` | New authenticated, non-caching evidence endpoint. |
| `core/api_urls.py` | Evidence endpoint route. |
| `core/services/customer_diagnostic.py` | Replace ordinary string parsing with structured construction; preserve exception contracts. |
| `core/services/customer_conversation.py` | Persist the validated assistant presentation in customer messages. |
| `core/services/llm.py` | Distinguish reported elapsed symptoms from explicit numeric/time requirements. |
| `core/api_views.py` | Explicit ticket confirmation and current-contract resolution eligibility. |
| `core/diagnostics/api_views.py` | Explicit confirmation for diagnostic ticket requests. |
| `core/diagnostics/orchestrator.py` | A diagnostic escalation recommendation no longer creates a ticket automatically. |
| `frontend/src/components/CustomerAnswer.jsx` | Shared four-point renderer, immediate warning, source dialog, local inability notice, keyboard focus and stale-response handling. |
| `frontend/src/components/customerAnswerContract.js` | Fail-closed contract validation and removal of unused citation padding. |
| `frontend/src/components/CustomerAnswer.css` | Responsive typography, citation targets, focus indicators and modal layout. |
| `frontend/src/pages/CustomerSupportPage.jsx` | Adopt the contract, explicit workflow intent, hide ineligible resolution, discard stale site/asset responses. |
| `frontend/src/pages/AssetDetailPage.jsx` | Adopt customer renderer, preserve staff answer display, bind controls to equipment and distinguish recommendation from ticket creation. |
| `core/tests_customer_answer.py` | Numeric, formatting, citation, approval and no-write security regressions on synthetic fixtures. |
| `core/tests_customer_answer_redteam.py` | 1,000 distinct deterministic backend/rule cases across 25 categories. |
| `core/tests_customer_mutations.py` | Independent source approval survives classifier corruption; explicit synthetic ticket intent. |
| `core/tests_real_company_isolation.py` | Preserve isolation assertions; remove assumptions that generated prose authorizes actions. |
| `core/tests_customer_rag_endtoend.py` | Explicit ticket/resolution intent and source-backed non-invasive observation fixtures. |
| `core/tests_customer_acceptance_scenarios.py` | Preserve retrieval assertions while rejecting unapproved physical procedures. |
| `core/tests_diagnostics.py` | Prove diagnostic recommendations create no ticket/passport; explicit requests retain recovery passports. |
| `frontend/test/customerAnswerContract.test.mjs` | Malformed, oversized, misnumbered, fabricated-source and hazard-preservation checks. |
| `frontend/test/customer-answer.html` | Synthetic browser harness entry point. |
| `frontend/test/customer-answer-preview.jsx` | Synthetic fixtures for missing evidence, hazards, source failure, delay and equipment changes. |
| `tools/run_customer_answer_checks.py` | Reproducible in-memory SQLite/new isolated Chroma test runner and failure ledger. |
| `tools/servy_answer_simulation.mjs` | 100,000 deterministic static assignments; no model calls or human-usability claims. |
| `CUSTOMER_ANSWER_REDESIGN.md` | This implementation and validation report. |

## B. Before and after

The reported old presentation treated headings such as “LOW-PRESSURE OR
CAPACITY FAULT” and “Possible causes” as numbered troubleshooting actions.
The new component shows these four section titles, with individual source links:

1. **What could be wrong?** Possible documented cause: restricted airflow. [1]
2. **What can I safely check or report?** No physical procedure is approved. Report what you observed.
3. **What is still unknown?** This is not a confirmed diagnosis.
4. **What should I do next?** Ask a technician if the issue continues. No service ticket has been created.

This is the executed synthetic browser fixture, not a claim about a company
manual or a confirmed VRF diagnosis. The backend synthetic tests separately
validate source-backed cause sentences, observations and fault-code definitions.
Real source text and source metadata are never invented by the renderer.

## C. Security and safety findings

Confirmed and fixed:

- “Cannot perform a check” invoked escalation. It now changes local UI state only.
- Diagnostic branch transitions created ServiceCalls without a separate request.
  They now retain a terminal recommendation; a confirmed request creates the call
  and passport atomically and retains existing idempotency.
- Ticket endpoints accepted requests without explicit confirmation. All three
  customer-facing paths now require JSON `confirm_ticket_creation: true`.
- Resolution did not require the current answer's `can_resolve` and an explicit
  customer report. Both are enforced. A report does not become a verified repair.
- Ordinary prose parsing could promote headings or model-issued text into
  guidance. Claims now originate from scoped persisted source passages.
- A “check” word could disguise compound physical work as an observation. The
  new presentation requires non-invasive wording and rejects physical operations
  and internal access. Approval remains separate from document authenticity.
- Numeric enumeration cleanup could corrupt decimals. Enumeration now requires
  whitespace; decimal and signed values are preserved.
- An elapsed symptom duration was treated as a mandatory specification. Explicit
  timing requirements retain a time dimension and cannot match pressure units.
- Browser testing found Tab focus escaping the single-control source dialog.
  Explicit containment now passes forward/reverse Tab and Escape checks.
- Stale responses could update equipment presentation or its workflow controls.
  Source views remount on equipment/contract changes; page requests and controls
  now check equipment identity before applying results or acting.

No unauthorized company disclosure was observed. This is a bounded test result,
not a proof that all possible inputs or source documents are safe.

## D. Executed validation

| Validation | Designed/executed | Passed | Failed | Skipped/blocked |
| --- | --- | --- | --- | --- |
| Backend unittest executions | 176 / 176 | 176 | 0 | 0 / 0 |
| Distinct adversarial cases (included in backend batch) | 1,000 / 1,000 | 1,000 | 0 | 0 / 0 |
| Frontend Node test groups | 5 / 5 | 5 | 0 | 0 / 0 |
| Static synthetic assignments | 100,000 / 100,000 | 100,000 | 0 | 0 / 0 |
| Grouped browser interaction checks | 10 / 10 | 10 | 0 | 0 / 0 |

The live company customer journey was not executed; it is separate from these
synthetic counts. Build passed. Lint exited successfully with 41 warnings still
present, including existing hook/dependency warnings in the application pages.
No real user testing was conducted.

Report paths under ignored `private_training_data/customer_answer_validation/`:

- `validation_summary.json`: consolidated final results and execution limits.
- `backend_1793295996804250000.json`: final backend suite, 176/176.
- `redteam_1793295932544800000.json`: all 1,000 individual case IDs/categories.
- `simulation_20261029_03.json`: 100,000 assignments and dimension coverage.
- `browser_20261029.json`: observed interaction results and semantics/geometry.
- `failure_resolution_ledger.json`: prior recorded failures, fixes and final regression status.
- `frontend_build_final.log`, `frontend_lint_final.log`: actual tool output.

Earlier failed runs are retained. A 100,000-scenario run initially failed 3,197
assertions because it expected malformed ordinary guidance to suppress even an
independent urgent warning. The corrected invariant preserves the warning and
rejects ordinary claims; it is covered by the five Node test groups as well.

Reproduce from the repository root using installed dependencies:

```powershell
.venv\Scripts\python.exe tools/run_customer_answer_checks.py
node --test frontend/test/customerAnswerContract.test.mjs
node tools/servy_answer_simulation.mjs private_training_data/customer_answer_validation/simulation_new_run.json
Set-Location frontend
npm run build
npm run lint
```

Use a new simulation output filename: reports are never overwritten. The backend
runner uses only synthetic in-memory SQLite records, a new private Chroma
directory, offline model flags and a fast test-only password hasher; authentication
and authorization checks stay enabled. Existing cached models can be loaded.
Missing collections in this isolated index exercise sparse fallback rather than
reading company vectors. These runs do not establish live dense retrieval quality.

The 1,000 adversarial cases are a batch inside the backend suite, not 1,000
separately discovered unittest methods. The 100,000 profiles are static contract
assignments across experience, reading proficiency, question complexity, symptoms,
device, evidence, citations, safety, authorization and accessibility. They do not
measure comprehension, satisfaction, screen-reader usability or model accuracy.

Browser fixture: `http://localhost:5173/test/customer-answer.html`.
Observed checks cover desktop/mobile layout, four section headings, source labels,
exact synthetic excerpts, HTML escaping, inability notice, failure/refetch, delayed
responses after equipment changes, modal keyboard behavior, immediate warning
semantics and concise insufficient evidence. Controls make the checks repeatable.

## E. Citation verification

Synthetic backend tests verify exact excerpt containment and claim IDs, duplicate
references, wrong document/chunk IDs, user/customer/tenant mismatch, signature
tampering, expiration, index-version/content changes, withdrawal, sharing changes,
quarantine, confidentiality, deletion and unapproved procedures. The API rechecks
the valid signed token after permissions change. Fault definitions in a shared
chunk must match the queried code; required quantities must match the cited span.

The browser displayed a literal synthetic `<script>` string as text, with zero
script elements in the excerpt. Opening a previously successful citation after
simulated failure showed the error instead of the previous excerpt.

## F. Workflow safety

The 1,000-case batch checks unchanged counts of ServiceCall, CustomerInteraction,
CustomerConversation and DiagnosticSession around read-only contract construction.
Source opening also preserves those counts. The synthetic browser fixture makes
no operational API calls. Missing confirmation is rejected on all three ticket
paths; isolated legacy tests retain explicit ticket creation, deduplication,
authentic complaint notes, cross-customer rejection and customer-reported resolution.

Normal customer question POSTs still create their existing conversation and
diagnostic records. This redesign is not a new read-only internal tester portal.
No customer question or operational write was performed against company data.

## G. Known limitations and release scope

- No real customer account was provisioned or impersonated, and no company query
  was submitted. The live company customer journey remains unverified in this run.
- The original no-reload Django server was not restarted. The changed backend
  has been tested in isolation; a controlled application restart is still needed.
- Genuine company authorization scope and permission to persist test conversations
  must be established before live customer UI testing. No access policy was bypassed.
- No human participants, industrial expert approvals or physical screen-reader
  sessions were fabricated. Full WCAG 2.2 AA conformance is not established.
- Formatting and claim extraction are conservative English rules. Ambiguous,
  unsupported, conflicting or poorly structured evidence may require abstention
  and expert review. The suite does not prove general contradiction detection.
- Free-form QA cannot establish physical prerequisites. Procedures with such
  prerequisites remain withheld from this presentation; use an approved workflow.
  Physical procedures also require an exact source-bound approved description.
- Signed citations expire after one hour. Changed or old evidence may require a
  new authorized query. An already-open passage is not continuously monitored for
  permission changes; every subsequent opening revalidates.
- Existing unrelated frontend lint warnings remain. No production readiness,
  retrieval accuracy improvement or synthetic-profile usability rate is claimed.

## H. Release recommendation

NOT READY — BLOCKING ISSUES REMAIN

The code passed the listed isolated checks and is available for review. Live customer
release remains blocked on an authorized staging/customer session, controlled
backend restart and end-to-end evidence/safety review with the active company
collection. These were not replaced by simulation results.
