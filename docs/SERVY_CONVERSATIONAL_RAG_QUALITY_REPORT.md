# Conversational RAG and dual-API quality engineering — report

Date: 2026-11-14. Local development only (`servy_dev`, `test_servy_dev`, isolated
development Chroma). Real BGE-small, hybrid retrieval, V5 and Qwen
`qwen3.5:2b-q4_K_M` (digest `1e2d21a4…`). Nothing committed, pushed or deployed.

## Root causes found

| Symptom | Root cause |
|---|---|
| Answers read like rigid document reports | The answer was a **fixed four-section template** (`TITLES` in `customer_answer.py`) joined into the message text. Qwen only reworded individual claims, so it could not shape the reply. |
| "Explain the second cause" still listed every cause | The stored `contract["answer"]` was assembled before the cause focus was applied. A new retrieval for the cause also added unrelated causes from other sections. |
| "Sorry, I meant ice, not water" answered about water | Corrections were not recognised. Symptoms only accumulated, and "water" in the message still matched leakage claims. |
| "What about a communication fault instead?" merged with the previous alarm | "instead" was treated as a back-reference, so the old topic was merged in. |
| Multi-part "what causes X and how often should Y…" dropped the schedule | Any cause question set `wants_facts = False`. |
| "The module shows a high-pressure alarm" returned overview trivia | A symptom report without the word "cause" was classed as a fact question, and the facts-over-causes rule then removed the documented cause. |
| An ambiguous "It is not working properly" got a flat refusal | No clarification path existed. |
| Five concurrent confirmations all returned `201 Created` | Duplicates were judged from a pre-lock check. (The database already prevented duplicate calls: only one was created.) |
| The technician description differed from the reviewed preview | Creation wrapped the reviewed text in a header, the confirmation note, a duplicate source list and a repeat of the latest answer. |
| The ticket's "questions asked" would capture a whole reply | The question fallback split the message on "?" only. |

## Changes

**API 1, conversational RAG**
- **`core/services/customer_response_composer.py` (new).** Composes a natural, cited reply only from claims that are already validated. Every technical segment is an unchanged validated claim with exactly one citation, and connective text is fixed wording chosen by intent:
  - a symptom gets its cause list;
  - a fact or schedule gets a direct statement;
  - an ordinal reference gets only the selected cause;
  - a subject follow-up ("What about the filters?") gets only claims on that subject;
  - other follow-ups don't repeat the previous turn's claims;
  - the limitation line is not repeated on consecutive turns.

  The role distinction is explicit: only `customer_procedure` claims appear as "Checks the manual approves for customers". Observations appear as "Details worth noting for the technician".
- **Ambiguous messages** get a clarifying question that offers only the asset's own documented symptom section titles, under the same customer-visibility filter.
- **`conversation_context.topic_shift_query`** answers corrections and explicit topic changes for the new subject, removing a trailing negated clause. The stored customer message is unchanged.
- **The ordinal focus** also rebuilds the stored answer text.
- **`customer_answer.py`:**
  - symptom reports are treated as cause requests;
  - frequency questions keep their facts, so multi-part questions work;
  - the facts-over-causes rule no longer fires for cause questions or symptom reports.
- **The fixed `sections` remain in the contract** for older clients, the ticket builder and evaluations. The new `presentation.response` is additive.

**API 2, conversation to ServiceCall**
- The official ticket view returns `created` (true/false) and an accurate detail message, and the gateway uses it.
- The stored technician description is **exactly the reviewed preview**, plus at most a labelled "Customer note at confirmation" line (`ticketing.py`).
- `conversation_ticket._questions_from` reads the composed reply's question segment.

**Frontend**
- `customerAnswerContract.validatedResponse` checks every segment: technical segments must cite one evidence ID already used by a validated claim, connective segments must cite nothing, and length and kind are limited.
- `CustomerAnswer.jsx` renders the composed reply, falling back to the sections.

**Tools**
- `tools/servy_multiturn_eval_v1.json` with `servy_multiturn_eval.py`: a versioned multi-turn evaluation.
- `tools/servy_api_acceptance.py`: HTTP acceptance checks for API 1 and API 2.
- `tools/servy_retrieval_experiment.py`: the embedding-text experiment.
- `tools/servy_latency_profile.py`: per-stage latency.
- `tools/servy_three_asset_gateway_eval.py` now also records per-asset Qwen activity.

## Before / after measurements (real models, HTTP)

**Multi-turn evaluation v1** (8 scenarios, 18 turns):

| Metric | Baseline | Final |
|---|---:|---:|
| Turns passed | 10/18 | **18/18** |
| Turns showing the fixed report headings | 15/18 | **0/18** |
| Mean reply length (words) | ~118 | **45** |
| Ordinal-reference turns correct | 0/3 | 3/3 |
| Correction / topic change / multi-part / ambiguous | 0/4 | 4/4 |
| Repeated previous-turn evidence (mean share) | — | 0.32 |
| Cross-asset leaks / verbatim failures | 0 / 0 | 0 / 0 |

**Protected 67-turn evaluation:** 67/67 before and after (17/17, 12/12, 17/17 documented; 5/5, 6/6, 4/4 refused). Zero leaks, zero citation-open failures, zero verbatim failures.

**API acceptance (18 checks):** 16/18 at first, with the concurrency reporting and the preview-to-created mismatch failing. Final: **18/18**. The live workflow is 26/26.

**Qwen claim validation** is unchanged by this work (the previous fix stands): 84/97 claims accepted on identical outputs, with all 13 rejections being correct rejections; every documented-answer turn invoked Qwen. Measured false-rejection and false-acceptance rates are based on manual review of those 97 claims only:
- false rejections: 0 found among the 13 rejected;
- false acceptances: 0 found among a reviewed sample;
- the sample is small, so this is not a proven bound.

## Experiments rejected (previous implementation kept)

| Experiment | Result | Decision |
|---|---|---|
| Qwen writes the acknowledgement and clarifying question, under strict validation | Same 18/18 and 67/67. Acknowledgements only restated the customer; 0 questions passed validation. Mean latency **8.5 s → 21.5 s**. | Rejected; code archived outside the repository |
| Contextualized embeddings (title + heading + body) vs current (heading + body), 40 queries | hit@1 0.875 → 0.900, MRR 0.933 → 0.942 (one query), hit@3 1.0 for both | Not significant; the current serving format is kept. The end-to-end pipeline already finds every expected section. |

## Performance (median / p90, 9 turns, no other load)

| Measure | Median | p90 |
|---|---|---|
| End-to-end | 4.0 s | 5.5 s |
| First streamed event | under 60 ms | — |
| BGE dense retrieval | 27 ms | — |
| Sparse retrieval | 5 ms | — |
| V5 reranking | **1.8 s** | 2.8 s |
| Qwen claim wording | **2.4 s** | 3.3 s |

- **Memory:** the customer gateway uses about 2.1 GB of RAM and the technician gateway 180 MB; Qwen uses about 2.3 of 4.1 GB VRAM (RTX 3050 Ti).
- **Composer cost:** no model call.
- **Database connections:** return to 0 after the full suite, so the leak fix holds.

## API contracts (Customer gateway :8002, Technician :8003)

**Customer:**
- `POST demo-api/customer-support/conversations/` (`{asset_id}`) returns 201 `{conversation_uuid, …}`.
- `POST …/{uuid}/messages/` and `…/messages-stream/` (SSE `log` … `result`) both take `{message}` with an optional `X-Idempotency-Key`. The response is `{user_message, assistant_message, contract}`. `contract.presentation.response` (new, additive) holds `{text, segments[{kind: text|claim|list_item|question, text, citation_ids}], style, composer}`.
- `GET …/conversations/`, `GET …/{uuid}/` list and reopen conversations.
- `GET …/{uuid}/ticket/` is a read-only preview: `{preview: {sections (9), description, message_count, missing_fields, can_create}}`.
- `POST …/{uuid}/ticket/` takes `confirm_ticket_creation=true`, the form fields and 0–10 attachments. It returns 201 when created and 200 for a duplicate, with `{ticket, created, duplicate, detail}`.

**Technician:** `tech-api/technician/inbox/`, `…/{id}/`, `…/{id}/update/`, `…/{id}/attachments/{aid}/` (unchanged).

## Remaining limitations

- **Qwen's role is still claim-level wording.** The reply structure is fixed connective text chosen by intent, because Qwen-written connective text gave no measured gain at 2.5× latency on this 2B model.
- **The multi-turn set is small** (18 turns over 3 assets), and naturalness is measured by proxies: headings, length, repetition and questions. There is no human-rated sample yet, and no LLM judge was used.
- **Decomposition is selective:** cause plus schedule questions work. Arbitrary multi-section comparisons are not decomposed.
- **History handling:** conversation context uses the last 10 messages plus the first complaint. Long-conversation summarization was not implemented. The ticket description always uses the full transcript.
- **Browser acceptance** was verified through the real HTTP routes and by loading both UIs without errors. Logging in with the access codes was not automated, per your earlier instruction.

## Final regression (final code)

| Suite | Result |
|---|---|
| Backend, MySQL `test_servy_dev` | **894 OK**, 2 skipped (SQLite-mode mirrors), exit 0 |
| Backend, SQLite in-memory | **894 OK**, 2 skipped (MySQL-only mirrors, which pass in the MySQL run), exit 0 |
| CI tool tests | 321 OK, 1 skipped (Windows symlink privilege), exit 0 |
| Frontend | 76/76; lint exit 0 (48 pre-existing warnings, none added); build OK |
| Production Chroma | 0 mismatches against the baseline |
| Git | Nothing staged; both stashes intact |
