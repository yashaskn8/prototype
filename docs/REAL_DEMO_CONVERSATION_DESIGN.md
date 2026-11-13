# Ask Servy real-data RAG: grounded conversational responses

## Why the old demo repeated four answers

The shared Django `build_customer_answer` generated four fixed sections, and
`customerAnswerContract.js` validated *only* four sections. Even a question
that asked solely for documented possible causes therefore displayed causes,
checks, unknowns and next steps. A real conversation needs **answer relevance
and state**, not just an ever-growing array of chat bubbles.

## Design principles borrowed from current official documentation

- OpenAI's File Search and Retrieval guides discuss semantic + keyword
  search, filtering, query rewriting, ranking, and cited output:
  https://developers.openai.com/api/docs/guides/retrieval
  https://developers.openai.com/api/docs/guides/tools-file-search
- Anthropic's citation documentation emphasizes source-bound spans and
  claims attached to the exact supporting passages:
  https://platform.claude.com/docs/en/build-with-claude/citations
- Google's Gemini File Search describes chunking, indexing and grounding
  responses in retrieved documents with citation annotations:
  https://ai.google.dev/gemini-api/docs/file-search

These are **design patterns**, not claims that Servy runs GPT, Claude or
Gemini. The real-data demo uses only a localhost Ollama model when enabled;
it never sends customer text or approved source passages to a remote model.

## Implemented layers

1. **Authenticated equipment scope**: the selected asset/customer is checked
   before retrieval. Indexed documents must be positively verified and
   customer-authorized, and citations reauthorize the exact source on open.
2. **Contextual retrieval**: signed, asset-bound continuation state carries
   bounded original/recent customer reports, recognized symptoms, and
   attributable completed checks. Hybrid BGE-small + sparse retrieval and
   V5 reranking remain in the existing services.
3. **Source evidence contract**: `build_customer_answer` creates only
   citations to approved exact source spans. Policy, safety, model and
   procedure-approval gates remain authoritative.
4. **Intent-aware focused answer**: the read-only demo selects **only**
   relevant evidence for the current intent (possible causes, observations,
   safe next steps, fault-code meaning or document facts). A cause question
   produces one cited causes section rather than four unrelated questions.
   Fault-code source sentences must match the user's actual code.
5. **Conversation state and no-repeat handling**: a bounded set of hashes
   of previously shown source claims and previously asked follow-up questions
   is signed with the selected asset, customer and chat thread. `What other
   causes?` shows only additional supported claims and clearly acknowledges
   when none remain. Explanation requests do not trigger another generic
   diagnostic question.
6. **Missing detail and graceful refusal**: a vague or ambiguous reply like
   `I checked that` asks which check and observed outcome; a bare `yes`
   requests the displayed code. No completed check is invented. No technical
   answer is emitted when matching evidence is unavailable.
7. **Plain-language explanation**: for an explicit request to explain cited
   causes, the optional localhost Ollama model receives only the selected
   source sentences and equipment identity. It must return one explanation
   per existing claim plus an exact copy of the source sentence. Server
   validation rejects changed quotes, new numbers/codes, newly introduced
   equipment terms, repair/action language and diagnostic certainty. The
   generated explanation is displayed separately from the original cited
   manual statement. If generation is unavailable or fails validation, Servy
   keeps the extractive answer and its evidence limitation; it does not invent
   detail.
8. **Conversation UI**: source-marked answer paragraphs, a focused headline,
   at most one relevant customer observation question, context-sensitive
   suggested follow-up chips, New chat, and read-only ticket preview.

## Security invariants

The focused presentation's technical claims are **views of already verified
source claims**. Optional model-authored explanations are separate paraphrases
attached to those existing claim IDs; they cannot add claim IDs, citations,
procedures or diagnoses. Every displayed explanation remains beside its
original manual statement and its permitted citation. A missing,
invalid, expired, off-asset or out-of-scope citation cannot be fabricated
by the dialogue module. A human-readable question is not itself a manual
claim. No customer physical procedure can appear without prior procedure
approval in the underlying customer answer service. Customer responses,
claim history and follow-up suggestions remain in a signed continuation
and browser memory, never persisted to operational company records.

The new `view_mode: focused` is accepted only for a single cited section
under a small allowlist of headings. The main customer portal retains
the previous four-section presentation contract, so the demo upgrade
does not silently rewrite other customer workflows.

## Test / manual acceptance checks

Run `python -m unittest tools.test_servy_real_demo_dialogue -v`,
`python -m unittest tools.test_servy_real_demo_explainer -v`, and
`node --test frontend/test/customerAnswerContract.test.mjs`.
GitHub CI also runs the full real-data demo security tests and frontend build.

On the Real-data demo at `http://127.0.0.1:5175/` check:

- “My Split AC is running but not cooling. What possible causes are
  documented?” → a **single causes section**, with approved manual citations.
- “What can I safely observe?” → only customer-safe observations, no
  guessed physical procedure.
- “What does E-04 indicate?” → only evidence for E-04, not E-05.
- “What other causes are there?” → new cited claims only, or a clear
  no-additional-evidence response.
- “Could you explain each issue in detail?” → one plain-language explanation
  per source-backed cause, each shown with its exact manual statement and
  citation; no repeated diagnostic question.
- Repeating a follow-up question → choose a different unanswered question or
  omit the question when no useful one remains.
- “I checked that” → asks which check and its outcome; does not assume.
- “No error code” → seeks relevant next safe observations, not a repeat
  of the original four questions.
- A request to bypass a safety rule or obtain an unsupported precise
  setting still fails closed.

The test fixtures are synthetic. Real-company document coverage and
accuracy still require a live, authorized Windows demonstration, including
manual inspection of citations and unsupported-answer refusals.


## Asset-family regression scope

The local explanation contract is asset-agnostic and accepts only the claims
already returned by the selected asset's authorized retrieval path. Regression
fixtures cover Split AC Unit 2T, VRF Outdoor Module and Ductable AC 5T. Existing
retrieval regressions also exercise the three approved asset IDs (59, 60 and
61). These fixtures prove that the code path is shared and scoped; they do not
prove that every real manual contains enough detail for every requested
explanation. Each real source citation still requires a live authorized review.


## Complete approved-document review and supplementary web research

The customer demo performs the normal dense/hybrid ranked retrieval and a
separate exhaustive sparse pass across every customer-visible,
provenance-verified, searchable chunk in the selected asset's canonical
asset/customer scope. The scan checks the current issue, signed original
complaint, and previously cited cause phrases. Matching passages are
deduplicated and kept document-diverse. The normal customer answer builder
still reauthorizes each source chunk before a claim or citation is shown.

The UI reports how many approved documents and passages were searched. If
an approved document has no searchable chunks, a safe scan limit is exceeded,
or the scan fails, the UI marks coverage incomplete instead of claiming
every document was reviewed.

Supplementary online search is a separate, optional source channel. An
operator must configure these local environment settings:

- SERVY_BRAVE_SEARCH_API_KEY: the provider API key, stored only in the local
  environment or deployment secret store.
- SERVY_WEB_RESEARCH_ALLOWED_DOMAINS: comma-separated official manufacturer
  domains approved by the operator for the selected brand.

The search query contains only the public manufacturer name, exact asset
model, a small allowlisted symptom vocabulary, and an exact fault code when
one is reported. It does not send customer identity, asset code, serial
number, company documents, manual passages, or the raw conversation.
Returned sources must use HTTPS, belong to the configured manufacturer
domain allowlist, and contain the selected asset's exact model. When a fault
code is reported, accepted snippets must contain that exact code and no
other fault code. Snippets containing customer repair or physical-procedure
language are excluded.

External excerpts are labelled "Supplementary online information — not
company-approved". They are never inserted into approved manual claims,
manual citations, the company source viewer, customer action lists, or the
local source-bound explainer. They cannot override an approved manual,
confirm a diagnosis, or authorize a repair. Missing credentials, missing
manufacturer/model identity, provider errors, and non-matching results fail
closed with an explicit status.

Offline CI coverage:
- python -m unittest tools.test_servy_real_demo_exhaustive -v
- python -m unittest tools.test_servy_real_demo_web_research -v
- frontend/test/onlineResearchContract.test.mjs

Example PowerShell configuration (replace the domain with the verified
official domain; never commit the API key):

    $env:SERVY_BRAVE_SEARCH_API_KEY = "<your key>"
    $env:SERVY_WEB_RESEARCH_ALLOWED_DOMAINS = "<official-manufacturer-domain>"
