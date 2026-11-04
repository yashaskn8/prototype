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
Gemini. The local real-data demo stays offline and extractive by default.

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
   of previously shown source claims is signed with the selected asset and
   customer. `What other causes?` shows only additional supported claims
   and clearly acknowledges when none remain. Repeating the original
   question still permits the same source-backed explanation.
6. **Missing detail and graceful refusal**: a vague or ambiguous reply like
   `I checked that` asks which check and observed outcome; a bare `yes`
   requests the displayed code. No completed check is invented. No technical
   answer is emitted when matching evidence is unavailable.
7. **Conversation UI**: source-marked answer paragraphs, a focused headline,
   at most one relevant customer observation question, context-sensitive
   suggested follow-up chips, New chat, and read-only ticket preview.

## Security invariants

The focused presentation is a **view of already verified claims**, not an
LLM-authored statement. Every technical claim must have an evidence ID
linked to a permitted citation from the original source contract. A missing,
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

Run `python -m unittest tools.test_servy_real_demo_dialogue -v`
and `node --test frontend/test/customerAnswerContract.test.mjs`.
GitHub CI also runs the full real-data demo security tests and frontend build.

On the Real-data demo at `http://127.0.0.1:5175/` check:

- “My Split AC is running but not cooling. What possible causes are
  documented?” → a **single causes section**, with approved manual citations.
- “What can I safely observe?” → only customer-safe observations, no
  guessed physical procedure.
- “What does E-04 indicate?” → only evidence for E-04, not E-05.
- “What other causes are there?” → new cited claims only, or a clear
  no-additional-evidence response.
- “I checked that” → asks which check and its outcome; does not assume.
- “No error code” → seeks relevant next safe observations, not a repeat
  of the original four questions.
- A request to bypass a safety rule or obtain an unsupported precise
  setting still fails closed.

The test fixtures are synthetic. Real-company document coverage and
accuracy still require a live, authorized Windows demonstration, including
manual inspection of citations and unsupported-answer refusals.
