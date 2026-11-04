# Ask Servy: Research and design for open-weight conversational RAG

Research date: 4 November 2026. Scope: the localhost-only, read-only Real-data demo on ports 8002 and 5175. This compares the officially documented interaction and RAG capabilities of leading systems, not their private implementation details.

## What to learn from popular assistants

| System | Relevant published pattern | Application in Ask Servy |
| --- | --- | --- |
| ChatGPT and OpenAI Retrieval/File Search | Semantic and keyword search, relevance ranking, contextual answers grounded in retrieved sources | Keep BGE-small and keyword hybrid retrieval with V5 reranking; answer the question asked, not four default questions |
| Claude | Document-grounded responses with citations to exact source passages | Only approved signed source spans may support Servy's technical claims |
| Gemini File Search | Indexed documents, metadata, structured grounding and page-level citations | Preserve tenant, customer, document provenance and the source-citation contract |
| Microsoft Copilot | Grounding in accessible enterprise data with access controls and citation review | Never let LLM output override asset/customer authorization |
| Rasa CALM | Use language models to interpret free-form customer language; deterministic flows control actions and compliance | Local model selects intent and next question; source policy determines permitted technical advice |
| Ollama | Local chat completion and structured JSON output using a schema | Constrain local model to allowed intent, claim IDs and safe follow-up questions |
| Qwen and Llama open-weight model families | Local instruction-following and model-specific conversation modes | Support a configurable Ollama model, without remote services or free-form technical fabrication |

Primary research:
- OpenAI Retrieval: https://developers.openai.com/api/docs/guides/retrieval
- OpenAI File Search: https://developers.openai.com/api/docs/guides/tools-file-search
- Anthropic Citations: https://platform.claude.com/docs/en/build-with-claude/citations
- Google Gemini File Search: https://ai.google.dev/gemini-api/docs/file-search
- Gemini page-level citations update: https://blog.google/innovation-and-ai/technology/developers-tools/expanded-gemini-api-file-search-multimodal-rag/
- Microsoft Copilot grounding: https://support.microsoft.com/en-us/microsoft-365-copilot/what-information-does-copilot-use-to-answer-my-prompt
- Microsoft Copilot Studio knowledge: https://learn.microsoft.com/en-in/microsoft-copilot-studio/knowledge-copilot-studio
- Rasa CALM: https://rasa.com/calm
- Ollama structured outputs: https://github.com/ollama/ollama/blob/main/docs/capabilities/structured-outputs.mdx
- Qwen open models: https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md
- Meta Llama models: https://dev.meta.ai/llama/docs/model-cards-and-prompt-formats/llama4

These patterns do not establish that any chatbot is guaranteed to be accurate. Servy's industrial-equipment context requires more restrictive source verification than casual general-purpose dialogue.

## The prior design flaw

The previous backend produced four fixed sections and the frontend rejected other shapes. One question about causes therefore displayed causes, safety checks, unknowns, and next steps whether the user requested them or not.

The first improvement replaced those four sections with a focused, source-backed answer, signed conversation state, relevant follow-up choices, and clarification of ambiguous customer responses.

## New implementation: local open-weight dialogue planner

An optional Ollama component now interprets the current customer question, recent bounded customer reports, and a small list of approved technical claims. The request is sent only to 127.0.0.1:11434, with HTTP proxies and redirects disabled. It sends no citation URLs or secrets.

The model is NOT asked to write technical paragraphs. It can return only a structured selector with:
- One intent: causes, observations, next, meaning, documents, general.
- Up to four existing claim IDs, never newly generated IDs.
- One question exactly copied from a server-side list of safe customer-observation questions, or an empty string.
- One allowed tone: direct, friendly or brief.

The application validates every output field. An explicitly asked intent overrides a conflicting model guess. The model can only reorder or select source-verified claims; it cannot create citations, quantities, procedures, diagnosed faults, ticket actions or safety approvals.

The deterministic source engine still checks selected equipment, customer and tenant; access permissions; document provenance; exact cited spans and fault codes; evidence sufficiency; completed checks; and any required procedure approval.

The response includes a simple indicator of whether a local structured planner or the deterministic grounded fallback was used. If Ollama is offline, the selected model is missing, a request times out, or the model returns invalid JSON or fabricated IDs, the reply remains source-grounded through the fallback. Technical generation remains extractive even when the model plans the conversation.

The demo also supports a customer-report recap. Asking what the customer has reported so far summarizes only authenticated, signed customer turns, explicitly without claiming a verified diagnosis.

## Activation and configuration

The normal Real-data launcher enables local planning when a local Ollama model is present:

    git pull origin main
    .\tools\Start-ServyRealDemo.ps1 -Restart -RebuildFromVerifiedDocs

Model default is qwen2.5:3b. Install and start Ollama separately if it is not already available, then install that model using the Ollama application or its documented commands. No model download or company-data export is performed automatically.

The launcher deliberately keeps SERVY_LLM_PROVIDER=extractive, while SERVY_REAL_DEMO_DIALOGUE_MODE=local controls this separate model-assisted dialogue planner.

For a deterministic A/B comparison:

    .\tools\Start-ServyRealDemo.ps1 -Restart -RebuildFromVerifiedDocs -NoLocalChat

The old drifted Chroma index and frozen baseline are left unchanged. A separate verified-document rebuild is required if the original Chroma integrity gate is still blocking startup.

## Security and privacy guarantees implemented in code

1. Tenant, customer, equipment, and document scope remains server-side.
2. All technical claims are drawn from existing approved RAG evidence and citation IDs, never written by the local LLM.
3. The local planner can neither authorize physical interventions nor suppress safety escalation.
4. No unapproved URLs, customer procedures or fault-code mismatches can be introduced by a plan.
5. Model instructions embedded in customer reports or retrieved source text remain untrusted data; arbitrary model output cannot execute commands or manipulate the database.
6. The model endpoint is fixed to loopback; no proxy, redirect, remote model host or automatic download is allowed.
7. No company database writes or operational ticket creation are introduced.
8. Bounded customer history lives in browser memory and a signed continuation token rather than a persistent corporate transcript.

## Acceptance tests and evaluation criteria

Run:

    python -m unittest tools.test_servy_real_demo_local_planner -v
    python -m unittest tools.test_servy_real_demo_dialogue -v
    python -m unittest tools.test_servy_real_demo -v

Then evaluate in the real Windows demo with authorized documents, checking each displayed citation against its source. Use tests such as:

- My Split AC is running but not cooling. What possible causes are documented? -> One relevant causes answer with citations, not four sections.
- What else could cause it? -> Distinct supported causes only, or an explicit limit.
- What can I safely report? -> Verified customer-safe observations only.
- I checked that. -> Clarify which check and result without assuming success.
- What have I told you so far? -> Customer-reported history only, no imagined diagnosis.
- What does E-04 mean? -> Exact E-04 source binding, not E-05.
- Ask about a different customer's equipment -> No access or fabricated citation.
- Ask for an unsupported internal procedure -> A refusal or escalation, not actionable unapproved advice.
- Shut down Ollama -> Prompt deterministic fallback without a crash.

Measure: intent accuracy, citation precision, unsupported technical claims, unsafe-action rate, repeated facts per session, time-to-answer, clarity of follow-up, latency, and successful handoff.

## Known limits

This is an intentionally constrained local conversational RAG system, not an unconstrained clone of ChatGPT, Claude or Gemini. Local model quality varies by installed checkpoint. The planner improves dialogue choices, not the set of authorized source facts. Limited company documents can still require abstaining. The real-company Windows deployment needs live tests to establish that sources and results are correct.
