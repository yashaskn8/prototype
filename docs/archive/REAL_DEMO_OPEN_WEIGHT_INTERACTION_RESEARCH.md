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

## Current real-data demo configuration

The approved Windows real-data presentation runs without a chat-generation server. The verified local retrieval stack is:

- **BGE-small embeddings** for dense semantic retrieval over the approved company collection.
- **Hybrid retrieval** that also considers lexical/sparse matches.
- **The verified Servy V5 reranker** to order the retrieved candidates.
- **The deterministic dialogue controller** for intent focus, follow-up context, clarifications, evidence limits, safe handoff and conversation recap.

BGE-small and V5 serve retrieval/ranking roles. They do not generate conversational prose. Technical responses are assembled from approved source passages and retain their citations; dialogue behavior is handled by deterministic server-side logic.

The optional Ollama planner and source-bound explanation modules remain separate legacy capabilities in the repository. They are not enabled by the real-data launcher. That launcher sets `SERVY_LLM_PROVIDER=extractive` and `SERVY_REAL_DEMO_DIALOGUE_MODE=rules`, does not probe port 11434, and never depends on a Qwen chat model. This avoids a false startup warning and avoids an unnecessary service dependency during the presentation.

## Activation and configuration

Use the existing authorized real-data launcher:

    git pull --ff-only origin main
    .\tools\Start-ServyRealDemo.ps1 -Restart

The launcher keeps the read-only company data path, BGE/V5 retrieval readiness gate, customer/asset authorization checks, and cited extractive answer contract. It explicitly reports that deterministic source-grounded dialogue is active. The backward-compatible `-NoLocalChat` switch may still be passed, but it is no longer required.

No Ollama installation, Qwen chat download, external API key or remote model endpoint is required for this demo. The general application still has a separately configurable optional LLM integration; it is not part of this presentation runtime.

The old drifted Chroma index and frozen baseline are left unchanged. A separate verified-document rebuild is used only when the original Chroma integrity gate requires it:

    .\tools\Start-ServyRealDemo.ps1 -Restart -RebuildFromVerifiedDocs

## Security and privacy guarantees implemented in code

1. Tenant, customer, equipment, and document scope remains server-side.
2. All technical claims are drawn from existing approved RAG evidence and citation IDs, never written by the local LLM.
3. No chat model can authorize physical interventions or suppress safety escalation in this runtime; the deterministic server policy remains authoritative.
4. No unapproved URLs, customer procedures or fault-code mismatches can be introduced by a plan.
5. Instructions embedded in customer reports or retrieved source text remain untrusted data and cannot override server policy or manipulate the database.
6. The presentation runtime makes no chat-model network request and performs no automatic model download.
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
- Run with no Ollama installation -> Normal deterministic, source-grounded startup and dialogue; no Ollama probe or warning.

Measure: intent accuracy, citation precision, unsupported technical claims, unsafe-action rate, repeated facts per session, time-to-answer, clarity of follow-up, latency, and successful handoff.

## Known limits

This is an intentionally constrained conversational RAG system, not an unconstrained text-generation assistant. The real-data demo uses retrieval models plus deterministic dialogue logic; it does not claim to run a chat-generation model. Limited company documentation can still require abstaining. The real-company Windows deployment needs live query tests to establish source accuracy and answer quality.
