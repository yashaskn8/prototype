# Servy RAG architecture decision record — 2026-11-05

Status: **provisional**, pending benchmark and real-document review. No new provider, container platform, or live company-data writes authorized.

## Sources checked (public, not indexed as customer evidence)

- Microsoft, *Relevance and Ranking Overview*, ms.date 2026-08-06: https://learn.microsoft.com/en-us/azure/search/search-relevance-overview — keyword + vector + ranker; agentic retrieval described as **preview**, not an incumbent dependency.
- Microsoft, *Hybrid Search Overview*: https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview — lexical retrieval is useful for codes and identifiers; RRF fuses lexical and vector results.
- Anthropic, *Contextual Retrieval*, **2024-09-19** (historical work, not a 2026 invention): https://www.anthropic.com/research/contextual-retrieval — context-enriched chunks and reranking are candidates; small-corpus full-context alternative needs a cost/latency/permission benchmark.
- OWASP, *2025 Top 10 for LLM Applications*: https://genai.owasp.org/llm-top-10/ — prompt injection, sensitive information disclosure, improper output handling, excessive agency, vector/embedding risks.

## Architecture decision

Problem: the current read-only demo supports trusted extractive claims with citations but can repeat context-misaligned answers and cannot fulfill a real ticket-creation contract. Preserve the existing Django / React / authorized source boundary. Continue using pre-retrieval server-side permission checks, signed asset-scoped continuation, bounded hybrid retrieval and reranking, source-span validation before display, and explicit read-only ticket draft labeling. Do not auto-add an LLM that generates technical claims.

Alternatives: (a) mandatory remote managed agent, **reject** (external data/provider approval absent); (b) unvalidated free-form local model output, **reject** (citation and safety proof missing); (c) existing local extractive pipeline plus bounded natural dialogue improvements, **adopt provisionally**; (d) semantic caching, graph retrieval, late interaction, multimodal retrieval, corrective/agentic search, **defer** pending held-out per-slice benchmark and complexity measurements.

## Coverage matrix

| Area | Present or candidate | Release decision |
| --- | --- | --- |
| PDF/DOCX/CSV extraction and provenance | Existing parsing | Verify page/section/table/OCR failures, version hash and stable IDs |
| OCR, complex tables and units | Unproven | BLOCKED until approved fixtures and human source review |
| Chunking and contextual summaries | Existing chunk pipeline | Benchmark heading/procedure preservation; summaries must never substitute for source proof |
| Exact identifiers | Existing lexical + code checks | Hold out exact E/F codes and asset models |
| Vector + hybrid + reranking | Existing local models | Benchmark lexical-only/vector-only/hybrid/rerank on same permissions and corpus |
| Query rewriting and decomposition | Existing bounded follow-up logic | Decompose only compound requests; check negations, units and cost |
| Conversational state | Signed asset-bound continuation | Regression-test vague reports, corrections, past checks and topic changes |
| Source citation validation | Existing answer contract | Independent semantic support and source-view access review |
| Access/revocation/cache | Existing tenancy boundaries | Security review and revocation-in-flight test, no assumptions |
| Structured UI and ticket actions | React demo preview-only | Keep no-write demo; authorize separate staging ticket integration |
| Graph, late interaction, multimodal, caching, agentic search | Advanced options | DEFER; no measured gain yet |
| Evaluation and production operations | Historic scripts and CI | Freeze thresholds before held-out testing and collect real staging evidence |

## Tradeoffs and release controls

The extractive architecture minimizes unsupported equipment facts and preserves offline execution, but cannot guarantee fluent, deep answers unsupported by approved manuals. Any planner may select only approved claim identifiers; it cannot mint citations or operational side effects. Timing, corpus size, parser error rates, per-slice citation support and refusal rates must be measured before selecting additional retrieval strategies.

The real-data Windows demo's SELECT-only database login must never be promoted to write access as a shortcut. Real ticket submission requires a separately authorized nonproduction API, actor/asset/field-bound confirmation, idempotency key and uncertain-commit reconciliation. Screen text or historical audits cannot prove these gates.

A release candidate record must contain exact commit, config, corpus hash, model versions, dataset provenance, commands and results. Review roles A–F are independent checklists unless actual sub-agent execution is available; no fabricated parallel-agent signoff.
