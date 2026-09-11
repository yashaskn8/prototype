# Architecture

```text
Customer / Engineer
        |
        v
Django templates + forms
        |
        v
Authenticated Tenant + Customer + Asset context
        |
        +------------------------------------+
        |                                    |
        v                                    v
Knowledge metadata filter             Call Register context
(tenant/customer/product/asset)       + previous resolved cases
        |                                    |
        +------------------+-----------------+
                           v
                 Local hashing-vector retrieval
                 (content relevance + scope score)
                           |
                           v
                Optional loopback Ollama LLM
                OR grounded extractive fallback
                           |
                           +--> answer + server-verified references
                           |
                           +--> if unresolved: create ServiceCall
                                + SLA + least-active technician
```

## Why metadata filtering comes first

The supplied Knowledge Base contains useful manuals alongside test/duplicate/noisy entries. The RAG service therefore never performs a global unscoped search. It first enforces authenticated tenant scope, customer visibility, confidentiality, quarantine state, product/asset relevance and then ranks only permitted evidence.

## Knowledge scope

Documents may be:

- tenant-wide,
- domain/category-wide,
- product-wide,
- asset-specific,
- customer-specific,
- confidential staff-only.

Product-wide manuals intentionally use `asset = NULL`, allowing the same approved guide to support every asset of that model. Asset-specific SOPs can still override/rank above shared product material.

## Call Register memory

Closed calls with a non-empty resolution are converted into a sanitized `ServiceResolutionIndex`. Direct customer names, phone numbers and email addresses are redacted before those resolutions become reusable RAG evidence. Customer self-service only sees previous cases from that same customer; staff mode may reuse sanitized same-product cases across customers in the same tenant.

## Local / zero-cost vector layer

`core/services/vectors.py` uses scikit-learn `HashingVectorizer`, storing sparse vectors directly in SQLite JSON fields. This avoids an external vector database and works offline. It is stronger than plain keyword matching but is still intentionally lightweight.

## Upgrade path

`core/services/retriever.py` and `core/services/vectors.py` are isolated service boundaries. They can later be replaced with sentence-transformer embeddings + FAISS/Chroma/pgvector without changing the Django views or customer workflow.
