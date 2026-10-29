# SERVY Real Company Data Audit

**Date:** 2026-10-29  
**Commit:** `0e1f187895f8144b3d2b1c9ed463b1896c2cdbef`  
**Evaluator:** Independent Red Team (Automated)  
**Mode:** MODE A — Synthetic Staging

---

## 1. Data Authorization Status

| Item | Status |
|------|--------|
| Real company data authorization | **NOT ESTABLISHED** |
| Starlly API credentials present | Yes (CRM integration only) |
| ML evaluation permission documented | No |
| External LLM API data transmission approved | No |
| Production database access | Not attempted |

> [!IMPORTANT]
> **All evaluation was conducted in MODE A (Synthetic Staging).** No external company data was accessed. All tests used the existing authorized database content. Real-data validation is BLOCKED until authorization is established.

## 2. Data Sources Used

| Source | Description | Authorization |
|--------|-------------|---------------|
| MySQL database (servy_db) | 20 tenants, 229 retrievable chunks, 139 knowledge documents | Existing dev/staging data |
| ChromaDB (chroma.sqlite3) | 4 active collections, 229 vectors each | Local vector index |
| V5 Reranker (best_model/) | XLM-RoBERTa-large cross-encoder, 2.27 GB | Local model artifact |
| BGE-small embeddings | BAAI/bge-small-en-v1.5 via HuggingFace | Public model |
| MiniLM embeddings | all-MiniLM-L6-v2 via HuggingFace | Public model |
| Ollama LLM | qwen2.5:3b — OFFLINE | Not tested |

## 3. Tenant Inventory

| Tenant ID | Name | Chunks | RAG Docs | Products | Customers | Assets |
|-----------|------|--------|----------|----------|-----------|--------|
| 6 | Starlly Tester | 132 | 43 | 11 | 5 | 51 |
| 9 | Skydream | 69 | 9 | 12 | — | — |
| Others (18) | Various | 28 | — | — | — | — |

## 4. Sensitive Data Handling

| Control | Status |
|---------|--------|
| Secret-value redaction | ✅ .env not committed |
| Minimal private-data extraction | ✅ Only aggregate metrics in reports |
| Private results directories | ✅ outputs/ is gitignored |
| No private docs to public GitHub | ✅ Verified |
| No unauthorized LLM API transmission | ✅ Ollama offline; no external API used |
| Source-document fingerprints | ✅ Content SHA verified for 20-chunk sample |

## 5. Conclusion

Real company data evaluation is **BLOCKED**. All measurements reported in this red team are development-environment results on synthetic/staging data. They cannot be cited as evidence of production accuracy on genuine company questions.

To proceed to MODE B or MODE C, the following must be established:
1. Written authorization from data owner for ML evaluation use
2. Approved data snapshot scope and retention policy
3. Access-controlled evaluation environment
4. Explicit prohibition on training with evaluation data
5. Audit trail for all data access
