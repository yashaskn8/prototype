# SERVY MySQL-ChromaDB Integrity Report

**Date:** 2026-10-29  
**Commit:** `0e1f187`  
**Database:** MySQL 8.0.41 (servy_db)  
**Vector Store:** ChromaDB (chroma_db/chroma.sqlite3)

---

## 1. Executive Summary

**ALL INTEGRITY CHECKS PASSED.** Zero anomalies detected across FK integrity, duplicate detection, index alignment, and content verification.

---

## 2. Foreign Key Integrity

| Check | Result |
|-------|--------|
| Chunks with no document | 0 ✅ |
| Chunks with no tenant | 0 ✅ |
| Documents with no tenant | 0 ✅ |
| Chunk-document tenant mismatches | 0 ✅ |
| Assets with no product | 0 ✅ |
| Document-product tenant mismatches | 0 ✅ |

---

## 3. Duplicate Detection

| Check | Result |
|-------|--------|
| Duplicate (same tenant + text) chunk groups | 0 ✅ |

---

## 4. RAG-Disabled Document Leak Check

| Check | Result |
|-------|--------|
| Non-quarantined chunks from RAG-disabled docs | 0 ✅ |

---

## 5. ChromaDB-MySQL Alignment

| Collection | Vectors | Status |
|-----------|---------|--------|
| servy_dense_kb | 229 | ✅ ALIGNED |
| servy_dense_bge_small_v1 | 229 | ✅ ALIGNED |
| servy_company_dense_kb | 229 | ✅ ALIGNED |
| servy_company_bge_small_v1 | 229 | ✅ ALIGNED |

All 4 active collections contain exactly 229 vectors, matching the 229 retrievable MySQL chunks (non-quarantined + RAG-enabled).

- **Stale vectors (in Chroma, not MySQL):** 0
- **Missing vectors (in MySQL, not Chroma):** 0
- **ID format:** ChromaDB uses `chunk-{id}`, MySQL uses plain `{id}` — mapping verified

---

## 6. Content SHA Verification

20-chunk sample from `servy_company_bge_small_v1`:

| Check | Result |
|-------|--------|
| Content mismatches | 0/20 ✅ |

ChromaDB document text matches MySQL chunk text for all sampled records.

---

## 7. Data Quality

| Check | Result |
|-------|--------|
| Chunks with null text | 0 ✅ |
| Chunks with empty text | 0 ✅ |
| Chunks with null heading | 0 ✅ |
| Chunks with empty heading | 1 ⚠️ |

One chunk has an empty heading (not null). This is a minor cosmetic issue — the chunk text is present and retrievable.

---

## 8. Embedding Model Field

MySQL `KnowledgeChunk.embedding_model` values:
- `hashing-word12-v1` (majority — sparse/BM25 hashing)
- `starlly-real-v1` (some chunks)

> [!NOTE]
> The `embedding_model` field records the sparse hashing model used for BM25 tokenization, not the dense embedding model. Dense embeddings are stored in ChromaDB collections named by profile (e.g., `servy_dense_bge_small_v1`).

---

## 9. Collection Inventory

| Collection | Vectors | Active | Purpose |
|-----------|---------|--------|---------|
| servy_dense_kb | 229 | ✅ | MiniLM dense vectors |
| servy_dense_bge_small_v1 | 229 | ✅ | BGE-small dense vectors |
| servy_company_dense_kb | 229 | ✅ | Company MiniLM vectors |
| servy_company_bge_small_v1 | 229 | ✅ | Company BGE-small vectors |
| 13 others | various | ❌ | Experimental/backup |

Total ChromaDB collections: 17  
Active collections: 4  
Total active vectors: 916 (229 × 4)

---

## 10. Conclusion

The MySQL-ChromaDB pipeline is fully synchronized. All retrievable chunks are correctly indexed across all 4 active collections. No orphaned, stale, duplicate, or missing vectors were detected. Content integrity is verified for the sampled subset.
