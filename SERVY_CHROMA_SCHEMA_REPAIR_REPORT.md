# SERVY ChromaDB Schema Repair Report

> [!IMPORTANT]
> This report documents the repair of ChromaDB HNSW indexes and schema 2.0 metadata.

## Problem Statement

Legacy MiniLM collections contained schema 1.0 metadata missing required fields.
The Django retriever's `_matches_current_bge_vector()` validation rejected these vectors,
causing fallback to sparse-only retrieval even when dense vectors existed.

Additionally, after a previous reindex session, HNSW index files were NOT persisted to disk.
Direct `.query()` calls threw: `chromadb.errors.InternalError: Error creating hnsw segment reader: Nothing found on disk`

## Root Causes

### 1. Missing Schema 2.0 Metadata
Legacy vectors lacked: `chunk_id`, `document_id`, `chunk_index`, `tenant_id`,
`document_version`, `content_sha256`, `embedding_model`, `embedding_revision`

### 2. HNSW Not Persisted
Previous session added vectors to ChromaDB's SQLite storage but HNSW index files
were never written to the `chroma_db/` segments directory.

## Repair Process

### Script: `scratch/repair_hnsw.py`

For each of 4 collections:
1. Extracted all records via `.get()` (IDs, embeddings, metadatas, documents)
2. Deleted the collection entirely
3. Recreated with identical metadata (HNSW config, distance function)
4. Re-added all records to force HNSW rebuild
5. Verified `.query()` returns correct results with proper distances

### Collections Repaired

| Collection | Vectors | Schema | HNSW Status |
|-----------|---------|--------|-------------|
| `servy_dense_kb` | 229 | 2.0 ✓ | Working ✓ |
| `servy_company_dense_kb` | 229 | 2.0 ✓ | Working ✓ |
| `servy_dense_bge_small_v1` | 229 | 2.0 ✓ | Working ✓ |
| `servy_company_bge_small_v1` | 229 | 2.0 ✓ | Working ✓ |

### Verification

Direct ChromaDB queries confirmed working:
- Tenant-filtered queries return correct results
- Document-ID filtered queries return correct results
- Unfiltered queries return results with proper cosine distances
- All metadata fields present and valid

## Impact on Retrieval

### Before Repair
- All 42 queries showed `bk=fallback` (sparse-only)
- Dense path returned 0 results due to HNSW read errors

### After Repair
- 40/42 queries show `bk=hybrid` (true dense+sparse RRF fusion)
- Dense retrieval contributes to ranking
- OP26-SLV-145-2 promoted from rank=2 to HIT@1 through dense contribution

## Backup Collections Preserved

Historical and backup collections were NOT deleted:
- 4 backup collections (`*_backup_*`)
- 9 experimental collections (`op26`, `op27`, etc.)

These are preserved for rollback if needed.

## Schema 2.0 Metadata Fields

All 229 vectors in each collection now include:

```json
{
  "chunk_id": <int>,
  "document_id": <int>,
  "chunk_index": <int>,
  "tenant_id": <int>,
  "document_version": <str>,
  "content_sha256": "<sha256hex>",
  "embedding_model": "<model_name>",
  "embedding_revision": "<revision>"
}
```

Verified via `scratch/check_version_mismatch.py`.

---

*Generated: 2026-10-29 | Commit: 18d0277*
