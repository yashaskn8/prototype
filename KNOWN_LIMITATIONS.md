# Prototype limitations

- Tenant switching in the header is intentionally a demo control. Production must derive tenant from authenticated organisation membership.
- Local hashing-vector retrieval is deliberately lightweight and zero-cost. The service boundary is ready to replace it with sentence-transformer embeddings + pgvector/FAISS/Chroma later.
- The Ollama integration is optional. Model weights are not bundled in this ZIP because they are large and hardware-specific.
- Uploaded files should be malware-scanned and size-limited before production use.
- Video URLs are stored as Knowledge Base items but transcript ingestion is not included in this first prototype.
- Fine-grained role/object permissions, audit trails, queues/background indexing, backups and production observability are outside the PoC scope.
