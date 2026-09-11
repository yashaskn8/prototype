# Manager requirements → implemented prototype mapping

| Requirement from discussion | Implementation in this project |
|---|---|
| Build locally before touching production Servy | Standalone Django project with SQLite |
| Mimic current Servy schema/screens | Assets, projects, inventory, part requests, users concept, KB and call register models + seeded data |
| Call Register download option | XLSX export button; XLSX import workflow and template |
| Build a dummy database | Django seed command + standalone sample SQLite DB + Excel workbook |
| Knowledge Base with customer documents | Upload/create page with tenant/customer/product/asset metadata |
| Asset-wise / product-wise sub-classification | Domain → Category → Brand → Product → Asset |
| Two domains first | Dairy Equipment / Milk Analyzer and Material Handling / Aerolift |
| Clear installation/troubleshooting docs | Curated installation, cleaning, maintenance, troubleshooting and user guides |
| RAG should answer based on question + asset | Asset-aware metadata filtering before local hashing-vector semantic retrieval |
| Accurate references | Every answer returns the exact document and section/chunk used |
| If KB doesn't solve it, redirect to ticket | Customer page has “No — create service call”; interaction history is attached to ticket |
| Technician assignment | Simple least-active technician assignment on escalation |
| Go beyond traditional Servy | Customer self-service AI layer before call registration + Engineer Copilot |
| Multi-tenant SaaS | Every business object has a tenant; retrieval hard-filters tenant and customer scope |
| Confidential KB items | Customer portal excludes confidential content; staff assistant may use it |
| Zero API spend | Local retrieval; optional Ollama local LLM; deterministic extractive fallback |
| Customer uploads own docs | KB item supports customer-specific file/text/link metadata and local indexing |
| Existing noisy/test KB data | Seed has 118 KB items; curated RAG docs are enabled, legacy/test fillers are excluded |
| Similar screenshot volumes | Seed command creates 51 assets, 26 projects, 118 KB items, 532 inventory items and 7,665 calls |
