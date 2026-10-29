# SERVY Real-World Red Team Fix Log

**Date:** 2026-10-29  
**Commit:** `0e1f187`  
**Evaluator:** Independent Red Team

---

## Summary

| Severity | Found | Fixed | Deferred |
|----------|-------|-------|----------|
| Critical (Release Blocker) | 2 | 0 | 2 |
| High | 2 | 0 | 2 |
| Medium | 3 | 0 | 3 |
| Low | 2 | 0 | 2 |
| **Total** | **9** | **0** | **9** |

> [!IMPORTANT]
> **No code changes were made during this red team.** The prompt explicitly prohibited modifying a working model. All findings are documented for the development team to address.

---

## Critical (Release Blockers)

### DEFECT-001: Manifest SHA Misattribution
- **File:** `SERVY_PRODUCTION_RELEASE_MANIFEST.json`
- **Description:** V5 reranker SHA-256 `b1cf23ba...` is attributed to the `company-bge-ft` embedding model. The actual embedding SHA is `7d394413...`.
- **Impact:** Model identity verification would match the wrong artifact.
- **Fix:** Update manifest with correct SHA assignments.
- **Status:** DEFERRED — metadata only, no code change needed.

### DEFECT-002: Serving Config Discrepancy
- **Files:** `.env`, deployment documentation
- **Description:** `SERVY_SERVING_EMBEDDING_PROFILE` is not set. Default resolves to `minilm`, but the 90.48% benchmark was measured with `bge-small`.
- **Impact:** Production would serve with 85.71% accuracy instead of the reported 90.48%.
- **Fix:** Add `SERVY_SERVING_EMBEDDING_PROFILE=bge-small` to production `.env`.
- **Status:** DEFERRED — requires deployment team action.

---

## High

### DEFECT-003: Ollama LLM Offline
- **Description:** The qwen2.5:3b LLM is not running. Answer generation, abstention, citation quality, and faithfulness are all untested.
- **Impact:** Cannot verify end-to-end answer correctness. Retrieval accuracy alone does not prove answer quality.
- **Fix:** Start Ollama and test answer quality. Consider a fallback behavior when LLM is unavailable.
- **Status:** DEFERRED — infrastructure dependency.

### DEFECT-004: No Independent Evaluation Dataset
- **Description:** All evaluation datasets (42-query dev benchmark, 40-case red team) were designed by the same team with knowledge of the system.
- **Impact:** Cannot claim independent accuracy validation. Results may be optimistically biased.
- **Fix:** Commission an external evaluator to create a blind held-out dataset.
- **Status:** DEFERRED — requires external resources.

---

## Medium

### DEFECT-005: Policy Guard Pattern Gaps
- **File:** `core/services/content_safety.py` → `query_is_control_abuse()`
- **Description:** Misses DAN/jailbreak patterns ("You are DAN"), role-assumption ("ADMIN OVERRIDE"), and social engineering ("Forget your rules").
- **Impact:** Some adversarial queries bypass the first line of defense. The LLM prompt still provides protection, but defense-in-depth is weakened.
- **Fix:** Add regex patterns for:
  - `you are (DAN|admin|root|system)`
  - `(ADMIN|SYSTEM|ROOT)\s*OVERRIDE`
  - `forget (your|all) (rules|instructions|constraints)`
  - `pretend you (have no|don't have) restrictions`
- **Status:** DEFERRED.

### DEFECT-006: No Document Revocation Audit Trail
- **File:** `core/models.py` → `KnowledgeDocument`
- **Description:** No `revoked_at` timestamp field. Document deactivation is via `is_rag_enabled=False` with no record of when/why.
- **Impact:** Cannot audit document lifecycle. Cannot detect if a document was improperly reactivated.
- **Fix:** Add `revoked_at` and `revoked_by` fields to KnowledgeDocument.
- **Status:** DEFERRED.

### DEFECT-007: MiniLM Dense Component Net-Negative
- **Description:** Sparse-only (88.1% Hit@1) outperforms MiniLM hybrid (85.7% Hit@1). The MiniLM dense component introduces ranking noise that harms accuracy.
- **Impact:** If BGE-small is not configured, the hybrid mode is worse than sparse-only.
- **Fix:** Either set `bge-small` as default or tune RRF weights to reduce MiniLM's contribution.
- **Status:** DEFERRED.

---

## Low

### DEFECT-008: Cold Start Latency (28-33s)
- **Description:** First query takes 28-33 seconds due to V5 reranker model loading.
- **Impact:** First user after a cold start or restart experiences unacceptable latency.
- **Fix:** Implement model pre-warming on startup. Or use lazy loading with a health check endpoint.
- **Status:** DEFERRED.

### DEFECT-009: Empty Heading on 1 Chunk
- **Description:** One KnowledgeChunk has an empty (but not null) heading field.
- **Impact:** Cosmetic only — chunk text is present and retrievable.
- **Fix:** Audit and populate missing headings.
- **Status:** DEFERRED.
