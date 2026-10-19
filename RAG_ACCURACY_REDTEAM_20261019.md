# Servy retrieval-accuracy red-team: October 2026

This hardening is **not** evidence of a measured Hit@1 increase.

## Verified code-level risks addressed

1. **Stale or revoked content:** versioned index eligibility now excludes revoked
   knowledge documents. Retrieval still independently checks authorization,
   document status, tenant, product/asset scope and vector identity.
2. **Uncalibrated similarity reporting:** when a dense-only path runs, preserve
   the original cosine similarity rather than overwriting it with a metadata-
   boosted score. Downstream evidence diagnostics must not confuse them.
3. **Loss of retrieval candidates under RRF:** optional **Qwen3 dense-first**
   ablation preserves Qwen3 candidate order and backfills authorized sparse-only
   passages. When a passage occurs in both, preserve its sparse evidence fields.
   The ablation is gated by **all** of the existing experimental-Qwen3 safety
   flags and a separate `SERVY_QWEN3_DENSE_FIRST=1` flag (default OFF).
4. **Company index updates:** all company embedding profiles preserve stale
   historical vectors instead of deleting them. Valid refreshed passages may
   still be upserted non-destructively; a strict full-index verification must
   flag obsolete rows before experimental serving is approved.
5. **Destructive API:** standalone vector deletion is prohibited for all
   company embedding profiles, not just company-Qwen3.

## How to evaluate the Qwen3 dense-first hypothesis

Only after the separate local Qwen3 model and its index are verified:

- Keep current MiniLM/retrieval and V5 checkpoints intact.
- Freeze the independent, human-reviewed holdout and all safety constraints.
- Compare `SERVY_QWEN3_DENSE_FIRST=0` (current RRF) against
  `SERVY_QWEN3_DENSE_FIRST=1` (dense first, sparse backfill) at exactly
  the same top-10 candidates, V5 reranker weights, tenant eligibility,
  per-source diversity policy, and query set.
- Capture Recall@10, Hit@1, MRR@10, citation correctness, abstention safety,
  p50/p95 inference latency and unauthorized candidate count (must be zero).
- Promote only if independent results justify it; keep it OFF otherwise.

The prior 42-question research set is already reused many times. One more
correct question changes Hit@1 by about **2.38 percentage points**, so it
cannot substantiate a 0.01% accuracy gain. Neither changes to CI nor
synthetic tests prove production answer quality.

## Regression testing

```powershell
python manage.py test core.tests_accuracy_redteam core.tests_qwen3_accuracy core.tests_v5_reranker -v 2
```

Use the normal MySQL CI for the complete backend suite. No local Docker and
no private model or enterprise data are required for these unit tests.

**Operational readiness** remains conditional on private source integrity,
Qwen3 full-index verification, browser-level RAG usage, and independent answer
evaluation. The V5 research preflight code exists in
`tools/servy_v5_training_preflight.py`; it must be run locally on the original
private artifacts, not on example/mock files. Do not retrain to optimize the
reused development benchmark.
