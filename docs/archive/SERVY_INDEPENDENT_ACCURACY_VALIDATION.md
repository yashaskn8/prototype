# SERVY Independent Accuracy Validation Report

**Date:** 2026-10-29  
**Status:** ⚠️ **INDEPENDENT VALIDATION DOES NOT YET EXIST**

---

## Current Evaluation State

### Existing Evaluation Sets

| Set | Size | Purpose | Independent? |
|-----|------|---------|-------------|
| 42-query historical (C2 tenant) | 42 queries | Full development benchmark | ❌ Used during model development |
| 41-query authorized | 41 queries | Production accuracy (minus confidential) | ❌ Subset of above |
| 64-case benchmark | 64 cases | Cross-equipment evaluation | ⚠️ Partially — but 9 equipment name mismatches, most fell back to sparse-only |

### Current Verified Accuracy (NOT Independent)

| Configuration | Hit@1 (41 authorized) | MRR@10 |
|--------------|----------------------|--------|
| Sparse only (BM25) | 37/41 = 90.24% | 0.921 |
| Hybrid + V5 (pool=5) | 38/41 = 92.68% | 0.935 |

**These numbers are development-set accuracy, NOT independent validation.**

---

## Why Independent Validation Is Required

1. **Selection bias:** Pool size (5), reranker threshold, hybrid weights, and RRF parameters were all tuned on the 42-query set. Reporting accuracy on the tuning set overstates real-world performance.

2. **Gold label quality:** 2 of the 3 remaining failures are gold label errors/ambiguities. This suggests the evaluation set itself has quality issues, making it unreliable as a benchmark.

3. **Equipment coverage:** The 42-query set covers a narrow slice of equipment families. Production queries will span all equipment in the tenant.

4. **Adversarial queries:** The evaluation set contains no adversarial, ambiguous, or out-of-domain queries. Real users will submit these.

---

## Independent Benchmark Design Requirements

### Minimum Requirements
- **100+ queries** from equipment families NOT used in any prior training or model selection
- **Multiple annotators** for gold labels (inter-annotator agreement ≥ 0.8 Cohen's κ)
- **Query types:** factual, procedural, troubleshooting, safety, specification, edge-case
- **No overlap** with the 42-query development set or the 64-case cross-equipment set

### Recommended Methodology
1. Sample equipment families from the production database that were NOT in the training/evaluation pipeline
2. Recruit 2+ domain experts to write queries and label gold answers independently
3. Resolve disagreements via discussion, recording both the consensus label and the disagreement rate
4. Run the benchmark ONCE — do not iterate on it (treat as a held-out test set)
5. Report accuracy with 95% confidence intervals

### Expected Effort
- **Query creation:** 5-10 hours (domain expert time)
- **Gold labeling:** 5-10 hours (2 annotators)
- **Benchmark execution:** ~1 hour (automated)
- **Analysis and reporting:** 2-4 hours

---

## Previous 64-Case Benchmark Issues

The existing 64-case benchmark (`outputs/hybrid_benchmark_results.json`) had significant issues:
1. **9 equipment name mismatches** — benchmark equipment names didn't match DB product names, causing `no_asset_match` failures
2. **All backends were "fallback"** — HNSW indexes were broken at the time, so dense retrieval wasn't used
3. **No ground truth labels** — the benchmark measured retrieval feasibility, not accuracy

These issues mean the 64-case results cannot serve as independent validation.

---

## Conclusion

**Do not claim 95% accuracy has been achieved.** The verified ceiling is 92.68% on a 41-query development set. True accuracy on independent data is unknown and likely lower due to selection bias.

The next step is to create a proper independent benchmark following the requirements above.
