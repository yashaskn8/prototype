# Servy — Why a forced 42/42 accuracy claim would be invalid

This is a source-code investigation, not an execution of the private 42-query
checkpoint benchmark. No new V5 weights, private records, gold labels, or
model-inference results are included in this commit.

## Audit of the recorded failures

- **OP26-SLV-52-0:** Its expected gold evidence was classified confidential in
  the C2 evaluation. Under that context, denying access is **correct behavior**.
  Do not expand permissions to score this case as a hit.
- **OP26-SLV-143-0:** The selected top passage and silver gold are neighboring
  chunks of the same troubleshooting section. This requires independent SME
  adjudication of valid alternative evidence, not an arbitrary rank override.
- **OP26-SLV-76-2:** The selected passage concerns escalation and may be more
  relevant than the silver calibration label. Preserve the original label until
  an authorized reviewer confirms the expected evidence.
- **OP26-SLV-144-2:** The semi-synthetic question about "operating v specification
  5" does not describe the intended USB communication-loss evidence clearly.
  A V5 rank-4 result may reflect query/label mismatch rather than a simple
  reranker defect. Do not hardcode this query or boost its gold identifier.

The historic **38/42** and authorized **38/41** use different denominators.
Do not compare them as though they measure identical populations. The second
figure is **92.68%** on *reused development questions*, not an independent
production result.

After *separate expert review*, two of the three authorized misses could be
judged valid at top-1 under revised relevance labels; that would represent a
conditional **40/41**, not a measured or independently proven improvement and
not 100% production accuracy. A remaining disputed case still needs investigation.

## Implemented reliability correction

The retriever captures a frozen `ServingConfig` once per request. Previously
the dense scorer then reread `SERVY_SERVING_EMBEDDING_PROFILE` directly from
Django settings rather than using the captured value. The model choice could
therefore diverge from the request snapshot during configuration changes and
controlled comparisons.

The scorer now uses `serving_config.embedding_profile` when provided; callers
without a snapshot retain their existing settings fallback. An isolated
synthetic regression test verifies that a later settings override does not
change the embedding profile of an in-progress request. No tenant filter,
vector validation, model weights, retrieval thresholds, or relevance labels
were modified.

## Next required experiment (private authorized environment)

1. Verify the deployment actually uses the intended BGE and V5 weights.
2. Run `python manage.py test core.tests_serving_truth -v 2` plus full
   MySQL tests; inspect the GitHub Actions result.
3. Reproduce the frozen authorized 41-case dev metrics with model/index
   fingerprints, per-stage rankings, and query-scope evidence.
4. Prepare independent SME review packages for the two ambiguous labels and
   the semi-synthetic prompt. Do not change the frozen benchmark.
5. On unseen, source-family-disjoint company cases, test real asset-scoped
   customer/engineer APIs and generated answers. Record Hit@1, Recall@10,
   MRR@10, answer correctness, citations, abstention, isolation and latency.

**No 100% accuracy assertion or production promotion is justified by this patch.**
