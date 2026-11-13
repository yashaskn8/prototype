# SERVY — Accuracy evidence and fine-tuning gates

## What this PR actually fixes

* V6 now writes V5 and V6 per-query evaluation records, a paired movement ledger,
  and a candidate-recall/head-pool ceiling report into its **private experiment
  directory**. Equal aggregate Hit@1 cannot conceal offsetting query regressions.
* Frozen candidate identifiers, training subset, original V5 weights, corpus
  and development split are SHA-256 checked both before and after training.
  An in-place mutation aborts reporting a valid comparison.
* Research-only counterfactuals reuse **already computed logits** to estimate
  Hit@1/5/10 and MRR at pool sizes 5, 8 and 10. They never change model
  weights, production retrieval, tenant permissions or serving pool settings.
* Synthetic tests cover upper bounds, top-1 exchanges with unchanged aggregate
  scores, candidate drift, label mismatch, missing examples and invalid logits.

## Findings from the existing V5/V6 result

On the reported 42-query frozen development candidates:

* Hit@1: 35/42 = 83.33% for both models
* Hit@5: 37/42 = 88.10%
* Candidate Recall@10: 38/42 = 90.48%
* Four queries have no gold evidence in the frozen top ten
* The first five candidates contain gold evidence for at most 37/42 queries:
  **88.10% is the head-five top-one oracle ceiling** on this frozen set.
* The earlier reported 37/42 V5 Hit@1 is from a different/unreconciled
  evaluation. Do **not** overwrite one metric with the other.

These are DEVELOPMENT diagnostics on provisional labels, **not independent
holdout or production metrics**. Improving reranker training cannot retrieve
chunks entirely absent from the frozen candidate set. Any production candidate
retrieval change needs separate authorization, source integrity, latency and
quality tests.

## How to diagnose the EXISTING V5/V6 experiment without retraining

The original V6 experiment has already completed and does not contain
the new per-query outputs. The existing read-only checkpoint audit can replay
the two *known* local checkpoints against the same frozen candidates:

```bash
python tools/servy_checkpoint_audit.py \
  --cases /content/servy_v6_inputs/recovery_v4/family_resplit_20261018_234238/dev_unapproved_candidate.jsonl \
  --corpus /content/servy_v6_inputs/recovery_v4/servy_corpus_recovered_20261018T150933Z.jsonl \
  --candidates /content/servy_v6_inputs/recovery_v4/research_artifacts_resplit_20261019/frozen_candidates.jsonl \
  --model v5=/content/servy_v6_inputs/best_model \
  --model v6=/content/drive/MyDrive/Servy_V6_Authorized/outputs/v6_research_first_run/best_model \
  --pool-size 5 \
  --top-k 10 \
  --out-dir /content/drive/MyDrive/Servy_V6_Authorized/outputs/v5_v6_independent_replay_01
```

Use a **NEW** destination directory; the existing audit tool can write
to an existing directory, so choose a unique path and inspect it first.
This replays development evaluation only; it does not certify independence,
answer grounding, production serving, or the company permission scope.

Output is private and includes source/query IDs and text. Do not commit
private per-query output, training files or model weights to GitHub.

## Further training decision gate

**Retain the existing verified V5 model. Do not automatically fine-tune V7.**

1. Identify whether each top-one miss is due to absent candidate evidence,
   a gold passage outside the eligible head pool, or a reranker ordering miss.
2. Compare per-query V5 and V6 results on the same checkpoint hashes, candidate
   SHA, pool size, split SHA and model inference settings.
3. If retraining is proposed, select a new label-reviewed and genuinely
   independent evaluation population. Do not reuse known dev failures as
   training examples and then claim gains on the same dev set.
4. Keep independent holdout and human label approval false unless supported by
   independent records. Never bypass the existing permission gate.
5. Measure runtime CPU/GPU latency and memory, safety, citations and tenant
   isolation before considering any production rollout.

No code change or increased pool limit is an accuracy improvement until a real
replay demonstrates it. No company-private artifact is available in Git CI.

## New regression test

```bash
python -m unittest discover -s tools -p "test_servy_accuracy_diagnostics.py" -v
```
