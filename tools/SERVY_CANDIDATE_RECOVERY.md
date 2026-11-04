# Offline candidate recovery experiment

No training, network model fetch, serving integration, or label changes. Both
experimental switches default off. Run from the repository root. All outputs
must use a NEW directory under `private_training_data`; inputs are read-only.

In Colab, set `ROOT` to the existing authorized experiment folder, and run:

```bash
ROOT=/content/drive/MyDrive/servy_experiments/servy_v4_bge_20261018T193940Z
python tools/servy_candidate_recovery.py \
  --cases "$ROOT/recovered_v6_inputs/split/dev_unapproved_candidate.jsonl" \
  --corpus "$ROOT/recovered_v6_inputs/corpus.jsonl" \
  --candidates "$ROOT/private_training_data/v6_qwen_candidate_snapshot_001/frozen_candidates.jsonl" \
  --recorded "$ROOT/v6_research_run_001/v6_dev_per_query.jsonl" \
  --families "$ROOT/recovered_v6_inputs/families.jsonl" \
  --manifest "$ROOT/v6_research_run_001/experiment_manifest.json" \
  --model "$ROOT/v6_research_run_001/best_model" --device cuda \
  --out-dir "$ROOT/private_training_data/candidate_recovery_NEW" \
  --enable-lexical-rescue --enable-context-rerank
```

Add `--generate-only` to verify the recorded baseline and generate candidates
without importing torch, loading a checkpoint, or measuring new answer rankings.
This is sufficient to measure candidate coverage, not reranker accuracy.

The runner pins the original snapshot, corpus, development split, recorded V6
results and V6 weights. The original hashes are embedded in the runner; the
manifest must agree. It fingerprints the family ledger and every input before
and after execution. No source-family reassignment occurs. Independent
holdout/technical label approval and customer authorization remain unverified.

The one declared lexical strategy keeps Qwen's first three candidates, adds up
to two BM25 top-five candidates, then deduplicates/truncates the remaining Qwen
order to ten. Voltage units are normalized symmetrically. This is not RRF.
Context is retained only from explicit corpus fields, never gold query titles.
Contextual Qwen representations are available via `build_qwen_candidates(...,
contextual=True)` but are NOT run by this script. No Qwen weights are needed.
Contextual reranking is a separate arm on unchanged candidates, at pool five.

Local generation on the verified 42-query dataset REJECTED the lexical arm:
head-five coverage 38/42 -> 37/42; top-ten coverage stays 40/42. Thus even a
perfect reranker cannot exceed the old 37/42 Hit@1 with this new head pool.
No new V6 model inference was performed. The corpus has no separate title,
heading or section fields; existing Markdown headings are already in passages,
so this input cannot test a new title enrichment. No serving patch is proposed.

Reports contain every improved/regressed query and a conservative rejection
gate. Actual scoring always scores new candidates with the existing local V6;
baseline order drift aborts comparison. Proprietary outputs stay private.

```bash
python -m unittest discover -s tools -p 'test_servy_candidate_recovery.py' -v
```
