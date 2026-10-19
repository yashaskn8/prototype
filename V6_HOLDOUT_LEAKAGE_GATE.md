# Servy V6 — holdout leakage gate and defensible mentor evidence

**Status: independent holdout NOT available in the known corpus.** This is a
dataset-design limitation, not a model-training bug. GitHub cannot create an
unseen source family by renaming a file, changing the split, or erasing records.

The owner reported the following read-only audit on locally held PRIVATE data:

- 18 proposed canonical source families in the recovered corpus.
- Original full train uses 13 families, dev uses 5, with 0 overlap. There
  are **0 unused families** in that frozen source corpus.
- `frozen_independent_holdout_benchmark.jsonl` contains 64 questions, but
  10/10 mapped holdout families overlap the selected 111-question training
  subset; 40/64 questions directly share chunk references.
- Holdout reference overlap categories include 21 holdout-negative chunk
  IDs that appeared as training-positive IDs and 9 holdout-negative chunk
  IDs that appeared as training-negative IDs. **These cross-question role
  overlaps are not by themselves proof of incorrect labels.**
- The first combined report flagged **22 questions with missing OR empty
  gold**, but the separate unresolved-reference count was **1**. This
  indicates 21 questions have no positive references, while 1 cites a
  missing corpus chunk. Empty-gold questions **could be valid abstention
  cases**; they cannot be declared broken or scorable until the task and
  label intent are confirmed. The one unresolved gold reference requires
  source reconciliation. These are not interchangeable error classes.
- The file's `review_status=approved` metadata and filled `reviewer`
  fields do **not** independently authenticate the review provenance.
- `op21_accuracy_audit/rerank_holdout.json` is an *already-scored*
  historical diagnostic, not a new independent final test.

These are user-reported values; the GitHub repository has **no private files**
and cannot verify the actual 64-question evidence. To reproduce and independently
confirm the calculations, use the new offline auditor below.

## Run the local, read-only audit (Windows PowerShell, no Docker)

From `C:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype`:

```powershell
$root = ".\private_training_data"
$base = "$root\recovery_v4"
python .\tools\servy_v6_holdout_audit.py `
  --holdout "$root\frozen_independent_holdout_benchmark.jsonl" `
  --selected-train "$base\v5_selected_111\train_keep_ai_provisional.jsonl" `
  --original-train "$base\family_resplit_20261018_234238\train_unapproved_candidate.jsonl" `
  --dev "$base\family_resplit_20261018_234238\dev_unapproved_candidate.jsonl" `
  --corpus "$base\servy_corpus_recovered_20261018T150933Z.jsonl" `
  --families "$base\ai_review_evidence_resplit_20261018_234537\ai_source_family_recommendations.jsonl"
```

The command runs on your own computer and **prints counts and SHA-256 file
fingerprints only**, not private questions, passages, or reviewer identities.
It does not move files, relabel examples, score models, run training, access
the Internet, or change Git.

An optional `--out "path/to/new_local_report.json"` writes metadata only
and refuses to overwrite a report. If a missing corpus gold chunk is present,
it is **flagged**, not dropped, invented, or injected into candidates.
The `evidence_integrity` section separately reports
`holdout_questions_with_no_positive_references`,
`holdout_questions_with_unresolved_positive_references`, and
`holdout_questions_with_resolvable_positive_references`. A no-positive
question is marked `UNDETERMINED_POSSIBLE_ABSTENTION` until evidence
establishes whether it intentionally tests refusal/no-answer behavior.

### Strict gate interpretation

- `BLOCKED_HOLDOUT_LEAKAGE_OR_INCOMPLETE_EVIDENCE`: Do not report
  independent Hit@k, MRR, or model-comparison findings based on this dataset.
- `STRUCTURAL_ISOLATION_ONLY_CHRONOLOGY_UNVERIFIED`: No structural overlap
  was detected by these checks, but chronological/model-selection independence,
  reviewer identity and organization approval still require external proof.
- `independent_holdout_verified` is **always false**. A program cannot
  establish usage history or genuine reviewer sign-off from only these files.
- `review_metadata_unverified` contains counts, **not** approval status.
- The auditor examines both the actual selected train subset and the
  historical **original** 353-query train. This avoids incorrectly treating
  material used in prior V4 experiments as fully unseen.

## What can be reported to a mentor today?

Report the V5 pretrained-to-finetuned exploratory comparison only with its
exact candidate set and 42-question **reused development** denominator:
V5 top-1 37/42 (88.10%). The previous Qwen3 40/42 candidate Recall@10
comes from a different retrieval experiment; do not call it an independent
V6 accuracy score. The 64-question leaked holdout can support *explicitly
labeled historical diagnostics*, not claims of generalized accuracy.

An independent final benchmark requires *new authorized source families*
disjoint from ALL earlier training/dev source families, valid answer evidence,
and a verifiable history of non-use in model selection. If new sources are
unavailable, document the limitation; do not randomly reshuffle existing
questions, drop 40 contaminated questions, or train on the five reused
development failures to manufacture a score.

Private-data training remains blocked by
`PENDING_AUTHORIZED_SIGN_OFF`. The V6 continuation script is a read-only
plan by default and preserves the existing V5 checkpoint.

## Unit tests

```bash
python -m unittest discover -s tools -p "test_servy_v6_holdout_audit.py" -v
```

Synthetic CI tests cover unseen-family hypothetical data, direct
positive-negative reference overlap, duplicated queries, missing gold,
cross-tenant IDs, historical exposure, and non-promotion invariants.
