# Servy V5 — the four saved Colab files, audited without changing them

This GitHub change addresses the four file locations confirmed in Drive:

```text
servy_experiments/servy_v4_bge_20261018T193940Z/
├── qwen3_candidates_top10.jsonl
├── hybrid_rrf60_candidates_top10.jsonl
├── v5_gemini_semantic_audit_v1/
│   └── input_snapshot/split/train_unapproved_candidate.jsonl
└── v5_training_candidates_20261019T083815_575655Z/
    └── train_keep_ai_provisional.jsonl
```

The four files are **not part of the GitHub repository** and GitHub cannot
modify, certify, or retrain on them. The structural auditor runs locally or in
the owner's Colab against the files already mounted from Google Drive.

## One-command structural check (no model, GPU, Docker or training)

After mounting Google Drive and cloning/pulling this repository into Colab,
run from the repository working directory:

```bash
python tools/servy_v5_artifact_audit.py \
  --experiment-root "/content/drive/MyDrive/servy_experiments/servy_v4_bge_20261018T193940Z"
```

Optional: persist the **metadata-only** report privately in Drive:

```bash
python tools/servy_v5_artifact_audit.py \
  --experiment-root "/content/drive/MyDrive/servy_experiments/servy_v4_bge_20261018T193940Z" \
  --out "/content/drive/MyDrive/servy_experiments/v5_private_artifact_audit.json"
```

The command **refuses to overwrite** an existing report. Keep all private
questions, source passages, model weights, and private datasets out of Git.

The audit checks:

- There is **exactly one** selected V5 training-subset snapshot in the
  expected location (no arbitrary folder chosen when multiple snapshots exist)
- Each selected train row matches an original train row **exactly**, including
  the references and labels; duplicate query identities are rejected
- Each selected query has at least one positive and one active negative,
  and no positive/negative conflict or quarantined active negatives
- Both Qwen3 and hybrid candidate exports have unique nonempty lists of up to
  ten chunk IDs; both cover the same query IDs
- When the companion dev file is present, its query IDs match the candidate
  exports; when the resplit report is present, its frozen train SHA is checked
- The output includes per-file SHA-256 fingerprints, record counts and
  top-1/order agreement **but does not print the underlying private text**

A successful report reads `STRUCTURAL_AUDIT_PASSED_RESEARCH_ONLY`.
It explicitly records `training_authorized=false` and
`fine_tuning_admitted=false`: a structural audit **is not** enough to start
fine-tuning, claim accurate labels, or promote a model.

## What these four files cannot prove

The four observed paths **do not identify** the complete corpus export,
source-family recommendations, quarantined negative ledger, signed resplit
report or independent holdout. Some additional files might exist under names
not captured by the earlier filename search. Do not fabricate these files.

The full strict preflight remains:

```bash
python tools/servy_v5_training_preflight.py \
  --split-dir "PRIVATE_SNAPSHOT_SPLIT_DIR" \
  --corpus "PRIVATE_CORPUS.jsonl" \
  --families "PRIVATE_SOURCE_FAMILY_RECOMMENDATIONS.jsonl" \
  --candidates "PRIVATE_QWEN3_CANDIDATES.jsonl" \
  --candidate-train "PRIVATE_SELECTED_TRAIN.jsonl" \
  --manifest "PRIVATE_SELECTION_MANIFEST.json"
```

The original V5 `experiment_manifest.json` lists metrics and
`expert_approved=false`. It is **not** a selection-integrity manifest and
cannot be silently substituted: it lacks the candidate subset's SHA, the
source-train SHA and the accepted-query count. Regenerate provenance only
from the actual private source/selection records after a full data audit,
then have an authorized reviewer approve the intended research use.
Never invent an expert sign-off or an independent holdout.

The previously measured 88.10% Hit@1 (37/42 development queries) and the
reused top-5/8/10 pool experiment are **diagnostic**, not an independent
assessment of production RAG accuracy. Those five known dev failures must
not be copied into a training split to inflate the same benchmark.

## Offline unit tests

```bash
python -m unittest discover -s tools -p "test_servy_v5_artifact_audit.py" -v
python -m unittest discover -s tools -p "test_servy_v5_training_preflight.py" -v
```

GitHub CI runs both tests on synthetic data only. No Google Drive access,
credential, training input, checkpoint or environment change is required.
