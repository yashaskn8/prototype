# Servy V4 — blocked-by-default research workflow

This is **not a production rollout**. No private dataset, company manual,
trained weight, credentials, or reviewer-identifying information belongs in
this public repository. The 32 historical development questions have been
repeatedly consulted and cannot be called an untouched independent holdout.
The original V2/V3 checkpoints must be considered unavailable until their
actual bytes and SHA-256 digests are recovered from an **authorized** store.

## Evidence needed BEFORE starting Colab

1. Written company approval identifying exactly which data may leave the
   company environment, the authorized execution account, named reviewer
   credentials, and authorized persistent checkpoint destination. Approval
   is external to this Git repository and is **not inferred from a file path**.
2. Exact JSONL train/dev/corpus exports with `tenant_id`, stable
   `query_id`, `document_id`, `chunk_id`, `source_group`, `source_version`,
   positive IDs and matching text. Every answer-bearing alternate requires
   explicit review; do **not** turn an AI-screened negative into an approved
   label. Each approved pair needs `review_status: "approved"` and a real
   `reviewer_id`; a negative requires `negative_review_status: "approved"`
   and `negative_reviewer_id`.
3. The exact Stage 2 triplets used previously, if claiming to audit Stage 2.
   Neither the 50-row triage workbook nor an approximate reconstruction is
   a replacement. A new V4 need not reuse those rows.
4. New, independently reviewed evaluation cases. Do not use the reused
   32-case development set to claim generalization or 95% answer accuracy.
5. Check the actual installed dependency APIs and Colab GPU configuration;
   the gated runner has **not** yet undergone a genuine GPU training run.

## Offline no-Docker commands (Windows PowerShell)

From the cloned repository root:

```powershell
git fetch origin
git switch audit/servy-fixed-candidate-checkpoints-20261018
git pull --ff-only origin audit/servy-fixed-candidate-checkpoints-20261018

python -m unittest discover -s tools -p "test_servy_finetune_preflight.py" -v
python -m unittest discover -s tools -p "test_servy_training_preflight.py" -v
python -m unittest discover -s tools -p "test_servy_v4_colab_train.py" -v
python -m unittest discover -s tools -p "test_op26_negative_governance.py" -v
```

Keep all private paths outside the Git checkout, including reports.

Example **schema** for a privately held model configuration
(`BASE_SHA` and `TOKENIZER_SHA` MUST be replaced with verified 40-character
repository commit SHAs; this is intentionally not a valid ready-to-run config):

```json
{
  "base_model_id": "BAAI/bge-reranker-v2-m3",
  "base_revision": "BASE_SHA",
  "tokenizer_id": "BAAI/bge-reranker-v2-m3",
  "tokenizer_revision": "TOKENIZER_SHA",
  "sentence_transformers_version": "VERSION_VERIFIED_IN_ENVIRONMENT",
  "loss_name": "BinaryCrossEntropyLoss",
  "max_seq_length": 256,
  "candidate_pool": 10,
  "query_format": "raw-question",
  "passage_format": "approved-frozen-corpus"
}
```

The private authorization manifest schema is:

```json
{
  "approval_reference": "ACTUAL_CORPORATE_APPROVAL_REFERENCE",
  "approved_persistent_destination": "/approved/persistent/servy-v4-UNIQUE-RUN-ID",
  "authorized_reviewer_ids": ["ACTUAL_APPROVED_REVIEWER_ID"],
  "authorized_model_config_sha256": "REAL_SHA256_OF_PRIVATE_CONFIG",
  "authorized_input_sha256": {
    "train": "REAL_SHA256",
    "dev": "REAL_SHA256",
    "corpus": "REAL_SHA256",
    "candidates": "REAL_SHA256"
  }
}
```

These fields are **claims supplied by your data custodian**; code cannot
authenticate the human expertise of reviewers or establish storage permission.
Do not submit an example manifest or a fabricated reviewer approval.

Run the model-free strict gate **before initializing CUDA**:

```powershell
python tools/servy_training_preflight.py `
  --train "C:/company-approved/servy/train.jsonl" `
  --dev "C:/company-approved/servy/dev.jsonl" `
  --corpus "C:/company-approved/servy/corpus.jsonl" `
  --model-config "C:/company-approved/servy/model.json" `
  --authorization "C:/company-approved/servy/authorization.json" `
  --artifact-dir "C:/company-approved/models/servy-v4-UNIQUE-RUN-ID" `
  --report "C:/company-approved/reports/servy-v4-preflight.json"
```

The strict gate exits **0** only on
`READY_FOR_RESEARCH_EXPERIMENT`; errors exit nonzero with a private JSON
report. Experimental readiness is not production eligibility. Use a fresh
report path because overwriting output evidence is prohibited. Add
`--stage2` only with **exact**, reviewed Stage 2 inputs; `--holdout` may
be used for structural leakage checks only, never for checkpoint selection.

## Frozen candidate construction and comparison

`tools/servy_checkpoint_audit.py` can build research candidates using
Qwen3 `encode_query()` and `encode_document()`, or accept a frozen file.
**It does not implement live customer/site/asset/document ACL equivalence.**
Use this step only with approved, nonproduction research data.

```bash
python tools/servy_checkpoint_audit.py \
  --cases /approved/data/dev.jsonl \
  --corpus /approved/data/corpus.jsonl \
  --train /approved/data/train.jsonl \
  --model pretrained=BAAI/bge-reranker-v2-m3 \
  --pool-size 5 --top-k 10 \
  --out-dir /approved/experiments/BASELINE-UNIQUE-ID
```

Preserve the generated `frozen_candidates.jsonl` unmodified. Obtain its
SHA-256 and have the data custodian authorize the file hash. Do **not**
silently inject gold chunks. The comparator's baseline metrics are retrieval
diagnostics on reused development cases, not evidence of safe technician advice.

## Experimental runner (only after an explicitly approved admission)

On **company-approved Colab or equivalent L4-enabled execution** install
and pin the versions recorded in the private config. First verify
`python -c "import sentence_transformers; print(sentence_transformers.__version__)"`
and the actual CUDA, BF16 support, dependency compatibility, tokenizer
revision, and retrieval passage formatting. No version is assumed verified here.

```bash
python tools/servy_v4_colab_train.py \
  --train /approved/data/train.jsonl \
  --dev /approved/data/dev.jsonl \
  --corpus /approved/data/corpus.jsonl \
  --model-config /approved/data/model.json \
  --authorization /approved/data/authorization.json \
  --candidates /approved/data/frozen_candidates.jsonl \
  --artifact-dir /approved/persistent/servy-v4-UNIQUE-RUN-ID \
  --run-id servy-v4-UNIQUE-RUN-ID \
  --epochs 2 --batch-size 4 --learning-rate 0.0000007
```

The runner rejects absent/mismatched inputs before model initialization,
trains only against reviewed binary labels with
`BinaryCrossEntropyLoss`, selects best development loss (not final epoch),
saves epoch checkpoints and selected weights, reloads and checks model logits,
and compares fixed candidates. It never promotes a checkpoint to serving.

**Known limitations:** The training loop itself has not been exercised on
Colab in this audit. The default 256 token limit needs document-specific
truncation checks; verify actual package APIs in a disposable *synthetic*
GPU smoke run before any approved company-data experiment. The runner does
not enforce external filesystem access policies or true reviewer expertise.
A final answer-level independent evaluation remains a separate blocker.

## Backup and recovery policy

The module `tools/servy_checkpoint_backup.py` copies checkpoint bytes to
the approved persistent destination with overwrite refusal, SHA-256 and byte
count verification, and a second post-rename verification. If Colab Drive
is selected it refuses to claim persistence when `/content/drive` is not
mounted. **A mounted personal Google Drive is still unauthorized unless
company policy explicitly permits it.** An unmounted folder named Drive
is not a backup.

Never disconnect an ephemeral runtime with the only weights in `/content`.
The training runner aborts if a mandatory backup fails; inspect
`best_model`, all epoch checkpoint directories, `run_manifest.json`,
`baseline.json`, `comparison.json`, and per-query files in the APPROVED
persistent destination. Do not claim historic V2 or V3 checkpoint recovery
unless the actual bytes are found and verified.

## Rollback

All changes are on the audit branch and PR #2 remains draft. Production
MySQL data, Chroma indexes, currently served MiniLM reranker, and `main`
must not be modified by this workflow. Revert a candidate patch on the
audit branch by a normal `git revert <verified-commit-sha>`; do not force-push
or deploy research weights. Abort any planned training if a gate changes or
a frozen dataset hash fails verification.
