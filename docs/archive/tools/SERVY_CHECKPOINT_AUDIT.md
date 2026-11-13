# Servy checkpoint-comparison audit — research only

This branch adds an offline, read-only comparator. **No fine-tuning, database writes, index replacement, Docker installation or production model promotion.** The model results from previous Colab experiments are user-reported, **not reproduced in this repository**.

## Code-path findings and limits

| Severity | Status | Evidence | Consequence |
|---|---|---|---|
| Critical | FIXED in this branch | `tools/servy_checkpoint_audit.py:normalize_inputs` | Missing exported gold chunks now abort instead of quietly lowering recall. |
| Critical | FIXED in this branch | `check_split_overlap` | Training positive/negative chunks, traceable review passages, guide IDs and exact query overlap now checked. Text-only negatives that cannot be matched to the exported corpus produce an unknown-provenance result, not verified separation. |
| High | FIXED in this branch | `rerank_fixed` and `evaluate_one` | Every candidate gets a raw logit, but only the head pool is reordered; original Qwen tail order remains fixed. |
| High | FIXED in this branch | `checkpoint_fingerprint` | Local checkpoint directory SHA-256 is recorded. A Hugging Face model ID without pinned revision remains explicitly unverified. |
| High | VERIFIED integration mismatch, not changed | `core/ml/reranker.py` | The serving reranker is `cross-encoder/ms-marco-MiniLM-L6-v2`, gated by `SERVY_RERANKER_ENABLED`, and reranks up to eight authoritative document candidates. **This is not automatically V2/V3 BGE**. Offline 5-candidate comparison is not production validation. |
| High | VERIFIED methodological limitation | `core/ml/benchmark_registry.py`; previous Colab experiments | The repeatedly used 32-question development benchmark is not an untouched, expert-adjudicated final test. Do not present Hit@1 on it as answer accuracy. |
| High | VERIFIED technical limitation | `core/services/document_loader.py` | PDFs use `pypdf` extractable text; scanned PDFs with no extractable text yield no passages. Actual incidence in proprietary Servy guides is **unknown** without document-level source inspection. |
| Medium | HYPOTHESIS ONLY | `core/services/chunking.py` | Character-bounded chunks and overlapping sections could separate tables, numeric units or procedures. No particular false answer is proven without original guide passages. |
| High | UNKNOWN | Private Colab V2/V3 weights, exactly used Stage 2 27 triplets, 32 dev labels | Exact Stage 1 vs Stage 2 regression, overlap with stage-2 negatives, and root cause of V3's lower score remain unverified. Do not train V4 based on guesses. |

Other production retrieval logic lives in `core/services/retriever.py`, the Chroma index lifecycle in `core/ml/indexer.py` and `core/services/chroma_store.py`, and training-label review in `core/ml/labels.py`. This branch does **not** replace their authorization or answer-safety checks. Document-level matches are diagnostic surrogates and **not proof of correct technical answers**.

## Install-free preflight (Windows PowerShell)

Run from the nested Git repository directory. The evaluator itself uses only Python standard-library modules in preflight mode:

```powershell
python -m unittest discover -s tools -p "test_servy_checkpoint_audit.py" -v
python tools/servy_checkpoint_audit.py --validate-only `
  --cases private_training_data/servy_colab_dev.jsonl `
  --corpus private_training_data/servy_colab_corpus.jsonl `
  --train private_training_data/servy_colab_train.jsonl `
  --out-dir ml_artifacts/servy_checkpoint_preflight
```

Provide the **exact 27 used Stage 2 examples** as another `--train` input (JSONL/CSV/XLSX); the 50-row AI triage workbook is not proof of which 27 were actually trained on. The `--train` argument may be repeated. Missing sources or passage-text mismatches block by default, as they should. The `--allow-untraceable-training` escape hatch marks provenance as potentially incomplete; it is for exploratory diagnosis, never for promotion.

If inputs were only stored in Colab `/content` and that runtime was deleted, re-export them from the approved local system before running. This command does **not** restore deleted checkpoints or private exports.

## Model comparison (only after approved checkpoint restoration)

On a GPU-enabled **approved** environment, run the same frozen corpus and training preflight, then:

```bash
python tools/servy_checkpoint_audit.py \
  --cases /content/servy_colab_dev.jsonl \
  --corpus /content/servy_colab_corpus.jsonl \
  --train /content/servy_colab_train.jsonl \
  --train /content/v3_stage2_exact_27.jsonl \
  --model pretrained=BAAI/bge-reranker-v2-m3 \
  --model v2=/content/servy_reranker_provisional_v2 \
  --model v3_stage1=/content/servy_reranker_v3_stage1 \
  --model v3_final=/content/servy_reranker_v3_research \
  --pool-size 5 --top-k 10 \
  --out-dir /content/servy_comparison
```

Local saved checkpoints must exist; the evaluator **never retrains them**. If no independent weights backup exists, do not claim the lost weights were evaluated. To compare exactly identical retrieval candidates on subsequent runs, pass `--candidates <previous output>/frozen_candidates.jsonl` and compare its digest in `summary.json`.

The comparator exports `per_query.jsonl`, `per_query.csv`, `frozen_candidates.jsonl`, and `summary.json`; `--validate-only` exports `preflight.json`. These may contain proprietary technician questions; retain them in company-approved storage. No private training examples should be committed to Git.

## Known remaining gaps (explicitly out of scope)

1. The source-authority, site/customer/equipment, quarantine, and confidentiality gates from live Servy serving are not reproduced by tenant-only Qwen research candidate generation. Never claim production security from this tool.
2. Exact text equality and source/document IDs do not rule out paraphrased query leakage, duplicate guides with changed IDs, or source-family variants that lack `source_group` metadata.
3. Neither provisional AI triage nor similarity scores constitute expert approval. The 32 cases may have incomplete or wrong positive sets.
4. This report contains **no actual CPU/GPU evaluation on the private Servy corpus or recovered V2/V3 checkpoints**. Source-level findings are separate from measured ML results.

**Promotion gate:** Independently reviewed train/dev/test manifests, verified source isolation, saved and hashed model artifacts, production-equivalent scoped candidate generation, CI green and correct-evidence/answer-safety evaluations. Until then, leave production unchanged.


## V4 fine-tuning admission: run BEFORE starting a Colab GPU session

The separate standard-library-only preflight does not load a model. Run it
against the **exact** exported Stage 1 and dev records and the same 123-chunk
corpus used for those exports. Do not upload this proprietary data to GitHub.

Windows PowerShell from the nested `prototype` Git checkout:

```powershell
python -m unittest discover -s tools -p "test_servy_finetune_preflight.py" -v

python tools/servy_finetune_preflight.py `
  --train private_training_data/servy_colab_train.jsonl `
  --dev private_training_data/servy_colab_dev.jsonl `
  --corpus private_training_data/servy_colab_corpus.jsonl `
  --artifact-dir "C:/approved-model-storage/servy-v4" `
  --report private_training_data/servy_v4_preflight.json
```

If the exact 27 **used** Stage 2 hard-negative triples have been reconstructed
with tenant IDs, add `--stage2 private_training_data/v3_stage2_exact_27.jsonl`.
Do **not** substitute the original 50-row AI-triage Excel spreadsheet. The
validator accepts selected JSONL or CSV, not the review workbook, because it
needs to establish which exact examples actually influenced the weights.

Optional `--holdout private_training_data/frozen_independent_holdout_benchmark.jsonl`
checks only for structural overlap (including unanswerable holdout queries); it
does **not** score, select on, or inspect final holdout performance.

Outcomes:
- `BLOCKED` (exit 2): the supplied data cannot safely enter training as-is.
  Inspect the report; do not weaken checks or silently remove failures.
- `REVIEW_REQUIRED` (exit 0): no mechanically provable blocker remains in
  the submitted subset, but labels, hard negatives, source authority and
  independent accuracy remain unverified until expert adjudication.
- `CHECK_TOKENIZER_TRUNCATION_FOR_LONG_PASSAGES` is only a character-count
  heuristic; examine model-specific tokenizer lengths separately.
- `PERSISTENT_CHECKPOINT_DESTINATION_UNSET` or
  `CHECKPOINT_TARGET_IS_EPHEMERAL_COLAB_STORAGE` alerts to lost checkpoint risk.
  A Google Drive folder is not by itself proof of company permission.

The preflight checks exact text and document identity, not semantic
paraphrases, guide-version lineage, or row-level customer ACLs. Passing it does
not imply "zero training loss", no overfitting or 95% retrieval accuracy. Use
best-validation-checkpoint selection, bounded tuning, immutable saved artifacts,
and an independent expert-reviewed test set to control regression.
