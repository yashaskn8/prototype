# Servy V6 continuation: code prepared, training not authorized

**Status: engineering-ready dry run; no V6 training executed.** The October 19
private review package reports 57 sources, 762 technical positive/negative
verdicts and zero source-family collisions. Every one of 762 human decision
fields was blank and the package says `PENDING_AUTHORIZED_SIGN_OFF`.

**Technical consistency checks are not a substitute for organizational
permission** to train on private employer content, particularly on Google
Colab or another external cloud processor. ChatGPT cannot impersonate an
authorized approver, fill in a fake reviewer, or change that decision. No
private data, human approvals, models or source documents are in Git.

## What is ready now

The new `tools/servy_v6_continuation.py` implements a default **read-only**
V6 plan and an explicit, separately gated training path. The dry-run plan
reuses the existing strict V5 data preflight:

- The original frozen train split, development split, quarantine ledger,
  corpus, proposed source families and frozen candidates must pass the
  existing full signed-hash/source-isolation audit.
- The selected candidate train file must be **exactly** the 111 original
  unmodified train rows (111 positive, 161 active negative references).
- The verified *V5* `model.safetensors` SHA-256 must be
  `b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda`.
- Planned V6 output must be a **new** directory, not nested in V5.
- V5 remains a rollback; no Django serving/profile or model promotion changes.
- The reused 42-question dev set is always marked non-independent.

The training branch, **if it is later genuinely authorized**, initializes
from that V5 checkpoint, not from pretrained BGE and not from all 353 source
queries. Its output is a separate V6 research checkpoint with loss and
frozen-candidate dev-only metrics. It requires a CUDA GPU, pinned
`sentence-transformers==5.1.0`, conservative one/two-epoch bounds, and
never promotes a checkpoint to production automatically.

## One local dry-run command (no GPU, no Docker, no writes)

Run in PowerShell from your cloned `prototype` root. Use a NEW path for the
planned output: this command does not create it.

```powershell
$base = ".\private_training_data\recovery_v4"
python tools\servy_v6_continuation.py `
  --split-dir "$base\family_resplit_20261018_234238" `
  --corpus "$base\servy_corpus_recovered_20261018T150933Z.jsonl" `
  --families "$base\ai_review_evidence_resplit_20261018_234537\ai_source_family_recommendations.jsonl" `
  --candidates "$base\research_artifacts_resplit_20261019\frozen_candidates.jsonl" `
  --candidate-train "$base\v5_selected_111\train_keep_ai_provisional.jsonl" `
  --manifest "$base\v5_selected_111\selection_integrity_manifest.json" `
  --v5-model ".\best_model" `
  --out-dir ".\private_training_data\v6_research_new"
```

Expected status: `V6_PLAN_READY_TRAINING_NOT_AUTHORIZED`.

This output is a *plan*, not a trained model. If `--train` is requested
without an authentic, separately scoped organizational approval record,
the tool fails closed **before loading CUDA models or writing output**.
The record must explicitly name an approving authority, actual approval
reference, the selected subset's SHA-256, and approved environment. External
Google Colab use also needs explicit external-processor authorization.
This script performs structural checks on a record; it cannot authenticate
a signer's identity. The operator must verify permission with the
organization and cannot use ChatGPT-generated text as approval.

## Independent evaluation gate

The 64-question `frozen_independent_holdout_benchmark.jsonl` has
source-family/chunk overlap with training and a missing gold reference.
**Do not claim independent accuracy from it.** Run the metadata-only
`tools/servy_v6_holdout_audit.py` as documented in
[`V6_HOLDOUT_LEAKAGE_GATE.md`](V6_HOLDOUT_LEAKAGE_GATE.md).
No unseen source families remain in the current 18-family frozen corpus.

## Quality interpretation

Earlier V5 Hit@1 = 37/42 (88.10%) is a *reused research development set*,
not independent holdout accuracy. The Qwen3 and frozen local candidate
exports are distinct. No improvement should be claimed until an untouched,
appropriately source-family-isolated expert-evaluated set is available.
Do not train on the five known development failures or substitute
machine-generated "approved" decisions for human sign-off.

## Synthetic CI

```bash
python -m unittest discover -s tools -p "test_servy_v6_continuation.py" -v
```

This test requires no private files, checkpoint, GPU, provider account, or
permission artifact. It verifies read-only behavior, SHA-256 enforcement,
non-overlap, pending-signoff rejection, permission scope and external
processor gates.
