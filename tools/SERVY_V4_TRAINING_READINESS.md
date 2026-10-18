# Servy V4 fine-tuning readiness — no training performed

This branch audits code paths and adds CPU-only admission checks. It cannot guarantee zero training loss or a higher Hit@1. Any unreviewed labels and missing checkpoints remain blockers to credible promotion.

## Verified code issues addressed

1. Op27 training selected final weights automatically. The training script now selects best validation loss, using all development questions as anchor-positive pairs rather than only those with a negative.
2. The Op27 index builder ignored content_first and stamped a legacy metadata passage format even when encoding heading/body only. Both Op27 training and index building now call the identical passage-format function in tools/op27_passages.py, with the correct format stamp.
3. Op27 source resolution previously skipped missing positives and negatives. Missing chunks, tenant mismatch, forbidden chunks, and contradictory positive/negative passages now cause explicit failure.
4. Op27 research checkpoint and vector collections are now overwrite-protected.
5. Separate admission validators tools/servy_finetune_readiness.py and tools/servy_finetune_preflight.py check structural split, passage, hard-negative and source-document isolation, without GPU or database writes.
6. tools/servy_checkpoint_audit.py compares fixed candidate sets, documents, chunks, logits, rankings and checkpoints without training.

## Remaining blockers

- All 198 train, 32 dev and 27 Stage2 examples previously described were weak/AI-supervised, not independently expert-approved. Source correctness and equipment applicability need expert review.
- The exact Stage2 27 triplets used in Colab must be preserved for auditing. The original Excel workbook of 50 examples is not an exact substitute.
- V2 and V3 checkpoint files apparently vanished with the Colab runtime; the saved weights cannot be compared unless recovered from approved persistent storage.
- The 32-query dev set has been repeatedly consulted. It cannot validate independent model generalization or technical answer correctness.
- Op26 weak-negative selection is heuristic; false negatives can actively harm contrastive training.
- The research corpus is tenant-filtered, not production customer/asset/confidentiality-authorized.

## Windows PowerShell commands — run inside the Git checkout

~~~powershell
python -m unittest discover -s tools -p "test_servy_checkpoint_audit.py" -v
python -m unittest discover -s tools -p "test_servy_finetune_readiness.py" -v
python -m unittest discover -s tools -p "test_servy_finetune_preflight.py" -v
python -m unittest discover -s tools -p "test_op27_passages.py" -v

python tools/servy_finetune_readiness.py --train private_training_data/servy_colab_train.jsonl --dev private_training_data/servy_colab_dev.jsonl --corpus private_training_data/servy_colab_corpus.jsonl --out-dir private_training_data/readiness_v4

python tools/servy_finetune_preflight.py --train private_training_data/servy_colab_train.jsonl --dev private_training_data/servy_colab_dev.jsonl --corpus private_training_data/servy_colab_corpus.jsonl --report private_training_data/preflight_v4.json
~~~

Both tools require real private exports already present locally. They do not retrieve missing documents. Avoid putting private data in the public repository.

If the exact 27 Stage2 examples are available, append --stage2 private_training_data/v3_stage2_exact_27.jsonl to each admission command. Do not substitute all 50 review candidates.

For a real independent holdout, source-document isolation must be verified using a full authorized corpus spanning all benchmark tenants; a one-tenant 123-chunk export is insufficient.

## Conditions before another L4 experiment

- Structural preflight has no blockers, and source references/document group isolation are demonstrated.
- Training labels and negatives are individually adjudicated where correctness is ambiguous.
- Models and tokenizers use pinned revisions; matching normalization/chunking/query-prefix policies are recorded.
- Training monitors grouped validation loss/ranking, stores the best checkpoint, stops on non-finite loss, and never tunes on a final untouched test.
- Every checkpoint and report is saved to approved persistent storage and SHA-256 verified BEFORE the Colab runtime is deleted.
- Reproduction includes fixed candidate sets and per-question failure reports.
- Serving authorization, citations, safety and correct-equipment actions remain separate production acceptance gates.

No fine-tuning, private model loading, production document mutation or vector-index rewriting was performed in this audit.
