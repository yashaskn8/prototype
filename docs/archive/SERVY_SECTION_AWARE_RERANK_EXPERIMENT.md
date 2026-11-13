# Servy experimental section-aware reranker

## What changed

The default serving path is unchanged. Dense + sparse/BM25 retrieval, RRF,
authorization and the existing reranker continue to run as before unless the
opt-in flag is explicitly enabled.

The experiment adds:
- Multi-intent section classification for safety, pre-installation, installation,
  commissioning, initial checks, diagnostics, measurements, maintenance,
  inspection and record keeping.
- Context-enriched CrossEncoder passages containing asset identity, document
  title, section heading and document type. Stored source/citation text is unchanged.
- An experimental score blending incoming rank (60%) with the existing
  evidence-aware technical reranker rank (40%), with bounded section alignment
  adjustments. Unclassified candidates are neutral.
- No candidate filtering and no change to tenant/customer/asset/source authorization.
- Per-candidate section diagnostics.

## Enable only in an isolated evaluation instance

Environment flag: SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK=1
Default: SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK=0

Do not enable in production until evaluation has been reviewed. No model weights
are trained or modified.

## Tests

python manage.py test core.tests_section_aware_rerank -v 2

## Offline benchmark replay

Questions JSONL fields: query_id, asset_id, question, question_type,
gold_chunk_ids (or expected_chunk), baseline_ranked_chunk_ids
(or reranked_top10_chunks). Candidate IDs are ordered and may contain up to 20.
Corpus JSONL fields: chunk_id, asset_id, document_id, title, heading, text, doc_type.

python tools/evaluate_section_aware_rerank.py --questions questions.jsonl --corpus target_corpus.jsonl --output report.json

The evaluator reports candidate Recall@5/@10/@20, Hit@1/@5/@10, MRR,
per-asset and per-question-type metrics, latency, and input SHA-256 hashes.

## Evaluation caveats

The V2.2 150-question bank, complete candidate export and 64-passage corpus
were preserved in Google Drive, not this repository. Therefore no accuracy
claim is made by this code change alone. Replay must use those exact files,
converted to the JSONL schema above. Gold labels are preserved; no model-generated
judgments are promoted to human-approved ground truth.

This is an internal diagnostic benchmark, not independent accuracy validation.
No production checkpoint, database, deployment or source document is modified.
