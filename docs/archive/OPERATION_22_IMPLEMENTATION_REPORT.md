# Operation 22 — accuracy engineering implementation

**Verdict: `EXPERIMENTAL_IMPROVEMENT_NOT_INDEPENDENTLY_VERIFIED`.** The experimental evidence ranker produced no Hit@1 gain over fair MiniLM on six weakly supervised development cases. It improved historical exact-label Hit@5 from 35/43 to 36/43, but the 43 cases have been repeatedly inspected, contain questionable labels, and cannot establish a real-world gain. No 95% claim is supported. Customer serving, production MiniLM, and the Operation 20 BGE checkpoint remain unchanged.

## A. Starting accuracy and population

Operation 21's local controlled, same-authorized-corpus comparison on 43 repeatedly inspected single-label relevance cases found MiniLM **24/43 Hit@1, 35/43 Hit@5**; base BGE **14/43, 30/43**; and Operation 20 BGE FT **16/43, 30/43**. These are exact assigned-chunk metrics, not adjudicated answer-bearing relevance, endpoint performance, or independent company accuracy. The original frozen benchmark is preserved.

## B. Confirmed issues addressed

- `tools/op22_starlly_snapshot.py` and `tools/op22_provenance_audit.py` establish the authenticated organisation-273 attachment-to-document-to-chunk-to-vector chain. Nine documents/69 chunks have byte and mapping matches. The 157 other local documents and empty provenance table are explicitly unresolved; no verification status was written into production data. Details are in `OPERATION_22_SOURCE_AND_EVALUATION_INTEGRITY.md`.
- `tools/op22_prepare_review.py` creates source-backed private review queues with actual passage text, six alternate candidates, 11 suspect-support cases, and five real complaint candidates. Every judgment remains `PENDING_REVIEW`.
- `core/ml/op22_review_workflow.py` and `tools/op22_freeze_reviewed_dataset.py` enforce reviewer allowlist, approval evidence, positive/negative consistency, source identity, and a frozen pre-existing-source exclusion list. This prevents the current local corpus from being repackaged as a new independent source-separated test.
- `core/ml/op22_technical_evidence.py` adds an isolated deterministic parser, equipment/source gating, bounded fault-code matching, specification/procedure checks, query-conditioned overview preference, fixed MiniLM/lexical feature ranking, and fail-closed machine answer-support decisions. It is **not connected to serving**. The rule is deliberately conservative about numerical and hazardous procedures. A machine `ANSWER_SUPPORTED` decision is only a candidate for human assessment, not an approved answer.
- `tools/op22_rank_experiment.py` compares identical authorized candidate universes and writes per-query ranks. `tools/op22_company_diagnostic.py` runs a separate strict verified-source check on authentic complaints. `tools/op22_latency_benchmark.py` measures warm local retrieval costs. No new model was trained or checkpoint overwritten.

## C. Controlled results

| Retrieval system | Six weak development Hit@1 / Hit@5 | Independent reviewed Hit@1 | Historical exact-label Hit@1 / Hit@5 | Grounded answer accuracy | Warm offline p95 |
|---|---:|---:|---:|---:|---:|
| Fair MiniLM | 4/6 / 6/6 | Not available | 24/43 / 35/43 | Not measured | 23.32 ms* |
| Base BGE-small | 4/6 / 6/6 | Not available | 14/43 / 30/43 | Not measured | 21.63 ms† |
| Operation 20 BGE FT | 4/6 / 6/6 | Not available | 16/43 / 30/43 | Not measured | 20.50 ms† |
| Experimental evidence ranker over MiniLM + lexical | 4/6 / 6/6 | Not available | 24/43 / 36/43 | Not measured | 22.97 ms* |

* 36 repeated runs over six development questions; warm MiniLM, SQLite authorization/candidate query, query encoding, in-memory dot products, and lexical/feature ranking for the experimental method. Excludes Chroma I/O, network, corpus encoding, citation construction, and answer generation. The loaded model and corpus process used about 650 MiB RSS. † Operation 21's different warm in-memory dense-only measurement on 64 questions, excluding SQLite/Chroma; these latency values must not be compared directly to the starred values.

The experimental ranker's historical MRR@10 was 0.6826 versus MiniLM 0.6727; nDCG@10 was 0.7374 versus 0.7139; Recall@10 was 39/43 versus 36/43; and Recall@20 was 43/43 versus 41/43. These are contaminated exact-label diagnostics. The six weak development cases had identical MiniLM and evidence-ranker ranks for labeled positives, with MRR@10 0.8056 and nDCG@10 0.8552. There is **no validated Hit@1 lift**, and no independent professionally reviewed set exists.

The experimental verifier marked six of 43 historical relevance queries machine `ANSWER_SUPPORTED` and abstained on all 21 historical unsupported/security cases. This is not a 21/21 validated abstention score: those cases have no independently approved support judgments, and answer generation was not evaluated. Complaint 43's strict-source search found 23 asset-specific permitted chunks and put a noise/vibration passage first but withheld answer coverage; complaints 39–42 require asset clarification. Complaint accuracy and grounded-answer correctness remain unmeasured.

## D. Remaining per-query errors

The private `op22_review/historical_per_query.jsonl` file preserves full ranks, features, decisions, and reasons for all 64 cases. Among the 43 assigned relevance labels, the experimental ranker still misses the assigned chunk at rank 1 on 19 cases: `EVAL-ERR-04`, `EVAL-SYM-04/06`, `EVAL-TRAP-02/03/05/06`, `EVAL-MSP-01/02/05/06`, `EVAL-MLT-01/02/04`, `EVAL-AMB-03`, `EVAL-TECH-02/03/04`, and `EVAL-XTLK-02`. These are **assigned-label misses**, not confirmed wrong technical answers.

- Suspected missing technical support or wrong equipment: `EVAL-TRAP-02/03/06`, `EVAL-AMB-03`, `EVAL-TECH-02`, `EVAL-XTLK-02`; the reviewer must decide whether the question is answerable from approved sources.
- Plausible alternate passages or split procedure context: `EVAL-ERR-04`, `EVAL-SYM-04/06`, `EVAL-MSP-02`, `EVAL-MLT-01`, `EVAL-TECH-03`. The model's top passage is not silently promoted to a valid positive.
- Remaining exact-label ranking ambiguity: `EVAL-TRAP-05`, `EVAL-MSP-01/05/06`, `EVAL-MLT-02/04`, `EVAL-TECH-04`; reviewer adjudication and more representative development labels are needed.

Against MiniLM, historical labeled-positive rank improved on `EVAL-TRAP-02/06`, `EVAL-MSP-01/05/06`, `EVAL-MLT-04`, `EVAL-AMB-03`, `EVAL-TECH-02`, and `EVAL-XTLK-02`; it worsened on `EVAL-SYM-04` and `EVAL-TECH-03`. The rank-1 count remained 24/43. No feature weights were fitted on these historical labels.

## E. Fine-tuning recommendation

Retain MiniLM as the baseline. The 31 unreviewed training rows, six weak development rows, and current complaint descriptions do not justify another BGE fine-tuning run or a learned ranker. Collect reviewer-approved source-backed positive spans, all valid alternatives, and genuinely invalid competing negatives by equipment family; use new documents for the sealed test. Only then compare a lightweight ranker or modest fine-tuning against MiniLM on reviewed development and once on the independent test. The suggested 50–100 answerable examples per equipment family is a data-collection pilot, not an accuracy guarantee.

## F. Reproduction, tests, Git and release state

With the existing local SQLite snapshot and cached models, run `python tools/op22_prepare_review.py`, `python tools/op22_rank_experiment.py`, `python tools/op22_company_diagnostic.py`, and `python tools/op22_latency_benchmark.py`. Private results live under `private_training_data/op22_review/`; source bytes, labels, checkpoints, and secrets are Git-ignored. `core/tests_op22_evidence.py` covers identifier and model mismatch, missing specifications, source/equipment gates, safety abstention, ranking, reviewer validation, false-negative guard, and pagination. Operation 21 index-integrity fixes and tests remain present.

The remote `main` ref was checked before staging and remained `74bb93ed09db1cc72759d7d0c255ec0f3d4b1b76`. Local tests use SQLite. MySQL 8.0 compatibility requires the existing GitHub Actions workflow on a pushed commit; no local MySQL result is claimed. No customer deployment or promotion occurred.

## G. Accuracy verdict

`EXPERIMENTAL_IMPROVEMENT_NOT_INDEPENDENTLY_VERIFIED`. The code and source/review safeguards are implemented and locally measurable, while independent test Hit@1, human adjudicated passage accuracy, grounded answer correctness, and 95% attainment remain unknown. The immediate blockers to an accuracy verdict are a named authorized technical reviewer, independently sampled new sources/questions, and approved multi-positive relevance and answer-support judgments.
