<USER_REQUEST>
# OPERATION 27 — SERVY RAG: MAXIMUM ACCURACY RECOVERY
## Gemini 3.8 Flash | HIGH Thinking | Autonomous ML and Retrieval Engineering

### SYSTEM ROLE

You are acting as a principal ML researcher, industrial information-retrieval specialist, senior RAG architect, evaluation scientist, and autonomous software engineering agent.

You are not here to produce another impressive-looking report.

You are here to identify and eliminate the actual causes of low retrieval accuracy, train better embedding models where justified, improve grounded technical answering, and deliver reproducible evidence of the strongest achievable performance.

**TARGET: 95%+ genuine technical RAG accuracy on an independently reviewed evaluation population.**

This target is an objective, not an outcome you are permitted to invent.

Repository: `yashaskn8/prototype`

Last verified remote HEAD: `75f38b2fcdc863339988a86e38ed1a0f5a564ad8`

Environment: Windows, 16 GB RAM, native MySQL 8.0, ChromaDB, CPU.

Models:
- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-small-en-v1.5`

Current production retrieval: MiniLM + sparse/BM25 + RRF + technical conditional reranking.

Operation 26 trained both models locally. The checkpoints are reported at:

- `ml_models/checkpoints/op26_minilm_ft/final`
- `ml_models/checkpoints/op26_bge_ft/final`

Treat local reports as claims to reproduce, not as proof.

## ABSOLUTE OPERATING RULES

1. Execute implementation and experiments, not just audits.
2. Inspect actual local code, MySQL records, Chroma collections, training manifests and checkpoints.
3. Keep a reproducible experiment ledger across context compaction.
4. Maintain an explicit distinction between embedding accuracy, retrieval accuracy, evidence completeness and final-answer correctness.
5. Never optimize on independent final-test labels.
6. Never manufacture technical facts, labels, reviewer approvals, benchmark data, or accuracy results.
7. Never bypass tenant, customer, asset, provenance, or safety controls to improve scores.
8. Keep private company documents and checkpoints out of public repositories and unauthorized external services.
9. Avoid unrelated architecture rewrites, approval frameworks, or generic infrastructure projects.
10. Use the existing MySQL installation. Do not reinstall or migrate the database again.
11. Preserve the original production models, vectors and application behavior.
12. Continue useful work when a single experiment fails, but stop unproductive experiment families after measured evidence of a plateau.
13. Train MiniLM and BGE sequentially, not simultaneously.
14. Do not deploy two dense embedding models to production simply to increase benchmark scores.
15. Deliver actual code, real measurements and a justified final model recommendation.

---

# PHASE 0 — REPRODUCE OPERATION 26 BEFORE TRUSTING IT

This is mandatory and must be completed before further expensive training.

Operation 26 reported:

| Model | Reported Recall@1 | Recall@5 | MRR@10 |
|---|---:|---:|---:|
| Base MiniLM | 0.3438 | 0.5000 | 0.4213 |
| Fine-tuned MiniLM | 0.3281 | 0.5781 | 0.4229 |
| Base BGE | 0.2031 | 0.3906 | 0.2951 |
| Fine-tuned BGE | 0.3281 | 0.4688 | 0.3944 |

Previously, the Operation 24 complete hybrid serving system reported 32/43 Hit@1.

These numbers must not be compared directly until their denominators, corpus, retrieval depth, dataset roles and retrieval pipelines are reconciled.

## P0.1 — Audit the denominator

The historical benchmark contains 64 total cases, previously partitioned into 43 relevance cases and 21 abstention/security cases.

Operation 26 reports values such as 0.3281, numerically consistent with approximately 21/64.

Determine whether its retrieval metrics incorrectly use all 64 cases as the denominator.

If so, repair the evaluator.

Report separately:

- Answerable relevance queries.
- Unsupported queries.
- Equipment-clarification queries.
- Security and authorization tests.

Metrics for relevance must use the valid relevance denominator.

Unsupported-question evaluation must use its own denominator.

Never mix different tasks into one undefined accuracy score.

## P0.2 — Audit benchmark independence

Earlier Servy operations repeatedly inspected the historical 43 relevance questions.

The benchmark is therefore not an untouched independent final evaluation merely because its file is frozen.

Check training/validation/test overlap by:

- Exact query text.
- Semantic paraphrase.
- Positive passage.
- Source section.
- Document version.
- Source family.
- Equipment family.
- Near-duplicate manual.
- Generated question templates.

Preserve original benchmark results as historical diagnostics.

Do not manipulate the ground truth.

## P0.3 — Reproduce the trained checkpoints

Verify that both fine-tuned checkpoints:

- Exist.
- Have the recorded content fingerprints.
- Load independently.
- Produce finite 384-dimensional embeddings.
- Use the correct query prefixes and passage formatting.
- Match their corresponding experimental Chroma indexes.
- Produce the reported query-level rankings under the recorded conditions.

If the original 92-step training runs cannot be reproduced, identify precisely why.

Do not claim an experiment was verified solely because its JSON report exists.

## P0.4 — Produce the real baseline

Reproduce, on the same eligible source universe:

1. Base MiniLM dense.
2. Op26 MiniLM fine-tuned dense.
3. Base BGE dense.
4. Op26 BGE fine-tuned dense.
5. Existing production hybrid retrieval.
6. Hybrid retrieval using Op26 fine-tuned MiniLM.
7. Hybrid retrieval using Op26 fine-tuned BGE.

The production comparison must invoke the real retrieval entry point, including candidate admission, fusion, reranking, deduplication and final truncation.

Save actual ranked chunk IDs and their eligibility decisions.

**No additional training begins until the comparative baseline is defensible.**

---

# PHASE 1 — FIND THE TRUE ACCURACY BOTTLENECKS

The current dense models are far below the 95% target.

Do not assume the problem is insufficient training epochs.

For every failed answerable diagnostic query, identify exactly one primary failure class, with secondary causes if needed.

Failure classes:

A. No authoritative answer exists in the eligible corpus.

B. Correct source not ingested.

C. Document parsed incorrectly.

D. Correct technical facts split across chunks.

E. Incorrect equipment applicability.

F. Incorrect query interpretation.

G. Dense retrieval failed to retrieve the correct source.

H. Sparse retrieval failed to retrieve it.

I. Candidate discarded by relevance gate.

J. Candidate retrieved but fusion ranked it too low.

K. Correct passage lost during deduplication or truncation.

L. Reranker promoted a misleading passage.

M. Correct multiple-source evidence incorrectly evaluated against a single chunk.

N. Ground truth is incorrect or incomplete.

O. Correct retrieval but incorrect final generated answer.

Measure the relative contribution of each failure class.

For every case, record whether the expected answer is actually supported by original source text.

## CRITICAL RULE

If the correct answer-bearing passage does not exist in the eligible corpus, no embedding model can reliably recover it.

Report such cases as source-coverage gaps, not model failures to be solved by fabricated text.

Prioritize the highest-frequency recoverable causes.

Do not spend another operation adding arbitrary heuristic bonuses.

---

# PHASE 2 — REPAIR THE TRAINING DATA

This is the highest-impact model-training stage.

Operation 26 reported 353 training examples, 42 development examples and 207 indexed chunks.

It also used 343 same-document hard negatives.

These counts do not prove high-quality supervision.

## 2.1 Verify source authenticity

Inspect the actual original-document provenance for every training pair.

Confirm each source is authorized for model training.

Distinguish:

- Verified original company manual.
- Historical local document with incomplete provenance.
- Source-derived silver question.
- Human-reviewed relevance label.
- Synthetic or demonstrative material.
- Duplicated or revised document.

Do not treat all MySQL records as independently verified company originals.

## 2.2 Audit positive examples

For every training positive, verify:

- The requested technical fact exists in the passage.
- The correct equipment and model apply.
- The answer is not merely semantically related.
- The passage is complete enough for the intended question.
- The source hash matches.
- The cited section/version is correct.

Reject unsupported positives.

Permit several valid positives where multiple source sections genuinely answer the same question.

## 2.3 Audit hard negatives

The current same-document negative mining can easily create false negatives.

Inspect all 343 reported hard negatives.

Do not assume that different section IDs imply different relevance.

Identify cases where the negative:

- Contains an alternative correct answer.
- Is a continuation of the same procedure.
- Provides a prerequisite needed to answer the question.
- Defines the fault while the positive describes the remedy.
- Contains another valid measurement.
- Is a near-duplicate of the positive.

Implement false-negative-aware sampling.

Use multiple-positive-aware training where the loss supports it.

Hard-negative examples should be difficult because they are deceptively relevant but genuinely wrong or insufficient, not because they have been incorrectly labeled.

## 2.4 Improve supervision diversity

Use authorized original manuals and genuine technician questions.

Build source-backed examples for:

- Exact fault codes.
- Equipment-specific symptoms.
- Component failures.
- Maintenance intervals.
- Repair procedures.
- Technical specifications.
- Thresholds and ranges.
- Electrical and mechanical measurements.
- Procedure prerequisites.
- Multi-symptom diagnosis.
- Technician abbreviations.
- Paraphrases and typos.
- Wrong-model distractors.

Do not create hundreds of nearly identical examples from the same passage just to increase dataset size.

Measure distinct question intents, source sections, documents and equipment families.

Do not export proprietary passages to an external generation service without authorization.

## 2.5 New dataset discipline

Create separate:

- TRAIN
- DEVELOPMENT
- HISTORICAL_DIAGNOSTIC
- INDEPENDENT_FINAL_TEST

Use grouped splitting to avoid leaking source-family duplicates.

Keep all machine-generated examples explicitly SILVER.

Human-reviewed evidence requires genuine approval from an authorized reviewer.

If the final-test set cannot be populated independently, report independent accuracy as NOT MEASURED.

---

# PHASE 3 — ACCURACY-FIRST DOCUMENT REPRESENTATION

Before retraining, investigate whether the models receive enough information to learn the technical task.

Compare:

A. Existing chunk representation.

B. Heading + body.

C. Equipment model + section hierarchy + body.

D. Answer-bearing technical span + verified surrounding context.

E. Parent-section retrieval with child-passage ranking.

Retain original source offsets and citations.

Technical documents may contain:

- Multi-page procedures.
- Tables.
- Error-code matrices.
- Equipment-specific settings.
- Safety prerequisites.
- Numeric ranges.
- Component-specific values.

A chunk containing only half a procedure may be insufficient even when it ranks first.

Implement robust document structure handling where failures justify it.

Do not invent missing table cells or OCR content.

For extraction failures, preserve the original document and record the unresolved source gap.

Test whether better representations improve pretrained MiniLM and BGE before attributing everything to insufficient fine-tuning.

---

# PHASE 4 — TRAIN BOTH MODELS MORE INTELLIGENTLY

Use the repaired source-backed dataset.

Do not blindly continue from the previous four-epoch checkpoints.

For both architectures, compare training from the pretrained base and, where justified, controlled continuation from the Operation 26 checkpoint.

Train MiniLM first, BGE second.

## 4.1 Loss-function experiments

Evaluate retrieval-appropriate objectives compatible with the actual data.

Candidates include:

- MultipleNegativesRankingLoss.
- CachedMultipleNegativesRankingLoss where resource appropriate.
- Explicit hard-negative contrastive training.
- Margin-based ranking using verified negative labels.
- Multiple-positive-aware objectives when needed.

Do not choose a loss merely because its training value decreases.

Check whether the loss treats another valid positive as an implicit negative.

Check whether batches contain collisions.

Check whether the selected negatives are genuinely incorrect for the query.

Only run controlled configurations justified by development evidence.

## 4.2 MiniLM optimization

Start with base MiniLM and the repaired dataset.

Use model-specific text formatting.

Evaluate bounded learning-rate, batch-size and epoch experiments.

Avoid catastrophic forgetting of general semantic relevance.

Prioritize:

- Correct first-ranked technical passage.
- Better top-10 candidate recall.
- Wrong-model discrimination.
- Fault-code and specification grounding.

Select checkpoints using held-out development retrieval performance, not training loss alone.

## 4.3 BGE optimization

Repeat the same scientifically controlled process for BGE-small.

Preserve its query instruction prefix.

Compare against original BGE and the Operation 26 checkpoint.

Investigate where BGE improves specific industrial nomenclature and where it still falls behind MiniLM.

Do not infer that a tokenizer's fragmented subwords are conclusively the root cause without experimental evidence.

## 4.4 Training schedule

Use short, meaningful experimental rounds.

After each round:

1. Save the checkpoint.
2. Rebuild its isolated vector index.
3. Run development retrieval.
4. Compare against the preceding checkpoint.
5. Inspect query-level improvements and regressions.
6. Decide whether to continue, modify the dataset, change the objective, or stop.

Do not repeat training when the observed problem is missing source coverage or incorrect retrieval filtering.

Reject a checkpoint that improves training loss but degrades meaningful retrieval.

---

# PHASE 5 — RECOVER ANSWER-BEARING CANDIDATES

Fine-tuning is only one part of RAG accuracy.

Inspect the existing serving system:

- `_knowledge_candidates`
- `_dense_score_candidates`
- `_sparse_score_candidates`
- `_reciprocal_rank_fusion`
- `conditional_swap_rerank`
- Final reranker.
- Deduplication.
- Final answer context selection.

Measure the correct evidence's rank at every stage.

## Candidate retrieval experiments

Test:

1. Fine-tuned MiniLM with current dense candidate gate.
2. Fine-tuned MiniLM with controlled evidence-aware candidate rescue.
3. MiniLM + existing BM25 and character n-grams.
4. Exact fault-code and equipment identifier retrieval.
5. Section-aware technical evidence retrieval.
6. Best candidate union with bounded RRF.
7. Equivalent experiments using fine-tuned BGE.

Measure Recall@5, @10, @20 and @50 where meaningful.

Do not globally lower similarity thresholds to increase recall without measuring irrelevant retrieval.

Maintain strict authorization before exposing candidates to ranking.

Do not loosen tenant or asset boundaries.

The immediate retrieval objective is to ensure that actual answer-bearing evidence reaches the final ranking stage.

---

# PHASE 6 — ELIMINATE WEAK TECHNICAL RANKING

Replace benchmark-specific ranking shortcuts with generalizable evidence selection.

Review the current conditional swap logic.

Investigate:

- Repeated swap stability.
- Procedural-density bias.
- First-mentioned-symptom bias.
- Generic-heading demotion.
- Technical phrase matching.
- Numeric matching.
- Wrong-model promotions.

Implement only corrections supported by failure evidence.

Prioritize:

### Query intent awareness

Distinguish:

- What does this fault mean?
- Why is this happening?
- How should the fault be diagnosed?
- What exact specification applies?
- Which procedure is authorized?
- What evidence is missing?

### Technical entity matching

Recognize:

- Equipment.
- Manufacturer.
- Model.
- Component.
- Fault code.
- Symptom.
- Unit.
- Numeric value.
- Measurement condition.
- Safety constraint.

### Technical contradiction awareness

Differentiate:

- High versus low.
- Above versus below.
- Open versus closed.
- Allowed versus prohibited.
- Minimum versus maximum.
- Normal versus abnormal.
- Correct model versus similar model.

### Multi-evidence selection

Some questions require two or more supporting passages.

Allow compatible multi-passage evidence without inventing a shared diagnosis.

Preserve procedure continuity and source identity.

A top-ranked passage is not automatically the correct answer.

---

# PHASE 7 — IMPROVE GROUNDED ANSWER CORRECTNESS

The business needs a reliable troubleshooting system, not just an embedding benchmark.

Test the complete workflow:

Technician question
→ Equipment resolution
→ Authorized candidate retrieval
→ Evidence ranking
→ Evidence completeness
→ Answer generation
→ Technical claim validation
→ Citations or abstention.

## Grounding requirements

Every consequential technical claim must be supported by authorized source evidence.

Verify:

- Fault meaning.
- Correct equipment identity.
- Component-to-specification association.
- Numeric values and units.
- Diagnosis and remedy distinction.
- Required safety prerequisites.
- Procedure completeness.
- Correct source citations.

Do not generate unsafe maintenance procedures from loosely related text.

Do not infer missing torque, voltage, pressure or temperature specifications.

If source evidence is incomplete, state exactly what is missing.

For genuinely answerable questions, avoid needless abstention by recovering and combining complete verified evidence.

Evaluate both answer correctness and answer coverage.

Do not raise the apparent correctness rate by refusing nearly every question.

---

# PHASE 8 — REAL EVALUATION, NOT SCORE THEATER

Define the final metrics before making further optimization decisions.

Evaluate three separate tasks.

## A. Retrieval accuracy

On answerable, source-supported questions:

- Hit@1.
- Hit@5.
- Recall@10.
- Recall@20.
- MRR@10.
- nDCG@10.
- Correct equipment selection.
- Correct authoritative source selection.
- Complete evidence coverage.

## B. Grounded-answer correctness

Using independently reviewed technical references where available:

- Factually correct answer rate.
- Complete answer rate.
- Correct numerical specification rate.
- Correct procedure rate.
- Citation support rate.
- Unsupported technical claim rate.

## C. Abstention and safety

On unsupported and unsafe requests:

- Correct abstention.
- Wrong-equipment rejection.
- Unauthorized evidence rejection.
- Contradiction detection.
- Clarification behavior.
- False unsupported-answer acceptance.

Report real counts alongside percentages.

Preserve the original relevance benchmark as a historical diagnostic.

Never describe that benchmark as fresh, untouched evidence.

Do not tune and evaluate on the same query population while claiming generalization.

Prefer new independently reviewed technician cases for final certification.

Require a sufficiently representative sample across equipment families and query types.

When the sample is small, report uncertainty and limitations.

## SUCCESS TARGETS

Engineering targets:

- At least 95% correct first-ranked answer-bearing retrieval on qualifying independent questions.
- At least 95% grounded-answer correctness on qualifying independent answerable questions.
- High supported-answer coverage.
- Strong unsupported-query rejection.
- Zero observed unauthorized cross-tenant disclosures in tested scenarios.
- Acceptable CPU latency and memory.

These are goals, not unconditional pass conditions to manipulate.

If the verified result is below target, preserve it accurately.

---

# PHASE 9 — TEST REAL-WORLD GENERALIZATION

Build difficult new diagnostic tests from authentic authorized technical material, keeping development tests separate from final evaluation.

Include:

- Unseen equipment descriptions.
- Previously unseen fault codes.
- Technician slang.
- Typographical mistakes.
- Different manufacturer models.
- Incorrect-model distractors.
- Multiple concurrent symptoms.
- Measurements with similar numbers.
- Conflicting operating ranges.
- Table-based troubleshooting facts.
- Procedures spanning multiple chunks.
- Missing safety prerequisites.
- Unsupported faults.
- Incomplete equipment context.
- Contradictory document revisions.
- Near-duplicate source sections.

Where these cases are machine-derived, label them SILVER.

Do not claim they represent independent human-reviewed field performance.

Test the exact production-intended retrieval path.

---

# PHASE 10 — REPRODUCIBILITY AND ANTI-FABRICATION GATES

Every published metric must be backed by an executable evaluation run.

Record:

- Git commit.
- Model checkpoint SHA-256.
- Dataset SHA-256.
- Document/index fingerprint.
- Query and passage preprocessing versions.
- Experiment configuration.
- Actual model outputs.
- Actual ranked source IDs.
- Expected evidence IDs.
- Query-level pass/fail results.
- Aggregate metric calculation.
- Latency and resource measurements.

Add verification that independently recomputes metrics from raw per-query ranking outputs.

Fail if:

- The denominator is inconsistent.
- Results lack a valid evaluation record.
- A benchmark score is hardcoded into the evaluator.
- The expected positive is inserted into the ranking.
- An evaluated query leaked into training.
- A model uses another model's embeddings.
- One model is evaluated against a different authorized corpus.
- A weakly supervised label is presented as human-reviewed.
- A checkpoint fingerprint does not match its vector collection.
- Final-test data is used for hyperparameter selection.

Passing unit tests alone is not proof of accuracy.

The final scorecard must be reproducible from permitted local artifacts.

---

# PHASE 11 — KEEP COMPANY INFRASTRUCTURE COST LOW

Train both models offline.

For production, recommend one dense model.

Use the strongest justified choice among:

- Base MiniLM.
- Fine-tuned MiniLM.
- Base BGE-small.
- Fine-tuned BGE-small.

Combine the chosen dense model with existing low-cost sparse retrieval and bounded technical evidence selection.

No mandatory GPU.

No paid external embedding API.

No paid reranking API.

No permanent second dense embedding model unless separately authorized after a demonstrated material benefit.

Measure end-to-end latency, not only isolated model encoding time.

Avoid unbounded reranking over entire corpora.

Make the architecture practical for an initial 200-document company deployment.

---

# PHASE 12 — PREPARE THE WINNING MODEL FOR THE ORIGINAL SERVY REPOSITORY

The company intends to integrate the trained model into its original Servy repository.

Therefore, ensure the selected checkpoint is portable and not dependent on the prototype's database schema.

Create a local release candidate bundle containing:

- Model weights.
- Tokenizer.
- Model configuration.
- Base model identity and revision.
- Checkpoint SHA-256.
- Query-prefix configuration.
- Passage-format specification.
- Embedding normalization requirements.
- Embedding dimension.
- Dependency versions.
- Vector reindexing requirements.
- Inference example.
- CPU performance report.
- Evaluation evidence.
- Known failure cases.
- Integration validation checklist.

Provide a simple loading adapter using Sentence Transformers.

Do not include:

- MySQL data.
- SQLite data.
- Company documents.
- Sensitive evaluation questions.
- API credentials.
- Customer records.
- Full prototype source tree unless separately authorized.

The original company repository must re-embed its authorized documents using the selected checkpoint and identical preprocessing.

Ensure the model bundle can be validated without requiring the entire prototype.

Do not confuse portability with automatic production readiness.

---

# PHASE 13 — FINAL ENGINEERING EXECUTION

Complete all justified actionable experiments.

Protect existing user changes and experimental checkpoints.

Do not force-push.

Do not delete original Chroma collections.

Do not mutate original company documents.

Keep newly trained checkpoints in private, versioned paths.

Keep customer-serving configuration unchanged until approved.

Add targeted tests for verified changes.

Run relevant local tests and MySQL integration tests.

Commit and push only authorized, non-sensitive source changes.

Verify the actual remote commit and GitHub Actions conclusion.

If CI fails, diagnose and repair failures introduced by this operation.

Do not report incomplete CI as passed.

## MANDATORY FINAL SCORECARD

Provide separate results for historical diagnostics, development, silver generalization tests, and independently reviewed final evaluation.

| Configuration | Dataset | Hit@1 | Recall@10 | MRR@10 | Grounded correctness | P95 |
|---|---|---|---|---|---|---|
| Base MiniLM | Actual | Actual | Actual | Actual | Actual | Actual |
| Op26 MiniLM FT | Actual | Actual | Actual | Actual | Actual | Actual |
| Improved MiniLM FT | Actual | Actual | Actual | Actual | Actual | Actual |
| Base BGE | Actual | Actual | Actual | Actual | Actual | Actual |
| Op26 BGE FT | Actual | Actual | Actual | Actual | Actual | Actual |
| Improved BGE FT | Actual | Actual | Actual | Actual | Actual | Actual |
| Existing hybrid production | Actual | Actual | Actual | Actual | Actual | Actual |
| Best complete RAG pipeline | Actual | Actual | Actual | Actual | Actual | Actual |

Never substitute reported historical metrics for freshly measured results.

For unavailable metrics, write NOT MEASURED.

## FINAL REPORT REQUIREMENTS

Provide:

1. Corrected Operation 26 metric denominators.
2. Reproduced baseline scores.
3. Training data quality findings.
4. Specific retrieval failure categories and counts.
5. Models actually trained in this operation.
6. Actual optimizer steps, training loss and development scores.
7. Model checkpoint fingerprints.
8. New retrieval methods implemented.
9. Before-and-after query-level improvements and regressions.
10. Grounded-answer results.
11. Independent evaluation status.
12. Resource and company cost impact.
13. Winning model and evidence supporting its selection.
14. Portable company-integration bundle path.
15. Tests executed.
16. Final Git commit and CI verification.
17. Remaining limitations preventing a genuine 95% claim.

---

# HARD COMPLETION CONTRACT

You are not finished merely because:

- Two models have been trained.
- Training losses decreased.
- A benchmark says 95%.
- A synthetic dataset says 100%.
- New heuristics were implemented.
- 500+ Django tests passed.
- A model checkpoint loads.
- A report is marked COMPLETE.

You are finished when the available legitimate engineering improvements have been implemented and evaluated, experiments have been accepted or rejected on measured evidence, and all final accuracy claims are properly qualified.

If independent 95% accuracy is not established, say so explicitly and provide the strongest verifiable result.

**Never misrepresent a source-derived silver score as production accuracy.**

**Never sacrifice technical safety or source authorization to inflate accuracy.**

**Never invent missing evidence.**

## FINAL DIRECTIVE — EXECUTE

STOP treating additional fine-tuning epochs as the default solution.

FIRST: Audit the Operation 26 metric denominator and data leakage.

SECOND: Repair positive/negative supervision.

THIRD: Recover actual answer-bearing passages.

FOURTH: Fine-tune MiniLM and BGE using legitimate development evidence.

FIFTH: Compare models against the complete hybrid serving system.

SIXTH: Improve grounded answer correctness.

SEVENTH: Package the strongest model for original Servy repository integration.

Spend your effort on real experiments, implementation and measurable results—not paperwork.

**BUSINESS GOAL: 95%+ RELIABLE TECHNICAL RAG ACCURACY.**

**ENGINEERING RULE: PROVE THE IMPROVEMENT OR REPORT THE LIMITATION.**

Begin immediately with the actual local Operation 26 artifacts and the 64-versus-43 denominator investigation. Do not begin another training run until the scoring discrepancy is resolved.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-17T15:54:56+05:30.

The user's current state is as follows:
Active Document: c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\core\tests_operation_16_finetuning_integrity.py (LANGUAGE_PYTHON)
Cursor is on line: 1
Other open documents:
- c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\core\ml\embeddings.py (LANGUAGE_PYTHON)
- c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\tools\verify_all_records.py (LANGUAGE_PYTHON)
- c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\tools\verify_mysql_connection.py (LANGUAGE_PYTHON)
- c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\tools\inspect_chroma.py (LANGUAGE_PYTHON)
- c:\Users\prana\OneDrive\Desktop\SURVY-RAG\prototype\servy_rag\settings.py (LANGUAGE_PYTHON)
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.8 Flash (Medium) to Claude Opus 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>