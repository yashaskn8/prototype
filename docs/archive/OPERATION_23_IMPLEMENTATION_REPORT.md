# Operation 23 implementation and measurement

## Release boundary

This operation changes private capture, source admission, review, reconciliation,
evaluation, and offline evidence experiments. It does not change customer serving,
the production MiniLM selection, Chroma collections, or the Starlly read-only API.
The starting point was `80e06d86c51df73544ada97ca829ba84523081ac` on `main`.

## Authenticated source admission

The existing authorised organisation-273 Starlly key was used only for read-only
collection. The 2026-10-16 private capture contains 12 assets, five complaints,
32 site-asset relationships, and nine downloaded attachments. The collector
records the organisation and tenant, hashes captured catalogue files and raw
and extracted attachment bytes, and keeps source files under ignored
`private_training_data/`. Admission rechecks the catalogued asset/attachment
relationship, organisation, tenant, metadata, byte and text checksums. It
extracts passages with stable content-bound IDs and records the source version.

The v2 sealed prior-source ledger includes 58 historical raw hashes and
normalized text/shingles/manual families for **all 166** local documents,
including 108 text-only records without a raw checksum. The earlier private
v1 ledger is preserved. New sources can be admitted as
`INDEPENDENT_SOURCE_CANDIDATE` without an existing Servy database document;
historical bytes, near duplicates, and revisions are quarantined. An executable
synthetic test proves a genuinely new authorized source can be frozen in a
final-test set, while historical and near-duplicate sources fail. The current
real capture has **zero** independent new sources: all nine attachments are
`HISTORICAL_EXCLUDED`. This is a data finding, not a workflow failure. The
original Operation 22 exclusion manifest is unchanged.

For a later organization-273 capture, run `tools/op23_capture_sources.py`
with `--expected-org 273` and a new path under `private_training_data/`,
then `tools/op23_admit_sources.py --capture ... --output ...`. Admission
refuses to overwrite an existing manifest. Generate a reviewer packet with
`tools/op23_prepare_reviewer_package.py --capture ... --admission ...
--output ...` (optionally `--suggestions ...` for old machine proposals).
Only private paths are accepted by these commands.

The capture manifest records authenticated collection but is not a signature
issued by Starlly. Admission relies on the authorized capture process and
byte checks; independent human approval is still required for ground truth.

## Complaint and review state

A private five-case reviewer package contains each original complaint,
authenticated asset relationship, available manual passages, machine
suggestions, clarification requests, and an unsigned `review_case_template`
with the fields expected by the freezer. Templates deliberately have no
technical positives and an incomplete index until reviewed. Complaint 43 has external asset 332,
whose catalogue relationship was checked against the existing Servy mapping.
Complaints 39–42 have no explicit asset relationship in the complaint or
inspected site-asset records. They remain equipment-clarification cases with
no technical positives. All five are `PENDING_REVIEW`; no reviewer or approval
was invented.

Ground-truth freezing requires an externally provisioned Ed25519 public key
with the `technical_reviewer` role, subject, organisation, tenant, and active
authorization window. The reviewer signature covers every case field except
the signature object, including case ID, exact passages, alternatives,
negatives, source version, answerability, timestamp, and query provenance.
No trusted real reviewer key is configured. Local edits cannot create an
approved case without the corresponding private signing key.

An answerable case requires exact positive passage IDs and source versions.
An unanswerable case may have no primary source or positive passage, but must
bind an authenticated complaint, resolved asset, and complete indexed source
universe. Missing-asset complaints enter only a clarification evaluation task.
Cross-role query/source-family overlap and historical final-test sources are
rejected. The frozen JSONL and digest manifest use exclusive creation.
An identity owner must supply `SERVY_REVIEW_TRUST_ROOT_JSON` from a trusted
location; candidates cannot create that trust root. After a reviewer signs
the exact case object, `tools/op23_freeze_reviewed_dataset.py` rechecks the
signature and admission before creating frozen files. The evaluator checks
them again on every run.

### Human actions still required

1. Name and authorize a technical reviewer and provision the trusted public
   key outside candidate files. Retain the private signing key with the
   company's identity owner.
2. Review complaint 43's exact equipment, manual passages, proposed positive
   and alternative passages, negative passages, answerability and safety facts;
   correct the machine suggestions where necessary, then sign the full case.
3. Resolve equipment context for complaints 39–42 through an explicit
   authoritative relationship, or review them strictly as clarification cases.
   Do not assign technical positives from inferred equipment.
4. Collect newly authorized, genuinely independent manuals and complaints for
   a final set. Review exact answer-bearing spans and approve unanswerable cases
   only after confirming the complete source universe and index.
5. Freeze a development configuration with a separately authorized evaluation
   owner's signed configuration lock before independent final evaluation.
   Separately judge generated answers and unsafe actions; retrieval passage
identity alone does not establish grounded answer correctness.

## Passage and corpus decisions

The offline verifier now returns passage support states separately from
corpus-level evidence states. A missing identifier in one passage cannot
produce a no-answer conclusion. A definitive no-answer requires a complete
authorized source universe, complete index, and complete candidate search.
The support checker inspects fault-code definitions, explicit model,
component/specification association, numeric values and units, procedure
detail, neighboring passages within the same document section, negation,
conflicting specifications, and safety-sensitive actions. These remain
machine judgments, not authenticated technical labels.

The historical diagnostic's machine support coverage moved from 6/43 in
Operation 22 to 19/43 with the corrected passage logic; 21/21 historical
unsupported/security cases still abstained in this diagnostic. Neither count
proves answer correctness, and the historical labels are repeatedly inspected.

## Offline retrieval experiment

The fair MiniLM reference remains 4/6 Hit@1 on six weak development questions
and 24/43 Hit@1, 35/43 Hit@5 on the historical relevance set. The fixed
offline evidence ranker remains 4/6 and 24/43, 36/43 respectively. A
categorical support-tier ordering was implemented as an **optional offline
experiment** to favor machine-classified answer-bearing passages without
adjusting keyword weights. It did not improve weak development (4/6 Hit@1),
and regressed historical Hit@1 to 21/43 and Hit@5 to 35/43. Across the 43
historical questions it improved zero positive-passage ranks, worsened five,
and left 38 unchanged. It is rejected for deployment; default ranking and
serving remain unchanged.

On this Windows development machine, the fixed feature-ranking stage had
5.052 ms P95 and the support-tier stage 5.492 ms P95. These timings exclude
model encoding, database/Chroma candidate retrieval, and answer generation;
they are **not** end-to-end latency. No independent reviewed accuracy or 95%
result exists.

## Provenance and evaluation

The read-only company audit verifies nine original organisation-273 documents
and their 69 chunks. A new reconciliation dry run proposes nine
`WOULD_INSERT_PENDING_AUTHORIZED_WRITE` mappings after checking tenant, asset,
external attachment, raw/text hashes, document, and every chunk. The other
157 local documents remain unresolved; no provenance row was written. MySQL
CI runs disposable-record mapping and uniqueness tests.

The reviewed evaluator keeps TRAIN, DEVELOPMENT, HISTORICAL_DIAGNOSTIC, and
INDEPENDENT_FINAL_TEST denominators separate. It supports multiple positive
passages, Hit@1/5/20, MRR@10, nDCG@10, candidate ceiling, asset/source
selection, answer coverage, evidence selection, unanswerable abstention,
clarification, observed cross-tenant leakage, and P95 end-to-end latency when
provided. Grounded answer correctness and unsafe-action rate remain unset
until separately authenticated human judgments are available. The final set
cannot run without a frozen configuration digest signed by an externally
trusted `evaluation_owner` key.

`tools/op23_evaluate_reviewed.py --generate-minilm` can produce a read-only
private MiniLM passage ranking directly from the admitted source universe,
after it validates the signed frozen cases. The configuration must identify
`op23-private-minilm-passage-v1`; an optional private JSONL output preserves
per-query predictions. This baseline uses original extracted passages and
does not write Chroma. It makes no answer-generation claim, so answer coverage
is zero and end-to-end latency remains unset for this retrieval-only mode.
Externally generated predictions can also be supplied for a separately
validated retrieval or answer pipeline.

## Verification and remaining limits

The targeted synthetic tests cover new and historical source admission,
near duplicates, wrong tenant/attachment, forged reviewer approval, changed
source, no-positive unanswerable, missing-asset clarification, source/passage
decision separation, incomplete index, procedure overlap, equivalent
positives, safety actions, provenance mappings, and evaluation math. The
broader `core` regression suite and MySQL/frontend GitHub Actions are release
gates. Private captures, reviewer packets, snapshots, labels, and model
checkpoints remain ignored by Git.

The blocker for an independent accuracy claim is the absence of both a new
independent company source set and authenticated technical approvals. The
software workflow can accept them when available. A 95% target is not claimed.
