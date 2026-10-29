# Servy RAG — blockers and source-fix evidence

## Completed in source control

- Embedding default: `bge-small`; V5 reranker head-pool default: five.
  V5 inference remains **opt-in** until its approved private weights are present.
- The release manifest now separates the experimental company fine-tuned BGE
  embedding SHA from the V5 cross-encoder model fingerprint.
- Synthetic role-spoofing patterns have been patched and tested.
- `python manage.py check_rag_serving`: safe read-only config diagnostic.
- `python manage.py check_rag_serving --strict`: checks BGE and company-BGE
  schema-valid Chroma indexes plus verified V5 inference with synthetic input.
  Neither command measures accuracy or grants data access.
- CI invokes the diagnostic and synthetic regression tests on MySQL.

## Deployment prerequisites

On the authorized host, verify the actual BGE index and tenant/asset scope.
Securely install the approved local V5 checkpoint and configure
`SERVY_RERANKER_ENABLED=1`, pool five, pinned SHA and model path. Run
the strict serving check and real customer/engineer Django APIs. Code
defaults and CI tests do not establish an active BGE+V5 production system.

The local Ollama generator must be made available by the infrastructure
owner, then independently evaluated for technical correctness, grounding,
citation validity and appropriate abstention. Do not call retrieval Hit@1
answer accuracy.

## Expert-reviewed relevance labels (not altered)

- `OP26-SLV-143-0`: silver gold 143, current top 142; neighboring chunks of
  the same troubleshooting section. Confirm whether both are valid evidence.
- `OP26-SLV-76-2`: silver gold 76 (calibration), top 79 (escalation).
  Check question intent against the complete approved source.
- `OP26-SLV-144-2`: semi-synthetic voltage query may not describe the
  intended USB communication-loss passage. Do not manufacture a rank-one hit.

An SME must document approved evidence, reviewer, source SHA, and rationale.
Preserve the frozen historical labels; create a separately versioned reviewed
benchmark if gold evidence genuinely changes. Confidential
`OP26-SLV-52-0` remains excluded from an unauthorized evaluation.

The last historical authorized development result is **38/41=92.68%**,
not a new result from this patch. No independent 95%+ result exists.

## External blockers

1. New source-family-disjoint expert-reviewed frozen benchmark.
2. Written company permission for private data evaluation/training.
3. End-to-end answer quality validation with an actual local generator.
4. Verified deployed BGE/V5, scope and complete MySQL CI.
5. Human review of the three disputed old examples.

**No production promotion is authorized by these code changes.**
