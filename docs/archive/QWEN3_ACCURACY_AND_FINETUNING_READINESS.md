# Servy: evidence-driven Qwen3 + V5 accuracy experimentation

**Status: source integration and preflight only.** GitHub CI has no private model
checkpoints, enterprise source files or human-labeled independent evaluation.
This feature is disabled by default. It is not a claim that live accuracy exceeds
the baseline, or that the 88.10% 42-question research Hit@1 transfers to Django.

## Architecture and protection

1. Django authorizes tenant, customer, asset and source eligibility **before**
   vector search. Chroma queries remain restricted to authorized document IDs.
2. The Qwen3-Embedding-0.6B 1024-dimensional embedder retrieves candidates
   using distinct `encode_query` / `encode_document` APIs, normalized
   vectors and max sequence length 256, as in the research.
3. Dedicated versioned collections:
   `servy_dense_qwen3_06b_v1` (non-company data) and
   `servy_company_qwen3_06b_v1` (real-company data).
   They are never mixed with the legacy 384-dimensional MiniLM/BGE collections.
4. Per-vector tenant/document/chunk identity, content fingerprint, index version
   and embedding model must match the current authorized database chunk.
   A full Chroma collection identity manifest is mandatory for Qwen3.
5. The existing V5 BGE *cross-encoder* ranks only authorized top-five passages.
   Retain the legacy MiniLM candidate retrieval when Qwen3 is unavailable or
   experiment flags are disabled. No changes to V4/V5 weights.
6. The experimental Qwen3 relevance gate defaults to 0.35. Its optimal value
   **has not been evaluated**; tune on separate development data, then freeze
   before holdout measurement.

## Local Windows setup (no Docker)

**Do not activate before a full Qwen3 index is complete.** The user's
`prototype/best_model` is V5 **reranker** weights; it is **not** a Qwen3
embedding checkpoint. A separate locally downloaded full Qwen3 embedding model
is required. The model is loaded **offline only**. Do not place weights in Git.

In your local `prototype/.env` (untracked), set the Qwen path; leave the
serving feature flags disabled at first:

```ini
SERVY_QWEN3_MODEL_PATH=ml_models/checkpoints/qwen3_embedding_0_6b
SERVY_QWEN3_ENABLED=0
SERVY_QWEN3_INDEX_VERIFIED=0
SERVY_USE_EXPERIMENTAL_RAG=0
SERVY_SERVING_EMBEDDING_PROFILE=qwen3
SERVY_QWEN3_DENSE_GATE=0.35
```

Verify the model is genuinely `Qwen/Qwen3-Embedding-0.6B`, outputs 1024-D
vectors and has working `encode_query/encode_document` methods. Ensure
the correct `sentence-transformers` dependency and sufficient local memory.
The existing V5 cross-encoder loads separately and needs its own verified hash.

The actual index build is an **explicit opt-in write** to a new collection,
never to MiniLM's collection; schedule it only when company data/index
operations are authorized and a snapshot/backup is available. Example:

```powershell
python manage.py shell -c "from core.ml.indexer import build_index; print(build_index('company-qwen3'))"
python manage.py verify_qwen3_index --profile company-qwen3
```

For non-company/synthetic tenant collections, use `qwen3` instead of
`company-qwen3` in both commands. The full index build does not delete
stale vectors for real-company Qwen3; any unresolved stale/orphan vectors
must be investigated separately, not forcibly purged.

Only after `QWEN3 INDEX VERIFIED` succeeds and you explicitly approve a
local experimental run, set:

```ini
SERVY_QWEN3_ENABLED=1
SERVY_QWEN3_INDEX_VERIFIED=1
SERVY_USE_EXPERIMENTAL_RAG=1
SERVY_SERVING_EMBEDDING_PROFILE=qwen3
```

Django routes to `company-qwen3` for real company tenants, and `qwen3` for
other tenants. Index completeness must be separately verified for **each**
serving tenant category. An explicit flag is an operator assertion, not
automatic proof of accuracy. Missing/incompatible model or collection falls
back to existing sparse retrieval; do not represent it as Qwen3 success.

## Training readiness (research only)

AI triage earlier selected 111 training queries from 353; 242 cases remain
unresolved. The 42 development questions are reused, not an independent
holdout. No automated proposal is expert approval.

Use the following local preflight with **your actual PRIVATE paths**. Do not
commit the input files:

```powershell
python tools/servy_v5_training_preflight.py --split-dir "PATH_TO_PRIVATE/split" --corpus "PATH_TO_PRIVATE/corpus.jsonl" --families "PATH_TO_PRIVATE/families.jsonl" --candidates "PATH_TO_PRIVATE/candidates.jsonl" --candidate-train "PATH_TO_V5/train_keep_ai_provisional.jsonl" --manifest "PATH_TO_V5/manifest.json"
```

Expected output status if everything matches:
`READY_FOR_EXPLORATORY_FINE_TUNING`, `production_promotion_allowed=false`,
`expert_labels_approved=false`, `independent_holdout_verified=false`.

**Do not run another supervised fine-tuning epoch on the 42 dev questions.**
Build a completely new human-reviewed, source-family-isolated evaluation set
and compare MiniLM+V5, Qwen3+V5, and Qwen3 alone at equal candidate budgets.
Track candidate Recall@10, Hit@1, MRR@10, citation correctness, abstentions,
tenant/equipment isolation, and p50/p95 latency. Do not optimize to a 0.01%
difference on 42 questions: one answer changes Hit@1 by 2.38 percentage
points, and a claimed 0.01% delta cannot be resolved by this sample size.

Only fine-tune further if failure evidence reveals recoverable ranking errors,
with expert-validated positives and hard negatives from the training split.
Retain V5 as rollback checkpoint. A CI pass verifies software regression,
**not** the model's quality on private enterprise data.

## Automated source-code verification

```powershell
python manage.py test core.tests_qwen3_accuracy -v 2
```

No Docker, model downloads, or private files are required for these mocked
unit checks. Real-weight tests must be run locally with the authorized company
records and a separate reproducible evaluation. Keep the current stable
candidate retrieval available until those checks pass.
