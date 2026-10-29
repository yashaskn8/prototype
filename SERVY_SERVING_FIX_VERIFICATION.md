# Servy serving configuration and transparency correction

This change addresses confirmed mismatches from the October 29, 2026
synthetic red-team audit. It does **not** modify any live private deployment
environment, install model weights, start Ollama, or claim a new benchmark result.

## Source changes

- Default `SERVY_SERVING_EMBEDDING_PROFILE=bge-small`; real tenants select
  their separate `company-bge-small` index. Explicit `minilm` remains supported.
- Default V5 head pool 5, but `SERVY_RERANKER_ENABLED` remains **off** unless
  the approved private checkpoint has been installed and explicitly activated.
- Correct distinct model fingerprints in the release manifest.
- Block specific role-spoofing queries without broadly rejecting legitimate
  equipment questions.
- Report Ollama failure by exception class (no prompts or secrets); optional
  mode returns `extractive-local`. With provider `ollama` or `auto` and
  `SERVY_LLM_REQUIRE_OLLAMA=1`, an offline or failing generator abstains with
  engine `llm-unavailable` instead of quietly substituting extractive output.

## Deploying the previously evaluated BGE+V5 configuration

1. Verify MySQL authorization, the pinned BGE-small 384-D embedding model and
   schema-2.0/hash-valid BGE Chroma index. Do not trust collection name alone.
2. Confirm `SERVY_SERVING_EMBEDDING_PROFILE=bge-small` in the **running
   environment**. An existing `.env` can override code defaults.
3. Verify/install the approved private V5 checkpoint and fingerprint. Set
   `SERVY_RERANKER_ENABLED=1`, `SERVY_RERANKER_POOL_SIZE=5`,
   `SERVY_RERANKER_MODEL_PATH`, and `SERVY_RERANKER_EXPECTED_SHA256`.
   Strict pinned deployments can set `SERVY_RERANKER_REQUIRED=1`.
4. Run `python manage.py verify_v5_reranker` and
   `python manage.py test core.tests_serving_truth core.tests_production_rag_accuracy -v 2`
   on an isolated MySQL test database, then run the full core test suite.
5. Run actual asset-scoped customer and engineer API checks with the deployed
   model/index. Compare retrieval diagnostics and verify citations.
6. When the local Ollama model is available, set `SERVY_LLM_PROVIDER=ollama`.
   Optional `SERVY_LLM_REQUIRE_OLLAMA=1` prevents substitution of an
   extractive answer. Evaluate generated answer correctness, grounding,
   citation validity and abstention before user-visible deployment.

## Evidence limitations

The historical 38/41 = 92.68% authorized development Hit@1 used BGE+V5,
not the old default MiniLM-only configuration and not this commit.
The earlier MiniLM+V5 36/42 = 85.71% run used the 42-question historical
population. Neither result measures independently verified answer accuracy.
The historical API test report records 35/38 pass, 1 failed malformed request
(missing `site_id`), and 2 warnings. The separate security report records
28/30 passes and 2 warnings. They remain prior evidence until rerun.

**Release remains blocked** pending deployed environment verification,
independent company-data evaluation, data authorization, and end-to-end
answer-quality validation. No model has been promoted by this commit.
