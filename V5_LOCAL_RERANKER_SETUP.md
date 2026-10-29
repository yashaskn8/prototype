# Servy V5 BGE reranker — local integration

This integrates the **V5 BGE cross-encoder reranker** into the already-authorized
Django retrieval pipeline. It does NOT install Qwen3 as the serving embedding
retriever; existing MiniLM/Chroma retrieval remains the default. A historical V5 diagnostic reported **88.10% Hit@1**, while the latest
V5/V6 comparison reported **83.33% Hit@1 (35/42)** and **88.10% Hit@5 (37/42)**
on the reused development set. The difference in evaluation configurations is
not yet reconciled. Neither is an independent live-UI accuracy guarantee.

## Windows (no Docker)

1. Download the **entire** `best_model/` from your verified Google Drive V5
   experiment, including `model.safetensors`, `config.json`, tokenizer files.
   Keep the Google Drive copy.

2. Recommended local location, relative to the directory containing `manage.py`:

   ```text
   SURVY-RAG/prototype/ml_models/checkpoints/servy_v5/best_model/
   ```

   `ml_models/checkpoints/` is already excluded by `.gitignore`, so weights
   cannot be committed by an ordinary `git add`. If your model is instead
   `SURVY-RAG/prototype/best_model/`, set `SERVY_RERANKER_MODEL_PATH=best_model`.
   If it is under `SURVY-RAG/ml_models/` (a **sibling** of `prototype/`),
   use an explicit absolute path or `../ml_models/.../best_model`.
   Relative paths resolve from the Django repository root.

   For a new install, use the offline verified installer instead of manually
   copying a checkpoint into the active location. Once you have the complete
   V5 `best_model` directory on your Windows machine, run from `prototype`:

   ```powershell
   python tools/install_servy_v5_checkpoint.py --source "C:\\path\\to\\verified\\best_model"
   ```

   The installer copies into `ml_models/checkpoints/servy_v5/best_model` by
   default, verifies the **exact V5 weights SHA-256 before and after copying**,
   rejects symlinks, and never overwrites an existing installation. It is
   offline and does not upload checkpoints or labels to GitHub. If the V5 folder
   is currently only in Colab, first transfer that same verified folder from
   your approved Drive storage to the serving machine; the installer cannot
   access your private Colab filesystem. **Do not install the V6 checkpoint.**

3. In your local `prototype/.env` (not tracked in Git), configure:

   ```ini
   SERVY_RERANKER_ENABLED=1
   SERVY_RERANKER_REQUIRED=1
   SERVY_RERANKER_MODEL_PATH=ml_models/checkpoints/servy_v5/best_model
   SERVY_RERANKER_EXPECTED_SHA256=b1cf23ba0806f0057d73a5ab93fde88d634fc278f71b27f3a2a915afd8f8bcda
   SERVY_RERANKER_DEVICE=cpu
   SERVY_RERANKER_POOL_SIZE=5
   ```

   V5 weights are approximately 2.27 GB; local CPU inference may be slow or
   memory-intensive on a 16 GB laptop. `SERVY_RERANKER_DEVICE=cuda` is possible
   only when compatible CUDA/PyTorch and GPU memory are actually available.
   Do not assume Colab's L4 exists on your Windows laptop.

4. From the `prototype` directory, with your normal virtual environment
   and `requirements.txt` installed, verify the model without running a live
   customer request:

   ```powershell
   python manage.py verify_v5_reranker
   ```

   Expected: `SERVY V5 RERANKER VERIFIED: ...`. This executes the actual
   serving-time reranker on one **synthetic knowledge passage**, including the
   SHA-256-pinned offline checkpoint load, raw-logit scoring, and verified V5
   model provenance. Missing weights, fallback, or invalid scores make the
   command fail rather than produce a false success. It does **not** certify
   tenant authorization, HTTP requests, benchmark accuracy, or deployment.

5. Start your normal Django/MySQL and React processes. Log in and use
   **Customer AI Support** or **Engineer Copilot**. Authorized retrieved
   technical chunks can now be reranked by V5. The UI does not automatically
   display a verified model badge or the research benchmark score.
   Both API responses expose `reranker_applied`: it is `true` only when a
   knowledge passage received a finite cross-encoder score in that request.
   For auditability, API responses also expose `reranker_model`. The value
   `servy_v5_bge_verified` is emitted only after finite scores from the
   locally loaded checkpoint with the known V5 SHA-256; `legacy_ms_marco`
   and `local_cross_encoder` distinguish other opt-in configurations.
   The React UI displays a V5 status line only for verified V5 scoring.
   This indicates model usage, **not** answer correctness or source validity.
   It is `false` when V5 is disabled, unavailable, or falls back after a
   scoring error. `retrieval_backend` describes candidate retrieval, not V5.

## Safety and rollout

- If `SERVY_RERANKER_ENABLED=0` (the default), existing behavior is unchanged.
- For deployment, set `SERVY_RERANKER_REQUIRED=1` **in addition to** the enabled flag, full local path, and approved digest. Django startup then runs one synthetic passage through the actual SHA-pinned V5 scorer and refuses to start if the checkpoint is absent, incompatible, or returns invalid scores. This opt-in strict mode is off by default. It is a model smoke check, not an independent benchmark or a live user-query test.
- With the feature enabled, a nonexistent checkpoint or wrong weights SHA-256
  makes the reranker fail closed to **the original, already-authorized ranking**.
  It must never silently download or switch to a different remote model in
  that case. Review backend logs if model verification fails.
- V5 reranks at most 5 authorized technical documents with the example
  configuration; it cannot retrieve missing passages or change access scope.
  The implementation accepts an explicitly configured pool size of **up to 10**.
  Test 5 vs 8 vs 10 against the same frozen authorized candidates, including
  answer correctness, latency, memory and cite integrity, before deploying a
  larger size. The research ceiling is not a measured production improvement.
  A configured SHA-256 without a local checkpoint path now fails closed, rather
  than silently loading the legacy MS MARCO model.
  Historical service resolution records are not sent to V5 for scoring.
- Changing `SERVY_RERANKER_MODEL_PATH` does **not** replace the Qwen3 embedding
  model, build a Qwen3 Chroma index, or reproduce the 88.10% research conditions.
  Qwen3 requires a separate gated indexing and retrieval integration with
  consistent tenant filtering, versioned collections and evaluation.
- Do not promote to production without verifying real-document citation
  fidelity, tenant isolation, fallback behavior, latency and an independent
  answer-quality evaluation.

## Tests

```powershell
python manage.py test core.tests_v5_reranker -v 2
python manage.py test core.tests_v5_activation -v 2
python -m unittest tools.test_install_servy_v5_checkpoint -v
```

These tests mock the model and do not require the 2.27 GB weights. They are
additionally covered by the repository's normal CI suite.

To exercise the actual checkpoint through authorization, retrieval, RAG, and
both Django API endpoints using disposable synthetic records, run:

```powershell
$env:SERVY_DB_ENGINE = "sqlite3"
$env:SERVY_TEST_REAL_V5 = "1"
python manage.py test core.tests_v5_runtime -v 2
```

The real-weight test is skipped in CI when `SERVY_TEST_REAL_V5` is unset.
It does not measure independent accuracy or verify the React browser UI.
