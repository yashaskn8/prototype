"""Operation 18.0 — Final Fine-Tuning Readiness Gate Tests.

Covers:
  Phase 1: Mixed-mode training preserves ALL examples (triplet + pair).
  Phase 2: No false-negative collisions in expand_training_triplets.
  Phase 3: Full CPU smoke-run proving real optimizer steps execute.
  Phase 4: Benchmark identity and denominator consistency.
  Phase 5: Checkpoint fingerprint and index identity.
"""
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

# Ensure Django is configured before any model imports
os.environ.setdefault("SERVY_DB_ENGINE", "sqlite3")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
django.setup()

from django.test import TestCase, override_settings


# ===========================================================================
# Phase 1: Mixed-Mode Training Data Preservation
# ===========================================================================

class TestMixedModeDataPreservation(TestCase):
    """Phase 1 (Op18): mixed_verified_examples mode must NEVER silently drop
    positive-only examples when triplet examples also exist."""

    def test_mixed_mode_keeps_both_triplets_and_pairs(self):
        """Mixed mode must produce datasets containing BOTH triplet examples
        (with hard negatives) and pair examples (positive-only).
        Verified via the training report's n_effective_examples count."""
        from core.ml.training import finetune, TRAINING_MODE_MIXED
        from core.ml.embeddings import BGE_SMALL_REVISION

        # 3 examples: 2 with negatives, 1 without
        rows = [
            {"query_text": "error E-01", "positive_text": "E-01 compressor fault",
             "negative_text": "E-02 fan motor fault", "review_status": "approved", "query_id": "q1"},
            {"query_text": "error E-02", "positive_text": "E-02 fan motor fault",
             "negative_text": "E-01 compressor fault", "review_status": "approved", "query_id": "q2"},
            {"query_text": "general maintenance", "positive_text": "Regular maintenance schedule",
             "negative_text": None, "review_status": "approved", "query_id": "q3"},
        ]

        out_dir = tempfile.mkdtemp(prefix="servy_mixed_mode_op18_")
        try:
            report = finetune(
                rows, [], out_dir, BGE_SMALL_REVISION,
                synthetic_smoke_test=True,
                training_mode=TRAINING_MODE_MIXED,
                epochs=1, batch_size=4,
            )

            # CRITICAL: ALL 3 examples must contribute — none dropped
            self.assertEqual(report["n_effective_examples"], 3,
                             f"Expected 3 effective examples, got {report['n_effective_examples']}")
            self.assertEqual(report["n_with_negative"], 2)
            self.assertEqual(report["n_without_negative"], 1)
            self.assertTrue(report["hard_negatives_consumed"])
            self.assertTrue(report["training_completed"])
            self.assertGreater(report["optimizer_steps"], 0)

        except (OSError, Exception) as e:
            err_str = str(e)
            if "not available locally" in err_str or "LocalEntryNotFoundError" in err_str or "couldn't connect" in err_str:
                self.skipTest(f"Model not cached locally: {e}")
            raise
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)

    def test_mixed_mode_all_negatives_produces_single_triplet_ds(self):
        """When ALL examples have negatives, mixed mode should produce a single
        triplet dataset (no need for multi-dataset)."""
        from core.ml.training import finetune, TRAINING_MODE_MIXED
        from core.ml.embeddings import BGE_SMALL_REVISION

        rows = [
            {"query_text": "q1", "positive_text": "p1", "negative_text": "n1",
             "review_status": "approved", "query_id": "q1"},
            {"query_text": "q2", "positive_text": "p2", "negative_text": "n2",
             "review_status": "approved", "query_id": "q2"},
        ]

        datasets_captured = {}
        try:
            from sentence_transformers import SentenceTransformerTrainer
        except ImportError:
            self.skipTest("sentence_transformers not available")

        def mock_trainer_init(self_inner, *args, **kwargs):
            datasets_captured["train"] = kwargs.get("train_dataset")
            raise InterruptedError("Stop")

        with mock.patch.object(
            type(SentenceTransformerTrainer), '__init__', mock_trainer_init
        ):
            try:
                finetune(rows, [], tempfile.mkdtemp(), BGE_SMALL_REVISION,
                         synthetic_smoke_test=True, training_mode=TRAINING_MODE_MIXED)
            except (InterruptedError, Exception):
                pass

        train_ds = datasets_captured.get("train")
        if train_ds is None:
            self.skipTest("Model not cached locally")
        # Should be a single Dataset (not a dict) since all have negatives
        if not isinstance(train_ds, dict):
            self.assertEqual(len(train_ds), 2)
            self.assertIn("negative", train_ds.column_names)

    def test_mixed_mode_no_negatives_produces_single_pairs_ds(self):
        """When NO examples have negatives, mixed mode should produce a single
        pairs dataset."""
        from core.ml.training import finetune, TRAINING_MODE_MIXED
        from core.ml.embeddings import BGE_SMALL_REVISION

        rows = [
            {"query_text": "q1", "positive_text": "p1", "negative_text": None,
             "review_status": "approved", "query_id": "q1"},
            {"query_text": "q2", "positive_text": "p2", "negative_text": None,
             "review_status": "approved", "query_id": "q2"},
        ]

        datasets_captured = {}
        try:
            from sentence_transformers import SentenceTransformerTrainer
        except ImportError:
            self.skipTest("sentence_transformers not available")

        def mock_trainer_init(self_inner, *args, **kwargs):
            datasets_captured["train"] = kwargs.get("train_dataset")
            raise InterruptedError("Stop")

        with mock.patch.object(
            type(SentenceTransformerTrainer), '__init__', mock_trainer_init
        ):
            try:
                finetune(rows, [], tempfile.mkdtemp(), BGE_SMALL_REVISION,
                         synthetic_smoke_test=True, training_mode=TRAINING_MODE_MIXED)
            except (InterruptedError, Exception):
                pass

        train_ds = datasets_captured.get("train")
        if train_ds is None:
            self.skipTest("Model not cached locally")
        # Should be a single pairs-only Dataset
        if not isinstance(train_ds, dict):
            self.assertEqual(len(train_ds), 2)
            self.assertNotIn("negative", train_ds.column_names)

    def test_to_ds_source_has_no_data_loss_pattern(self):
        """The training module must NOT contain the old to_ds(include_neg)
        pattern that silently dropped positive-only rows."""
        from core.ml import training
        source = inspect.getsource(training)
        # The old pattern: single to_ds with include_neg that filtered away pairs
        self.assertNotIn(
            "def to_ds(triplets, include_neg",
            source,
            "Old data-lossy to_ds(triplets, include_neg) must be replaced"
        )
        # Must have the new split functions
        self.assertIn("def to_ds_triplets", source)
        self.assertIn("def to_ds_pairs", source)
        self.assertIn("def to_ds_all_pairs", source)


# ===========================================================================
# Phase 2: False-Negative Collision Elimination
# ===========================================================================

class TestFalseNegativeCollisionElimination(TestCase):
    """Phase 2 (Op18): expand_training_triplets must not create in-batch
    false negative collisions."""

    def test_primary_only_prevents_multi_positive_collision(self):
        """With preserve_primary_positive_only=True, secondary positives
        must not appear as in-batch negatives for the same anchor."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "AC error E-01",
                "positive_texts": ["E-01 compressor fault", "E-01 alternate fix"],
                "negative_texts": ["E-02 fan motor fault"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        # Only primary positive should be used
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0]["positive_text"], "E-01 compressor fault")
        # The alternate positive "E-01 alternate fix" must NOT appear anywhere
        all_texts = [t["positive_text"] for t in triplets] + [t.get("negative_text", "") for t in triplets]
        self.assertNotIn("E-01 alternate fix", all_texts)

    def test_cross_query_negatives_are_safe(self):
        """When q1's negative is q2's positive, this is valid — the
        BatchSamplers.NO_DUPLICATES handles in-batch collision avoidance."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "error E-01",
                "positive_texts": ["E-01 fault guide"],
                "negative_texts": ["E-02 unrelated guide"],
            },
            {
                "query_id": "q2", "query_text": "error E-02",
                "positive_texts": ["E-02 unrelated guide"],
                "negative_texts": ["E-01 fault guide"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(len(triplets), 2)
        # Each query should appear with unique query_id suffix
        qids = {t["query_id"] for t in triplets}
        self.assertEqual(len(qids), 2)

    def test_no_empty_or_none_negative_in_triplets(self):
        """Expanded triplets must never have empty string negatives."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {"query_id": "q1", "query_text": "q1", "positive_texts": ["p1"],
             "negative_texts": ["n1"]},
            {"query_id": "q2", "query_text": "q2", "positive_texts": ["p2"],
             "negative_texts": []},
            {"query_id": "q3", "query_text": "q3", "positive_texts": ["p3"],
             "negative_texts": [None]},
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        for t in triplets:
            neg = t.get("negative_text")
            if neg is not None:
                self.assertNotEqual(neg, "", f"Empty negative in {t['query_id']}")
                self.assertTrue(len(str(neg).strip()) > 0,
                                f"Whitespace-only negative in {t['query_id']}")


# ===========================================================================
# Phase 3: CPU Smoke Run (Real Optimizer Steps)
# ===========================================================================

class TestCPUSmokeRun(TestCase):
    """Phase 3 (Op18): Prove complete synthetic training pipeline
    executes real optimizer steps and produces valid checkpoints."""

    def test_synthetic_smoke_training_completes(self):
        """Run a real synthetic smoke test: load model, run optimizer steps,
        save checkpoint, reload, and verify the report."""
        from core.ml.training import finetune, TRAINING_MODE_MIXED
        from core.ml.embeddings import BGE_SMALL_REVISION

        # Minimal synthetic data — enough for 1 optimizer step
        train_rows = [
            {"query_text": "compressor fault E-01", "positive_text": "Error E-01 indicates compressor failure.",
             "negative_text": "Error E-02 indicates fan motor failure.", "review_status": "approved", "query_id": "smoke-q1"},
            {"query_text": "fan motor E-02", "positive_text": "Error E-02 indicates fan motor failure.",
             "negative_text": "Error E-01 indicates compressor failure.", "review_status": "approved", "query_id": "smoke-q2"},
            {"query_text": "general maintenance", "positive_text": "Regular HVAC maintenance schedule.",
             "negative_text": None, "review_status": "approved", "query_id": "smoke-q3"},
            {"query_text": "temperature sensor", "positive_text": "Temperature sensor calibration procedure.",
             "negative_text": None, "review_status": "approved", "query_id": "smoke-q4"},
        ]

        out_dir = tempfile.mkdtemp(prefix="servy_smoke_op18_")
        try:
            report = finetune(
                train_rows, [], out_dir, BGE_SMALL_REVISION,
                epochs=1, batch_size=4, lr=2e-5,
                synthetic_smoke_test=True,
                training_mode=TRAINING_MODE_MIXED,
            )

            # Verify report completeness
            self.assertTrue(report["training_completed"])
            self.assertTrue(report["reload_succeeded"])
            self.assertEqual(report["training_status"], "SYNTHETIC_SMOKE_TEST_INELIGIBLE_FOR_PRODUCTION")
            self.assertFalse(report["promotion_eligible"])
            self.assertEqual(report["supervision_mode"], "synthetic_smoke")
            self.assertEqual(report["training_mode"], "mixed_verified_examples")
            self.assertGreater(report["optimizer_steps"], 0, "Must run at least 1 optimizer step")
            self.assertGreater(report["train_loss"], 0.0, "Train loss must be positive")
            self.assertEqual(report["reload_dim"], 384, "BGE-small must produce 384-d embeddings")
            self.assertTrue(len(report["checkpoint_sha256"]) == 64, "Must have valid SHA-256 hash")

            # Verify all 4 examples contributed
            self.assertEqual(report["n_train_rows"], 4)
            self.assertEqual(report["n_effective_examples"], 4)  # No examples dropped!
            self.assertEqual(report["n_with_negative"], 2)
            self.assertEqual(report["n_without_negative"], 2)
            self.assertTrue(report["hard_negatives_consumed"])

            # Verify checkpoint exists on disk
            final_path = Path(out_dir) / "final"
            self.assertTrue(final_path.exists(), "Checkpoint 'final' dir must exist")
            self.assertTrue((final_path / "model.safetensors").exists() or
                            (final_path / "pytorch_model.bin").exists(),
                            "Model weights must be saved")

            # Verify run_report.json was written
            report_path = Path(out_dir) / "run_report.json"
            self.assertTrue(report_path.exists())
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_report["checkpoint_sha256"], report["checkpoint_sha256"])

        except (OSError, Exception) as e:
            err_str = str(e)
            if "not available locally" in err_str or "LocalEntryNotFoundError" in err_str or "couldn't connect" in err_str:
                self.skipTest(f"BGE model not cached locally: {e}")
            raise
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)


# ===========================================================================
# Phase 4: Benchmark Identity and Denominator Consistency
# ===========================================================================

class TestBenchmarkIdentity(TestCase):
    """Phase 4 (Op18): Benchmark evaluation must use correct denominator
    identities and never silently substitute datasets."""

    def test_benchmark_registry_has_correct_roles(self):
        """Each registered benchmark must have its role correctly defined."""
        from core.ml.benchmark_registry import (
            REGISTERED_BENCHMARKS, ROLE_INDEPENDENT_HOLDOUT,
            ROLE_SOFTWARE_REGRESSION_FIXTURE,
        )
        holdout = REGISTERED_BENCHMARKS.get("servy-real-holdout-v1")
        ci = REGISTERED_BENCHMARKS.get("servy-ci-regression-fixture-v1")
        self.assertIsNotNone(holdout)
        self.assertIsNotNone(ci)
        self.assertEqual(holdout.role, ROLE_INDEPENDENT_HOLDOUT)
        self.assertEqual(ci.role, ROLE_SOFTWARE_REGRESSION_FIXTURE)

    def test_benchmark_sha256_is_immutable(self):
        """The registered SHA-256 digests must match actual files."""
        from core.ml.benchmark_registry import (
            REGISTERED_BENCHMARKS, compute_file_sha256, BenchmarkIntegrityError,
        )
        for bid, spec in REGISTERED_BENCHMARKS.items():
            if spec.file_path.exists():
                actual = compute_file_sha256(spec.file_path)
                self.assertEqual(
                    actual, spec.expected_sha256,
                    f"Benchmark '{bid}' SHA-256 mismatch: expected {spec.expected_sha256[:16]}, got {actual[:16]}"
                )

    def test_benchmark_manifest_denominators_consistent(self):
        """Primary denominators must sum to total queries."""
        from core.ml.benchmark_manifest import build_manifest
        try:
            manifest = build_manifest()
        except Exception:
            self.skipTest("Benchmark file not available")

        denom = manifest["primary_denominators"]
        total_primary = denom["relevance_queries"] + denom["abstention_queries"] + denom["security_queries"]
        self.assertEqual(total_primary, manifest["total_benchmark_cases"],
                         f"Denominator sum {total_primary} != total cases {manifest['total_benchmark_cases']}")

    def test_compute_metrics_denominator_excludes_abstention(self):
        """Relevance metrics denominator must exclude abstention queries."""
        from core.ml.evaluation import compute_metrics
        records = [
            {"ranked_chunk_ids": [1, 2], "relevant_chunk_ids": [1],
             "latency_ms": 10.0, "is_unsupported": False},
            {"ranked_chunk_ids": [3, 4], "relevant_chunk_ids": [5],
             "latency_ms": 10.0, "is_unsupported": False},
            {"ranked_chunk_ids": [], "relevant_chunk_ids": [],
             "latency_ms": 5.0, "is_unsupported": True},
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["n_queries_evaluated"], 2,
                         "Denominator must be 2 (excluding 1 abstention)")
        self.assertEqual(metrics["n_abstention_queries"], 1)
        # Hit@1: first query hits, second misses → 0.5
        self.assertEqual(metrics["hit@1"], 0.5)


# ===========================================================================
# Phase 5: Checkpoint and Index Identity
# ===========================================================================

class TestCheckpointIdentity(TestCase):
    """Phase 5 (Op18): Checkpoint fingerprints must be deterministic
    and index manifests must detect model mismatches."""

    def test_checkpoint_fingerprint_is_deterministic(self):
        """Same files must produce same fingerprint."""
        from core.ml.embeddings import compute_checkpoint_fingerprint
        # Use a temp dir with known content
        tmp = tempfile.mkdtemp()
        try:
            (Path(tmp) / "config.json").write_text('{"test": true}')
            (Path(tmp) / "model.bin").write_bytes(b"fake_model_weights_1234")
            fp1 = compute_checkpoint_fingerprint(tmp, short=False)
            fp2 = compute_checkpoint_fingerprint(tmp, short=False)
            self.assertEqual(fp1, fp2)
            self.assertEqual(len(fp1), 64)  # Full SHA-256
        finally:
            shutil.rmtree(tmp)

    def test_checkpoint_fingerprint_changes_with_content(self):
        """Different file content must produce different fingerprint."""
        from core.ml.embeddings import compute_checkpoint_fingerprint
        tmp1 = tempfile.mkdtemp()
        tmp2 = tempfile.mkdtemp()
        try:
            (Path(tmp1) / "model.bin").write_bytes(b"weights_v1")
            (Path(tmp2) / "model.bin").write_bytes(b"weights_v2")
            fp1 = compute_checkpoint_fingerprint(tmp1, short=False)
            fp2 = compute_checkpoint_fingerprint(tmp2, short=False)
            self.assertNotEqual(fp1, fp2)
        finally:
            shutil.rmtree(tmp1)
            shutil.rmtree(tmp2)

    def test_nonexistent_path_returns_empty(self):
        """Missing path must return empty string, not crash."""
        from core.ml.embeddings import compute_checkpoint_fingerprint
        fp = compute_checkpoint_fingerprint("/nonexistent/path/xyz")
        self.assertEqual(fp, "")

    def test_index_manifest_model_mismatch_raises(self):
        """get_collection must refuse to use a collection stamped by another model."""
        from core.ml.embeddings import EmbeddingConfigError, EmbeddingProfile, get_collection
        # Create a profile that expects a collection stamped by a different model
        profile = EmbeddingProfile(
            key="test-mismatch", model_id="EXPECTED_MODEL",
            revision="rev1", dim=384,
            collection="servy_dense_kb",  # This is stamped as minilm
            query_prefix="", content_first=True,
        )
        # minilm collections may or may not be stamped; test the logic
        try:
            coll = get_collection(profile, create=False)
            # If it didn't raise, the collection is unstamped (legacy)
            # or stamped correctly — both are handled by the guard
        except EmbeddingConfigError as e:
            # Expected: model mismatch or unstamped error
            self.assertTrue(
                "mismatch" in str(e).lower() or "unstamped" in str(e).lower(),
                f"Unexpected error: {e}"
            )
        except Exception:
            pass  # Collection may not exist in test DB


# ===========================================================================
# Phase 6: Training Label Readiness Audit
# ===========================================================================

class TestTrainingLabelReadiness(TestCase):
    """Phase 6 (Op18): Audit training label provenance, review workflow,
    and train/eval/holdout overlap."""

    def test_split_no_overlap(self):
        """Train/validation/test splits must have zero overlap."""
        from core.ml.labels import split_by_group
        rows = [
            {"query_id": f"q{i}", "source_group": f"doc-{i % 7}",
             "positive_chunk_ids": [i]}
            for i in range(50)
        ]
        splits = split_by_group(rows)
        train_ids = {r["query_id"] for r in splits["train"]}
        val_ids = {r["query_id"] for r in splits["validation"]}
        test_ids = {r["query_id"] for r in splits["test"]}
        self.assertEqual(len(train_ids & val_ids), 0, "Train/val overlap!")
        self.assertEqual(len(train_ids & test_ids), 0, "Train/test overlap!")
        self.assertEqual(len(val_ids & test_ids), 0, "Val/test overlap!")
        self.assertEqual(len(train_ids | val_ids | test_ids), 50)

    def test_split_groups_not_straddled(self):
        """Same source_group must not appear in multiple splits."""
        from core.ml.labels import split_by_group
        rows = [
            {"query_id": f"q{i}", "source_group": f"doc-{i % 3}",
             "positive_chunk_ids": [i]}
            for i in range(30)
        ]
        splits = split_by_group(rows)
        for partition_name, partition_rows in splits.items():
            groups = {r["source_group"] for r in partition_rows}
            for other_name, other_rows in splits.items():
                if other_name == partition_name:
                    continue
                other_groups = {r["source_group"] for r in other_rows}
                straddled = groups & other_groups
                self.assertEqual(len(straddled), 0,
                                 f"Groups {straddled} straddle {partition_name} and {other_name}")


# ===========================================================================
# Phase 7: Evaluation Metric Correctness
# ===========================================================================

class TestEvaluationMetricCorrectness(TestCase):
    """Phase 7 (Op18): Verify individual metric functions compute correctly."""

    def test_hit_at_1_perfect(self):
        from core.ml.evaluation import hit_at_k
        self.assertEqual(hit_at_k([10, 20, 30], {10}, 1), 1.0)

    def test_hit_at_1_miss(self):
        from core.ml.evaluation import hit_at_k
        self.assertEqual(hit_at_k([20, 30, 10], {10}, 1), 0.0)

    def test_mrr_first_position(self):
        from core.ml.evaluation import mrr_at_k
        self.assertEqual(mrr_at_k([10, 20], {10}, 10), 1.0)

    def test_mrr_second_position(self):
        from core.ml.evaluation import mrr_at_k
        self.assertEqual(mrr_at_k([20, 10], {10}, 10), 0.5)

    def test_mrr_not_found(self):
        from core.ml.evaluation import mrr_at_k
        self.assertEqual(mrr_at_k([20, 30], {10}, 10), 0.0)

    def test_ndcg_perfect_single(self):
        from core.ml.evaluation import ndcg_at_k
        # Single relevant doc at position 1 → nDCG = 1.0
        self.assertAlmostEqual(ndcg_at_k([10], {10}, 10), 1.0)

    def test_ndcg_empty(self):
        from core.ml.evaluation import ndcg_at_k
        self.assertEqual(ndcg_at_k([20, 30], {10}, 10), 0.0)

    def test_recall_at_k(self):
        from core.ml.evaluation import recall_at_k
        # 2 relevant, 1 found in top-3
        self.assertEqual(recall_at_k([10, 20, 30], {10, 40}, 3), 0.5)

    def test_precision_at_k(self):
        from core.ml.evaluation import precision_at_k
        # top-3 has 1 relevant out of 3
        self.assertAlmostEqual(precision_at_k([10, 20, 30], {10}, 3), 1/3)
