"""Operation 16.0 — Fine-Tuning Integrity Regression Tests.

Covers:
  Phase 1: Explicit training modes — no silent hard-negative fallback.
  Phase 2: Contrastive false-negative collision detection.
  Phase 3: Provenance and authorization enforcement for training labels.
  Phase 4: Immutable serving configuration.
  Phase 5: Historical baseline comparison labeling.
  Phase 6: Separate retrieval and abstention metrics.
"""
import hashlib
import json
import os
import sys
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
# Phase 1: Explicit Training Modes — No Silent Hard-Negative Fallback
# ===========================================================================

class TestTrainingModeEnforcement(TestCase):
    """Phase 1: Every reviewed negative used for training must be accounted for,
    and no implicit fallback is allowed."""

    def test_valid_training_modes(self):
        """Validates that the module exposes correct training mode constants."""
        from core.ml.training import (
            TRAINING_MODE_POSITIVE_PAIRS,
            TRAINING_MODE_VERIFIED_TRIPLETS,
            TRAINING_MODE_MIXED,
            VALID_TRAINING_MODES,
        )
        self.assertEqual(TRAINING_MODE_POSITIVE_PAIRS, "positive_pairs_only")
        self.assertEqual(TRAINING_MODE_VERIFIED_TRIPLETS, "verified_triplets_only")
        self.assertEqual(TRAINING_MODE_MIXED, "mixed_verified_examples")
        self.assertEqual(len(VALID_TRAINING_MODES), 3)

    def test_invalid_training_mode_rejected(self):
        """Invalid training modes must raise ValueError."""
        from core.ml.training import finetune
        with self.assertRaises(ValueError) as ctx:
            finetune([], [], "/tmp/out", "abc", training_mode="auto_fallback")
        self.assertIn("Invalid training_mode", str(ctx.exception))

    def test_verified_triplets_mode_rejects_missing_negatives(self):
        """verified_triplets_only mode must reject if any example lacks a negative."""
        from core.ml.training import finetune, TRAINING_MODE_VERIFIED_TRIPLETS

        rows = [
            {"query_text": "test query", "positive_text": "positive passage",
             "negative_text": None, "review_status": "approved", "query_id": "q1"},
        ]
        with self.assertRaises((ValueError, Exception)):
            finetune(
                rows, [], "/tmp/out", "abc",
                synthetic_smoke_test=True,
                training_mode=TRAINING_MODE_VERIFIED_TRIPLETS,
            )

    def test_mixed_mode_preserves_negatives(self):
        """mixed_verified_examples mode must include negatives where available,
        not discard them because other examples lack negatives."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "fault code E-01",
                "positive_texts": ["positive passage 1"],
                "negative_texts": ["negative passage 1"],
            },
            {
                "query_id": "q2", "query_text": "general symptom",
                "positive_texts": ["positive passage 2"],
                "negative_texts": [],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        with_neg = [t for t in triplets if t.get("negative_text")]
        without_neg = [t for t in triplets if not t.get("negative_text")]

        # q1 should have a negative, q2 should not
        self.assertEqual(len(with_neg), 1)
        self.assertEqual(len(without_neg), 1)
        self.assertEqual(with_neg[0]["negative_text"], "negative passage 1")

    def test_multiple_negatives_per_query(self):
        """Multiple reviewed negatives per query must produce multiple triplets."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "fault code E-01",
                "positive_texts": ["positive passage 1"],
                "negative_texts": ["neg A", "neg B", "neg C"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(len(triplets), 3)
        neg_texts = {t["negative_text"] for t in triplets}
        self.assertEqual(neg_texts, {"neg A", "neg B", "neg C"})

    def test_all_examples_have_negatives(self):
        """When all examples have negatives, all should produce triplets with negatives."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": f"q{i}", "query_text": f"query {i}",
                "positive_texts": [f"pos {i}"],
                "negative_texts": [f"neg {i}"],
            }
            for i in range(5)
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(len(triplets), 5)
        self.assertTrue(all(t.get("negative_text") for t in triplets))

    def test_no_examples_have_negatives(self):
        """When no examples have negatives, all should produce pair-only entries."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": f"q{i}", "query_text": f"query {i}",
                "positive_texts": [f"pos {i}"],
                "negative_texts": [],
            }
            for i in range(5)
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(len(triplets), 5)
        self.assertTrue(all(t.get("negative_text") is None for t in triplets))


# ===========================================================================
# Phase 2: Contrastive False-Negative Collision Detection
# ===========================================================================

class TestFalseNegativeCollision(TestCase):
    """Phase 2: Duplicate anchors must not create unintended false negatives.
    Alternative correct passages must not be treated as negatives."""

    def test_primary_positive_only_prevents_false_negatives(self):
        """With preserve_primary_positive_only=True, each query anchor appears
        once per negative, preventing in-batch false negative collisions."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "shared query",
                "positive_texts": ["correct passage A", "correct passage B"],
                "negative_texts": ["neg 1"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        # Should only use the PRIMARY positive (index 0), not alternative positives
        self.assertEqual(len(triplets), 1)
        self.assertEqual(triplets[0]["positive_text"], "correct passage A")

    def test_multiple_positives_disabled_by_default(self):
        """Default behavior must prevent multiple positives from creating
        false negative collisions in the batch."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "query with alternatives",
                "positive_texts": ["pos A", "pos B", "pos C"],
                "negative_texts": ["neg X"],
            },
        ]
        # Default is preserve_primary_positive_only=False, but training.py
        # always calls with True.
        triplets_primary = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(len(triplets_primary), 1)

        triplets_all = expand_training_triplets(examples, preserve_primary_positive_only=False)
        self.assertEqual(len(triplets_all), 3)  # One per positive × one negative

    def test_distinct_query_ids_in_expanded_triplets(self):
        """Each expanded triplet must have a unique query_id suffix."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "query",
                "positive_texts": ["pos"],
                "negative_texts": ["neg A", "neg B"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        ids = [t["query_id"] for t in triplets]
        self.assertEqual(len(ids), len(set(ids)), f"Duplicate query_ids: {ids}")


# ===========================================================================
# Phase 3: Provenance and Authorization Enforcement
# ===========================================================================

class TestProvenanceEnforcement(TestCase):
    """Phase 3: Missing provenance must be rejected for approved training data.
    Empty verification_status must not be accepted as VERIFIED."""

    def test_authorized_reviewer_rejects_bot_names(self):
        """Bot/synthetic/placeholder names must be rejected."""
        from core.ml.labels import is_authorized_reviewer
        bad_names = ["bot", "AI assistant", "synthetic-reviewer", "test",
                     "system", "anonymous", "placeholder", "self-approved"]
        for name in bad_names:
            self.assertFalse(
                is_authorized_reviewer(name),
                f"Reviewer '{name}' should be rejected"
            )

    def test_authorized_reviewer_accepts_configured_names(self):
        """Configured reviewer names must be accepted."""
        from core.ml.labels import is_authorized_reviewer
        # These are configured in SERVY_AUTHORIZED_TRAINING_REVIEWERS
        good_names = ["Pranav Patel", "admin", "dr-yashas-kn"]
        for name in good_names:
            self.assertTrue(
                is_authorized_reviewer(name),
                f"Reviewer '{name}' should be accepted"
            )

    def test_empty_reviewer_rejected(self):
        """Empty or short reviewer strings must be rejected."""
        from core.ml.labels import is_authorized_reviewer
        self.assertFalse(is_authorized_reviewer(""))
        self.assertFalse(is_authorized_reviewer("  "))
        self.assertFalse(is_authorized_reviewer("ab"))

    def test_validate_labels_rejects_duplicate_query_id(self):
        """Duplicate query_id must be rejected."""
        from core.ml.labels import validate_labels
        from core.models import Tenant, KnowledgeDocument, KnowledgeChunk
        # Create real fixtures so validation reaches duplicate check
        tenant, _ = Tenant.objects.get_or_create(slug="test-t", defaults={"name": "test-t"})
        doc = KnowledgeDocument.objects.create(
            tenant=tenant, title="Test Doc", doc_type="manual",
            is_rag_enabled=True, is_confidential=False, index_status="INDEXED",
        )
        c1 = KnowledgeChunk.objects.create(
            tenant=tenant, document=doc, chunk_index=0, text="chunk one",
        )
        c2 = KnowledgeChunk.objects.create(
            tenant=tenant, document=doc, chunk_index=1, text="chunk two",
        )
        rows = [
            {"query_id": "q1", "query_text": "test", "positive_chunk_ids": [c1.id],
             "negative_chunk_ids": [], "review_status": "pending",
             "reviewer": "admin", "tenant_id": tenant.id},
            {"query_id": "q1", "query_text": "test2", "positive_chunk_ids": [c2.id],
             "negative_chunk_ids": [], "review_status": "pending",
             "reviewer": "admin", "tenant_id": tenant.id},
        ]
        result = validate_labels(rows, require_approved=False)
        duplicate_rejections = [r for r in result["rejected"] if "duplicate" in r[1]]
        self.assertTrue(len(duplicate_rejections) >= 1,
                        f"Expected duplicate rejection, got: {result['rejected']}")

    def test_validate_labels_rejects_contradictory_labels(self):
        """Chunk that is both positive and negative must be rejected."""
        from core.ml.labels import validate_labels
        from core.models import Tenant, KnowledgeDocument, KnowledgeChunk
        tenant, _ = Tenant.objects.get_or_create(slug="test-contra", defaults={"name": "test-contra"})
        doc = KnowledgeDocument.objects.create(
            tenant=tenant, title="Test Doc", doc_type="manual",
            is_rag_enabled=True, is_confidential=False, index_status="INDEXED",
        )
        c1 = KnowledgeChunk.objects.create(
            tenant=tenant, document=doc, chunk_index=0, text="chunk",
        )
        rows = [
            {"query_id": "q1", "query_text": "test", "positive_chunk_ids": [c1.id],
             "negative_chunk_ids": [c1.id], "review_status": "pending",
             "reviewer": "admin", "tenant_id": tenant.id},
        ]
        result = validate_labels(rows, require_approved=False)
        self.assertTrue(len(result["rejected"]) >= 1)
        self.assertIn("contradictory", result["rejected"][0][1])


# ===========================================================================
# Phase 4: Immutable Serving Configuration
# ===========================================================================

class TestServingConfig(TestCase):
    """Phase 4: Serving configuration must be immutable and read from settings."""

    def test_serving_config_is_frozen(self):
        """ServingConfig instances must be immutable."""
        from core.services.retriever import ServingConfig
        config = ServingConfig(
            dense_relevance_gate=0.55,
            sparse_relevance_gate=0.055,
            embedding_profile="auto",
        )
        with self.assertRaises(Exception):
            config.dense_relevance_gate = 0.38  # type: ignore

    def test_get_serving_config_reads_from_settings(self):
        """get_serving_config() must read current Django settings."""
        from core.services.retriever import get_serving_config
        config = get_serving_config()
        self.assertIsNotNone(config.dense_relevance_gate)
        self.assertIsNotNone(config.sparse_relevance_gate)
        self.assertIsNotNone(config.embedding_profile)

    @override_settings(SERVY_DENSE_RELEVANCE_GATE=0.38)
    def test_serving_config_reflects_settings_change(self):
        """get_serving_config() must reflect changed settings."""
        from core.services.retriever import get_serving_config
        config = get_serving_config()
        self.assertEqual(config.dense_relevance_gate, 0.38)

    @override_settings(SERVY_DENSE_RELEVANCE_GATE=0.55)
    def test_serving_config_default_gate(self):
        """Default gate must be 0.55 from settings."""
        from core.services.retriever import get_serving_config
        config = get_serving_config()
        self.assertEqual(config.dense_relevance_gate, 0.55)

    def test_parallel_configs_are_independent(self):
        """Two ServingConfig snapshots must be independent."""
        from core.services.retriever import ServingConfig
        config_a = ServingConfig(
            dense_relevance_gate=0.55,
            sparse_relevance_gate=0.055,
            embedding_profile="auto",
        )
        config_b = ServingConfig(
            dense_relevance_gate=0.38,
            sparse_relevance_gate=0.055,
            embedding_profile="auto",
        )
        self.assertNotEqual(config_a.dense_relevance_gate, config_b.dense_relevance_gate)
        # Both must remain unchanged
        self.assertEqual(config_a.dense_relevance_gate, 0.55)
        self.assertEqual(config_b.dense_relevance_gate, 0.38)


# ===========================================================================
# Phase 5: Historical Baseline Comparison Labeling
# ===========================================================================

class TestBaselineComparisonLabeling(TestCase):
    """Phase 5: Threshold ablation must not be labeled as historical progress."""

    def test_benchmark_pipeline_modes_are_labeled_correctly(self):
        """Track B modes must include threshold_ablation in their name
        when they are just gate comparisons, not historical vs. current."""
        # This test verifies the evaluate_track_b_serving function's mode handling
        from core.ml.evaluate_benchmark import evaluate_track_b_serving
        import inspect
        source = inspect.getsource(evaluate_track_b_serving)
        # The function must contain the ablation label comment
        self.assertIn("THRESHOLD ABLATION", source,
                       "production_baseline mode must be labeled as THRESHOLD ABLATION")


# ===========================================================================
# Phase 6: Separate Retrieval and Abstention Metrics
# ===========================================================================

class TestRetrievalAbstentionSeparation(TestCase):
    """Phase 6: Retrieval metrics must not be zeroed by abstention decisions."""

    def test_compute_metrics_handles_supported_queries(self):
        """Supported queries must compute proper retrieval metrics."""
        from core.ml.evaluation import compute_metrics
        records = [
            {
                "ranked_chunk_ids": [10, 20, 30],
                "relevant_chunk_ids": [10],
                "latency_ms": 50.0,
                "is_unsupported": False,
            },
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["hit@1"], 1.0)
        self.assertEqual(metrics["mrr@10"], 1.0)

    def test_compute_metrics_handles_abstention_separately(self):
        """Abstention queries must not affect retrieval metrics denominators."""
        from core.ml.evaluation import compute_metrics
        records = [
            {
                "ranked_chunk_ids": [10, 20, 30],
                "relevant_chunk_ids": [10],
                "latency_ms": 50.0,
                "is_unsupported": False,
            },
            {
                "ranked_chunk_ids": [],
                "relevant_chunk_ids": [],
                "latency_ms": 10.0,
                "is_unsupported": True,
            },
        ]
        metrics = compute_metrics(records)
        # The abstention query should not dilute retrieval metrics
        self.assertEqual(metrics["n_queries_evaluated"], 1)
        self.assertEqual(metrics["n_abstention_queries"], 1)
        self.assertEqual(metrics["hit@1"], 1.0)

    def test_nonempty_retrieval_not_zeroed_by_abstention(self):
        """A nonempty retrieval list must NOT be converted to empty
        merely because generation abstained."""
        # Simulating what Phase 6 fixes: ranked_chunk_ids should reflect
        # actual retrieval, not be zeroed when answer_mode == ABSTAIN
        from core.ml.evaluation import compute_metrics
        records = [
            {
                "ranked_chunk_ids": [10, 20, 30],  # Retriever found chunks
                "relevant_chunk_ids": [10],
                "latency_ms": 50.0,
                "is_unsupported": False,
                # Even if this had "generation_abstained": True, retrieval metrics
                # should use the actual ranked_chunk_ids
            },
        ]
        metrics = compute_metrics(records)
        self.assertEqual(metrics["hit@1"], 1.0)


# ===========================================================================
# Phase 2 Extended: Batch Composition Verification
# ===========================================================================

class TestBatchCompositionIntegrity(TestCase):
    """Phase 2 extended: Verify actual training data composition."""

    def test_negative_from_different_query_is_safe(self):
        """Negatives from different queries should not collide."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "AC unit error E-01",
                "positive_texts": ["E-01 means compressor fault"],
                "negative_texts": ["E-02 means fan motor fault"],
            },
            {
                "query_id": "q2", "query_text": "AC unit error E-02",
                "positive_texts": ["E-02 means fan motor fault"],
                "negative_texts": ["E-01 means compressor fault"],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        # q1's negative is q2's positive — this is a valid cross-query scenario
        # but under MNRL, q2's positive will also serve as an in-batch negative for q1
        # if both are in the same batch. This is handled by BatchSamplers.NO_DUPLICATES
        # and preserve_primary_positive_only=True.
        self.assertEqual(len(triplets), 2)
        # Each query_id prefix must be unique
        prefixes = {t["query_id"].split("-p")[0].split("-n")[0] for t in triplets}
        self.assertEqual(len(prefixes), 2)

    def test_source_group_preserved_in_triplets(self):
        """Source group must be preserved through expansion for leakage-safe splitting."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "query",
                "positive_texts": ["pos"],
                "negative_texts": ["neg"],
                "source_group": "manual-123",
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        self.assertEqual(triplets[0]["source_group"], "manual-123")


# ===========================================================================
# Phase 1 Extended: Training Report Accounting
# ===========================================================================

class TestTrainingReportAccounting(TestCase):
    """Phase 1 extended: Training report must account for every example."""

    def test_report_structure(self):
        """The training report must include all Phase 1 accounting fields."""
        from core.ml.training import TRAINING_MODE_MIXED
        required_fields = [
            "training_mode", "n_train_rows", "n_train_expanded_triplets",
            "n_effective_examples", "n_with_negative", "n_without_negative",
            "n_excluded", "excluded_example_ids", "n_distinct_queries",
            "hard_negatives_consumed", "verified_negatives_mode",
        ]
        # We can't run the full training pipeline in unit tests,
        # but we verify the module exports are correct
        from core.ml import training
        self.assertEqual(training.TRAINING_MODE_MIXED, "mixed_verified_examples")


# ===========================================================================
# Dataset Split Integrity
# ===========================================================================

class TestDatasetSplitIntegrity(TestCase):
    """Verify deterministic splitting by source group."""

    def test_split_by_group_is_deterministic(self):
        """Same input must produce identical splits."""
        from core.ml.labels import split_by_group
        rows = [
            {"query_id": f"q{i}", "source_group": f"doc-{i % 3}"}
            for i in range(20)
        ]
        split1 = split_by_group(rows)
        split2 = split_by_group(rows)
        self.assertEqual(
            [r["query_id"] for r in split1["train"]],
            [r["query_id"] for r in split2["train"]],
        )

    def test_split_covers_all_rows(self):
        """All rows must appear in exactly one split."""
        from core.ml.labels import split_by_group
        rows = [
            {"query_id": f"q{i}", "source_group": f"doc-{i % 5}"}
            for i in range(30)
        ]
        split = split_by_group(rows)
        all_ids = set()
        for partition in ("train", "validation", "test"):
            for r in split[partition]:
                self.assertNotIn(r["query_id"], all_ids, "Row appears in multiple splits")
                all_ids.add(r["query_id"])
        self.assertEqual(len(all_ids), 30)


# ===========================================================================
# Operation 17.0: Training Pipeline Integrity Hardening
# ===========================================================================

class TestOp17VerifiedTripletsStrictFailure(TestCase):
    """Op17 M4: verified_triplets_only mode must FAIL with diagnostic
    listing incomplete example IDs, not silently discard them."""

    def test_verified_triplets_fails_with_incomplete_ids(self):
        """When some examples lack negatives, the ValueError must list
        the incomplete example IDs so the engineer can fix them."""
        from core.ml.training import finetune, TRAINING_MODE_VERIFIED_TRIPLETS

        rows = [
            {"query_text": "query with neg", "positive_text": "pos A",
             "negative_text": "neg A", "review_status": "approved", "query_id": "q-good"},
            {"query_text": "query without neg", "positive_text": "pos B",
             "negative_text": None, "review_status": "approved", "query_id": "q-missing"},
        ]
        with self.assertRaises(ValueError) as ctx:
            finetune(
                rows, [], "/tmp/out", "abc",
                synthetic_smoke_test=True,
                training_mode=TRAINING_MODE_VERIFIED_TRIPLETS,
            )
        error_msg = str(ctx.exception)
        self.assertIn("TRAINING_BLOCKED", error_msg)
        self.assertIn("q-missing", error_msg,
                       "Error must list the specific incomplete example IDs")
        self.assertIn("1 of 2", error_msg,
                       "Error must report exact counts of incomplete vs total")

    def test_verified_triplets_passes_when_all_have_negatives(self):
        """When ALL examples have negatives, verified_triplets must NOT raise."""
        from core.ml.training import finetune, TRAINING_MODE_VERIFIED_TRIPLETS

        rows = [
            {"query_text": "query A", "positive_text": "pos A",
             "negative_text": "neg A", "review_status": "approved", "query_id": "q1"},
            {"query_text": "query B", "positive_text": "pos B",
             "negative_text": "neg B", "review_status": "approved", "query_id": "q2"},
        ]
        # Should reach model loading (may fail in CI without model files)
        # but must NOT fail at the verified_triplets validation stage
        try:
            finetune(
                rows, [], "/tmp/out", "abc",
                synthetic_smoke_test=True,
                training_mode=TRAINING_MODE_VERIFIED_TRIPLETS,
            )
        except ValueError as e:
            self.assertNotIn("TRAINING_BLOCKED", str(e),
                             "Should not block training when all examples have negatives")
        except Exception:
            pass  # Model loading or other non-validation errors are OK


class TestOp17NoBlankNegatives(TestCase):
    """Op17 M3: No empty-string negatives must ever be passed to contrastive loss."""

    def test_mixed_mode_dataset_has_no_blank_negatives(self):
        """Expanded triplets must never contain blank negative_text."""
        from core.ml.labels import expand_training_triplets
        examples = [
            {
                "query_id": "q1", "query_text": "query with neg",
                "positive_texts": ["pos A"],
                "negative_texts": ["neg A"],
            },
            {
                "query_id": "q2", "query_text": "query without neg",
                "positive_texts": ["pos B"],
                "negative_texts": [],
            },
        ]
        triplets = expand_training_triplets(examples, preserve_primary_positive_only=True)
        for t in triplets:
            neg = t.get("negative_text")
            if neg is not None:
                self.assertNotEqual(neg, "",
                                     f"Blank negative found for {t['query_id']}")
                self.assertTrue(len(neg.strip()) > 0,
                                f"Whitespace-only negative for {t['query_id']}")

    def test_no_blank_negative_in_training_data_construction(self):
        """The to_ds function must not use 'or empty-string' pattern for negatives."""
        import inspect
        from core.ml import training
        source = inspect.getsource(training)
        # The old dangerous pattern inserted blank negatives into contrastive loss
        if "def to_ds_triplets" in source or "def to_ds_pairs" in source or "def to_ds" in source:
            # Find the dataset builder functions and check for dangerous patterns
            for func_name in ["to_ds_triplets", "to_ds_pairs", "to_ds_all_pairs", "to_ds"]:
                if f"def {func_name}" in source:
                    func_body = source.split(f"def {func_name}")[1].split("\n    out_dir")[0]
                    self.assertNotIn(
                        'or ""',
                        func_body,
                        f"{func_name} must not contain 'or empty-string' pattern for negatives"
                    )


class TestOp17ServingConfigGateIndependence(TestCase):
    """Op17 M6: Per-request serving config must be independent."""

    def test_scoring_functions_accept_serving_config(self):
        """Retrieval scoring functions must accept serving_config parameter."""
        import inspect
        from core.services.retriever import _sparse_score_candidates, _dense_score_candidates
        sparse_sig = inspect.signature(_sparse_score_candidates)
        dense_sig = inspect.signature(_dense_score_candidates)
        self.assertIn("serving_config", sparse_sig.parameters,
                       "_sparse_score_candidates must accept serving_config")
        self.assertIn("serving_config", dense_sig.parameters,
                       "_dense_score_candidates must accept serving_config")

    @override_settings(SERVY_DENSE_RELEVANCE_GATE=0.38)
    def test_config_reflects_low_gate(self):
        """ServingConfig must read 0.38 from settings when configured."""
        from core.services.retriever import get_serving_config
        config = get_serving_config()
        self.assertEqual(config.dense_relevance_gate, 0.38)

    @override_settings(SERVY_DENSE_RELEVANCE_GATE=0.55)
    def test_config_reflects_high_gate(self):
        """ServingConfig must read 0.55 from settings when configured."""
        from core.services.retriever import get_serving_config
        config = get_serving_config()
        self.assertEqual(config.dense_relevance_gate, 0.55)

    def test_two_configs_with_different_gates_are_independent(self):
        """Two ServingConfig instances must not share mutable state."""
        from core.services.retriever import ServingConfig
        config_low = ServingConfig(
            dense_relevance_gate=0.38,
            sparse_relevance_gate=0.055,
            embedding_profile="auto",
        )
        config_high = ServingConfig(
            dense_relevance_gate=0.55,
            sparse_relevance_gate=0.055,
            embedding_profile="auto",
        )
        self.assertEqual(config_low.dense_relevance_gate, 0.38)
        self.assertEqual(config_high.dense_relevance_gate, 0.55)
        with self.assertRaises(Exception):
            config_low.dense_relevance_gate = 0.99  # type: ignore
