import tempfile
import unittest
from pathlib import Path
from django.test import TestCase

from core.ml.benchmark_registry import (
    load_and_verify_benchmark,
    compute_file_sha256,
    get_evaluation_provenance_metadata,
    BenchmarkIntegrityError,
    HOLDOUT_SHA256,
    CI_REGRESSION_FIXTURE_SHA256,
    REGISTERED_BENCHMARKS,
    ROLE_INDEPENDENT_HOLDOUT,
    ROLE_SOFTWARE_REGRESSION_FIXTURE,
    BASE_DIR,
)

PRIVATE_HOLDOUT_PATH = REGISTERED_BENCHMARKS["servy-real-holdout-v1"].file_path


class BenchmarkRegistryTests(TestCase):
    def test_registered_ci_fixture_loading(self):
        """Public CI regression fixture must load and verify with authentic digest."""
        rec = load_and_verify_benchmark("servy-ci-regression-fixture-v1")
        self.assertEqual(rec["dataset_id"], "servy-ci-regression-fixture-v1")
        self.assertEqual(rec["dataset_role"], ROLE_SOFTWARE_REGRESSION_FIXTURE)
        self.assertEqual(rec["sha256_digest"], CI_REGRESSION_FIXTURE_SHA256)
        self.assertEqual(rec["number_of_cases"], 64)
        self.assertEqual(len(rec["query_ids"]), 64)
        self.assertEqual(len(rec["category_distribution"]), 14)

    def test_incorrect_public_fixture_digest_fails_closed(self):
        """Tampering with public fixture bytes raises BenchmarkIntegrityError."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tf:
            tf.write('{"query_id": "tampered", "tenant_id": 9, "query_text": "fake"}\n')
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(BenchmarkIntegrityError) as ctx:
                load_and_verify_benchmark("servy-ci-regression-fixture-v1", custom_path=tf_path)
            self.assertIn("Benchmark integrity violation", str(ctx.exception))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_missing_private_holdout_fails_closed(self):
        """Private holdout must fail explicitly when required and unavailable."""
        missing_path = BASE_DIR / "nonexistent_dir" / "missing_holdout.jsonl"
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            load_and_verify_benchmark("servy-real-holdout-v1", custom_path=missing_path)
        self.assertIn("Production benchmark", str(ctx.exception))

    @unittest.skipUnless(PRIVATE_HOLDOUT_PATH.exists(), "Private holdout only available in authorized evaluation environment")
    def test_registered_holdout_digest_and_loading_isolated(self):
        """Must load and verify canonical independent holdout when authorized environment has file."""
        rec = load_and_verify_benchmark("servy-real-holdout-v1")
        self.assertEqual(rec["dataset_id"], "servy-real-holdout-v1")
        self.assertEqual(rec["sha256_digest"], HOLDOUT_SHA256)
        self.assertEqual(rec["number_of_cases"], 64)
        self.assertEqual(len(rec["query_ids"]), 64)
        self.assertEqual(rec["dataset_role"], ROLE_INDEPENDENT_HOLDOUT)
        self.assertEqual(len(rec["category_distribution"]), 14)
        self.assertEqual(rec["tenant_distribution"][9], 44)
        self.assertEqual(rec["tenant_distribution"][6], 20)

    def test_corrupted_json_benchmark_fails_closed(self):
        """Invalid JSON syntax in benchmark raises BenchmarkIntegrityError."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tf:
            tf.write('{"query_id": "q1", INVALID_JSON\n')
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(BenchmarkIntegrityError) as ctx:
                load_and_verify_benchmark("servy-ci-regression-fixture-v1", custom_path=tf_path)
            # Will fail on sha256 or json decode
            self.assertTrue(isinstance(ctx.exception, BenchmarkIntegrityError))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_attempted_dataset_substitution_fails_closed(self):
        """Passing public fixture file for private holdout dataset ID fails closed with digest mismatch."""
        ci_fixture_path = REGISTERED_BENCHMARKS["servy-ci-regression-fixture-v1"].file_path
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            load_and_verify_benchmark("servy-real-holdout-v1", custom_path=ci_fixture_path)
        self.assertIn("Benchmark integrity violation", str(ctx.exception))

    def test_duplicate_query_ids_fails_closed(self):
        """Duplicate query IDs must fail closed to prevent skewed denominators."""
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tf:
            tf.write('{"query_id": "DUP-1", "tenant_id": 9, "query_text": "q1"}\n')
            tf.write('{"query_id": "DUP-1", "tenant_id": 9, "query_text": "q2"}\n')
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(BenchmarkIntegrityError) as ctx:
                load_and_verify_benchmark("servy-ci-regression-fixture-v1", custom_path=tf_path)
            self.assertTrue(isinstance(ctx.exception, BenchmarkIntegrityError))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_unregistered_benchmark_fails_closed(self):
        """Unregistered benchmark dataset_id raises BenchmarkIntegrityError."""
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            load_and_verify_benchmark("unknown-fake-benchmark")
        self.assertIn("Unregistered benchmark", str(ctx.exception))

    def test_evaluation_provenance_metadata(self):
        """Evaluation provenance must capture code commit SHA, serving config, and digests."""
        prov = get_evaluation_provenance_metadata()
        self.assertIn("evaluation_code_sha", prov)
        self.assertIn("serving_configuration", prov)
        self.assertIn("source_document_fingerprints", prov)
        self.assertIn("chroma_collection_manifest", prov)
        self.assertIn("checkpoint_fingerprints", prov)
        self.assertIn("SERVY_RAG_TOP_K", prov["serving_configuration"])
