import tempfile
from pathlib import Path
from django.test import TestCase

from core.ml.benchmark_registry import (
    load_and_verify_benchmark,
    compute_file_sha256,
    get_evaluation_provenance_metadata,
    BenchmarkIntegrityError,
    HOLDOUT_SHA256,
    REGISTERED_BENCHMARKS,
    ROLE_INDEPENDENT_HOLDOUT,
    ROLE_SOFTWARE_REGRESSION_FIXTURE,
)


class BenchmarkRegistryTests(TestCase):
    def test_registered_holdout_digest_and_loading(self):
        # Must load and verify canonical independent holdout
        rec = load_and_verify_benchmark("servy-real-holdout-v1")
        self.assertEqual(rec["dataset_id"], "servy-real-holdout-v1")
        self.assertEqual(rec["sha256_digest"], HOLDOUT_SHA256)
        self.assertEqual(rec["number_of_cases"], 64)
        self.assertEqual(len(rec["query_ids"]), 64)
        self.assertEqual(rec["dataset_role"], ROLE_INDEPENDENT_HOLDOUT)
        self.assertEqual(len(rec["category_distribution"]), 14)
        self.assertEqual(rec["tenant_distribution"][9], 44)
        self.assertEqual(rec["tenant_distribution"][6], 20)

    def test_registered_ci_fixture_loading(self):
        # CI regression fixture must be explicitly labeled as regression fixture
        rec = load_and_verify_benchmark("servy-ci-regression-fixture-v1")
        self.assertEqual(rec["dataset_id"], "servy-ci-regression-fixture-v1")
        self.assertEqual(rec["dataset_role"], ROLE_SOFTWARE_REGRESSION_FIXTURE)
        self.assertEqual(rec["sha256_digest"], HOLDOUT_SHA256)
        self.assertEqual(rec["number_of_cases"], 64)

    def test_mismatched_sha256_raises_integrity_error(self):
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".jsonl") as tf:
            tf.write('{"query_id": "tampered", "tenant_id": 9, "query_text": "fake"}\n')
            tf_path = Path(tf.name)

        try:
            with self.assertRaises(BenchmarkIntegrityError) as ctx:
                load_and_verify_benchmark("servy-real-holdout-v1", custom_path=tf_path)
            self.assertIn("Benchmark integrity violation", str(ctx.exception))
        finally:
            if tf_path.exists():
                tf_path.unlink()

    def test_unregistered_benchmark_fails_closed(self):
        with self.assertRaises(BenchmarkIntegrityError) as ctx:
            load_and_verify_benchmark("unknown-fake-benchmark")
        self.assertIn("Unregistered benchmark", str(ctx.exception))

    def test_missing_benchmark_file_fails_closed(self):
        missing_path = Path("non_existent_benchmark_file.jsonl")
        with self.assertRaises(BenchmarkIntegrityError):
            load_and_verify_benchmark("servy-real-holdout-v1", custom_path=missing_path)

    def test_evaluation_provenance_metadata(self):
        prov = get_evaluation_provenance_metadata()
        self.assertIn("evaluation_code_sha", prov)
        self.assertIn("serving_configuration", prov)
        self.assertIn("source_document_fingerprints", prov)
        self.assertIn("chroma_collection_manifest", prov)
        self.assertIn("checkpoint_fingerprints", prov)
        self.assertIn("SERVY_RAG_TOP_K", prov["serving_configuration"])
