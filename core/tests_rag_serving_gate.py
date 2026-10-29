"""Synthetic-only regression tests for BGE+V5 serving readiness."""
from io import StringIO
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, override_settings

from core.management.commands.check_rag_serving import serving_configuration_issues
from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256
from core.services.retriever import get_serving_config


ACTIVE = dict(
    SERVY_SERVING_EMBEDDING_PROFILE="bge-small",
    SERVY_RETRIEVAL_MODE="hybrid",
    SERVY_RERANKER_ENABLED=True,
    SERVY_RERANKER_POOL_SIZE=5,
    SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
    SERVY_RERANKER_MODEL_PATH="private/synthetic_checkpoint",
)


class ServingReadinessTests(SimpleTestCase):
    @override_settings(**ACTIVE)
    def test_evaluated_configuration_passes_static_gate(self):
        from django.conf import settings
        self.assertEqual(serving_configuration_issues(get_serving_config(), settings), [])

    @override_settings(
        SERVY_SERVING_EMBEDDING_PROFILE="minilm",
        SERVY_RETRIEVAL_MODE="sparse_only",
        SERVY_RERANKER_ENABLED=False,
    )
    def test_strict_gate_fails_on_unmatched_serving_config(self):
        with self.assertRaises(CommandError):
            call_command("check_rag_serving", strict=True, stdout=StringIO())

    @override_settings(**ACTIVE)
    def test_strict_gate_runs_offline_verification(self):
        out = StringIO()
        with mock.patch("core.management.commands.check_rag_serving.verify_local_serving_artifacts") as verify:
            call_command("check_rag_serving", strict=True, stdout=out)
        verify.assert_called_once_with()
        self.assertIn("LOCAL BGE+V5 VERIFIED", out.getvalue())

    @override_settings(**ACTIVE)
    def test_strict_gate_does_not_leak_artifact_failure_details(self):
        out = StringIO()
        with mock.patch(
            "core.management.commands.check_rag_serving.verify_local_serving_artifacts",
            side_effect=RuntimeError("private/path/with/sensitive/data")
        ):
            with self.assertRaises(CommandError) as caught:
                call_command("check_rag_serving", strict=True, stdout=out)
        self.assertIn("RuntimeError", str(caught.exception))
        self.assertNotIn("private/path", str(caught.exception))

    def test_non_strict_is_read_only_without_artifact_load(self):
        out = StringIO()
        with mock.patch("core.management.commands.check_rag_serving.verify_local_serving_artifacts") as verify:
            call_command("check_rag_serving", stdout=out)
        verify.assert_not_called()
        self.assertIn("Diagnostic only", out.getvalue())

    @override_settings(**ACTIVE)
    def test_disabled_v5_cannot_pass_even_with_bge_profile(self):
        from django.conf import settings
        with override_settings(SERVY_RERANKER_ENABLED=False):
            issues = serving_configuration_issues(get_serving_config(), settings)
        self.assertIn("V5 reranker not enabled", issues)
