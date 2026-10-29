"""The V5 activation check must exercise the real scoring path, not a substitute.

These tests are synthetic and mock the model weights. The optional real-weight
HTTP and database test is in core.tests_v5_runtime.
"""
from io import StringIO
from pathlib import Path
from unittest import mock

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase, override_settings

from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256


class V5ActivationCommandTests(SimpleTestCase):
    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_EXPECTED_SHA256="0" * 64,
    )
    def test_rejects_non_v5_digest_before_loading(self):
        with mock.patch(
            "core.management.commands.verify_v5_reranker._local_checkpoint",
            return_value=Path("/synthetic/best_model"),
        ), mock.patch("core.ml.reranker.get_reranker") as load:
            with self.assertRaisesRegex(CommandError, "approved V5"):
                call_command("verify_v5_reranker")
        load.assert_not_called()

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_DEVICE="cpu",
        SERVY_RERANKER_POOL_SIZE=5,
        SERVY_RERANKER_MODEL_PATH="best_model",
    )
    def test_verified_v5_uses_serving_raw_logits(self):
        model = mock.Mock()
        model.predict.return_value = [1.5]
        output = StringIO()
        with mock.patch(
            "core.management.commands.verify_v5_reranker._local_checkpoint",
            return_value=Path("/synthetic/best_model"),
        ), mock.patch(
            "core.ml.reranker.get_reranker",
            return_value=model,
        ):
            call_command("verify_v5_reranker", stdout=output)

        self.assertIn("SERVY V5 RERANKER VERIFIED", output.getvalue())
        model.predict.assert_called_once()
        args, kwargs = model.predict.call_args
        self.assertEqual(
            args[0],
            [[
                "How should I inspect the hydraulic pump?",
                "Inspect the hydraulic pressure gauge before servicing the pump.",
            ]],
        )
        import torch
        self.assertIsInstance(kwargs["activation_fn"], torch.nn.Identity)
        self.assertIs(kwargs["show_progress_bar"], False)

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_POOL_SIZE=5,
        SERVY_RERANKER_MODEL_PATH="best_model",
    )
    def test_model_unavailable_is_not_misreported_as_verified(self):
        with mock.patch(
            "core.management.commands.verify_v5_reranker._local_checkpoint",
            return_value=Path("/synthetic/best_model"),
        ), mock.patch("core.ml.reranker.get_reranker", return_value=None):
            with self.assertRaisesRegex(CommandError, "serving-path verification failed"):
                call_command("verify_v5_reranker")

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_POOL_SIZE=5,
        SERVY_RERANKER_MODEL_PATH="best_model",
    )
    def test_nonfinite_prediction_is_not_misreported_as_verified(self):
        model = mock.Mock()
        model.predict.return_value = [float("nan")]
        with mock.patch(
            "core.management.commands.verify_v5_reranker._local_checkpoint",
            return_value=Path("/synthetic/best_model"),
        ), mock.patch("core.ml.reranker.get_reranker", return_value=model):
            with self.assertRaisesRegex(CommandError, "serving-path verification failed"):
                call_command("verify_v5_reranker")

    @override_settings(SERVY_RERANKER_ENABLED=False)
    def test_disabled_model_is_not_misreported_as_verified(self):
        with mock.patch("core.ml.reranker.get_reranker") as load:
            with self.assertRaisesRegex(CommandError, "SERVY_RERANKER_ENABLED=1"):
                call_command("verify_v5_reranker")
        load.assert_not_called()

    @override_settings(SERVY_RERANKER_REQUIRED=True, SERVY_RERANKER_ENABLED=False)
    def test_required_v5_blocks_disabled_configuration(self):
        from core.ml.reranker import require_verified_v5_reranker
        with self.assertRaisesRegex(RuntimeError, "SERVY_RERANKER_ENABLED"):
            require_verified_v5_reranker()

    @override_settings(
        SERVY_RERANKER_REQUIRED=True, SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_MODEL_PATH="best_model",
        SERVY_RERANKER_EXPECTED_SHA256="0" * 64,
    )
    def test_required_v5_blocks_unapproved_digest_without_loading(self):
        from core.ml.reranker import require_verified_v5_reranker
        with mock.patch("core.ml.reranker.get_reranker") as loader:
            with self.assertRaisesRegex(RuntimeError, "pinned V5"):
                require_verified_v5_reranker()
        loader.assert_not_called()

    @override_settings(
        SERVY_RERANKER_REQUIRED=True, SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_MODEL_PATH="best_model",
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_POOL_SIZE=5,
    )
    def test_required_v5_blocks_unavailable_checkpoint_fallback(self):
        from core.ml.reranker import require_verified_v5_reranker
        with mock.patch("core.ml.reranker.get_reranker", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "refusing to start"):
                require_verified_v5_reranker()

    @override_settings(
        SERVY_RERANKER_REQUIRED=True, SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_MODEL_PATH="best_model",
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_POOL_SIZE=5,
    )
    def test_required_v5_executes_actual_serving_score(self):
        from core.ml.reranker import require_verified_v5_reranker
        model = mock.Mock()
        model.predict.return_value = [1.25]
        with mock.patch("core.ml.reranker.get_reranker", return_value=model):
            require_verified_v5_reranker()
        model.predict.assert_called_once()
        import torch
        self.assertIsInstance(model.predict.call_args.kwargs["activation_fn"], torch.nn.Identity)

    @override_settings(SERVY_RERANKER_REQUIRED=True)
    def test_django_ready_refuses_startup_on_required_model_failure(self):
        from django.apps import apps
        with mock.patch(
            "core.ml.reranker.require_verified_v5_reranker",
            side_effect=RuntimeError("required model unavailable"),
        ):
            with self.assertRaisesRegex(RuntimeError, "required model unavailable"):
                apps.get_app_config("core").ready()
