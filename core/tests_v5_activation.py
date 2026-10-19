"""The V5 verification command must never certify other checkpoint identities."""
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
        with mock.patch("core.management.commands.verify_v5_reranker._local_checkpoint",
                        return_value=Path("/synthetic/best_model")), mock.patch(
            "core.management.commands.verify_v5_reranker.get_reranker"
        ) as load:
            with self.assertRaisesRegex(CommandError, "approved V5"):
                call_command("verify_v5_reranker")
        load.assert_not_called()

    @override_settings(
        SERVY_RERANKER_ENABLED=True,
        SERVY_RERANKER_EXPECTED_SHA256=SERVY_V5_WEIGHTS_SHA256,
        SERVY_RERANKER_DEVICE="cpu",
    )
    def test_v5_digest_and_finite_prediction_succeed(self):
        model = mock.Mock()
        model.predict.return_value = [1.5]
        output = StringIO()
        with mock.patch("core.management.commands.verify_v5_reranker._local_checkpoint",
                        return_value=Path("/synthetic/best_model")), mock.patch(
            "core.management.commands.verify_v5_reranker.get_reranker",
            return_value=model,
        ):
            call_command("verify_v5_reranker", stdout=output)
        self.assertIn("SERVY V5 RERANKER VERIFIED", output.getvalue())
        model.predict.assert_called_once_with([["machine fault", "machine fault procedure"]])
