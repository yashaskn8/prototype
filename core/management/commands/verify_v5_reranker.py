"""Verify the configured local Servy V5 checkpoint (no DB or API calls)."""
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.ml.reranker import get_reranker, is_reranker_enabled, _local_checkpoint


class Command(BaseCommand):
    help = "Load and score one benign query with the local V5 cross-encoder."

    def handle(self, *args, **options):
        if not is_reranker_enabled():
            raise CommandError("SERVY_RERANKER_ENABLED=1 must be set for the V5 smoke check.")
        checkpoint = _local_checkpoint()
        if checkpoint is None:
            raise CommandError("Set SERVY_RERANKER_MODEL_PATH to your full best_model folder.")
        if not settings.SERVY_RERANKER_EXPECTED_SHA256:
            raise CommandError("Set SERVY_RERANKER_EXPECTED_SHA256 to the verified V5 weights digest.")
        model = get_reranker()
        if model is None:
            raise CommandError("V5 reranker unavailable: inspect Django logs for the exact cause.")
        try:
            scores = model.predict([["machine fault", "machine fault procedure"]])
            if len(scores) != 1:
                raise ValueError("Model did not return exactly one score")
            score = float(scores[0])
            import math
            if not math.isfinite(score):
                raise ValueError("Non-finite score")
        except Exception as exc:
            raise CommandError(f"V5 model inference failed: {exc}") from exc
        self.stdout.write(self.style.SUCCESS(
            f"SERVY V5 RERANKER VERIFIED: {checkpoint} (device={settings.SERVY_RERANKER_DEVICE})"
        ))
