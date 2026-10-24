"""Verify that the pinned local V5 model actually runs through Servy's scorer.

This is an offline, synthetic serving-path smoke check, not a production
promotion decision, independent accuracy evaluation, or HTTP authorization test.
"""
import math
from types import SimpleNamespace

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.ml.reranker import (
    SERVY_V5_WEIGHTS_SHA256,
    _local_checkpoint,
    is_reranker_enabled,
    rerank_candidates,
)


class Command(BaseCommand):
    help = "Verify the pinned V5 checkpoint by executing the real reranker scoring path."

    def handle(self, *args, **options):
        if not is_reranker_enabled():
            raise CommandError("SERVY_RERANKER_ENABLED=1 must be set for the V5 smoke check.")
        checkpoint = _local_checkpoint()
        if checkpoint is None:
            raise CommandError("Set SERVY_RERANKER_MODEL_PATH to your full best_model folder.")
        expected = str(
            getattr(settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or ""
        ).strip().lower()
        if expected != SERVY_V5_WEIGHTS_SHA256:
            raise CommandError("SERVY_RERANKER_EXPECTED_SHA256 must equal the approved V5 weights digest.")

        # No private document, customer text, network API, or database is used.
        # This calls the same scorer as a real RAG request, including raw-logit
        # inference and its fail-closed behavior. A fallback must NOT pass.
        probe = SimpleNamespace(
            source_kind="knowledge_document",
            text="Inspect the hydraulic pressure gauge before servicing the pump.",
            score=0.5,
        )
        try:
            ranked = rerank_candidates(
                "How should I inspect the hydraulic pump?",
                [probe],
                top_k=1,
            )
            if (
                len(ranked) != 1
                or ranked[0] is not probe
                or getattr(probe, "reranker_model", None) != "servy_v5_bge_verified"
                or not math.isfinite(float(getattr(probe, "rerank_logit")))
                or not math.isfinite(float(getattr(probe, "reranker_score")))
            ):
                raise ValueError("V5 scorer did not produce a verified finite score")
        except Exception as exc:
            raise CommandError(
                "V5 serving-path verification failed or fell back: "
                f"{exc}. Inspect the Django reranker logs."
            ) from exc

        self.stdout.write(self.style.SUCCESS(
            f"SERVY V5 RERANKER VERIFIED: {checkpoint} (device={settings.SERVY_RERANKER_DEVICE})"
        ))
