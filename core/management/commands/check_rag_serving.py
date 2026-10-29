"""Read-only BGE + V5 serving readiness command.

Without --strict, inspect configuration without loading private models.
--strict verifies the locally available BGE models and Chroma collections
and executes the pinned V5 scorer using synthetic input. It never downloads,
trains, mutates indexes or touches private source text.
"""
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256
from core.services.retriever import get_serving_config


def serving_configuration_issues(config, config_settings):
    """Why this process cannot claim to be running evaluated BGE+V5."""
    issues = []
    if config.embedding_profile != "bge-small":
        issues.append("expected embedding profile bge-small")
    if config.retrieval_mode != "hybrid":
        issues.append("expected hybrid retrieval mode")
    if not config.reranker_enabled:
        issues.append("V5 reranker not enabled")
    if int(getattr(config_settings, "SERVY_RERANKER_POOL_SIZE", 5)) != 5:
        issues.append("V5 pool size differs from evaluated pool five")
    expected = str(getattr(config_settings, "SERVY_RERANKER_EXPECTED_SHA256", "") or "").strip().lower()
    if expected != SERVY_V5_WEIGHTS_SHA256:
        issues.append("approved V5 checkpoint SHA-256 not configured")
    if not str(getattr(config_settings, "SERVY_RERANKER_MODEL_PATH", "") or "").strip():
        issues.append("private V5 checkpoint path not configured")
    return issues


def verify_local_serving_artifacts():
    """Verify BGE indexes with create=False and execute the pinned V5 scorer."""
    from core.ml.embeddings import get_collection, get_profile, load_model
    from core.ml.reranker import require_verified_v5_reranker

    for key in ("bge-small", "company-bge-small"):
        profile = get_profile(key)
        if load_model(profile, allow_download=False) is None:
            raise RuntimeError("Local BGE model not available")
        if get_collection(profile, create=False) is None:
            raise RuntimeError("BGE collection not available")
    require_verified_v5_reranker()


class Command(BaseCommand):
    help = "Inspect BGE+V5 serving configuration; --strict verifies offline local models and indexes."

    def add_arguments(self, parser):
        parser.add_argument("--strict", action="store_true",
                            help="Require evaluated config, local BGE indexes and verified V5 inference.")

    def handle(self, *args, **options):
        config = get_serving_config()
        issues = serving_configuration_issues(config, settings)
        self.stdout.write(
            f"Embedding={config.embedding_profile} mode={config.retrieval_mode} "
            f"V5_enabled={config.reranker_enabled} "
            f"pool={getattr(settings, 'SERVY_RERANKER_POOL_SIZE', 5)}"
        )
        for issue in issues:
            self.stdout.write(self.style.WARNING(f"NOT VERIFIED: {issue}"))
        if options["strict"]:
            if issues:
                raise CommandError(
                    f"Evaluated BGE+V5 configuration is not active ({len(issues)} issue(s))."
                )
            try:
                verify_local_serving_artifacts()
            except Exception as exc:
                # Never include raw model paths, customer documents or secret-bearing errors.
                raise CommandError(
                    f"Offline serving verification failed: {type(exc).__name__}"
                ) from exc
            self.stdout.write(self.style.SUCCESS(
                "LOCAL BGE+V5 VERIFIED: model/index identities and synthetic scorer only; "
                "accuracy, authorization, and answer quality not certified."
            ))
        else:
            self.stdout.write(
                "Diagnostic only; run --strict on the approved deployment to "
                "verify the actual local BGE+V5 serving path."
            )
