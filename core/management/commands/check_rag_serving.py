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


class ServingArtifactError(RuntimeError):
    """A named local serving artifact is missing or wrong. Carries no paths."""


def verify_local_serving_artifacts(required_collections=("bge-small", "company-bge-small"),
                                   optional_collections=()):
    """Verify the local BGE model and indexes, then execute the pinned V5 scorer.

    For each profile: the model loads offline, a probe embedding has the
    profile's dimension, and the collection exists (create=False) and is not
    empty. A collection listed as optional is reported rather than required.
    Then the SHA-pinned V5 checkpoint must run real scoring. Nothing is
    downloaded, created or modified. Returns what was verified.
    """
    from core.ml.embeddings import encode_passages, get_collection, get_profile, load_model
    from core.ml.reranker import SERVY_V5_WEIGHTS_SHA256, require_verified_v5_reranker

    verified = {"collections": {}, "embedding_dimensions": {}}
    for key in tuple(required_collections) + tuple(optional_collections):
        required = key in required_collections
        profile = get_profile(key)
        model = load_model(profile, allow_download=False)
        if model is None:
            raise ServingArtifactError(f"local BGE model for profile {key} is not available")
        vector = encode_passages(profile, model, ["servy serving readiness probe"])[0]
        if len(vector) != profile.dim:
            raise ServingArtifactError(
                f"profile {key} produced {len(vector)}-dimensional embeddings, expected {profile.dim}")
        verified["embedding_dimensions"][key] = profile.dim
        try:
            collection = get_collection(profile, create=False)
            count = collection.count() if collection is not None else 0
        except Exception as exc:
            if required:
                raise ServingArtifactError(
                    f"collection for profile {key} is not available ({type(exc).__name__})") from exc
            collection, count = None, 0
        if collection is None or count < 1:
            if required:
                raise ServingArtifactError(f"collection for profile {key} is missing or empty")
            verified["collections"][key] = "NOT_PRESENT"
        else:
            verified["collections"][key] = count
    try:
        require_verified_v5_reranker()
    except Exception as exc:
        raise ServingArtifactError(
            f"pinned V5 reranker did not run verified inference ({type(exc).__name__})") from exc
    verified["v5"] = {"inference": "verified", "weights_sha256": SERVY_V5_WEIGHTS_SHA256,
                      "model": "servy_v5_bge_verified"}
    return verified


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
