from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        from . import signals  # noqa: F401
        from .test_database_guard import install as install_test_database_guard

        # Database tests may only run against a throwaway test database.
        install_test_database_guard()

        # Production opt-in: do not report a healthy startup if the pinned
        # V5 checkpoint cannot run the exact serving scorer.
        from django.conf import settings
        if getattr(settings, "SERVY_RERANKER_REQUIRED", False):
            from core.ml.reranker import require_verified_v5_reranker
            require_verified_v5_reranker()

        # No embedding model is loaded here. Active serving uses the verified
        # local BGE profile, loaded on first use by core.ml.embeddings. The old
        # unconditional warm-up loaded the legacy MiniLM model in every Django
        # process (management commands and tests included) and could attempt a
        # network download; the legacy MiniLM path still loads it on demand.
