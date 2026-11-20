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

        # Serving processes opt in to warming BGE, V5 and Qwen in the background so
        # model loading never happens inside a customer request.
        import os
        import sys
        if os.getenv("SERVY_PRELOAD_MODELS") == "1" and "runserver" in sys.argv:
            from core.ml.preload import start_preload
            start_preload()

        # No embedding model is loaded here. Active serving uses the verified
        # local BGE profile, loaded offline on first use by core.ml.embeddings.
