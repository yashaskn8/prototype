import threading
from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "core"

    def ready(self):
        from . import signals  # noqa: F401

        # Production opt-in: do not report a healthy startup if the pinned
        # V5 checkpoint cannot run the exact serving scorer.
        from django.conf import settings
        if getattr(settings, "SERVY_RERANKER_REQUIRED", False):
            from core.ml.reranker import require_verified_v5_reranker
            require_verified_v5_reranker()

        # Pre-warm the embedding model in background thread so the first RAG query has 0s cold start
        def _warmup():
            try:
                from core.services.chroma_store import get_sentence_transformer
                get_sentence_transformer()
            except Exception:
                pass

        threading.Thread(target=_warmup, daemon=True).start()
