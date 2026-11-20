"""Warm the serving models once at server start, off the request path.

Measured cold costs on the development laptop: V5 SHA-256 verification + load
~43 s, BGE-small load ~0.3 s, Qwen load into GPU memory a few seconds. Paying
them inside the first customer request produced 35-50 s answers. Each step is
independent; a failure is logged and the request path keeps its own bounded
fallbacks. Nothing here reads tenant data.
"""
from __future__ import annotations

import logging
import threading
import time

from django.conf import settings

logger = logging.getLogger("servy.rag.preload")
_started = False
_lock = threading.Lock()
status = {"bge": "pending", "v5": "pending", "qwen": "pending"}


def _warm_bge():
    from core.ml.embeddings import encode_queries, get_profile, load_model
    profile = get_profile(str(getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "bge-small") or "bge-small"))
    model = load_model(profile)
    if model is None:
        raise RuntimeError("BGE model unavailable")
    vector = encode_queries(profile, model, ["equipment not cooling"])[0]
    if len(vector) == 0:
        raise RuntimeError("BGE produced an empty vector")
    # Open the vector collection and run one query so the first customer search
    # does not pay the collection-open cost (measured ~1 s). No results are used.
    from core.ml.embeddings import get_collection
    collection = get_collection(profile, create=False)
    if collection is not None and collection.count():
        collection.query(query_embeddings=[list(map(float, vector))], n_results=1)


def _warm_v5():
    from core.ml import reranker
    if not reranker.is_reranker_enabled():
        status["v5"] = "disabled"
        return
    # Verifies the pinned SHA-256, loads the checkpoint and runs real scoring.
    reranker.require_verified_v5_reranker()


def _warm_qwen():
    import requests
    from urllib.parse import urlsplit, urlunsplit
    if str(getattr(settings, "SERVY_CUSTOMER_ANSWER_GENERATION", "disabled")).lower() != "ollama":
        status["qwen"] = "disabled"
        return
    url = str(getattr(settings, "SERVY_CUSTOMER_GENERATION_URL", "http://127.0.0.1:11434/api/chat"))
    parts = urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError("Qwen warm-up is loopback only")
    generate = urlunsplit((parts.scheme, parts.netloc, "/api/generate", "", ""))
    session = requests.Session()
    session.trust_env = False
    try:
        # An empty prompt loads the model and keeps it resident; nothing is generated.
        response = session.post(generate, json={
            "model": str(getattr(settings, "SERVY_CUSTOMER_GENERATION_MODEL", "qwen3.5:2b-q4_K_M")),
            "keep_alive": str(getattr(settings, "SERVY_CUSTOMER_GENERATION_KEEP_ALIVE", "30m")),
        }, timeout=(2, 120), allow_redirects=False)
        response.raise_for_status()
    finally:
        session.close()


def _run():
    for name, step in (("bge", _warm_bge), ("qwen", _warm_qwen), ("v5", _warm_v5)):
        started = time.monotonic()
        try:
            step()
            if status[name] == "pending":
                status[name] = "ready"
            logger.info("Preloaded %s in %.1f s", name, time.monotonic() - started)
        except Exception as exc:
            status[name] = "failed"
            logger.warning("Preloading %s failed (%s); requests use bounded fallbacks", name, type(exc).__name__)


def start_preload() -> bool:
    """Start the warm-up thread once per process. Returns True when started."""
    global _started
    with _lock:
        if _started:
            return False
        _started = True
    threading.Thread(target=_run, name="servy-model-preload", daemon=True).start()
    return True
