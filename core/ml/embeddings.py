"""Embedding model / Chroma collection registry.

Each embedding model is bound to exactly one versioned Chroma collection.
Vectors from different models are NEVER written into or queried from the same
collection, even when dimensions match (MiniLM and BGE-small are both 384-d
but live in different embedding spaces).

The legacy MiniLM profile keeps the existing collection name ``servy_dense_kb``
so current production behaviour and data are untouched.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, Optional

from django.conf import settings


class EmbeddingConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class EmbeddingProfile:
    key: str
    model_id: str
    revision: Optional[str]
    dim: int
    collection: str
    query_prefix: str = ""      # BGE retrieval instruction for queries only
    content_first: bool = False  # embed body/heading only (no manual metadata)
    local_path: Optional[str] = None


try:
    if settings.configured:
        BGE_SMALL_REVISION = getattr(settings, "SERVY_BGE_SMALL_REVISION", "") or "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    else:
        BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
except Exception:
    BGE_SMALL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PROFILES: Dict[str, EmbeddingProfile] = {
    "minilm": EmbeddingProfile(
        key="minilm", model_id="all-MiniLM-L6-v2", revision=None, dim=384,
        collection="servy_dense_kb", content_first=False,
    ),
    "bge-small": EmbeddingProfile(
        key="bge-small", model_id="BAAI/bge-small-en-v1.5", revision=BGE_SMALL_REVISION,
        dim=384, collection="servy_dense_bge_small_v1",
        query_prefix=BGE_QUERY_PREFIX, content_first=True,
    ),
}


def register_finetuned(path: str, key: str = "bge-small-ft") -> EmbeddingProfile:
    """Register a locally saved fine-tuned checkpoint as its own profile/collection."""
    p = EmbeddingProfile(
        key=key, model_id=path, revision=None, dim=384,
        collection=f"servy_dense_{key.replace('-', '_')}_v1",
        query_prefix=BGE_QUERY_PREFIX, content_first=True, local_path=path,
    )
    PROFILES[key] = p
    return p


def get_profile(key: str) -> EmbeddingProfile:
    try:
        return PROFILES[key]
    except KeyError:
        raise EmbeddingConfigError(f"Unknown embedding profile '{key}'. Known: {sorted(PROFILES)}")


_models: Dict[str, object] = {}
_lock = threading.Lock()


def load_model(profile: EmbeddingProfile, allow_download: bool = False):
    """Load a SentenceTransformer for a profile. Offline by default."""
    if profile.key in _models:
        return _models[profile.key]
    with _lock:
        if profile.key in _models:
            return _models[profile.key]
        from sentence_transformers import SentenceTransformer
        kwargs = {"device": getattr(settings, "SERVY_EMBEDDING_DEVICE", "cpu")}
        if profile.revision:
            kwargs["revision"] = profile.revision
        try:
            m = SentenceTransformer(profile.local_path or profile.model_id,
                                    local_files_only=not allow_download, **kwargs)
        except Exception as exc:
            raise EmbeddingConfigError(
                f"Model '{profile.model_id}' not available locally. "
                f"Run: python core/ml/download_bge.py ml_artifacts/bge_small_manifest.json ({exc})"
            ) from exc
        actual = m.get_sentence_embedding_dimension()
        if actual != profile.dim:
            raise EmbeddingConfigError(f"{profile.model_id} dim {actual} != expected {profile.dim}")
        _models[profile.key] = m
        return m


def get_collection(profile: EmbeddingProfile, create: bool = True):
    """Return the profile's collection; refuse to reuse a collection stamped by another model."""
    from core.services.chroma_store import get_chroma_client
    client = get_chroma_client()
    meta = {"hnsw:space": "cosine", "servy_model": profile.model_id, "servy_dim": profile.dim}
    if create:
        coll = client.get_or_create_collection(name=profile.collection, metadata=meta)
    else:
        coll = client.get_collection(profile.collection)
    stamped = (coll.metadata or {}).get("servy_model")
    # Legacy servy_dense_kb has no stamp; it is only ever paired with minilm.
    if stamped and stamped != profile.model_id:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' belongs to model '{stamped}', not '{profile.model_id}'.")
    if not stamped and profile.key != "minilm":
        raise EmbeddingConfigError(f"Collection '{profile.collection}' is unstamped; refusing to use it.")
    return coll


def encode_queries(profile: EmbeddingProfile, model, queries):
    return model.encode([profile.query_prefix + q for q in queries],
                        normalize_embeddings=True, batch_size=getattr(settings, "SERVY_EMBEDDING_BATCH_SIZE", 8))


def passage_text(profile: EmbeddingProfile, chunk) -> str:
    """Content-first passage text: heading + body. No title/category/tags/filename."""
    if profile.content_first:
        heading = (chunk.heading or "").strip()
        body = chunk.text or ""
        return f"{heading}\n{body}".strip() if heading and heading not in body else body
    d = chunk.document
    prefix = " ".join(filter(None, [
        d.title, d.description, d.tags, d.doc_type,
        d.domain.name if d.domain else "", d.category.name if d.category else "",
        d.product.name if d.product else "", d.asset.name if d.asset else "",
        d.asset.model_number if d.asset else "",
    ]))
    return f"{prefix} {chunk.heading}\n{chunk.text}".strip()
