"""Embedding model / Chroma collection registry.

Each embedding model is bound to exactly one versioned Chroma collection.
Vectors from different models are NEVER written into or queried from the same
collection, even when dimensions match (MiniLM and BGE-small are both 384-d
but live in different embedding spaces).

The legacy MiniLM profile keeps the existing collection name ``servy_dense_kb``
so current production behaviour and data are untouched.
"""
from __future__ import annotations

import hashlib
import threading
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
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
    promotion_eligible: bool = False
    checkpoint_hash: str = ""
    training_status: str = "production"
    dataset_manifest_hash: str = ""
    base_revision: Optional[str] = None


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
        collection="servy_dense_kb", content_first=False, promotion_eligible=True,
    ),
    "bge-small": EmbeddingProfile(
        key="bge-small", model_id="BAAI/bge-small-en-v1.5", revision=BGE_SMALL_REVISION,
        dim=384, collection="servy_dense_bge_small_v1",
        query_prefix=BGE_QUERY_PREFIX, content_first=True, promotion_eligible=True,
    ),
    "company-bge-small": EmbeddingProfile(
        key="company-bge-small", model_id="BAAI/bge-small-en-v1.5", revision=BGE_SMALL_REVISION,
        dim=384, collection="servy_company_bge_small_v1",
        query_prefix=BGE_QUERY_PREFIX, content_first=True, promotion_eligible=True,
    ),
    "company-minilm": EmbeddingProfile(
        key="company-minilm", model_id="all-MiniLM-L6-v2", revision=None, dim=384,
        collection="servy_company_dense_kb", content_first=False, promotion_eligible=True,
    ),
    "company-bge-ft": EmbeddingProfile(
        key="company-bge-ft",
        model_id=str(getattr(settings, "BASE_DIR", Path(__file__).resolve().parent.parent.parent) / "ml_models" / "checkpoints" / "servy_company_bge_v1" / "final"),
        revision=None,
        dim=384,
        collection="servy_company_bge_ft_v1",
        query_prefix=BGE_QUERY_PREFIX,
        content_first=True,
        local_path=str(getattr(settings, "BASE_DIR", Path(__file__).resolve().parent.parent.parent) / "ml_models" / "checkpoints" / "servy_company_bge_v1" / "final"),
        promotion_eligible=False,
        training_status="COMPANY_DATA_WEAKLY_SUPERVISED_EXPERIMENT",
        checkpoint_hash="7d394413398be69d2d1093538e5eec6cafa58cbfeae858cac4abf608b5db0d33",
        base_revision=BGE_SMALL_REVISION,
    ),
}


def register_finetuned(
    path: str,
    key: str = "bge-small-ft",
    *,
    promotion_eligible: bool = False,
    training_status: str = "synthetic_smoke",
    dataset_manifest_hash: str = "",
    base_revision: Optional[str] = None,
    checkpoint_hash: str = "",
    query_prefix: str = BGE_QUERY_PREFIX,
    content_first: bool = True,
) -> EmbeddingProfile:
    """Register a locally saved fine-tuned checkpoint as its own profile/collection."""
    p = EmbeddingProfile(
        key=key, model_id=path, revision=None, dim=384,
        collection=f"servy_dense_{key.replace('-', '_')}_v1",
        query_prefix=query_prefix, content_first=content_first, local_path=path,
        promotion_eligible=promotion_eligible,
        training_status=training_status,
        dataset_manifest_hash=dataset_manifest_hash,
        base_revision=base_revision or BGE_SMALL_REVISION,
        checkpoint_hash=checkpoint_hash,
    )
    PROFILES[key] = p
    return p


def get_profile(key: str) -> EmbeddingProfile:
    try:
        return PROFILES[key]
    except KeyError:
        raise EmbeddingConfigError(f"Unknown embedding profile '{key}'. Known: {sorted(PROFILES)}")


_models: Dict[tuple, object] = {}
_lock = threading.Lock()


def load_model(profile: EmbeddingProfile, allow_download: bool = False):
    """Load a SentenceTransformer for a profile. Offline by default."""
    checkpoint_fp = cached_checkpoint_fingerprint(profile.local_path) if profile.local_path else ""
    if profile.local_path and not checkpoint_fp:
        raise EmbeddingConfigError(f"Local checkpoint is missing: {profile.local_path}")
    cache_key = (profile.key, profile.model_id, profile.revision, profile.local_path, checkpoint_fp)
    if cache_key in _models:
        return _models[cache_key]
    with _lock:
        if cache_key in _models:
            return _models[cache_key]
        from sentence_transformers import SentenceTransformer
        kwargs = {"device": getattr(settings, "SERVY_EMBEDDING_DEVICE", "cpu")}
        if profile.revision:
            kwargs["revision"] = profile.revision
        model_target = profile.local_path or profile.model_id
        if model_target == "all-MiniLM-L6-v2":
            model_target = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            m = SentenceTransformer(model_target, local_files_only=not allow_download, **kwargs)
        except Exception as exc:
            raise EmbeddingConfigError(
                f"Model '{profile.model_id}' not available locally. "
                f"Run: python core/ml/download_bge.py ml_artifacts/bge_small_manifest.json ({exc})"
            ) from exc
        actual = m.get_embedding_dimension() if hasattr(m, "get_embedding_dimension") else m.get_sentence_embedding_dimension()
        if actual != profile.dim:
            raise EmbeddingConfigError(f"{profile.model_id} dim {actual} != expected {profile.dim}")
        _models[cache_key] = m
        return m


INDEX_SCHEMA_VERSION = "2.0"
PASSAGE_FORMAT_VERSION = "heading_body_v1"


def compute_checkpoint_fingerprint(checkpoint_path: str, *, short: bool = True) -> str:
    """Compute a content-based SHA-256 fingerprint of an entire checkpoint directory.

    Hashes every regular file's relative path and bytes in sorted order, producing
    a deterministic digest that changes when any model weight, config, or tokenizer
    file changes. If the path does not exist, returns the empty string.
    """
    p = Path(checkpoint_path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    if p.is_file():
        h.update(p.name.encode())
        h.update(p.read_bytes())
    else:
        for f in sorted(p.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(p)).encode())
                h.update(f.read_bytes())
    digest = h.hexdigest()
    return digest[:16] if short else digest


@lru_cache(maxsize=16)
def _fingerprint_for_file_signature(checkpoint_path: str, file_signature: tuple) -> str:
    return compute_checkpoint_fingerprint(checkpoint_path)


def cached_checkpoint_fingerprint(checkpoint_path: str) -> str:
    """Rehash a checkpoint when any file's size or modification time changes."""
    root = Path(checkpoint_path)
    if not root.exists():
        return ""
    files = [root] if root.is_file() else sorted(f for f in root.rglob("*") if f.is_file())
    signature = tuple(
        (str(f.relative_to(root)) if root.is_dir() else f.name, f.stat().st_size, f.stat().st_mtime_ns)
        for f in files
    )
    return _fingerprint_for_file_signature(str(root.resolve()), signature)


@dataclass(frozen=True)
class IndexManifest:
    model_id: str
    revision: str
    dim: int
    normalization: str
    query_prefix: str
    passage_format_version: str
    index_schema_version: str
    checkpoint_fingerprint: str = ""

    def to_metadata(self) -> dict:
        return {
            "hnsw:space": self.normalization,
            "servy_model": self.model_id,
            "servy_revision": self.revision or "",
            "servy_dim": int(self.dim),
            "servy_query_prefix": self.query_prefix or "",
            "servy_passage_format": self.passage_format_version,
            "servy_schema_version": self.index_schema_version,
            "servy_checkpoint_fp": self.checkpoint_fingerprint or "",
        }


def get_collection(profile: EmbeddingProfile, create: bool = True):
    """Return the profile's collection; refuse to reuse a collection stamped by another model
    or having an incompatible manifest (revision, dimension, query_prefix, passage_format)."""
    from core.services.chroma_store import get_chroma_client
    client = get_chroma_client()

    # Compute real content-based fingerprint for local checkpoints
    if profile.local_path:
        cp_fp = cached_checkpoint_fingerprint(profile.local_path)
        if not cp_fp:
            raise EmbeddingConfigError(f"Local checkpoint is missing: {profile.local_path}")
    else:
        cp_fp = profile.revision or ""

    manifest = IndexManifest(
        model_id=profile.model_id,
        revision=profile.revision or "",
        dim=profile.dim,
        normalization="cosine",
        query_prefix=profile.query_prefix or "",
        passage_format_version=PASSAGE_FORMAT_VERSION if profile.content_first else "legacy_metadata_v0",
        index_schema_version=INDEX_SCHEMA_VERSION,
        checkpoint_fingerprint=cp_fp,
    )
    meta = manifest.to_metadata()
    if create:
        coll = client.get_or_create_collection(name=profile.collection, metadata=meta)
    else:
        coll = client.get_collection(profile.collection)

    stored_meta = coll.metadata or {}
    stamped_model = stored_meta.get("servy_model")

    # Legacy servy_dense_kb and company-minilm have no stamp; only ever paired with minilm.
    if profile.key in ("minilm", "company-minilm"):
        if stamped_model and stamped_model != profile.model_id:
            raise EmbeddingConfigError(
                f"Collection '{profile.collection}' belongs to model '{stamped_model}', not '{profile.model_id}'."
            )
        return coll

    if not stamped_model:
        raise EmbeddingConfigError(f"Collection '{profile.collection}' is unstamped; refusing to use it.")

    if stamped_model != profile.model_id:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' model mismatch: stamped '{stamped_model}', expected '{profile.model_id}'."
        )

    if profile.local_path and stored_meta.get("servy_checkpoint_fp") != cp_fp:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' checkpoint fingerprint mismatch: "
            f"stamped '{stored_meta.get('servy_checkpoint_fp')}', expected '{cp_fp}'."
        )

    # Validate Dimension
    stamped_dim = stored_meta.get("servy_dim")
    if stamped_dim and int(stamped_dim) != profile.dim:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' dim mismatch: stamped {stamped_dim}, expected {profile.dim}."
        )

    # Validate Revision if present
    stamped_rev = stored_meta.get("servy_revision")
    if stamped_rev and profile.revision and stamped_rev != profile.revision:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' revision mismatch: stamped '{stamped_rev}', expected '{profile.revision}'."
        )

    # Validate Query Prefix
    stamped_prefix = stored_meta.get("servy_query_prefix")
    if stamped_prefix is not None and stamped_prefix != (profile.query_prefix or ""):
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' query prefix mismatch: stamped '{stamped_prefix}', expected '{profile.query_prefix}'."
        )

    # Validate Passage Format
    stamped_format = stored_meta.get("servy_passage_format")
    expected_format = PASSAGE_FORMAT_VERSION if profile.content_first else "legacy_metadata_v0"
    if stamped_format and stamped_format != expected_format:
        raise EmbeddingConfigError(
            f"Collection '{profile.collection}' passage format mismatch: stamped '{stamped_format}', expected '{expected_format}'."
        )

    return coll


def encode_queries(profile: EmbeddingProfile, model, queries):
    return model.encode([profile.query_prefix + q for q in queries],
                        normalize_embeddings=True, batch_size=getattr(settings, "SERVY_EMBEDDING_BATCH_SIZE", 8))


def encode_passages(profile: EmbeddingProfile, model, passages):
    return model.encode(passages,
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
