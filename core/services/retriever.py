"""
Servy RAG Retriever — Hardened Implementation

Implements true hybrid retrieval (dense + sparse concurrently), Reciprocal Rank
Fusion (RRF), authorization-before-ranking for dense search, content relevance
gating (metadata bonus cannot rescue irrelevant content), source authority
tiering, and exact technical identifier matching.
"""

from dataclasses import dataclass
from typing import List

from django.db.models import Q
from django.conf import settings

from core.models import KnowledgeChunk, KnowledgeDocument, ServiceResolutionIndex
from .vectors import embed_text, cosine_sparse, compute_bm25_scores, compute_char_ngram_scores
from .identifiers import extract_error_codes, extract_identifiers, extract_document_ids


@dataclass
class RetrievedChunk:
    source_kind: str
    source_id: int
    document_id: int | None
    chunk_id: int | None
    title: str
    heading: str
    text: str
    score: float
    doc_type: str
    source_url: str
    confidential: bool
    service_call_id: int | None = None
    model_version: str = "minilm"
    dense_score: float | None = None
    dense_similarity: float | None = None
    dense_rank: int | None = None
    sparse_score: float | None = None
    sparse_relevance: float | None = None
    sparse_rank: int | None = None
    exact_identifier_match: bool = False
    applicable_equipment_context: bool = False
    fusion_score: float | None = None
    fusion_rank: int | None = None
    reranker_score: float | None = None
    final_rank: int | None = None
    evidence_support: bool = False
    source_verified: bool = False
    equipment_applicable: bool = False
    customer_action_authorized: bool = False
    source_eligible: bool = False

    @property
    def reference(self):
        section = f" > {self.heading}" if self.heading else ""
        return f"{self.title}{section}"


class RetrievedResultList(list):
    def __init__(self, items=(), retrieval_backend="semantic", reranker_applied=False, reranker_model=None):
        super().__init__(items)
        self.retrieval_backend = retrieval_backend
        self.reranker_applied = reranker_applied
        self.reranker_model = reranker_model if reranker_applied else None


def is_authorized_candidate(
    chunk,
    *,
    tenant,
    customer=None,
    asset=None,
    include_confidential=False,
    staff_mode=False,
    allow_failed=False,
    backend="all",
    verified_doc_ids=None,
) -> bool:
    """Canonical server-side authorization predicate for retrieved candidates.

    Every candidate MUST satisfy all mandatory constraints:
    1. Authorized tenant
    2. Authorized customer
    3. Authorized asset / product hierarchy
    4. Document visibility (is_rag_enabled=True, backend-specific index_status valid)
    5. Confidentiality (if not include_confidential, is_confidential=False)
    6. Chunk quarantine state (is_quarantined=False)
    7. Revocation status (revoked_at is None)
    8. Positive provenance in verified_real_only mode
    """
    if not chunk:
        return False
    doc = getattr(chunk, "document", None)
    if not doc:
        return False

    # 1. Tenant boundary
    if chunk.tenant_id != tenant.id or doc.tenant_id != tenant.id:
        return False

    # 2. Chunk quarantine state
    if getattr(chunk, "is_quarantined", False):
        return False

    # 3. Document visibility & index state
    if not doc.is_rag_enabled:
        return False

    if backend == "dense":
        # Dense index requires complete synchronization (INDEXED)
        if doc.index_status != "INDEXED":
            return False
    elif backend == "sparse" or allow_failed:
        # Sparse index may be valid even if dense synchronization failed
        if doc.index_status not in ("INDEXED", "INDEXING", "FAILED"):
            return False
        # Sparse retriever requires text or sparse embedding to exist
        if not (chunk.text or chunk.content_embedding or chunk.embedding):
            return False
    else:
        if doc.index_status not in ("INDEXED", "INDEXING"):
            return False

    if getattr(doc, "revoked_at", None) is not None:
        return False

    # 4. Confidentiality
    if not include_confidential and doc.is_confidential:
        return False

    # 5. Customer scoping
    if customer:
        if doc.customer_id is not None and doc.customer_id != customer.id:
            return False
    else:
        if not staff_mode and doc.customer_id is not None:
            if not asset or doc.customer_id != getattr(asset, "customer_id", None):
                return False

    # 6. Asset & Equipment scoping (canonical 5-level hierarchy)
    from .knowledge_scope import is_chunk_applicable_to_asset
    if not is_chunk_applicable_to_asset(chunk, asset):
        return False

    # 7. Real company provenance verification
    from .customer_isolation import get_customer_data_mode, is_real_tenant, is_verified_real_document
    mode = get_customer_data_mode()
    if not staff_mode and is_real_tenant(tenant) and mode == "verified_real_only":
        if verified_doc_ids is not None:
            if doc.id not in verified_doc_ids:
                return False
        elif not is_verified_real_document(doc, allow_failed=allow_failed):
            return False

    return True


def _knowledge_candidates(tenant, asset, customer, include_confidential, allow_failed=False, query_doc_ids=None, staff_mode=False, backend="all"):
    from .customer_isolation import get_customer_data_mode, is_real_tenant
    mode = get_customer_data_mode()
    is_real = is_real_tenant(tenant)
    is_real_customer_query = (customer is not None) and (mode == "verified_real_only")

    # In verified_real_only mode, customer queries on non-real tenants return nothing
    if not staff_mode and is_real_customer_query and not is_real:
        return KnowledgeChunk.objects.none()

    qs = KnowledgeChunk.objects.filter(
        tenant=tenant,
        document__is_rag_enabled=True,
        is_quarantined=False,
    ).select_related(
        "document", "document__product", "document__asset",
        "document__category", "document__domain", "document__customer",
    )

    verified_doc_ids = None
    if not staff_mode and is_real:
        from .customer_isolation import is_verified_real_document
        qs = qs.exclude(document__title__icontains="synthetic")\
               .exclude(document__tags__icontains="synthetic")\
               .exclude(document__tags__icontains="dummy")\
               .exclude(document__description__icontains="synthetic")\
               .exclude(document__description__icontains="dummy")
        # Positive provenance enforcement: only allow documents satisfying is_verified_real_document
        doc_candidates = KnowledgeDocument.objects.filter(tenant=tenant)
        verified_doc_ids = {d.id for d in doc_candidates if is_verified_real_document(d, allow_failed=allow_failed)}
        qs = qs.filter(document_id__in=verified_doc_ids)

    if not include_confidential:
        qs = qs.filter(document__is_confidential=False)

    if customer:
        qs = qs.filter(Q(document__customer__isnull=True) | Q(document__customer=customer))
    elif not staff_mode and asset and asset.customer_id:
        qs = qs.filter(Q(document__customer__isnull=True) | Q(document__customer_id=asset.customer_id))
    elif not staff_mode:
        qs = qs.filter(document__customer__isnull=True)

    if asset:
        product = getattr(asset, "product", None)
        category = getattr(product, "category", None) if product else None
        domain = getattr(category, "domain", None) if category else None

        scope = Q(document__asset=asset)
        if product:
            scope |= Q(document__asset__isnull=True, document__product=product)
        if category:
            scope |= Q(document__asset__isnull=True, document__product__isnull=True, document__category=category)
        if domain:
            scope |= Q(
                document__asset__isnull=True,
                document__product__isnull=True,
            ) & (Q(document__domain=domain) | Q(document__category__domain=domain))
        scope |= Q(
            document__asset__isnull=True,
            document__product__isnull=True,
            document__category__isnull=True,
            document__domain__isnull=True,
        )
        qs = qs.filter(scope)
    else:
        # NO asset context: restrict strictly to unscoped/general fleet documents
        scope = Q(
            document__asset__isnull=True,
            document__product__isnull=True,
            document__category__isnull=True,
            document__domain__isnull=True,
        )
        qs = qs.filter(scope)

    # Pre-filter candidate list to those passing canonical authorization predicate
    # Eliminate the [:3000] slice limitation (Part 7)
    candidates = list(qs)
    return [
        c for c in candidates
        if is_authorized_candidate(
            c,
            tenant=tenant,
            customer=customer,
            asset=asset,
            include_confidential=include_confidential,
            staff_mode=staff_mode,
            allow_failed=allow_failed,
            backend=backend,
            verified_doc_ids=verified_doc_ids,
        )
    ]



def _resolution_candidates(tenant, asset, customer, staff_mode):
    from .customer_isolation import get_customer_data_mode, is_real_tenant, is_verified_real_service_call
    mode = get_customer_data_mode()
    is_real = is_real_tenant(tenant)
    is_real_customer_query = (customer is not None) and (mode == "verified_real_only")
    if not staff_mode and is_real_customer_query and not is_real:
        return []

    qs = ServiceResolutionIndex.objects.filter(tenant=tenant).select_related(
        "service_call", "service_call__asset", "service_call__asset__product", "service_call__customer"
    )

    if not staff_mode and is_real:
        # Defense in depth: exclude obvious synthetic identifiers at DB level
        qs = qs.exclude(
            service_call__servy_id__startswith="SRV-SYN-"
        ).exclude(
            service_call__servy_id__startswith="DUMMY-"
        ).exclude(
            service_call__servy_id__startswith="SYN-"
        )

    if asset:
        if asset.product_id:
            qs = qs.filter(Q(service_call__asset=asset) | Q(service_call__asset__product=asset.product))
        else:
            qs = qs.filter(service_call__asset=asset)
    if customer and not staff_mode:
        qs = qs.filter(service_call__customer=customer)

    candidates = list(qs[:1500])
    if not staff_mode and is_real and mode == "verified_real_only":
        # Positive provenance verification: exclude any service call that fails positive verification
        candidates = [c for c in candidates if is_verified_real_service_call(c.service_call)]

    return candidates


import logging
logger = logging.getLogger("servy.rag.retriever")


# ---------------------------------------------------------------------------
# Configurable constants — can be overridden in settings.py
# ---------------------------------------------------------------------------

EXACT_ASSET_BONUS = 0.35
EXACT_PRODUCT_BONUS = 0.25
MODEL_BONUS = 0.15
CATEGORY_BONUS = 0.10
DOMAIN_BONUS = 0.05
CUSTOMER_BONUS = 0.10

# Phase 4 (Operation 16.0): Immutable serving configuration.
# Content relevance gates are read from settings at call time, NOT cached in
# mutable module-level globals. This prevents benchmark-induced shared global-state
# mutation from affecting parallel requests or production behavior.
from dataclasses import dataclass as _dc


@_dc(frozen=True)
class ServingConfig:
    """Immutable snapshot of all serving-time configuration.
    
    Created once per retrieval call from Django settings.
    Never mutated. Benchmark code must NOT modify module-level globals.
    """
    dense_relevance_gate: float
    sparse_relevance_gate: float
    embedding_profile: str
    rrf_k: int = 60
    rrf_weight_dense: float = 0.55
    rrf_weight_sparse: float = 0.45
    top_k: int = 6
    reranker_enabled: bool = False


def get_serving_config() -> ServingConfig:
    """Construct an immutable ServingConfig from current Django settings.
    
    This is called once per retrieval request. The returned object is frozen
    and cannot be mutated, ensuring parallel requests never cross-contaminate.
    """
    return ServingConfig(
        dense_relevance_gate=float(getattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.55)),
        sparse_relevance_gate=float(getattr(settings, "SERVY_SPARSE_RELEVANCE_GATE", 0.055)),
        embedding_profile=str(getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "auto")),
        top_k=int(getattr(settings, "SERVY_RAG_TOP_K", 6)),
        reranker_enabled=bool(getattr(settings, "SERVY_RERANKER_ENABLED", False)),
    )


# Legacy module-level accessors — read from settings each time for backward compat
# DO NOT assign to these; use get_serving_config() in new code.
def _get_sparse_gate():
    return float(getattr(settings, "SERVY_SPARSE_RELEVANCE_GATE", 0.055))


def _get_dense_gate():
    return float(getattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.55))


# Legacy names kept for backward compatibility with existing code paths
SPARSE_RELEVANCE_GATE = _get_sparse_gate()
DENSE_RELEVANCE_GATE = _get_dense_gate()

DOMAIN_TYPO_MAP = {
    "indor": "indoor",
    "leeking": "leaking",
    "watter": "water",
    "milkk": "milk",
    "temprature": "temperature",
    "riseing": "rising",
    "abov": "above",
    "voltcure": "voltcore",
    "invrter": "inverter",
    "falt": "fault",
    "pllet": "pallet",
    "hydralic": "hydraulic",
    "cavtation": "cavitation",
    "ductible": "ductable",
    "wek": "weak",
    "airflo": "airflow",
    "whistlng": "whistling",
    "noys": "noise",
    "outdor": "outdoor",
    "comuncation": "communication",
    "errer": "error",
    "contrler": "controller",
    "splitt": "split",
    "icce": "ice",
    "coolng": "cooling",
    "pip": "pipe",
    "viberation": "vibration",
    "deflecton": "deflection",
    "compresser": "compressor",
    "coyl": "coil",
}


def normalize_domain_terms(text: str) -> str:
    """Bounded domain-aware typo normalizer for customer equipment complaints.
    Preserves alphanumeric technical identifiers and fault codes exactly.
    """
    if not text:
        return ""
    words = text.split()
    corrected = []
    for w in words:
        clean = w.strip(".,!?:;\"'()[]{}")
        clean_lower = clean.lower()
        if clean_lower in DOMAIN_TYPO_MAP:
            replacement = DOMAIN_TYPO_MAP[clean_lower]
            corrected.append(w.replace(clean, replacement))
        else:
            corrected.append(w)
    return " ".join(corrected)

# RRF constant (standard value)
RRF_K = 60


def _format_chunk_text(chunk) -> str:
    """Ensure section heading is present at the top of chunk text for full context."""
    if chunk.heading and chunk.heading.strip() and chunk.heading.strip().lower() not in (chunk.text or "").lower():
        return f"## {chunk.heading.strip()}\n{chunk.text.strip()}"
    return chunk.text or ""




# ---------------------------------------------------------------------------
# Metadata bonus calculation
# ---------------------------------------------------------------------------

def _calculate_metadata_bonus(document, chunk, asset, customer) -> float:
    bonus = 0.0
    if asset:
        if document.asset_id == asset.id:
            bonus += EXACT_ASSET_BONUS
        elif document.product_id and asset.product_id == document.product_id:
            bonus += EXACT_PRODUCT_BONUS
        elif document.category_id and asset.product and asset.product.category_id == document.category_id:
            bonus += CATEGORY_BONUS
        elif document.domain_id and asset.product and asset.product.category and asset.product.category.domain_id == document.domain_id:
            bonus += DOMAIN_BONUS

        if asset.model_number:
            model_lower = asset.model_number.lower().strip()
            if model_lower and (
                model_lower in document.title.lower()
                or model_lower in (document.tags or "").lower()
                or (chunk and model_lower in (chunk.heading or "").lower())
                or (chunk and model_lower in (chunk.text or "").lower())
            ):
                bonus += MODEL_BONUS

    if customer and document.customer_id == customer.id:
        bonus += CUSTOMER_BONUS
    return bonus


# ---------------------------------------------------------------------------
# Exact identifier matching boost
# ---------------------------------------------------------------------------

def _identifier_boost(query_identifiers: set, chunk_text: str, chunk_heading: str, doc_title: str = "") -> float:
    """Give a deterministic boost when the chunk contains an exact identifier
    that the query explicitly asks about, parsed through the same canonical extractor."""
    if not query_identifiers:
        return 0.0
    chunk_identifiers = extract_identifiers(f"{chunk_heading or ''} {chunk_text or ''}")
    if query_identifiers & chunk_identifiers:
        return 0.40  # Strong boost for exact chunk identifier match
    if doc_title:
        doc_identifiers = extract_identifiers(doc_title)
        if query_identifiers & doc_identifiers:
            return 0.25  # Document-level identifier match
    return 0.0


# ---------------------------------------------------------------------------
# Sparse retrieval scoring
# ---------------------------------------------------------------------------

def _sparse_score_candidates(candidates, evidence_vector, scope_vector,
                             query_identifiers, asset, customer, min_score,
                             query_doc_ids=None, evidence_text=None,
                             serving_config=None):
    """Score all candidates using hybrid lexical scoring (BM25 + character n-grams + sparse hashing).

    Returns list of RetrievedChunk tuples sorted by score descending with explicit sparse_rank.
    """
    scored = []
    chunk_texts = [_format_chunk_text(c) for c in candidates]
    bm25_scores = compute_bm25_scores(evidence_text, chunk_texts) if evidence_text else [0.0] * len(candidates)
    char_scores = compute_char_ngram_scores(evidence_text, chunk_texts) if evidence_text else [0.0] * len(candidates)

    for i, chunk in enumerate(candidates):
        d = chunk.document
        hashing_content = cosine_sparse(evidence_vector, chunk.content_embedding or chunk.embedding)
        scope_score = cosine_sparse(scope_vector, chunk.embedding)

        # Lexical score blends BM25 (exact keyword match) and char n-grams (misspelling/morphology robustness)
        bm25_val = bm25_scores[i] if i < len(bm25_scores) else 0.0
        char_val = char_scores[i] if i < len(char_scores) else 0.0

        # Background char-ngram threshold: character overlap below 0.22 is random English noise
        effective_char = char_val if char_val >= 0.22 else 0.0
        if bm25_val > 0.0:
            # Exact words matched: BM25 with term coverage governs, with bounded char assistance
            lexical_content = max(bm25_val, min(bm25_val * 1.5, effective_char * 0.5))
        else:
            # Zero exact words matched: pure misspelling/typo recovery mode
            lexical_content = effective_char
        content_score = max(hashing_content, lexical_content)

        # Content relevance gate: candidate must have minimum content match
        # BEFORE metadata bonus can apply (Defect #3 fix)
        base_score = (0.72 * content_score) + (0.28 * scope_score)
        id_boost = _identifier_boost(query_identifiers, chunk.text, chunk.heading, getattr(d, "title", ""))
        if query_doc_ids and d.id in query_doc_ids:
            id_boost = max(id_boost, 0.60)

        # Content relevance gate (Defect #3 fix): candidate must pass minimum content match
        # before metadata bonus can apply. Exact identifier matches are always relevant.
        # Use per-request config gate (Op17 fix) — NOT the stale module-level constant.
        _sparse_gate = serving_config.sparse_relevance_gate if serving_config else _get_sparse_gate()
        if content_score < _sparse_gate and id_boost <= 0.0:
            continue

        # Prevent single-word distractors from receiving metadata bonus alone
        if content_score < 0.15 and id_boost <= 0.0:
            boost = 0.0
        else:
            boost = _calculate_metadata_bonus(d, chunk, asset, customer)
        final_score = float(base_score + boost + id_boost)

        if final_score < min_score:
            continue

        from .knowledge_scope import is_chunk_applicable_to_asset
        is_equip_applicable = is_chunk_applicable_to_asset(chunk, asset)

        from .customer_isolation import is_real_tenant, is_verified_real_document
        is_verified = (
            is_verified_real_document(d, allow_failed=True)
            if is_real_tenant(d.tenant)
            else bool(d.is_rag_enabled and not d.is_confidential and d.checksum_sha256)
        )
        is_eligible = bool(
            d.is_rag_enabled
            and not d.is_confidential
            and not getattr(chunk, "is_quarantined", False)
        )
        # customer_action_authorized must NEVER be granted merely from document eligibility.
        # It defaults to False and can only be authorized by ProcedureApproval through safety authority.
        action_authorized = False

        scored.append(RetrievedChunk(
            source_kind="knowledge_document",
            source_id=chunk.id,
            document_id=d.id,
            chunk_id=chunk.id,
            title=d.title,
            heading=chunk.heading,
            text=_format_chunk_text(chunk),
            score=final_score,
            doc_type=d.get_doc_type_display(),
            source_url=d.source_url,
            confidential=d.is_confidential,
            sparse_score=final_score,
            sparse_relevance=final_score,
            exact_identifier_match=(id_boost > 0.0),
            applicable_equipment_context=(boost > 0.0),
            equipment_applicable=is_equip_applicable,
            customer_action_authorized=action_authorized,
            source_verified=is_verified,
            source_eligible=is_eligible,
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    for i, r in enumerate(scored, 1):
        r.sparse_rank = i
    return scored


def _matches_current_bge_vector(meta, chunk, profile) -> bool:
    """Match an indexed vector to the current authoritative BGE chunk."""
    import hashlib
    from core.ml.embeddings import passage_text, cached_checkpoint_fingerprint

    content_hash = hashlib.sha256(passage_text(profile, chunk).encode("utf-8")).hexdigest()
    # The collection header alone cannot establish which weights created a
    # specific vector. Qwen3 rows must also match the selected local checkpoint.
    qwen_fp = (
        cached_checkpoint_fingerprint(profile.local_path)
        if getattr(profile, "query_document_encoder", False) and profile.local_path
        else None
    )
    return bool(
        (qwen_fp is None or meta.get("model_checkpoint_fingerprint") == qwen_fp)
        and meta.get("chunk_id") == chunk.id
        and meta.get("document_id") == chunk.document_id
        and meta.get("chunk_index") == chunk.chunk_index
        and meta.get("tenant_id") == chunk.tenant_id
        and meta.get("document_version") == chunk.document.index_version
        and meta.get("content_sha256") == content_hash
        and meta.get("embedding_model") == profile.model_id
        and meta.get("embedding_revision") == (profile.revision or "")
    )


def _matches_current_minilm_vector(meta, chunk, profile) -> bool:
    """Reject stale legacy vectors before pairing them with current chunk text."""
    import hashlib

    return bool(
        meta.get("chunk_id") == chunk.id
        and meta.get("document_id") == chunk.document_id
        and meta.get("chunk_index") == chunk.chunk_index
        and meta.get("tenant_id") == chunk.tenant_id
        and str(meta.get("document_version")) == str(chunk.document.index_version or "1.0")
        and meta.get("content_sha256") == hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()
        and meta.get("embedding_model") == profile.model_id
    )


def _qwen3_profile_if_ready(settings_obj, is_real: bool) -> str:
    """Default-safe selector: three independent flags before Qwen3 serving.

    Returning the MiniLM profile is intentional when the experimental index
    has not been independently checked. This selector does not grant access to
    any document; authorization remains enforced in the retrieval query.
    """
    enabled = (
        getattr(settings_obj, "SERVY_USE_EXPERIMENTAL_RAG", False) is True
        and getattr(settings_obj, "SERVY_QWEN3_ENABLED", False) is True
        and getattr(settings_obj, "SERVY_QWEN3_INDEX_VERIFIED", False) is True
    )
    if enabled:
        return "company-qwen3" if is_real else "qwen3"
    return "company-minilm" if is_real else "minilm"


# ---------------------------------------------------------------------------
# Dense retrieval scoring
# ---------------------------------------------------------------------------

def _dense_score_candidates(candidates, allowed_doc_ids, allowed_chunks_by_id,
                            evidence_text, query_identifiers, asset,
                            customer, tenant, top_k, min_score,
                            query_doc_ids=None, staff_mode=False,
                            include_confidential=False,
                            serving_config=None):
    """Score candidates using ChromaDB dense embeddings.

    Authorization is enforced BEFORE ranking by passing allowed_doc_ids into
    the Chroma where-filter (Defect #2 fix).  Query embedding encodes the
    actual user question/complaint to prevent metadata-only documents from
    rescuing irrelevant content (Defect #3 fix).
    """
    try:
        from django.conf import settings
        from .customer_isolation import get_customer_data_mode, is_real_tenant
        mode = get_customer_data_mode()
        use_experimental = getattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
        is_real = is_real_tenant(tenant)
        is_real_customer_query = (customer is not None) and (mode == "verified_real_only")
        if not staff_mode and is_real_customer_query and not is_real:
            return []

        # Explicit serving model profile selection (Operation 13.0)
        serving_profile = getattr(settings, "SERVY_SERVING_EMBEDDING_PROFILE", "auto")
        if serving_profile in ("qwen3", "company-qwen3"):
            profile_key = _qwen3_profile_if_ready(settings, is_real)
        elif serving_profile in ("company-bge-ft", "bge-small-ft"):
            profile_key = "company-bge-ft"
        elif serving_profile in ("company-bge-small", "bge-small"):
            profile_key = "company-bge-small" if is_real else "bge-small"
        elif serving_profile in ("company-minilm", "minilm"):
            profile_key = "company-minilm" if is_real else "minilm"
        elif is_real:
            profile_key = "company-bge-small" if use_experimental else "company-minilm"
        else:
            profile_key = (
                getattr(settings, "SERVY_EXPERIMENTAL_EMBEDDING_MODEL", "bge-small")
                if use_experimental
                else getattr(settings, "SERVY_EMBEDDING_MODEL", "minilm")
            )
        from core.ml.embeddings import get_profile, load_model, get_collection, encode_queries

        profile = get_profile(profile_key)
        # Phase 7 Fix: Enforce model promotion gating at serving time
        if not getattr(profile, "promotion_eligible", False) and not staff_mode:
            logger.warning(
                "Embedding profile '%s' is not promotion eligible (training_status='%s'). "
                "Rejecting unpromoted model and falling back to authorized baseline profile.",
                profile.key, getattr(profile, "training_status", "unknown"),
            )
            profile_key = "company-minilm" if is_real else "minilm"
            profile = get_profile(profile_key)

        model = load_model(profile)
        if model is None or not allowed_doc_ids:
            return []

        dense_coll = get_collection(profile, create=False)
        query_embedding = encode_queries(profile, model, [evidence_text])[0].tolist()

        # Phase 3: Authorization-before-ranking — pass allowed_doc_ids into
        # Chroma filter so unauthorized documents cannot crowd out authorized
        # chunks in the top-N retrieved.
        doc_id_list = [int(did) for did in allowed_doc_ids]

        # Scalable candidate retrieval (Phase 1 Fix):
        # Choose Chroma result depth based on authorized chunk count and bounded retrieval budget.
        # NEVER cap vector hits by the number of document IDs (prevents multi-chunk recall suppression).
        batch_size = 400
        doc_id_batches = [doc_id_list[i : i + batch_size] for i in range(0, len(doc_id_list), batch_size)]
        fusion_budget = min(max(top_k * 6, 40), 200)

        # Map (doc_id, chunk_index) -> KnowledgeChunk for O(1) post-query lookup and verification
        candidates_by_doc_chunk = {(c.document_id, c.chunk_index): c for c in candidates}
        # Deduplicate results across overlapping batches: key -> (metadata, distance)
        best_matches = {}

        for batch_ids in doc_id_batches:
            batch_chunk_count = sum(1 for (did, _) in candidates_by_doc_chunk.keys() if did in batch_ids)
            if batch_chunk_count == 0:
                continue
            n_results = min(fusion_budget, max(batch_chunk_count, 1))

            chroma_where = {
                "$and": [
                    {"tenant_id": int(tenant.id)},
                    {"document_id": {"$in": batch_ids}},
                ]
            }
            res = dense_coll.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=chroma_where,
            )
            if res and res.get("metadatas") and res["metadatas"][0]:
                for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
                    doc_id = meta.get("document_id")
                    chunk_idx = meta.get("chunk_index", 0)
                    if doc_id not in allowed_doc_ids:
                        continue
                    key = (doc_id, chunk_idx)
                    if key not in candidates_by_doc_chunk:
                        # Vector in Chroma is stale or unassociated with authorized candidate
                        continue
                    current_chunk = candidates_by_doc_chunk[key]
                    # A vector from an earlier chunk at the same document/index
                    # position must never be paired with today's source text.
                    identity_matches = (
                        _matches_current_bge_vector(meta, current_chunk, profile)
                        if profile.content_first
                        else _matches_current_minilm_vector(meta, current_chunk, profile)
                    )
                    if not identity_matches:
                        continue
                    if key not in best_matches or dist < best_matches[key][1]:
                        best_matches[key] = (meta, dist)

        scored = []
        if best_matches:
            for (doc_id, chunk_idx), (meta, dist) in best_matches.items():
                matched_chunk = candidates_by_doc_chunk.get((doc_id, chunk_idx))
                if not matched_chunk or matched_chunk.is_quarantined:
                    continue
                if not is_authorized_candidate(
                    matched_chunk,
                    tenant=tenant,
                    customer=customer,
                    asset=asset,
                    include_confidential=include_confidential,
                    staff_mode=staff_mode,
                    backend="dense",
                ):
                    continue

                d = matched_chunk.document
                sim = max(0.0, 1.0 - float(dist))
                id_boost = _identifier_boost(query_identifiers, matched_chunk.text, matched_chunk.heading, getattr(d, "title", ""))
                if query_doc_ids and d.id in query_doc_ids:
                    id_boost = max(id_boost, 0.60)

                # Content relevance gate (Defect #3 fix): if semantic similarity is too low
                # and no identifier matches, metadata bonus cannot rescue the document.
                # Use per-request config gate (Op17 fix) — NOT the stale module-level constant.
                _dense_gate = serving_config.dense_relevance_gate if serving_config else _get_dense_gate()
                if getattr(profile, "query_document_encoder", False):
                    # Qwen3's similarity scale differs from BGE/MiniLM. This
                    # gate is a tunable hypothesis, not a calibrated threshold.
                    # Relevance/answerability validation remains downstream.
                    _dense_gate = float(getattr(settings, "SERVY_QWEN3_DENSE_GATE", 0.35))
                    if not 0.0 <= _dense_gate <= 1.0:
                        raise ValueError("SERVY_QWEN3_DENSE_GATE must be in [0,1]")
                if not getattr(profile, "content_first", False):
                    # MiniLM vector similarities have a lower ceiling (~0.30-0.50) compared to BGE (~0.60-0.85).
                    # Avoid pruning genuine semantic matches when using non-content-first profiles.
                    _dense_gate = min(_dense_gate, 0.25)
                if sim < _dense_gate and id_boost <= 0.0:
                    continue

                if getattr(profile, "content_first", False):
                    # In content-first mode, rely primarily on semantic content match.
                    # Bounded metadata bonus is strictly capped.
                    boost = min(0.05, _calculate_metadata_bonus(d, matched_chunk, asset, customer) * 0.25)
                elif sim < _dense_gate and id_boost <= 0.0:
                    boost = 0.0
                else:
                    boost = _calculate_metadata_bonus(d, matched_chunk, asset, customer)
                final_score = float(sim + boost + id_boost)

                if final_score < min_score:
                    continue

                from .knowledge_scope import is_chunk_applicable_to_asset
                is_equip_applicable = is_chunk_applicable_to_asset(matched_chunk, asset)

                from .customer_isolation import is_real_tenant, is_verified_real_document
                is_verified = (
                    is_verified_real_document(d, allow_failed=False)
                    if is_real
                    else bool(d.is_rag_enabled and not d.is_confidential and d.checksum_sha256)
                )
                is_eligible = bool(
                    d.is_rag_enabled
                    and not d.is_confidential
                    and not getattr(matched_chunk, "is_quarantined", False)
                )
                # Phase 4 Fix: customer_action_authorized must NEVER be granted merely from document eligibility.
                # Only procedure-specific safety authority may authorize customer actions.
                action_authorized = False

                scored.append(RetrievedChunk(
                    source_kind="knowledge_document",
                    source_id=matched_chunk.id,
                    document_id=d.id,
                    chunk_id=matched_chunk.id,
                    title=d.title,
                    heading=matched_chunk.heading,
                    text=_format_chunk_text(matched_chunk),
                    score=final_score,
                    doc_type=d.get_doc_type_display(),
                    source_url=d.source_url,
                    confidential=d.is_confidential,
                    model_version=profile.key,
                    dense_score=final_score,
                    dense_similarity=sim,
                    exact_identifier_match=(id_boost > 0.0),
                    applicable_equipment_context=(boost > 0.0),
                    equipment_applicable=is_equip_applicable,
                    customer_action_authorized=action_authorized,
                    source_verified=is_verified,
                    source_eligible=is_eligible,
                ))

        scored.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(scored, 1):
            r.dense_rank = i
        return scored

    except Exception as exc:
        logger.warning(
            "Chroma semantic retrieval encountered exception: %s. "
            "Falling back to sparse retriever.", exc,
        )
        return []


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (Phase 2)
# ---------------------------------------------------------------------------

def _reciprocal_rank_fusion(dense_results, sparse_results, w_dense=0.55, w_sparse=0.45):
    """Fuse dense and sparse ranked lists using RRF.

    RRF(d) = sum(w_i / (K + rank_i(d))) for all lists i where d appears.
    Only lists in which the document appears may contribute to that document's fusion score.
    Missing-list entries must NOT receive phantom contributions.
    """
    dense_map = {}
    for i, r in enumerate(dense_results, 1):
        key = (r.source_kind, r.source_id)
        if key not in dense_map:
            r.dense_score = r.score
            # Preserve raw dense_similarity from vector search; do not overwrite with boosted score
            r.dense_rank = i
            dense_map[key] = r

    sparse_map = {}
    for i, r in enumerate(sparse_results, 1):
        key = (r.source_kind, r.source_id)
        if key not in sparse_map:
            r.sparse_score = r.score
            r.sparse_rank = i
            sparse_map[key] = r

    # Collect all unique chunks preserving discovery order
    all_keys = list(dict.fromkeys(list(dense_map.keys()) + list(sparse_map.keys())))
    fused = []

    for key in all_keys:
        d_item = dense_map.get(key)
        s_item = sparse_map.get(key)
        base_item = d_item if d_item is not None else s_item
        if d_item is not None and s_item is not None:
            base_item.sparse_score = s_item.sparse_score
            base_item.sparse_relevance = s_item.sparse_relevance
            base_item.sparse_rank = s_item.sparse_rank
            base_item.exact_identifier_match = d_item.exact_identifier_match or s_item.exact_identifier_match
            base_item.applicable_equipment_context = d_item.applicable_equipment_context or s_item.applicable_equipment_context
            base_item.dense_similarity = d_item.dense_similarity
            base_item.dense_score = d_item.dense_score

        rrf_score = 0.0
        if base_item.dense_rank is not None:
            rrf_score += w_dense / (RRF_K + base_item.dense_rank)
        if base_item.sparse_rank is not None:
            rrf_score += w_sparse / (RRF_K + base_item.sparse_rank)

        base_item.fusion_score = rrf_score
        # Preserve natural candidate relevance score on [0, 1+] scale so downstream evidence sufficiency
        # check does not reject fused candidates solely due to RRF rank scaling (Phase 1 Fix)
        cand_scores = [s for s in [base_item.dense_score, base_item.sparse_score] if s is not None]
        base_item.score = max(cand_scores) if cand_scores else rrf_score
        fused.append(base_item)

    # Sort strictly by fused rank value (descending fusion_score)
    fused.sort(key=lambda x: (x.fusion_score or 0.0), reverse=True)
    for i, item in enumerate(fused, 1):
        item.fusion_rank = i
        item.final_rank = i
    return fused


# ---------------------------------------------------------------------------
# Main retrieve function
# ---------------------------------------------------------------------------

def retrieve(
    tenant,
    question,
    asset=None,
    customer=None,
    include_confidential=False,
    top_k=6,
    min_score=0.04,
    service_call=None,
    include_past_resolutions=True,
    staff_mode=False,
) -> List[RetrievedChunk]:
    # Construct retrieval query dynamically using database records
    query_parts = []
    if asset:
        query_parts.append(f"Asset: {asset.name}")
        if asset.asset_code:
            query_parts.append(f"Asset code: {asset.asset_code}")
        if asset.model_number:
            query_parts.append(f"Model: {asset.model_number}")
        if asset.product:
            query_parts.append(f"Product: {asset.product.name}")
            if asset.product.category:
                query_parts.append(f"Category: {asset.product.category.name}")
    if service_call:
        if service_call.complaint_type:
            query_parts.append(f"Call type: {service_call.complaint_type}")
        if service_call.complaint_text:
            query_parts.append(f"Technician context: {service_call.complaint_text}")
    query_parts.append(f"Customer complaint: {question or ''}")
    dynamic_query_text = "\n".join(query_parts).strip()

    norm_question = normalize_domain_terms(question or "")
    evidence_text = f"{norm_question}".strip()

    # Engineer Copilot enrichment: when the technician asks a vague follow-up
    # question like "what could be the reason", the bare question has zero
    # content similarity to any knowledge chunk and every candidate fails the
    # content relevance gate.  Enrich the evidence text with the original
    # customer complaint from the service call so the embedding inherits the
    # technical symptom terms (e.g. "milk analyzer not working").
    if service_call and service_call.complaint_text:
        # Extract just the first line of the original customer complaint.
        # complaint_text may contain the full AI diagnostic dump appended
        # during escalation; we only want the short symptom description.
        raw_complaint = service_call.complaint_text.strip()
        # The escalation format prepends "Customer issue:\n<original question>"
        # followed by "\n\nAI diagnostic guidance..." — extract only the
        # customer issue portion.
        if "Customer issue:" in raw_complaint:
            after = raw_complaint.split("Customer issue:", 1)[1]
            # Take up to the first double-newline or "AI diagnostic" marker
            for sep in ["\n\nAI diagnostic", "\n\nDIAGNOSTIC", "\n\nSource documents"]:
                if sep in after:
                    after = after.split(sep, 1)[0]
            complaint_core = after.strip()
        else:
            # Fallback: use the first non-empty line of complaint_text
            complaint_core = raw_complaint.split("\n")[0].strip()

        # Only enrich if the complaint core adds meaningful terms not already
        # in the question (avoid duplicating when the question IS the complaint)
        if complaint_core and complaint_core.lower() not in evidence_text.lower():
            evidence_text = f"{complaint_core} — {evidence_text}"

    evidence_vector = embed_text(evidence_text)
    scope_vector = embed_text(dynamic_query_text)

    # Extract technical identifiers from the query for exact matching
    query_identifiers = extract_identifiers(question or "") | extract_identifiers(norm_question)
    query_error_codes = extract_error_codes(question or "") | extract_error_codes(norm_question)
    query_doc_ids = extract_document_ids(question or "")

    # Step 1: Django determines candidate pool according to tenant, customer,
    # confidentiality, and asset scope
    candidates = list(_knowledge_candidates(
        tenant, asset, customer, include_confidential,
        allow_failed=True, query_doc_ids=query_doc_ids,
        staff_mode=staff_mode,
    ))
    allowed_chunks_by_id = {c.id: c for c in candidates}
    allowed_doc_ids = {c.document_id for c in candidates}
    # For dense retrieval, only INDEXED documents have valid Chroma embeddings
    dense_candidates = [c for c in candidates if c.document.index_status == "INDEXED"]
    dense_chunks_by_id = {c.id: c for c in dense_candidates}
    dense_doc_ids = {c.document_id for c in dense_candidates}

    retrieval_backend = "fallback"

    # Construct immutable per-request serving config (Op17 Phase 2 fix).
    # This ensures each retrieval call uses its own snapshot of settings,
    # preventing cross-contamination between parallel requests or experiments.
    serving_config = get_serving_config()

    # Step 2: TRUE HYBRID RETRIEVAL — run dense AND sparse concurrently,
    # then fuse results using Reciprocal Rank Fusion (Phase 2).
    # This replaces the old "sparse only if dense returns nothing" approach.

    # 2a: Sparse/local retrieval (always runs)
    sparse_results = _sparse_score_candidates(
        candidates, evidence_vector, scope_vector,
        query_identifiers, asset, customer, min_score,
        query_doc_ids=query_doc_ids,
        evidence_text=evidence_text,
        serving_config=serving_config,
    )

    # 2b: Dense retrieval (may fail gracefully)
    try:
        dense_results = _dense_score_candidates(
            dense_candidates, dense_doc_ids, dense_chunks_by_id,
            evidence_text, query_identifiers, asset, customer,
            tenant, top_k, min_score,
            query_doc_ids=query_doc_ids,
            staff_mode=staff_mode,
            include_confidential=include_confidential,
            serving_config=serving_config,
        )
    except Exception as exc:
        logger.warning("Dense retrieval failed gracefully, falling back to sparse: %s", exc)
        dense_results = []

    # 2c: Fuse results using RRF
    qwen_dense_first = (
        bool(dense_results)
        and getattr(settings, "SERVY_QWEN3_DENSE_FIRST", False) is True
        and getattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False) is True
        and getattr(settings, "SERVY_QWEN3_ENABLED", False) is True
        and getattr(settings, "SERVY_QWEN3_INDEX_VERIFIED", False) is True
        and all(r.model_version in ("qwen3", "company-qwen3") for r in dense_results)
    )
    if qwen_dense_first:
        # Controlled accuracy experiment: Qwen3's raw dense ranking leads,
        # instead of allowing unrelated lexical hits to displace it via RRF.
        # Append only distinct sparse candidates after the dense hits, so
        # authorized fallback passages remain available for the reranker.
        dense_ids = {(r.source_kind, r.source_id) for r in dense_results}
        results = list(dense_results) + [
            r for r in sparse_results if (r.source_kind, r.source_id) not in dense_ids
        ]
        retrieval_backend = "qwen3_dense_first_experimental"
    elif dense_results and sparse_results:
        results = _reciprocal_rank_fusion(dense_results, sparse_results)
        retrieval_backend = "hybrid"
    elif dense_results:
        for i, r in enumerate(dense_results, 1):
            r.dense_score = r.score
            # Keep the original cosine similarity, not the metadata-boosted
            # display score. Evidence gates and diagnostics need both.
            if r.dense_similarity is None:
                r.dense_similarity = r.dense_score
            r.dense_rank = i
            r.fusion_score = r.score
            r.fusion_rank = i
            r.final_rank = i
        results = dense_results
        retrieval_backend = "semantic"
    elif sparse_results:
        for i, r in enumerate(sparse_results, 1):
            r.sparse_score = r.score
            r.sparse_relevance = r.score
            r.sparse_rank = i
            r.fusion_score = r.score
            r.fusion_rank = i
            r.final_rank = i
        results = sparse_results
        retrieval_backend = "fallback"
    else:
        results = []
        retrieval_backend = "fallback"

    # Defense-in-depth: Ensure NO chunk from outside authorized candidates can leak
    results = [
        r for r in results
        if r.source_kind != "knowledge_document" or (
            r.document_id in allowed_doc_ids and r.chunk_id in allowed_chunks_by_id
        )
    ]

    logger.info(
        "retrieval_backend = %s (retrieved %d knowledge chunks, "
        "dense=%d, sparse=%d)",
        retrieval_backend, len(results), len(dense_results), len(sparse_results),
    )

    # Step 2d (Operation 24): Conditional swap re-ranking.
    # Preserves the RRF ranking by default. Only swaps top-1 with a candidate
    # from positions 2-5 when high-confidence conditions are met (same-heading
    # procedural preference, generic overview demotion, fault code promotion).
    try:
        from core.ml.conditional_rerank import conditional_swap_rerank
        results = conditional_swap_rerank(results, question or "", swap_window=5)
    except Exception as exc:
        logger.warning("Conditional swap re-ranking skipped due to error: %s", exc)

    # Step 3: Engineer Copilot - Previous Service Cases (separate from authoritative manuals)
    # Disabled for Customer Self-Service (only enabled when include_past_resolutions is True)
    resolution_results = []
    if include_past_resolutions:
        for idx in _resolution_candidates(tenant, asset, customer, staff_mode):
            call = idx.service_call
            score = cosine_sparse(evidence_vector, idx.embedding)
            if score < 0.02:
                continue
            boost = 0.0

            if asset and call.asset_id == asset.id:
                boost += 0.20
            elif asset and call.asset and call.asset.product_id == asset.product_id:
                boost += 0.10
            if service_call and call.id == service_call.id:
                continue

            final_score = float(score + boost)
            if final_score < min_score:
                continue
            safe_text = idx.text
            resolution_results.append(RetrievedChunk(
                source_kind="past_resolution",
                source_id=idx.id,
                document_id=None,
                chunk_id=None,
                title=f"Previous service case #{call.servy_id}",
                heading="Previous service case",
                text=safe_text,
                score=final_score,
                doc_type="Previous Service Case",
                source_url="",
                confidential=True,
                service_call_id=call.id,
            ))

    # Step 4: Strict Source Authority Tiering — authoritative technical guidance
    # (Knowledge Documents) strictly outranks historical cases (Service Calls).
    # Historical anecdotes can NEVER authorize repair actions or outrank approved manuals.
    # Preserve the exact fused ranking of knowledge documents!
    sorted_resolutions = sorted(resolution_results, key=lambda r: -r.score)
    all_results = list(results) + sorted_resolutions

    # Step 5: Diversity & content-level deduplication
    selected = []
    seen_chunk_keys = set()
    seen_content_hashes = set()
    per_source = {}
    max_per_source = getattr(settings, "SERVY_MAX_CHUNKS_PER_SOURCE", top_k)

    for item in all_results:
        chunk_key = (item.source_kind, item.document_id, item.chunk_id)
        if chunk_key in seen_chunk_keys and item.chunk_id is not None:
            continue

        # Deterministic SHA-256 content deduplication (Phase 7 Repair)
        # Content deduplication ensures duplicate manual versions are deduplicated
        import hashlib
        norm_text = " ".join((item.text or "").strip().lower().split())
        content_digest = hashlib.sha256(norm_text.encode("utf-8")).hexdigest()
        digest_key = content_digest
        if digest_key in seen_content_hashes:
            continue

        source_key = (item.source_kind, item.document_id or item.service_call_id or item.source_id)
        if item.source_kind == "past_resolution" and per_source.get(source_key, 0) >= 2:
            continue
        if item.source_kind == "knowledge_document" and per_source.get(source_key, 0) >= max_per_source:
            continue

        selected.append(item)
        seen_chunk_keys.add(chunk_key)
        seen_content_hashes.add(digest_key)
        per_source[source_key] = per_source.get(source_key, 0) + 1
        # Bounded candidate pool for reranking (Operation 13.0)
        # Collect up to 3x top_k so reranker can recover relevant passages
        rerank_pool_limit = max(top_k * 3, 24)
        if len(selected) >= rerank_pool_limit:
            break

    try:
        from core.ml.reranker import rerank_candidates
        selected = rerank_candidates(evidence_text, selected, top_k=top_k)
    except Exception as exc:
        logger.warning("Reranker step skipped due to error: %s", exc)
        selected = selected[:top_k]

    if len(selected) > top_k:
        selected = selected[:top_k]

    for i, item in enumerate(selected, 1):
        item.final_rank = i

    return RetrievedResultList(
        selected,
        retrieval_backend=retrieval_backend,
        reranker_applied=any(
            item.source_kind == "knowledge_document" and item.reranker_score is not None
            for item in selected
        ),
        reranker_model=next(
            (getattr(item, "reranker_model", None) for item in selected
             if item.source_kind == "knowledge_document"
             and item.reranker_score is not None
             and getattr(item, "reranker_model", None)),
            None,
        ),
    )
