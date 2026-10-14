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
from .vectors import embed_text, cosine_sparse
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

    @property
    def reference(self):
        section = f" > {self.heading}" if self.heading else ""
        return f"{self.title}{section}"


class RetrievedResultList(list):
    def __init__(self, items=(), retrieval_backend="semantic"):
        super().__init__(items)
        self.retrieval_backend = retrieval_backend


def _knowledge_candidates(tenant, asset, customer, include_confidential, allow_failed=False, query_doc_ids=None):
    qs = KnowledgeChunk.objects.filter(
        tenant=tenant,
        document__is_rag_enabled=True,
        is_quarantined=False,
    ).select_related(
        "document", "document__product", "document__asset",
        "document__category", "document__domain", "document__customer",
    )

    if not include_confidential:
        qs = qs.filter(document__is_confidential=False)

    if customer:
        qs = qs.filter(Q(document__customer__isnull=True) | Q(document__customer=customer))
    else:
        qs = qs.filter(document__customer__isnull=True)

    if asset and asset.product:
        product = asset.product
        # Hierarchy & Category-Independent Scoping:
        # 1. Exact asset-bound documents: document__asset=asset
        # 2. Product-bound documents: document__asset__isnull=True, document__product=product
        # 3. Fleet/unassigned documents within tenant (documents not bound to a different product or different asset):
        #    document__asset__isnull=True, document__product__isnull=True
        scope = (
            Q(document__asset=asset)
            | Q(document__asset__isnull=True, document__product=product)
            | Q(document__asset__isnull=True, document__product__isnull=True)
        )
        if query_doc_ids:
            scope |= Q(document__id__in=query_doc_ids)
        qs = qs.filter(scope)
    elif asset:
        scope = Q(document__asset=asset) | Q(document__asset__isnull=True, document__product__isnull=True)
        if query_doc_ids:
            scope |= Q(document__id__in=query_doc_ids)
        qs = qs.filter(scope)
    else:
        # NO asset context: restrict to unscoped/general fleet documents only
        # unless an explicit document ID is requested
        scope = Q(document__asset__isnull=True, document__product__isnull=True)
        if query_doc_ids:
            scope |= Q(document__id__in=query_doc_ids)
        qs = qs.filter(scope)
    return qs[:3000]


def _resolution_candidates(tenant, asset, customer, staff_mode):
    qs = ServiceResolutionIndex.objects.filter(tenant=tenant).select_related(
        "service_call", "service_call__asset", "service_call__asset__product", "service_call__customer"
    )
    if asset:
        if asset.product_id:
            qs = qs.filter(Q(service_call__asset=asset) | Q(service_call__asset__product=asset.product))
        else:
            qs = qs.filter(service_call__asset=asset)
    if customer and not staff_mode:
        qs = qs.filter(service_call__customer=customer)
    return qs[:1500]


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

# Content relevance gates: chunk MUST exceed this threshold BEFORE metadata
# bonus is applied.  This prevents irrelevant documents with good metadata
# from being rescued by bonus alone (Defect #3).
SPARSE_RELEVANCE_GATE = getattr(settings, "SERVY_SPARSE_RELEVANCE_GATE", 0.055)
DENSE_RELEVANCE_GATE = getattr(settings, "SERVY_DENSE_RELEVANCE_GATE", 0.55)

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

def _identifier_boost(query_identifiers: set, chunk_text: str, chunk_heading: str) -> float:
    """Give a deterministic boost when the chunk contains an exact identifier
    that the query explicitly asks about."""
    if not query_identifiers:
        return 0.0
    combined = f"{chunk_heading} {chunk_text}".upper()
    for qid in query_identifiers:
        if qid.upper() in combined:
            return 0.40  # Strong boost for exact identifier match
    return 0.0


# ---------------------------------------------------------------------------
# Sparse retrieval scoring
# ---------------------------------------------------------------------------

def _sparse_score_candidates(candidates, evidence_vector, scope_vector,
                             query_identifiers, asset, customer, min_score,
                             query_doc_ids=None):
    """Score all candidates using sparse/local embeddings.

    Returns list of (rank_score, RetrievedChunk) tuples sorted by score descending.
    """
    scored = []
    for chunk in candidates:
        d = chunk.document
        content_score = cosine_sparse(evidence_vector, chunk.content_embedding or chunk.embedding)
        scope_score = cosine_sparse(scope_vector, chunk.embedding)

        # Content relevance gate: candidate must have minimum content match
        # BEFORE metadata bonus can apply (Defect #3 fix)
        base_score = (0.72 * content_score) + (0.28 * scope_score)
        id_boost = _identifier_boost(query_identifiers, chunk.text, chunk.heading)
        if query_doc_ids and d.id in query_doc_ids:
            id_boost = max(id_boost, 0.60)

        # Content relevance gate (Defect #3 fix): candidate must pass minimum content match
        # before metadata bonus can apply. Exact identifier matches are always relevant.
        if content_score < SPARSE_RELEVANCE_GATE and id_boost <= 0.0:
            continue

        if content_score < 0.08 and id_boost <= 0.0:
            boost = 0.0
        else:
            boost = _calculate_metadata_bonus(d, chunk, asset, customer)
        final_score = float(base_score + boost + id_boost)

        if final_score < min_score:
            continue

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
        ))

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Dense retrieval scoring
# ---------------------------------------------------------------------------

def _dense_score_candidates(candidates, allowed_doc_ids, allowed_chunks_by_id,
                            evidence_text, query_identifiers, asset,
                            customer, tenant, top_k, min_score,
                            query_doc_ids=None):
    """Score candidates using ChromaDB dense embeddings.

    Authorization is enforced BEFORE ranking by passing allowed_doc_ids into
    the Chroma where-filter (Defect #2 fix).  Query embedding encodes the
    actual user question/complaint to prevent metadata-only documents from
    rescuing irrelevant content (Defect #3 fix).
    """
    try:
        from django.conf import settings
        use_experimental = getattr(settings, "SERVY_USE_EXPERIMENTAL_RAG", False)
        profile_key = (
            getattr(settings, "SERVY_EXPERIMENTAL_EMBEDDING_MODEL", "bge-small")
            if use_experimental
            else getattr(settings, "SERVY_EMBEDDING_MODEL", "minilm")
        )
        from core.ml.embeddings import get_profile, load_model, get_collection, encode_queries
        profile = get_profile(profile_key)
        model = load_model(profile)
        if model is None or not allowed_doc_ids:
            return []

        dense_coll = get_collection(profile, create=False)
        query_embedding = encode_queries(profile, model, [evidence_text])[0].tolist()

        # Phase 3: Authorization-before-ranking — pass allowed_doc_ids into
        # Chroma filter so unauthorized documents cannot crowd out authorized
        # chunks in the top-N retrieved (Defect #2 fix).
        doc_id_list = [int(did) for did in allowed_doc_ids]

        # Chroma has a limit on $in list size; batch if necessary
        if len(doc_id_list) > 500:
            # For very large authorized sets, filter by tenant only and
            # revalidate afterwards (defense-in-depth)
            chroma_where = {"tenant_id": int(tenant.id)}
        else:
            chroma_where = {
                "$and": [
                    {"tenant_id": int(tenant.id)},
                    {"document_id": {"$in": doc_id_list}},
                ]
            }

        chroma_res = dense_coll.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 4, 30),
            where=chroma_where,
        )

        scored = []
        if chroma_res and chroma_res.get("metadatas") and chroma_res["metadatas"][0]:
            for meta, dist in zip(chroma_res["metadatas"][0], chroma_res["distances"][0]):
                doc_id = meta.get("document_id")
                chunk_idx = meta.get("chunk_index", 0)

                # Defense-in-depth: revalidate authorization on every row
                if doc_id not in allowed_doc_ids:
                    continue

                matched_chunk = next(
                    (c for c in candidates if c.document_id == doc_id and c.chunk_index == chunk_idx),
                    None,
                )
                if not matched_chunk or matched_chunk.is_quarantined:
                    continue

                d = matched_chunk.document
                sim = max(0.0, 1.0 - float(dist))
                id_boost = _identifier_boost(query_identifiers, matched_chunk.text, matched_chunk.heading)
                if query_doc_ids and d.id in query_doc_ids:
                    id_boost = max(id_boost, 0.60)

                # Content relevance gate (Defect #3 fix): if semantic similarity is too low
                # and no identifier matches, metadata bonus cannot rescue the document.
                if sim < DENSE_RELEVANCE_GATE and id_boost <= 0.0:
                    continue

                if getattr(profile, "content_first", False):
                    # In content-first mode, rely primarily on semantic content match.
                    # Bounded metadata bonus is strictly capped.
                    boost = min(0.05, _calculate_metadata_bonus(d, matched_chunk, asset, customer) * 0.25)
                elif sim < 0.58 and id_boost <= 0.0:
                    boost = 0.0
                else:
                    boost = _calculate_metadata_bonus(d, matched_chunk, asset, customer)
                final_score = float(sim + boost + id_boost)

                if final_score < min_score:
                    continue

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
                ))

        scored.sort(key=lambda r: r.score, reverse=True)
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

    RRF(d) = w_dense / (K + rank_dense(d)) + w_sparse / (K + rank_sparse(d))
    """
    # Build rank maps (1-indexed)
    dense_ranks = {}
    for i, r in enumerate(dense_results, 1):
        key = (r.source_kind, r.source_id)
        if key not in dense_ranks:
            dense_ranks[key] = i

    sparse_ranks = {}
    for i, r in enumerate(sparse_results, 1):
        key = (r.source_kind, r.source_id)
        if key not in sparse_ranks:
            sparse_ranks[key] = i

    # Collect all unique chunks
    all_chunks = {}
    for r in dense_results + sparse_results:
        key = (r.source_kind, r.source_id)
        if key not in all_chunks:
            all_chunks[key] = r

    # Compute RRF scores
    fused = []
    absent_rank = max(len(dense_results), len(sparse_results)) + 100
    for key, chunk in all_chunks.items():
        d_rank = dense_ranks.get(key, absent_rank)
        s_rank = sparse_ranks.get(key, absent_rank)
        rrf_rank_score = (w_dense / (RRF_K + d_rank)) + (w_sparse / (RRF_K + s_rank))
        # Keep chunk.score as the calibrated similarity+bonus score
        fused.append((rrf_rank_score, chunk))

    fused.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in fused]


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

    evidence_text = f"{question or ''}".strip()

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
    query_identifiers = extract_identifiers(question or "")
    query_error_codes = extract_error_codes(question or "")
    query_doc_ids = extract_document_ids(question or "")

    # Step 1: Django determines candidate pool according to tenant, customer,
    # confidentiality, and asset scope
    candidates = list(_knowledge_candidates(
        tenant, asset, customer, include_confidential,
        allow_failed=True, query_doc_ids=query_doc_ids,
    ))
    allowed_chunks_by_id = {c.id: c for c in candidates}
    allowed_doc_ids = {c.document_id for c in candidates}
    # For dense retrieval, only INDEXED documents have valid Chroma embeddings
    dense_candidates = [c for c in candidates if c.document.index_status == "INDEXED"]
    dense_chunks_by_id = {c.id: c for c in dense_candidates}
    dense_doc_ids = {c.document_id for c in dense_candidates}

    retrieval_backend = "fallback"

    # Step 2: TRUE HYBRID RETRIEVAL — run dense AND sparse concurrently,
    # then fuse results using Reciprocal Rank Fusion (Phase 2).
    # This replaces the old "sparse only if dense returns nothing" approach.

    # 2a: Sparse/local retrieval (always runs)
    sparse_results = _sparse_score_candidates(
        candidates, evidence_vector, scope_vector,
        query_identifiers, asset, customer, min_score,
        query_doc_ids=query_doc_ids,
    )

    # 2b: Dense retrieval (may fail gracefully)
    dense_results = _dense_score_candidates(
        candidates, allowed_doc_ids, allowed_chunks_by_id,
        evidence_text, query_identifiers, asset, customer,
        tenant, top_k, min_score,
        query_doc_ids=query_doc_ids,
    )

    # 2c: Fuse results using RRF
    if dense_results and sparse_results:
        results = _reciprocal_rank_fusion(dense_results, sparse_results)
        retrieval_backend = "hybrid"
    elif dense_results:
        results = dense_results
        retrieval_backend = "semantic"
    elif sparse_results:
        results = sparse_results
        retrieval_backend = "fallback"
    else:
        results = []
        retrieval_backend = "fallback"

    logger.info(
        "retrieval_backend = %s (retrieved %d knowledge chunks, "
        "dense=%d, sparse=%d)",
        retrieval_backend, len(results), len(dense_results), len(sparse_results),
    )

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

    # Step 4: Source Authority Ranking (Phase 6) — authoritative documentation
    # (Tiers 1-3) always outranks historical cases (Tier 4) when addressing
    # the same topic.  Apply a tier-based penalty to historical cases.
    all_results = results + resolution_results
    all_results.sort(key=lambda r: r.score, reverse=True)

    # Apply authority tier demotion: if authoritative docs exist, penalize
    # historical cases so they rank below authoritative content
    has_authoritative = any(r.source_kind == "knowledge_document" for r in all_results)
    if has_authoritative:
        for r in all_results:
            if r.source_kind == "past_resolution":
                # Demote historical cases below authoritative documents
                r.score = r.score * 0.6

    all_results.sort(key=lambda r: r.score, reverse=True)

    # Step 5: Diversity & deduplication: avoid returning multiple identical
    # chunks from the same document/call
    selected = []
    seen_texts = set()
    per_source = {}
    for item in all_results:
        text_sig = item.text[:120].strip()
        if text_sig in seen_texts:
            continue
        key = (item.source_kind, item.document_id or item.service_call_id or item.source_id)
        if per_source.get(key, 0) >= 2:
            continue
        selected.append(item)
        seen_texts.add(text_sig)
        per_source[key] = per_source.get(key, 0) + 1
        if len(selected) >= top_k:
            break

    try:
        from core.ml.reranker import rerank_candidates
        selected = rerank_candidates(evidence_text, selected, top_k=top_k)
    except Exception as exc:
        logger.warning("Reranker step skipped due to error: %s", exc)

    return RetrievedResultList(selected, retrieval_backend=retrieval_backend)
