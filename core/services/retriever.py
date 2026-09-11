from dataclasses import dataclass
from typing import List

from django.db.models import Q

from core.models import KnowledgeChunk, ServiceResolutionIndex
from .vectors import embed_text, cosine_sparse


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

    @property
    def reference(self):
        section = f" → {self.heading}" if self.heading else ""
        return f"{self.title}{section}"


class RetrievedResultList(list):
    def __init__(self, items=(), retrieval_backend="semantic"):
        super().__init__(items)
        self.retrieval_backend = retrieval_backend


def _knowledge_candidates(tenant, asset, customer, include_confidential):
    qs = KnowledgeChunk.objects.filter(
        tenant=tenant,
        document__is_rag_enabled=True,
        document__index_status="INDEXED",
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
        category = product.category
        domain = category.domain if category else None
        # Strict hierarchy scoping: exact asset > product > category > domain > general fleet
        # Disallow documents explicitly tied to a DIFFERENT product or DIFFERENT asset
        scope = Q(document__asset=asset) | Q(document__asset__isnull=True, document__product=product)
        if category:
            scope |= Q(document__asset__isnull=True, document__product__isnull=True, document__category=category)
        if domain:
            scope |= Q(
                document__asset__isnull=True,
                document__product__isnull=True,
                document__category__isnull=True,
                document__domain=domain,
            )
        scope |= Q(
            document__asset__isnull=True,
            document__product__isnull=True,
            document__category__isnull=True,
            document__domain__isnull=True,
        )
        qs = qs.filter(scope)
    elif asset:
        qs = qs.filter(Q(document__asset=asset) | Q(document__asset__isnull=True, document__product__isnull=True))
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


# Documented hybrid ranking metadata bonuses:
# exact_asset_bonus: 0.35 (chunk comes from a manual explicitly mapped to this asset)
# exact_product_bonus: 0.25 (chunk comes from a manual mapped to this asset's product)
# model_bonus: 0.15 (document or chunk explicitly matches the asset's model number)
# category_bonus: 0.10 (chunk comes from a manual mapped to this product category)
# domain_bonus: 0.05 (chunk comes from a manual mapped to this domain)
# customer_bonus: 0.10 (chunk is specifically tailored/scoped to this customer)

EXACT_ASSET_BONUS = 0.35
EXACT_PRODUCT_BONUS = 0.25
MODEL_BONUS = 0.15
CATEGORY_BONUS = 0.10
DOMAIN_BONUS = 0.05
CUSTOMER_BONUS = 0.10


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
    evidence_vector = embed_text(evidence_text)
    scope_vector = embed_text(dynamic_query_text)

    # Step 1: Django determines candidate pool according to tenant, customer, confidentiality, and asset scope
    candidates = list(_knowledge_candidates(tenant, asset, customer, include_confidential))
    allowed_chunks_by_id = {c.id: c for c in candidates}
    allowed_doc_ids = {c.document_id for c in candidates}

    results: list[RetrievedChunk] = []
    retrieval_backend = "fallback"

    # Step 2: Attempt dense semantic retrieval from ChromaDB if available
    try:
        from .chroma_store import get_sentence_transformer, get_dense_collection
        model = get_sentence_transformer()
        if model is not None and allowed_doc_ids:
            dense_coll = get_dense_collection()
            query_embedding = model.encode(dynamic_query_text, normalize_embeddings=True).tolist()
            chroma_res = dense_coll.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 4, 30),
                where={"tenant_id": int(tenant.id)}
            )
            
            if chroma_res and chroma_res.get("metadatas") and chroma_res["metadatas"][0]:
                for meta, dist in zip(chroma_res["metadatas"][0], chroma_res["distances"][0]):
                    doc_id = meta.get("document_id")
                    chunk_idx = meta.get("chunk_index", 0)
                    
                    # Step 3: Django authorization re-check - verify document is allowed
                    if doc_id not in allowed_doc_ids:
                        continue
                    
                    matched_chunk = next((c for c in candidates if c.document_id == doc_id and c.chunk_index == chunk_idx), None)
                    if not matched_chunk or matched_chunk.is_quarantined:
                        continue
                    
                    d = matched_chunk.document
                    sim = max(0.0, 1.0 - float(dist))
                    boost = _calculate_metadata_bonus(d, matched_chunk, asset, customer)
                    final_score = float(sim + boost)
                    if final_score < min_score:
                        continue
                    
                    results.append(RetrievedChunk(
                        source_kind="knowledge_document",
                        source_id=matched_chunk.id,
                        document_id=d.id,
                        chunk_id=matched_chunk.id,
                        title=d.title,
                        heading=matched_chunk.heading,
                        text=matched_chunk.text,
                        score=final_score,
                        doc_type=d.get_doc_type_display(),
                        source_url=d.source_url,
                        confidential=d.is_confidential,
                    ))
                if results:
                    retrieval_backend = "semantic"
    except Exception as exc:
        logger.warning("Chroma semantic retrieval encountered exception: %s. Falling back to sparse retriever.", exc)
        retrieval_backend = "fallback"

    # Step 4: Deterministic local lexical/sparse fallback if semantic search did not return results
    if not results:
        retrieval_backend = "fallback"
        for chunk in candidates:
            d = chunk.document
            content_score = cosine_sparse(evidence_vector, chunk.content_embedding or chunk.embedding)
            scope_score = cosine_sparse(scope_vector, chunk.embedding)
            if content_score < min(0.006, min_score + 0.001):
                continue
            score = (0.72 * content_score) + (0.28 * scope_score)
            boost = _calculate_metadata_bonus(d, chunk, asset, customer)
            final_score = float(score + boost)
            if final_score < min_score:
                continue
            results.append(RetrievedChunk(
                source_kind="knowledge_document",
                source_id=chunk.id,
                document_id=d.id,
                chunk_id=chunk.id,
                title=d.title,
                heading=chunk.heading,
                text=chunk.text,
                score=final_score,
                doc_type=d.get_doc_type_display(),
                source_url=d.source_url,
                confidential=d.is_confidential,
            ))

    logger.info("retrieval_backend = %s (retrieved %d knowledge chunks)", retrieval_backend, len(results))

    # Step 5: Engineer Copilot - Previous Service Cases (separate from authoritative manuals)
    # Disabled for Customer Self-Service (only enabled when include_past_resolutions is True)
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
            results.append(RetrievedChunk(
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

    results.sort(key=lambda r: r.score, reverse=True)
    # Diversity & deduplication: avoid returning multiple identical chunks from the same document/call
    selected = []
    seen_texts = set()
    per_source = {}
    for item in results:
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
    return RetrievedResultList(selected, retrieval_backend=retrieval_backend)
