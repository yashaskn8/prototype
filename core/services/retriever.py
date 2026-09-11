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


def retrieve(
    tenant,
    question,
    asset=None,
    customer=None,
    include_confidential=False,
    top_k=5,
    min_score=0.04,
    service_call=None,
    include_past_resolutions=True,
    staff_mode=False,
) -> List[RetrievedChunk]:
    asset_terms = ""
    if asset:
        product = asset.product.name if asset.product else ""
        brand = asset.product.brand.name if asset.product and asset.product.brand else ""
        category = asset.product.category.name if asset.product and asset.product.category else ""
        asset_terms = f" {asset.name} {asset.model_number} {product} {brand} {category}"
    call_terms = ""
    if service_call:
        call_terms = f" {service_call.complaint_type} {service_call.complaint_text} {service_call.technician_notes}"

    evidence_text = f"{question or ''}{call_terms}".strip()
    evidence_vector = embed_text(evidence_text)
    scope_text = f"{evidence_text}{asset_terms}".strip()
    scope_vector = embed_text(scope_text)

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
            query_embedding = model.encode(f"{evidence_text} {asset_terms}".strip(), normalize_embeddings=True).tolist()
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
                    boost = 0.0
                    if asset:
                        if d.asset_id == asset.id:
                            boost += 0.22
                        elif d.product_id and asset.product_id == d.product_id:
                            boost += 0.14
                        elif d.category_id and asset.product and asset.product.category_id == d.category_id:
                            boost += 0.07
                    if customer and d.customer_id == customer.id:
                        boost += 0.10
                    final_score = float(sim + boost)
                    if final_score < min_score:
                        continue
                    
                    results.append(RetrievedChunk(
                        source_kind="knowledge",
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
            if content_score < 0.006:
                continue
            score = (0.72 * content_score) + (0.28 * scope_score)
            boost = 0.0
            if asset:
                if d.asset_id == asset.id:
                    boost += 0.22
                elif d.product_id and asset.product_id == d.product_id:
                    boost += 0.14
                elif d.category_id and asset.product and asset.product.category_id == d.category_id:
                    boost += 0.07
            if customer and d.customer_id == customer.id:
                boost += 0.10
            final_score = float(score + boost)
            if final_score < min_score:
                continue
            results.append(RetrievedChunk(
                source_kind="knowledge",
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
    # Diversity: avoid returning multiple chunks from the same document/call.
    selected = []
    per_source = {}
    for item in results:
        key = (item.source_kind, item.document_id or item.service_call_id or item.source_id)
        if per_source.get(key, 0) >= 2:
            continue
        selected.append(item)
        per_source[key] = per_source.get(key, 0) + 1
        if len(selected) >= top_k:
            break
    return RetrievedResultList(selected, retrieval_backend=retrieval_backend)
