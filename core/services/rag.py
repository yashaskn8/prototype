from django.conf import settings

from .content_safety import query_is_control_abuse
from .retriever import retrieve
from .llm import generate_answer


def ask_rag(
    *,
    tenant,
    question,
    asset,
    customer=None,
    include_confidential=False,
    service_call=None,
    staff_mode=False,
    top_k: int | None = None,
):
    question = (question or "").strip()[:4000]
    if query_is_control_abuse(question):
        return {
            "answer": "I can help with product, asset, installation, maintenance, troubleshooting, and service questions. I cannot reveal system instructions, confidential data, or information from other customers or tenants.",
            "references": [],
            "engine": "policy-guard",
            "retrieved": [],
            "retrieval_backend": "policy-guard",
            "reranker_applied": False,
            "reranker_model": None,
        }

    include_past_resolutions = bool(staff_mode and service_call)
    effective_top_k = top_k if top_k is not None else settings.SERVY_RAG_TOP_K
    results = retrieve(
        tenant=tenant,
        question=question,
        asset=asset,
        customer=customer,
        include_confidential=include_confidential,
        top_k=effective_top_k,
        min_score=settings.SERVY_RAG_MIN_SCORE,
        service_call=service_call,
        include_past_resolutions=include_past_resolutions,
        staff_mode=staff_mode,
    )
    from .llm import INSUFFICIENT_EVIDENCE_MESSAGE

    answer, engine = generate_answer(question, asset, results, service_call=service_call)

    # Helper to serialize chunk metadata
    def _chunk_to_dict(r):
        return {
            "source_kind": r.source_kind,
            "source_id": r.source_id,
            "document_id": r.document_id,
            "chunk_id": r.chunk_id,
            "service_call_id": r.service_call_id,
            "title": r.title,
            "heading": r.heading,
            "reference": r.reference,
            "doc_type": r.doc_type,
            "score": round(r.score, 4),
            "dense_similarity": round(r.dense_similarity, 4) if getattr(r, "dense_similarity", None) is not None else None,
            "sparse_relevance": round(r.sparse_relevance, 4) if getattr(r, "sparse_relevance", None) is not None else None,
            "fusion_score": round(r.fusion_score, 6) if getattr(r, "fusion_score", None) is not None else None,
            "final_rank": getattr(r, "final_rank", None),
            "evidence_support": getattr(r, "evidence_support", False),
            "source_verified": getattr(r, "source_verified", False),
            "equipment_applicable": getattr(r, "equipment_applicable", False),
            "customer_action_authorized": getattr(r, "customer_action_authorized", False),
            "source_url": r.source_url,
            "model_version": getattr(r, "model_version", "minilm"),
        }

    is_abstention = (
        answer == INSUFFICIENT_EVIDENCE_MESSAGE
        or engine == "insufficient-evidence"
        or "couldn't find enough approved information" in (answer or "").lower()
        or (answer or "").startswith("No approved documentation found")
    )

    all_retrieved_candidates = [_chunk_to_dict(r) for r in results]
    supported_chunks = [r for r in results if getattr(r, "evidence_support", False)]
    claim_supporting_passages = [_chunk_to_dict(r) for r in supported_chunks]

    if is_abstention:
        refs = []
        context_passages = []
    else:
        refs = claim_supporting_passages
        context_passages = claim_supporting_passages or all_retrieved_candidates

    return {
        "answer": answer,
        "references": refs,
        "engine": engine,
        "retrieved": results,
        "retrieval_backend": getattr(results, "retrieval_backend", "fallback"),
        "reranker_applied": getattr(results, "reranker_applied", False),
        "reranker_model": getattr(results, "reranker_model", None),
        "retrieved_candidates": all_retrieved_candidates,
        "context_passages": context_passages,
        "claim_supporting_passages": [] if is_abstention else claim_supporting_passages,
        "cited_passages": refs,
    }
