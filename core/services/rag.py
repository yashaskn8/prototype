from django.conf import settings

from .content_safety import query_is_control_abuse
from .retriever import retrieve
from .llm import generate_answer


def ask_rag(*, tenant, question, asset, customer=None, include_confidential=False, service_call=None, staff_mode=False):
    question = (question or "").strip()[:4000]
    if query_is_control_abuse(question):
        return {
            "answer": "I can help with product, asset, installation, maintenance, troubleshooting, and service questions. I cannot reveal system instructions, confidential data, or information from other customers or tenants.",
            "references": [],
            "engine": "policy-guard",
            "retrieved": [],
        }

    include_past_resolutions = bool(staff_mode and service_call)
    results = retrieve(
        tenant=tenant,
        question=question,
        asset=asset,
        customer=customer,
        include_confidential=include_confidential,
        top_k=settings.SERVY_RAG_TOP_K,
        min_score=settings.SERVY_RAG_MIN_SCORE,
        service_call=service_call,
        include_past_resolutions=include_past_resolutions,
        staff_mode=staff_mode,
    )
    answer, engine = generate_answer(question, asset, results, service_call=service_call)
    refs = [
        {
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
            "source_verified": getattr(r, "source_verified", True),
            "equipment_applicable": getattr(r, "equipment_applicable", True),
            "customer_action_authorized": getattr(r, "customer_action_authorized", True),
            "source_url": r.source_url,
            "model_version": getattr(r, "model_version", "minilm"),
        }
        for r in results
    ]
    return {"answer": answer, "references": refs, "engine": engine, "retrieved": results}
