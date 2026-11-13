from django.conf import settings
from pathlib import Path
import hashlib

from rest_framework.exceptions import PermissionDenied

from .content_safety import query_is_control_abuse
from .retriever import retrieve
from .llm import generate_answer


NO_AUTHORIZED_ORIGINAL_DOCUMENTATION = "NO_AUTHORIZED_ORIGINAL_DOCUMENTATION"


def has_authorized_original_documentation(*, tenant, customer, asset):
    """Require an intact company attachment with a customer-visible source chunk."""
    from .customer_isolation import (is_verified_real_asset, is_verified_real_customer,
                                     is_verified_real_document, validate_customer_serving_policy)
    from .retriever import _knowledge_candidates
    from .customer_answer import source_chunk

    if (not tenant or not customer or not asset or asset.tenant_id != tenant.id
        or customer.tenant_id != tenant.id or asset.customer_id != customer.id
        or not is_verified_real_asset(asset) or not is_verified_real_customer(customer)):
        raise PermissionDenied("Customer and asset scope is not verified.")
    validate_customer_serving_policy(tenant, customer, asset)
    media_root = Path(settings.MEDIA_ROOT).resolve()
    verified_files = {}
    for chunk in _knowledge_candidates(tenant, asset, customer, False, backend="all"):
        document = chunk.document
        if document.id not in verified_files:
            valid = False
            if (document.source_type == "file" and document.file and document.checksum_sha256
                and not document.is_confidential and not document.disable_sharing
                and is_verified_real_document(document)):
                try:
                    file_path = Path(document.file.path).resolve()
                    if file_path.is_relative_to(media_root) and file_path.is_file():
                        valid = hashlib.sha256(file_path.read_bytes()).hexdigest().lower() == document.checksum_sha256.lower()
                except OSError:
                    pass
            verified_files[document.id] = valid
        if not verified_files[document.id]:
            continue
        try:
            source_chunk(tenant=tenant, customer=customer, asset=asset,
                         document_id=document.id, chunk_id=chunk.id)
        except Exception:
            continue
        return True
    return False


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
    conversation_context=None,
    original_question=None,
):
    question = (question or "").strip()[:4000]
    display_question = (original_question or question).strip()[:4000]
    if query_is_control_abuse(display_question) or query_is_control_abuse(question):
        return {
            "answer": "I can help with product, asset, installation, maintenance, troubleshooting, and service questions. I cannot reveal system instructions, confidential data, or information from other customers or tenants.",
            "references": [],
            "engine": "policy-guard",
            "retrieved": [],
            "retrieval_backend": "policy-guard",
            "reranker_applied": False,
            "reranker_model": None,
            "pre_rerank_candidate_chunk_ids": [],
        }

    from .customer_isolation import get_customer_data_mode, is_real_tenant
    if (not staff_mode and is_real_tenant(tenant)
        and get_customer_data_mode() == "verified_real_only"
        and not has_authorized_original_documentation(tenant=tenant, customer=customer, asset=asset)):
        return {
            "answer": "No authorized original company documentation is available for this equipment. I cannot provide technical troubleshooting, maintenance steps, specifications, or citations.",
            "references": [], "engine": "insufficient-evidence",
            "reason_code": NO_AUTHORIZED_ORIGINAL_DOCUMENTATION,
            "retrieved": [], "retrieved_candidates": [], "context_passages": [],
            "claim_supporting_passages": [], "cited_passages": [],
            "retrieval_backend": "none", "reranker_applied": False, "reranker_model": None,
            "pre_rerank_candidate_chunk_ids": [],
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

    answer, engine = generate_answer(
        question,
        asset,
        results,
        service_call=service_call,
        conversation_context=conversation_context,
        original_question=display_question,
        allow_generation=bool(staff_mode),
    )

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
        "pre_rerank_candidate_chunk_ids": list(
            getattr(results, "pre_rerank_candidate_chunk_ids", []) or []
        ),
        "retrieved_candidates": all_retrieved_candidates,
        "context_passages": context_passages,
        "claim_supporting_passages": [] if is_abstention else claim_supporting_passages,
        "cited_passages": refs,
    }
