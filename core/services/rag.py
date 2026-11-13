            "answer": "No authorized original company documentation is available for this equipment. I cannot provide technical troubleshooting, maintenance steps, specifications, or citations.",
            "references": [], "engine": "insufficient-evidence",
            "reason_code": NO_AUTHORIZED_ORIGINAL_DOCUMENTATION,
            "retrieved": [], "retrieved_candidates": [], "context_passages": [],
            "claim_supporting_passages": [], "cited_passages": [],
            "retrieval_backend": "none", "reranker_applied": False, "reranker_model": None,
            "pre_rerank_candidate_chunk_ids": [],
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
    with execution_stage(
        "retrieval_pipeline",
        "Retrieving evidence from authorized technical sources",
    ):
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
    emit_execution_event(
        "retrieval_summary", "Retrieval result summary", "info",
        metrics={
            "retrieved_count": len(results),
            "retrieval_backend": getattr(results, "retrieval_backend", "fallback"),
            "reranker_applied": bool(getattr(results, "reranker_applied", False)),
            "reranker_model": getattr(results, "reranker_model", None) or "",
            "confidential_scope": "permitted" if include_confidential else "restricted",
        },
    )
    from .llm import INSUFFICIENT_EVIDENCE_MESSAGE

    with execution_stage(
        "answer_generation",
        "Preparing the evidence-grounded recommendation",
    ):
        answer, engine = generate_answer(
            question,
            asset,
            results,
            service_call=service_call,
            conversation_context=conversation_context,
            original_question=display_question,
            allow_generation=bool(staff_mode),
        )
    emit_execution_event(
        "answer_engine", "Answer generation outcome", "info",
        metrics={"answer_engine": str(engine)[:80]},
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

    with execution_stage(
        "evidence_validation",
        "Validating passage support before citation",
    ):
        all_retrieved_candidates = [_chunk_to_dict(r) for r in results]
        supported_chunks = [r for r in results if getattr(r, "evidence_support", False)]
        claim_supporting_passages = [_chunk_to_dict(r) for r in supported_chunks]
    emit_execution_event(
        "evidence_summary", "Evidence validation summary", "info",
        metrics={
            "candidate_count": len(all_retrieved_candidates),
            "supported_count": len(supported_chunks),
        },
    )

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
