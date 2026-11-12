"""Exhaustive, asset-scoped search across approved customer-visible manual chunks."""
from __future__ import annotations

import hashlib
import math

MAX_APPROVED_CHUNKS = 3000
MAX_QUERY_VARIANTS = 10
MAX_REFERENCE_RESULTS = 300
MAX_PER_DOCUMENT = 40


def _query_variants(question, original_complaint, cause_claims):
    from core.services.content_safety import query_is_control_abuse
    values = [question, original_complaint]
    if isinstance(original_complaint, str) and original_complaint.strip():
        values.append(original_complaint + " documented causes troubleshooting context")
    for claim in list(cause_claims or [])[:8]:
        source = claim.get("source_text", "") if isinstance(claim, dict) else claim
        if isinstance(source, str) and source.strip():
            values.append(source)
    variants, seen = [], set()
    for value in values:
        if not isinstance(value, str):
            continue
        clean = " ".join(value.split())[:700]
        key = clean.casefold()
        if not clean or key in seen or query_is_control_abuse(clean):
            continue
        seen.add(key)
        variants.append(clean)
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
    return variants


def search_all_approved_documents(*, tenant, customer, asset, question,
                                  original_complaint="", cause_claims=()):
    """Score every eligible chunk in the asset's authorized company-document scope.

    When an approved document has no searchable chunks, or a safe scan limit
    is exceeded, coverage is marked incomplete. Returned IDs are still
    reauthorized by the normal customer answer builder before citation.
    """
    from django.conf import settings
    from core.services.content_safety import detect_document_prompt_injection
    from core.services.customer_isolation import is_verified_real_document
    from core.services.knowledge_scope import get_applicable_knowledge_queryset
    from core.services.retriever import (
        _knowledge_candidates, _sparse_score_candidates, get_serving_config,
        normalize_domain_terms,
    )
    from core.services.identifiers import extract_identifiers
    from core.services.vectors import embed_text

    document_queryset = get_applicable_knowledge_queryset(
        asset, {"tenant": tenant, "customer": customer, "role": "customer"}
    ).filter(is_rag_enabled=True, is_confidential=False, disable_sharing=False)
    approved_documents = [
        doc for doc in document_queryset
        if is_verified_real_document(doc, allow_failed=True)
    ]
    approved_by_id = {doc.id: doc for doc in approved_documents}

    candidates = list(_knowledge_candidates(
        tenant, asset, customer, False, allow_failed=True, backend="sparse",
    ))
    candidates = [
        chunk for chunk in candidates
        if chunk.document_id in approved_by_id
        and chunk.tenant_id == tenant.id
        and chunk.document.tenant_id == tenant.id
        and not chunk.document.is_confidential
        and not chunk.document.disable_sharing
        and chunk.document.is_rag_enabled
        and not getattr(chunk, "is_quarantined", False)
        and isinstance(chunk.text, str)
        and chunk.text.strip()
        and not detect_document_prompt_injection(chunk.text)
    ]

    chunks_by_document = {}
    for chunk in candidates:
        chunks_by_document.setdefault(chunk.document_id, []).append(chunk)
    searchable_document_ids = set(chunks_by_document)
    missing_chunk_docs = set(approved_by_id) - searchable_document_ids
    coverage = {
        "complete": not missing_chunk_docs,
        "scope": "selected-asset-approved-documents",
        "documents_searched": len(searchable_document_ids),
        "approved_documents": len(approved_documents),
        "documents_without_searchable_chunks": len(missing_chunk_docs),
        "passages_searched": len(candidates),
        "documents_with_matches": 0,
        "matching_passages": 0,
        "references_in_answer_pool": 0,
        "document_titles": sorted(
            [approved_by_id[key].title for key in searchable_document_ids],
            key=str.casefold,
        )[:30],
        "note": "",
    }
    if len(candidates) > MAX_APPROVED_CHUNKS:
        coverage.update({
            "complete": False,
            "note": (
                "The approved source set exceeds the safe complete-scan limit. "
                "Servy did not claim that every document was reviewed."
            ),
        })
        return {"references": [], "coverage": coverage}
    if not candidates:
        coverage["note"] = "No approved, searchable manual passages were available for this asset."
        return {"references": [], "coverage": coverage}

    variants = _query_variants(question, original_complaint, cause_claims)
    if not variants:
        coverage["complete"] = False
        coverage["note"] = (
            "A safe issue-specific query could not be formed; complete retrieval was not verified."
        )
        return {"references": [], "coverage": coverage}

    scope_parts = [
        getattr(asset, "name", ""),
        getattr(asset, "model_number", ""),
        getattr(getattr(asset, "product", None), "name", ""),
        getattr(getattr(getattr(asset, "product", None), "category", None), "name", ""),
    ]
    scope_vector = embed_text(" ".join(str(value).strip() for value in scope_parts if value))
    serving_config = get_serving_config()
    min_score = float(getattr(settings, "SERVY_RAG_MIN_SCORE", 0.055))
    best_by_chunk = {}
    # The sparse scorer evaluates the full candidate array on each query,
    # unlike normal dense/reranker retrieval's top-k result window.
    for variant in variants:
        evidence_text = normalize_domain_terms(variant).strip()
        if not evidence_text:
            continue
        scored = _sparse_score_candidates(
            candidates,
            embed_text(evidence_text),
            scope_vector,
            extract_identifiers(variant) | extract_identifiers(evidence_text),
            asset,
            customer,
            min_score,
            query_doc_ids=None,
            evidence_text=evidence_text,
            serving_config=serving_config,
        )
        for result in scored:
            key = (result.document_id, result.chunk_id)
            score = float(result.score)
            if not math.isfinite(score):
                continue
            previous = best_by_chunk.get(key)
            if previous is None or score > previous["score"]:
                best_by_chunk[key] = {
                    "document_id": int(result.document_id),
                    "chunk_id": int(result.chunk_id),
                    "score": score,
                }

    chunk_by_id = {chunk.id: chunk for chunk in candidates}
    groups, seen_text = {}, set()
    for row in best_by_chunk.values():
        chunk = chunk_by_id.get(row["chunk_id"])
        if chunk is None:
            continue
        normalized = " ".join(chunk.text.casefold().split())
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen_text:
            continue
        seen_text.add(digest)
        groups.setdefault(row["document_id"], []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda item: (-item["score"], item["chunk_id"]))

    # Round-robin across manuals so one long source cannot crowd out all others.
    references, document_ids = [], sorted(groups)
    for offset in range(MAX_PER_DOCUMENT):
        for document_id in document_ids:
            rows = groups[document_id]
            if offset < len(rows):
                references.append(rows[offset])
                if len(references) >= MAX_REFERENCE_RESULTS:
                    break
        if len(references) >= MAX_REFERENCE_RESULTS:
            break

    coverage.update({
        "documents_with_matches": len(groups),
        "matching_passages": len(best_by_chunk),
        "references_in_answer_pool": len(references),
    })
    if missing_chunk_docs:
        coverage["note"] = (
            "Servy searched " + str(len(searchable_document_ids))
            + " searchable approved document(s) and " + str(len(candidates))
            + " passages. " + str(len(missing_chunk_docs))
            + " approved document(s) had no searchable chunks, so complete coverage is not claimed."
        )
    elif references:
        coverage["note"] = (
            "Servy searched all " + str(len(searchable_document_ids))
            + " approved document(s) and " + str(len(candidates))
            + " passages; relevant evidence was found in " + str(len(groups))
            + " document(s)."
        )
    else:
        coverage["note"] = (
            "Servy searched all " + str(len(searchable_document_ids))
            + " approved document(s) and " + str(len(candidates))
            + " passages but found no sufficiently relevant passage for this question."
        )
    return {"references": references, "coverage": coverage}
