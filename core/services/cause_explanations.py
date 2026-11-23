"""Explain every previously listed cause, each from its own retrieved evidence.

"Explain each issue." after a cause list must not re-answer the original
question with the same list. For every cause the customer was shown (taken from
the last citation-backed answer, never from the customer's pasted text alone):

1. the cause sentence is re-authorized in its source chunk and cited again;
2. a cause-specific retrieval runs through the normal authorized retriever;
3. explanation sentences are selected only from re-authorized chunks, must be
   customer-safe descriptive statements, must share at least two content terms
   with the cause, and are never the cause sentence or another listed cause;
4. a cause without such a sentence is reported as an explicit evidence gap.

All technical wording is exact source text with its own citation; the only
generated text is fixed connective wording.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from django.conf import settings

MAX_EXPLANATIONS_PER_CAUSE = 1
RESPONSE_LIMIT_TEXT = "More explanation may exist in the approved manuals; ask about this cause separately for details."
GAP_TEXT = "The approved manuals do not explain this cause further."
UNVERIFIED_TEXT = "One previously listed cause could not be re-verified in the currently authorized manuals."
LIMITATION = "These are documented possibilities, not a confirmed diagnosis of your unit."

_EXTRA_STOP = frozenset(
    "possible cause causes documented unit module equipment system outdoor indoor when after during "
    "which that this these those there their have has been being will would could should".split())


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" .").casefold()


def _content_keys(text: str) -> set:
    from .llm import _STOPWORDS
    from .term_matching import match_keys, stem
    words = set(re.findall(r"[a-z0-9_-]{3,}", str(text or "").casefold())) - _STOPWORDS - _EXTRA_STOP
    return {stem(w) for w in match_keys(words)}


def _locate(chunk, text: str):
    """Exact (start, end) of a claim inside its chunk via the claim parser, else None."""
    from .customer_answer import source_claims
    target = _norm(text)
    for claim, _kind, start, end in source_claims(chunk.text):
        if _norm(claim) == target:
            return start, end
    return None


def _candidate_sentences(chunk, asset):
    """Customer-safe descriptive sentences of one re-authorized chunk with exact spans."""
    from .customer_answer import (
        _RESTRICTED_LEVELS, _sentence_spans, customer_safe_fact, source_claims)
    from .customer_diagnostic import classify_action_safety
    for text, kind, start, end in source_claims(chunk.text):
        if kind == "possible_cause":
            # Another documented cause, usually of a different symptom, does not
            # explain this one even when it shares words such as "dirty".
            continue
        parts = _sentence_spans(chunk.text, text, start, end) or [(text, start, end)]
        for part, part_start, part_end in parts:
            level, _ = classify_action_safety(part, asset)
            if level in _RESTRICTED_LEVELS:
                continue
            if customer_safe_fact(part, level) == "descriptive":
                yield part, part_start, part_end


def _retrieve_for_cause(*, tenant, customer, asset, cause):
    from .rag_execution_log import emit_execution_event
    from .retriever import retrieve
    query = f"{cause.get('heading') or ''} {cause['source_text']}".strip()
    results = retrieve(tenant=tenant, question=query, asset=asset, customer=customer,
                       include_confidential=False, top_k=settings.SERVY_RAG_TOP_K,
                       min_score=settings.SERVY_RAG_MIN_SCORE, include_past_resolutions=False,
                       staff_mode=False)
    emit_execution_event(
        "cause_retrieval_summary", "Cause-specific retrieval", "info",
        metrics={"retrieved_count": len(results),
                 "retrieval_backend": getattr(results, "retrieval_backend", "fallback"),
                 "reranker_applied": bool(getattr(results, "reranker_applied", False))})
    return [(r.document_id, r.chunk_id, float(getattr(r, "score", 0) or 0))
            for r in results if getattr(r, "source_kind", "document") != "past_resolution"]


def _authorized_chunk(*, tenant, customer, asset, document_id, chunk_id):
    from rest_framework.exceptions import NotFound
    from .customer_answer import source_chunk
    if document_id is None or chunk_id is None:
        return None
    try:
        return source_chunk(tenant=tenant, customer=customer, asset=asset,
                            document_id=document_id, chunk_id=chunk_id)
    except (NotFound, ValueError, TypeError):
        return None


def explain_causes(*, interaction, causes: List[Dict[str, Any]],
                   retrieve_fn=None) -> Dict[str, Any]:
    """Return a customer-answer contract explaining each cause, or raise ValueError if none is citable."""
    from .customer_answer import _citation, TITLES
    from .rag_execution_log import emit_execution_event, execution_stage

    from .customer_answer import MAX_SECTION_CLAIMS
    tenant, customer, asset = interaction.tenant, interaction.customer, interaction.asset
    # Preserve the complete visible list up to the client claim limit.
    # Explanations use remaining section slots; do not silently discard
    # causes just because the manual contains more detail for earlier ones.
    omitted = max(0, len(causes) - MAX_SECTION_CLAIMS)
    causes = causes[:MAX_SECTION_CLAIMS]
    detail_budget = MAX_SECTION_CLAIMS - len(causes)
    retrieve_fn = retrieve_fn or _retrieve_for_cause
    all_cause_texts = {_norm(c["source_text"]) for c in causes}
    citations: List[Dict[str, Any]] = []
    cause_claims, fact_claims, gap_claims = [], [], []
    explanations, used_sentences = [], set()
    unverified = 0

    def cite(chunk, start, end, kind):
        citation = _citation(chunk, start, end, asset=asset, customer=customer,
                             user_id=interaction.created_by_id, kind=kind)
        citation["id"] = f"s{len(citations) + 1}"
        citations.append(citation)
        return citation["id"]

    for position, cause in enumerate(causes):
        with execution_stage("cause_evidence_retrieval",
                             f"Retrieving evidence for listed cause {position + 1}"):
            origin = _authorized_chunk(tenant=tenant, customer=customer, asset=asset,
                                       document_id=cause.get("document_id"), chunk_id=cause.get("chunk_id"))
            span = _locate(origin, cause["source_text"]) if origin else None
            hits = retrieve_fn(tenant=tenant, customer=customer, asset=asset, cause=cause)
            if span is None:
                # The stored chunk id may be stale after re-indexing: find the
                # exact cause sentence in a currently authorized retrieved chunk.
                for document_id, chunk_id, _score in hits:
                    chunk = _authorized_chunk(tenant=tenant, customer=customer, asset=asset,
                                              document_id=document_id, chunk_id=chunk_id)
                    found = _locate(chunk, cause["source_text"]) if chunk else None
                    if found:
                        origin, span = chunk, found
                        break
        if span is None:
            unverified += 1
            explanations.append({"cause": None, "status": "unverified", "evidence": []})
            continue
        cause_id = cite(origin, span[0], span[1], "possible_cause")
        cause_claims.append({
            "id": f"c{len(citations)}", "text": "Possible documented cause: " + cause["source_text"],
            "source_text": cause["source_text"], "kind": "possible_cause", "evidence_ids": [cause_id],
            "source_heading": origin.heading, "source_document_title": origin.document.title})

        cause_keys = _content_keys(cause["source_text"])
        ranked = []
        chunks = [(origin.document_id, origin.id, 1.0)] + [h for h in hits if h[1] != origin.id]
        for rank, (document_id, chunk_id, _score) in enumerate(chunks):
            chunk = origin if chunk_id == origin.id else _authorized_chunk(
                tenant=tenant, customer=customer, asset=asset, document_id=document_id, chunk_id=chunk_id)
            if chunk is None:
                continue
            for text, start, end in _candidate_sentences(chunk, asset):
                normalized = _norm(text)
                if normalized in all_cause_texts or normalized in used_sentences:
                    continue
                overlap = len(cause_keys & _content_keys(text))
                if overlap < 2:
                    continue
                heading_bonus = len(cause_keys & _content_keys(chunk.heading))
                ranked.append((-(overlap + heading_bonus), rank, start, chunk, text, end))
        ranked.sort(key=lambda item: item[:3])
        evidence = []
        for _neg, _rank, start, chunk, text, end in ranked[:min(MAX_EXPLANATIONS_PER_CAUSE, detail_budget)]:
            detail_budget -= 1
            used_sentences.add(_norm(text))
            evidence_id = cite(chunk, start, end, "information")
            fact_claims.append({"id": f"c{len(citations)}", "text": text, "source_text": text,
                                "kind": "information", "disclosure": "descriptive",
                                "evidence_ids": [evidence_id]})
            evidence.append({"text": text, "citation_ids": [evidence_id]})
        status = ("explained" if evidence else
                  "response_limit" if ranked and detail_budget == 0 else "insufficient_evidence")
        if not evidence:
            gap = RESPONSE_LIMIT_TEXT if status == "response_limit" else GAP_TEXT
            gap_claims.append({"text": f"{gap} ({cause['source_text'].rstrip('.')})",
                               "kind": "uncertainty", "evidence_ids": []})
        explanations.append({"cause": cause["source_text"], "cause_citation_ids": [cause_id],
                             "status": status, "evidence": evidence})

    if not cause_claims:
        raise ValueError("No previously listed cause could be re-verified.")
    emit_execution_event(
        "cause_explanation_summary", "Per-cause explanation outcome", "info",
        metrics={"candidate_count": len(causes),
                 "supported_count": sum(1 for e in explanations if e["status"] == "explained"),
                 "citation_count": len(citations)})

    unknown = gap_claims + ([{"text": UNVERIFIED_TEXT, "kind": "uncertainty", "evidence_ids": []}]
                            if unverified else [])
    if omitted:
        unknown.append({"text": f"{omitted} more listed cause(s) were not expanded here; "
                                "ask about them by number (for example, \"Explain the fifth cause.\").",
                        "kind": "uncertainty", "evidence_ids": []})
    unknown = unknown or [{"text": "This chat has not verified which documented possibility applies to your unit.",
                           "kind": "uncertainty", "evidence_ids": []}]
    sections = [
        {"number": 1, "title": "What the document says", "claims": cause_claims + fact_claims},
        {"number": 2, "title": TITLES[1], "claims": [{
            "text": "No customer procedure has been approved in the supporting evidence. "
                    "Report what you observed to a technician.", "kind": "uncertainty", "evidence_ids": []}]},
        {"number": 3, "title": TITLES[2], "claims": unknown},
        {"number": 4, "title": TITLES[3], "claims": [{
            "text": "Ask an authorized technician to confirm which cause applies if the issue continues. "
                    "No ticket has been created.", "kind": "next_step", "evidence_ids": []}]},
    ]
    return {"sections": sections, "citations": citations, "explanations": explanations,
            "supporting_claims": cause_claims + fact_claims, "omitted": omitted}


def explanation_response(equipment: str, built: Dict[str, Any]) -> Dict[str, Any]:
    """Composed reply: each cause, then its cited explanation or an explicit gap."""
    from .customer_response_composer import _segment, render_text
    segments = [_segment("text", f"Here is what the {equipment} manuals say about each listed cause:")]
    for item in built["explanations"]:
        if item["status"] == "unverified":
            segments.append(_segment("text", UNVERIFIED_TEXT))
            continue
        segments.append(_segment("list_item", item["cause"], item["cause_citation_ids"]))
        if item["evidence"]:
            for evidence in item["evidence"]:
                segments.append(_segment("claim", evidence["text"], evidence["citation_ids"]))
        else:
            segments.append(_segment("text",
                                     RESPONSE_LIMIT_TEXT if item["status"] == "response_limit"
                                     else GAP_TEXT))
    if built.get("omitted"):
        segments.append(_segment("text", f"{built['omitted']} more listed cause(s) were not expanded here; "
                                         "ask about them by number."))
    segments.append(_segment("text", LIMITATION))
    segments = segments[:30]
    explanations = [e for e in built["explanations"] if e["status"] != "unverified"]
    return {"text": render_text(segments, built["citations"]), "segments": segments,
            "style": "cause_explanations", "composer": "deterministic", "explanations": explanations}


def previous_answer_texts(context) -> List[str]:
    """Plain texts of earlier assistant replies, used to recognise pasted history."""
    return [t.get("text") or "" for t in getattr(context, "history_turns", None) or []
            if t.get("role") == "assistant"]


def causes_for_request(context, quoted_items: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The previously approved causes the request refers to, in the order shown.

    Pasted items select among the approved causes; pasted text that matches no
    approved cause is never treated as evidence.
    """
    from .conversation_context import listed_causes
    causes = listed_causes(getattr(context, "last_cause_list_contract", None)
                           or getattr(context, "last_supported_contract", None) or {})
    if quoted_items:
        quoted = {_norm(q) for q in quoted_items}
        selected = [c for c in causes if _norm(c["source_text"]) in quoted]
        return selected or causes
    return causes
