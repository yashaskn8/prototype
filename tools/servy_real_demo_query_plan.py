"""Bounded, scoped second-pass query planning for Servy's read-only RAG.

This module does not retrieve, generate technical guidance, or authorize
anything. It makes exactly one alternative keyword-focused question from a
customer's *current* subject and previously reported equipment issue.
"""
from __future__ import annotations

import os
import re

from core.services.content_safety import query_is_control_abuse
from core.services.query_decomposition import relevance_text, retrieval_focus
from tools.servy_real_demo_dialogue import (
    is_explanation_followup, out_of_scope_response, question_topic,
)

_COMPONENT = re.compile(
    r"\b(?:compressor|fan|coil|filter|refrigerant|thermostat|sensor|"
    r"outdoor unit|condenser|drain|airflow|remote|controller|"
    r"power supply|electrical supply)\b", re.I
)
_CODE = re.compile(r"\b(?:E|F|ERR)[- ]?\d{1,3}\b", re.I)


def targeted_retry_query(question, asset, context, *, last_topic=None, cause_claims=()):
    """Generate one safe retrieval query or return None.

    The caller must enforce the same customer/asset permissions, source
    reauthorization, evidence sufficiency, and safety gates for both passes.
    A failed retry must not add new knowledge or weaken any rule.
    """
    if os.environ.get("SERVY_REAL_DEMO_TARGETED_RETRY") == "0":
        return None
    if not isinstance(question, str) or not 1 <= len(question) <= 420:
        return None
    q = " ".join(question.split())
    if not q or out_of_scope_response(q) or query_is_control_abuse(q):
        return None

    topic = question_topic(q, last_topic=last_topic)
    explanation_request = (
        last_topic == "causes" and is_explanation_followup(q)
    )
    observation_request = topic == "observations"
    # Keep the established component/code retry, and add two bounded
    # evidence-only retrieval intents: customer-visible observations and a
    # deeper source search for an explicitly referenced cause list.
    relevant = relevance_text(q)
    subject = _CODE.search(relevant) or _COMPONENT.search(relevant)
    if subject is None and not (observation_request or explanation_request):
        return None

    # This is a retrieval hint, not a new customer observation or diagnosis.
    # Only the selected asset and latest signed customer issue are considered.
    asset_name = " ".join(str(getattr(asset, "name", "") or "").split())[:90]
    asset_model = " ".join(str(getattr(asset, "model_number", "") or "").split())[:45]
    issue = " ".join(str(getattr(context, "original_complaint", "") or "").split())
    issue = retrieval_focus(issue) or relevance_text(issue)
    issue = issue[:160]
    if query_is_control_abuse(issue) or out_of_scope_response(issue):
        issue = ""

    if observation_request:
        clauses = [
            asset_name, asset_model, issue,
            "customer-visible safe observations external display fault code warning "
            "indicator operating mode visible status and visible air path",
        ]
    elif explanation_request:
        cause_focus = []
        cause_terms = []
        stop_words = {
            "the", "and", "or", "a", "an", "is", "are", "be", "been",
            "may", "might", "could", "can", "to", "of", "for", "by",
            "when", "if", "this", "that", "it", "as", "with", "from",
            "affect", "affects", "involved", "issue", "problem", "operation",
            "operating", "possible", "cause", "causes", "documented",
        }
        for claim in list(cause_claims or [])[:8]:
            if isinstance(claim, dict):
                source = claim.get("source_text", "")
            else:
                source = claim
            if not isinstance(source, str) or not source.strip():
                continue
            focused = retrieval_focus(source) or relevance_text(source)
            if focused and not query_is_control_abuse(focused):
                cause_focus.append(focused[:110])
            # Include a compact lexical form as well as the complete source
            # phrase. Hyphenated equipment terms are split for BM25 matching.
            tokens = re.findall(r"[a-z0-9]+", source.casefold())
            cause_terms.extend(
                token for token in tokens if len(token) >= 3 and token not in stop_words
            )
        normalized_cause_terms = " ".join(dict.fromkeys(cause_terms))[:180]
        clauses = [
            asset_name, asset_model, issue,
            " ".join(cause_focus),
            normalized_cause_terms,
            "detailed documented causes explanation, meaning, contributing conditions, "
            "manual context and limitations",
        ]
    else:
        anchor = subject.group(0)
        clauses = [asset_name, asset_model, anchor]
        if issue and issue.casefold() != q.casefold() and anchor.casefold() not in issue.casefold():
            clauses.append(issue)
        elif issue and issue.casefold() != q.casefold():
            clauses = [asset_name, asset_model, issue]

    # Do not repeat the transcript; the retry is still limited to one query.
    return " ".join(item for item in clauses if item).strip()[:350] or None
