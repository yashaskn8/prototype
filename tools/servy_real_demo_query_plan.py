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
from tools.servy_real_demo_dialogue import out_of_scope_response

_COMPONENT = re.compile(
    r"\b(?:compressor|fan|coil|filter|refrigerant|thermostat|sensor|"
    r"outdoor unit|condenser|drain|airflow|remote|controller|"
    r"power supply|electrical supply)\b", re.I
)
_CODE = re.compile(r"\b(?:E|F|ERR)[- ]?\d{1,3}\b", re.I)


def targeted_retry_query(question, asset, context):
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
    # Anchor on what is being asked/reported now. A component the customer says
    # is clean or already checked must not become the retry subject.
    relevant = relevance_text(q)
    subject = _CODE.search(relevant) or _COMPONENT.search(relevant)
    if subject is None:
        return None
    # This is a retrieval hint, not a new customer observation or a diagnosis.
    # Only the selected asset and latest signed customer issue are considered.
    asset_name = " ".join(str(getattr(asset, "name", "") or "").split())[:90]
    asset_model = " ".join(str(getattr(asset, "model_number", "") or "").split())[:45]
    issue = " ".join(str(getattr(context, "original_complaint", "") or "").split())
    issue = retrieval_focus(issue) or relevance_text(issue)
    issue = issue[:160]
    if query_is_control_abuse(issue) or out_of_scope_response(issue):
        issue = ""
    anchor = subject.group(0)
    clauses = [asset_name, asset_model, anchor]
    if issue and issue.casefold() != q.casefold() and anchor.casefold() not in issue.casefold():
        clauses.append(issue)
    elif issue and issue.casefold() != q.casefold():
        clauses = [asset_name, asset_model, issue]
    # Do not repeat the entire conversation; avoid noisy keyword saturation.
    return " ".join(item for item in clauses if item).strip()[:350] or None
