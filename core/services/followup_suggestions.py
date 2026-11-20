"""Deterministic follow-up suggestions for a validated customer answer.

No model call and no new facts: every suggestion is either a request the
conversation layer already resolves against the answer just shown ("Explain
each of these causes.", "Tell me more about the first cause.") or a question
about another documented symptom section of the same equipment, using the
manual's own section title from this asset's authorized evidence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

MAX_SUGGESTIONS = 3
MAX_CHARS = 120


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").casefold()))


def _label(section: str) -> str:
    label = re.sub(r"^\s*\d+[.)]\s*", "", str(section or ""))
    label = re.sub(r"^(?:symptom|fault)\s*:\s*", "", label, flags=re.I).strip(" .:")
    return label.lower()


def suggest(contract: Dict[str, Any], asked: List[str]) -> List[str]:
    """Up to three suggestions, or [] when none are appropriate."""
    presentation = contract.get("presentation") or {}
    if (contract.get("response_type") != "DIAGNOSTIC_GUIDANCE"
            or presentation.get("response_type") != "DIAGNOSTIC_GUIDANCE"
            or not presentation.get("citations")
            or (presentation.get("hazard") or {}).get("severity") == "immediate"):
        return []
    response = presentation.get("response") or {}
    style = response.get("style")
    listed = [s for s in response.get("segments") or [] if s.get("kind") == "list_item"]
    asked_norm = {_norm(q) for q in asked}
    candidates: List[str] = []
    if style == "symptom_causes" and len(listed) >= 2:
        candidates += ["Explain each of these causes.", "Tell me more about the first cause."]

    # Other documented symptom sections of this asset (manual titles only).
    shown = {_label(c.get("source_heading")) for section in presentation.get("sections") or []
             for c in section.get("claims") or [] if c.get("source_heading")}
    for section in contract.get("documented_cause_sections") or []:
        label = _label(section)
        if (label and label not in shown and 3 <= len(label) <= 60
                and not re.search(r"possible causes?|^causes?$", label)):
            candidates.append(f"What are the possible causes of {label}?")

    out, seen = [], set()
    for text in candidates:
        key = _norm(text)
        if key in seen or key in asked_norm or len(text) > MAX_CHARS:
            continue
        seen.add(key)
        out.append(text)
        if len(out) == MAX_SUGGESTIONS:
            break
    return out
