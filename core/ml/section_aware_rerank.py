"""Opt-in section-intent and evidence-aware reranking experiment.

Never filters authorized candidates. The default serving path is unchanged unless
SERVY_EXPERIMENTAL_SECTION_AWARE_RERANK=1 is explicitly configured.
"""
from __future__ import annotations

import re


# These prompts explicitly ask for safety guidance, but the relevant restriction
# may live inside a symptom/diagnostic passage rather than a section titled
# "Safety". Section-heading labels alone must not override evidence ranking.
_SAFETY_FOCUSED_QUERY = re.compile(
    r"\\b(?:safety|hazards?|prohibited\\s+actions?)\\b",
    flags=re.IGNORECASE,
)


def _is_safety_focused_query(query_text: str) -> bool:
    return bool(_SAFETY_FOCUSED_QUERY.search(str(query_text or "")))


_INTENT_PATTERNS = {
    "safety": (
        r"\bsafety\b", r"\bhazard", r"\bppe\b", r"\block\s*out\b",
        r"\blockout\b", r"\btag\s*out\b", r"\btagout\b",
        r"\bisolat(?:e|ion|ing)\b", r"\bdischarg(?:e|ing)\b",
    ),
    "pre_installation": (
        r"\bpre[-\s]?installation\b", r"\bbefore\s+(?:installing|installation)\b",
        r"\bsite\s+preparation\b", r"\bfacility\s+preparation\b",
    ),
    "installation": (
        r"\binstallation\b", r"\binstall(?:ing|ed)?\b", r"\bmount(?:ing|ed)?\b",
        r"\bwiring\b", r"\bcabling\b", r"\bpipework\b", r"\bpiping\b",
    ),
    "commissioning": (
        r"\bcommission(?:ing|ed)?\b", r"\bstart[-\s]?up\b",
        r"\bfirst\s+start\b", r"\bput(?:ting)?\s+into\s+service\b",
    ),
    "initial_checks": (
        r"\binitial\s+checks?\b", r"\bpre[-\s]?start\s+checks?\b",
        r"\bstart[-\s]?up\s+checks?\b", r"\bfirst[-\s]?run\s+checks?\b",
    ),
    "diagnostics": (
        r"\bdiagnos(?:is|tic|tics)\b", r"\btroubleshoot(?:ing)?\b",
        r"\bfault(?:s|finding)?\b", r"\balarm(?:s)?\b", r"\berror\s+codes?\b",
        r"\bsymptom(?:s)?\b", r"\bnot\s+cooling\b", r"\bnot\s+starting\b",
    ),
    "measurements": (
        r"\bmeasure(?:ment|ments|d|s|ing)?\b", r"\breadings?\b",
        r"\bvalues?\b", r"\btemperature\b", r"\bpressure\b", r"\bvoltage\b",
        r"\bcurrent\b", r"\bresistance\b", r"\bairflow\b", r"\bair\s+flow\b",
        r"\brpm\b", r"\bpsi\b", r"\bkpa\b", r"\bbar\b", r"\bvolt(?:s)?\b",
        r"\bamps?\b", r"\bmm\b", r"\btorque\b",
    ),
    "maintenance": (
        r"\bmaintenance\b", r"\bmaintain(?:ing|ed)?\b", r"\bclean(?:ing|ed)?\b",
        r"\blubricat(?:e|ion|ing)\b", r"\bservice\s+intervals?\b",
        r"\bmaintenance\s+intervals?\b", r"\breplacement\s+intervals?\b",
        r"\bfilter\s+replacement\b",
    ),
    "inspection": (
        r"\binspect(?:ion|ions|ing|ed)?\b", r"\broutine\s+checks?\b",
        r"\bperiodic\s+checks?\b", r"\bvisual\s+checks?\b",
    ),
    "record_keeping": (
        r"\brecord(?:s|ed|ing)?\b", r"\blogbooks?\b", r"\bservice\s+logs?\b",
        r"\bmaintenance\s+records?\b", r"\bdocumentation\b",
        r"\breport(?:s|ing)?\b", r"\bservice\s+history\b",
    ),
}
_INTENT_ORDER = (
    "safety", "pre_installation", "installation", "commissioning",
    "initial_checks", "diagnostics", "measurements", "maintenance",
    "inspection", "record_keeping",
)


def classify_section_intents(text: str) -> tuple[str, ...]:
    """Return explicit section intents in a stable order."""
    value = str(text or "").lower()
    if not value.strip():
        return ()
    found = {
        intent for intent, patterns in _INTENT_PATTERNS.items()
        if any(re.search(pattern, value, flags=re.IGNORECASE) for pattern in patterns)
    }
    if "pre_installation" in found:
        found.discard("installation")
    return tuple(intent for intent in _INTENT_ORDER if intent in found)


def _candidate_intents(candidate) -> tuple[str, ...]:
    metadata = " ".join(
        str(getattr(candidate, field, "") or "")
        for field in ("heading", "doc_type", "title")
    )
    return classify_section_intents(metadata)


def experimental_evidence_section_rerank(results: list, query_text: str, top_k: int = 10) -> list:
    """Blend incoming and evidence-aware ranks, then apply bounded section matching.

    Incoming order weight: 0.60. Evidence-aware order weight: 0.40.
    Explicit section match: up to +0.42; conflicting classified section: -0.20.
    Unknown sections remain neutral. No candidate is removed; ties keep input order.
    """
    if not results or len(results) < 2:
        return results
    baseline = list(results)
    limit = max(1, min(int(top_k), len(baseline)))
    try:
        from core.ml.technical_retrieval import evidence_aware_rerank
        evidence_order = evidence_aware_rerank(
            list(baseline), query_text, top_k=limit, calibrated=True,
        )
    except Exception:
        evidence_order = baseline

    evidence_rank = {id(item): rank for rank, item in enumerate(evidence_order, 1)}
    query_intents = classify_section_intents(query_text)
    scored = []
    for base_rank, item in enumerate(baseline[:limit], 1):
        evidence_position = evidence_rank.get(id(item), base_rank)
        score = (0.60 / base_rank) + (0.40 / evidence_position)
        candidate_intents = _candidate_intents(item)
        adjustment = 0.0
        reason = "evidence_rank_only"
        if _is_safety_focused_query(query_text):
            # Safety wording is frequently embedded in symptom-specific passages
            # (e.g. "do not chip ice" or "do not reset a tripping breaker").
            # A generic heading called "Safety" must not outrank that evidence
            # merely because the user's question contains the word "safety".
            # Retain the evidence-aware rank, but neutralize section-only bias.
            adjustment = 0.0
            reason = "safety_query_section_neutral"
        elif query_intents and candidate_intents:
            overlap = set(query_intents).intersection(candidate_intents)
            if overlap:
                coverage = len(overlap) / len(query_intents)
                adjustment = 0.42 * coverage
                reason = "section_match:" + ",".join(
                    intent for intent in query_intents if intent in overlap
                )
            else:
                adjustment = -0.20
                reason = "classified_section_mismatch"

        item.section_query_intents = query_intents
        item.section_candidate_intents = candidate_intents
        item.section_rerank_adjustment = adjustment
        item.section_rerank_score = score + adjustment
        item.section_rerank_reason = reason
        scored.append((score + adjustment, base_rank, item))

    scored.sort(key=lambda row: (-row[0], row[1]))
    ranked = [row[2] for row in scored] + baseline[limit:]
    for rank, item in enumerate(ranked, 1):
        item.final_rank = rank
    return ranked
