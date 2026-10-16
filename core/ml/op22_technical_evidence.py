"""Experimental, source-bound technical retrieval features and verifier.

This module is deliberately not wired into customer serving. All judgments are
conservative machine assessments, not human-approved technical ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.services.identifiers import extract_all_technical_identifiers, extract_error_codes, match_canonical_identifiers

COMPONENTS = frozenset({
    "battery", "belt", "bearing", "breaker", "capacitor", "coil", "compressor",
    "contactor", "controller", "fan", "filter", "heater", "manifold", "motor",
    "pump", "refrigerant", "relay", "sensor", "selector", "switch", "valve", "wiring",
})
ACTION_WORDS = frozenset({
    "adjust", "attach", "calibrate", "check", "clean", "disconnect", "discharge",
    "flush", "inspect", "isolate", "measure", "prime", "replace", "reset", "test", "verify",
})
SPEC_UNITS = {
    "torque": r"\b\d+(?:\.\d+)?\s*(?:n\s*[·.]?\s*m|nm|ft\s*[- ]?\s*lb)\b",
    "deflection": r"\b\d+(?:\.\d+)?\s*(?:mm|cm|inch|inches)\b",
    "pressure": r"\b\d+(?:\.\d+)?\s*(?:bar|psi|kpa)\b",
    "voltage": r"\b\d+(?:\.\d+)?\s*(?:v|volt|volts)\b",
    "temperature": r"\b\d+(?:\.\d+)?\s*(?:°\s*c|celsius|°\s*f|fahrenheit)\b",
}
SAFETY_ACTIONS = frozenset({"discharge", "isolate", "attach", "replace", "calibrate"})
GENERIC_HEADINGS = frozenset({"purpose", "general overview", "overview"})
TECHNICAL_ABBREVIATIONS = frozenset({"CIP", "HVAC", "VRF", "MCB", "RCCB", "VFD", "PCB", "RPM"})
SYMPTOM_WORDS = frozenset({"noise", "vibration", "leak", "leakage", "failure", "fault", "alarm", "overheating", "smoke", "sparking", "tripping", "cooling"})
PROCEDURE_TYPES = {
    "cleaning": ("clean", "cleaning", "wash", "sanitation"),
    "inspection": ("inspect", "inspection", "check", "diagnose"),
    "replacement": ("replace", "replacement", "install", "installation"),
    "reset": ("reset", "restart"),
    "measurement": ("measure", "measurement", "test", "verify"),
}
SAFE_SPELLING_NORMALIZATION = {
    "capcitor": "capacitor", "compresor": "compressor", "refrigrant": "refrigerant",
    "vibraton": "vibration", "bearng": "bearing",
}
WORD_RE = re.compile(r"\b[a-z][a-z0-9-]*\b", re.IGNORECASE)


@dataclass(frozen=True)
class TechnicalQuery:
    text: str
    identifiers: frozenset[str]
    fault_codes: frozenset[str]
    models: frozenset[str]
    components: frozenset[str]
    actions: frozenset[str]
    specification_types: frozenset[str]
    numeric_expressions: frozenset[str]
    negated_terms: frozenset[str]
    abbreviations: frozenset[str]
    symptoms: frozenset[str]
    procedure_types: frozenset[str]
    conditions: tuple[str, ...]
    technical_intent: bool


@dataclass(frozen=True)
class EvidenceAssessment:
    decision: str
    answer_bearing: bool
    exact_identifier_match: bool
    component_match: bool
    specification_present: bool
    procedure_present: bool
    generic_section: bool
    reasons: tuple[str, ...]
    support_spans: tuple[str, ...] = ()
    specification_values: tuple[str, ...] = ()


def parse_technical_query(text: str) -> TechnicalQuery:
    raw = text or ""
    words = {SAFE_SPELLING_NORMALIZATION.get(word.lower(), word.lower()) for word in WORD_RE.findall(raw)}
    codes = extract_error_codes(raw)
    identifiers = extract_all_technical_identifiers(raw)
    explicit_models = {m.upper() for m in re.findall(r"\bmodel\s*[:#-]?\s*([a-z0-9][a-z0-9.-]{2,})\b", raw, re.I)}
    components = words & COMPONENTS
    actions = words & ACTION_WORDS
    specs = {kind for kind in SPEC_UNITS if kind in words}
    if "rating" in words or "limit" in words or "specification" in words:
        specs.add("unspecified")
    numbers = {
        re.sub(r"\s+", "", value.lower())
        for value in re.findall(r"(?<!\w)\d+(?:\.\d+)?\s*(?:n\s*[·.]?\s*m|nm|ft\s*[- ]?\s*lb|mm|cm|bar|psi|kpa|v|volts?|°\s*c|°\s*f|rpm|%)\b", raw, re.I)
    }
    negated = {m.lower() for m in re.findall(r"\b(?:not|without|except|exclude)\s+(\w+)\b", raw, re.I)}
    abbreviations = {word.upper() for word in re.findall(r"\b[A-Za-z]{2,6}\b", raw)
                     if word.upper() in TECHNICAL_ABBREVIATIONS}
    symptoms = words & SYMPTOM_WORDS
    procedures = {kind for kind, cues in PROCEDURE_TYPES.items() if words.intersection(cues)}
    conditions = tuple(part.strip() for part in re.split(r"\s+(?:and|but|while)\s+|[,;]", raw, flags=re.I) if part.strip())
    technical = bool(codes or identifiers or components or actions or specs or numbers or symptoms or procedures or abbreviations)
    return TechnicalQuery(
        text=raw, identifiers=frozenset(identifiers), fault_codes=frozenset(codes),
        models=frozenset(explicit_models), components=frozenset(components),
        actions=frozenset(actions), specification_types=frozenset(specs),
        numeric_expressions=frozenset(numbers), negated_terms=frozenset(negated),
        abbreviations=frozenset(abbreviations), symptoms=frozenset(symptoms),
        procedure_types=frozenset(procedures), conditions=conditions,
        technical_intent=technical,
    )


def assess_passage(
    query: TechnicalQuery, *, heading: str, body: str,
    document: Any = None, asset: Any = None, source_verified: bool = False,
    action_approved: bool = False, related_passages: tuple[dict, ...] = (),
    chunk_index: int | None = None, section_id: str | None = None,
) -> EvidenceAssessment:
    """Classify a passage, never the existence of an answer in the corpus."""
    from core.services.knowledge_scope import is_document_applicable_to_asset

    context = []
    document_id = getattr(document, "id", None)
    for segment in related_passages:
        if (document_id is not None and chunk_index is not None and section_id is not None
                and segment.get("document_id") == document_id
                and segment.get("section_id") == section_id
                and isinstance(segment.get("chunk_index"), int)
                and abs(segment["chunk_index"] - chunk_index) == 1):
            context.append(segment.get("text") or "")
    text = f"{heading or ''}\n{body or ''}" + ("\n" + "\n".join(context) if context else "")
    lower = text.lower()
    words = {word.lower() for word in WORD_RE.findall(lower)}
    heading_key = re.sub(r"^(?:\d+[.)]\s*)", "", (heading or "").strip().lower())
    generic = heading_key in GENERIC_HEADINGS
    reasons = []

    if document is not None and not is_document_applicable_to_asset(document, asset):
        return EvidenceAssessment("PASSAGE_EQUIPMENT_MISMATCH", False, False, False, False, False, generic, ("document_not_applicable",))
    if not source_verified:
        return EvidenceAssessment("PASSAGE_SOURCE_UNVERIFIED", False, False, False, False, False, generic, ("source_not_verified",))

    matched, missing = match_canonical_identifiers(set(query.fault_codes), text)
    code_match = bool(matched) and not missing if query.fault_codes else False
    if query.fault_codes and not code_match:
        reasons.append("required_fault_code_absent")
    component_match = bool(query.components & words) if query.components else False
    symptom_match = bool(query.symptoms & words) if query.symptoms else False
    if query.components and not component_match:
        reasons.append("requested_component_absent")
    model_match = True
    if query.models:
        document_model = getattr(getattr(document, "asset", None), "model_number", "") if document else ""
        if document_model and document_model.upper() not in query.models:
            model_match = False
            reasons.append("different_explicit_equipment_model")
        elif not document_model and not any(
            re.search(rf"(?<![A-Z0-9]){re.escape(model)}(?![A-Z0-9])", text, re.I)
            for model in query.models
        ):
            model_match = False
            reasons.append("explicit_equipment_model_not_in_source")
    if not model_match:
        return EvidenceAssessment("PASSAGE_EQUIPMENT_MISMATCH", False, code_match, component_match, False, False, generic, tuple(reasons))

    spec_values = tuple(sorted({match.group().strip() for kind in query.specification_types if kind in SPEC_UNITS
                                for match in re.finditer(SPEC_UNITS[kind], lower, re.I)}))
    spec_present = bool(spec_values)
    if query.specification_types and not spec_present:
        reasons.append("requested_specification_value_absent")
    normalize_number = lambda value: re.sub(r"[^a-z0-9.]", "", value.casefold()).replace("volt", "v")
    normalized = normalize_number(lower)
    missing_numbers = [number for number in query.numeric_expressions if normalize_number(number) not in normalized]
    if missing_numbers:
        reasons.append("numeric_value_or_unit_conflicts_with_source")
    body_lower = (body or "").lower() + ("\n" + "\n".join(context).lower() if context else "")
    body_words = {word.lower() for word in WORD_RE.findall(body_lower)}
    windows = [window.strip() for window in re.split(r"[.!?;\n]+", body_lower) if window.strip()]
    window_words = [{word.lower() for word in WORD_RE.findall(window)} for window in windows]
    spec_component_aligned = not (query.specification_types and query.components) or any(
        (query.components & tokens or query.components & {word.lower() for word in WORD_RE.findall(heading or "")})
        and any(re.search(SPEC_UNITS[kind], window, re.I) for kind in query.specification_types if kind in SPEC_UNITS)
        for window, tokens in zip(windows, window_words))
    if not spec_component_aligned:
        reasons.append("specification_not_bound_to_requested_component")
    action_cues = bool(body_words & ACTION_WORDS) or bool(re.search(r"\bstep\s*\d+\b", body_lower))
    procedure_kind_match = any(words.intersection(PROCEDURE_TYPES[kind]) for kind in query.procedure_types)
    action_match = bool(query.actions & body_words) if query.actions else True
    action_component_aligned = not (query.actions and query.components) or any(
        query.actions & tokens and query.components & tokens for tokens in window_words)
    numbered_steps = len(re.findall(r"(?:^|\n)\s*(?:step\s*)?\d+[.)]\s+", body_lower))
    has_detail = len(WORD_RE.findall(body_lower)) >= 7 and bool(re.search(
        r"\b(?:for|before|after|until|then|with|against|from|to)\b", body_lower))
    procedure_present = bool((query.actions or query.procedure_types) and action_cues and action_match
                             and action_component_aligned
                             and (not query.procedure_types or procedure_kind_match)
                             and (not query.components or component_match)
                             and (numbered_steps >= 2 or has_detail))
    if (query.actions or query.procedure_types) and not procedure_present:
        reasons.append("complete_requested_procedure_absent")
    code_definition = bool(code_match and re.search(
        r"\b(?:e|f)[- ]?\d{1,4}\b\s*(?::|[-–])\s*[a-z][a-z ]{7,}", lower, re.I))
    if query.fault_codes and code_match and not code_definition and not procedure_present:
        reasons.append("fault_code_mentioned_without_definition")
    if query.negated_terms and any(term in words for term in query.negated_terms):
        if not all(re.search(rf"\b(?:not|no|without)\s+{re.escape(term)}\b", lower)
                   for term in query.negated_terms):
            reasons.append("negated_condition_not_supported")
    negative_instruction = bool(query.actions and re.search(
        r"\b(?:do\s+not|never)\s+(?:\w+\s+){0,2}(?:" + "|".join(map(re.escape, query.actions)) + r")\b", body_lower))
    if negative_instruction:
        reasons.append("source_prohibits_requested_action")
    support = bool(
        not missing_numbers and not negative_instruction
        and "negated_condition_not_supported" not in reasons
        and (not query.components or component_match)
        and (not query.fault_codes or code_match)
        and (not query.specification_types or (spec_present and spec_component_aligned))
        and (not (query.actions or query.procedure_types) or procedure_present)
        and (not query.fault_codes or query.actions or query.procedure_types or code_definition)
        and bool(query.fault_codes or query.actions or query.procedure_types or query.specification_types)
    )
    if not query.technical_intent and len(WORD_RE.findall(body_lower)) >= 8:
        support = True
    support_spans = tuple(window for window in windows if
                          (not query.components or query.components & {word.lower() for word in WORD_RE.findall(window)}
                           or query.components & {word.lower() for word in WORD_RE.findall(heading or "")})
                          and (not query.specification_types or any(re.search(SPEC_UNITS[kind], window, re.I)
                              for kind in query.specification_types if kind in SPEC_UNITS))) if support else ()
    if support and code_definition and not support_spans:
        support_spans = (heading.strip(),)
    if support and query.actions & SAFETY_ACTIONS and not action_approved:
        decision = "PASSAGE_SAFETY_REVIEW_REQUIRED"
        reasons.append("physical_action_approval_missing")
    elif support:
        decision = "PASSAGE_SUPPORTED"
    elif component_match or code_match or procedure_present or symptom_match:
        decision = "PASSAGE_PARTIALLY_SUPPORTED"
    else:
        decision = "PASSAGE_RELEVANT" if not query.technical_intent else "PASSAGE_NOT_SUPPORTING_QUERY"
    return EvidenceAssessment(decision, support, code_match, component_match,
                              spec_present, procedure_present, generic, tuple(reasons),
                              support_spans, spec_values)


def rank_authorized_candidates(
    query: TechnicalQuery, candidates: list, dense_ranks: dict[int, int],
    lexical_ranks: dict[int, int], *, verified_document_ids: set[int],
    asset: Any = None, rerank_mode: str = "fixed",
) -> list[dict]:
    """Interpretable research ranker; authorization is mandatory before this call.

    The optional support tier compares categorical evidence, not tuned keyword
    weights. It remains offline until reviewed development labels justify use.
    """
    if rerank_mode not in {"fixed", "technical_support"}:
        raise ValueError("Unknown experimental rerank mode")
    scored = []
    for chunk in candidates:
        if chunk.id not in dense_ranks and chunk.id not in lexical_ranks:
            continue
        assessment = assess_passage(
            query, heading=chunk.heading, body=chunk.text,
            document=chunk.document, asset=asset,
            source_verified=chunk.document_id in verified_document_ids,
        )
        if assessment.decision in {"PASSAGE_EQUIPMENT_MISMATCH", "PASSAGE_SOURCE_UNVERIFIED"}:
            continue
        dense_rank = dense_ranks.get(chunk.id, 1000)
        lexical_rank = lexical_ranks.get(chunk.id, 1000)
        score = 1.0 / dense_rank + 0.12 / lexical_rank
        if query.fault_codes and assessment.exact_identifier_match:
            score += 0.50
        if query.specification_types and assessment.specification_present:
            score += 0.25
        if (query.actions or query.procedure_types) and assessment.procedure_present:
            score += 0.18
        if query.components and assessment.component_match:
            score += 0.12
        if query.technical_intent and assessment.generic_section and not assessment.answer_bearing:
            score -= 0.15
        tier = (2 if assessment.decision == "PASSAGE_SUPPORTED" else
                1 if assessment.decision == "PASSAGE_PARTIALLY_SUPPORTED" else 0)
        scored.append({"chunk_id": chunk.id, "score": round(score, 6), "support_tier": tier,
                       "assessment": assessment,
                       "dense_rank": dense_rank, "lexical_rank": lexical_rank})
    if rerank_mode == "technical_support" and query.technical_intent:
        return sorted(scored, key=lambda r: (-r["support_tier"], -r["score"], r["chunk_id"]))
    return sorted(scored, key=lambda r: (-r["score"], r["chunk_id"]))


def verify_answer_evidence(
    query: TechnicalQuery, ranked: list[dict], *,
    index_complete: bool = False, source_universe_complete: bool = False,
    asset_resolved: bool = True, expected_candidate_count: int | None = None,
    searched_candidate_count: int | None = None,
) -> dict:
    """Make a corpus-level decision only after a complete authorized search."""
    if not asset_resolved:
        return {"decision": "EQUIPMENT_CLARIFICATION_REQUIRED", "chunk_ids": [], "answer_coverage": False}
    if (not index_complete or not source_universe_complete
            or expected_candidate_count is None or searched_candidate_count != expected_candidate_count):
        return {"decision": "INCOMPLETE_SEARCH_OR_INDEX", "chunk_ids": [], "answer_coverage": False}
    if any(r["assessment"].decision == "PASSAGE_SOURCE_UNVERIFIED" for r in ranked):
        return {"decision": "SOURCE_VERIFICATION_REQUIRED", "chunk_ids": [], "answer_coverage": False}
    if any(r["assessment"].decision == "PASSAGE_SAFETY_REVIEW_REQUIRED" for r in ranked):
        return {"decision": "SAFETY_APPROVAL_REQUIRED", "chunk_ids": [], "answer_coverage": False}
    supported = [r for r in ranked if r["assessment"].decision == "PASSAGE_SUPPORTED"]
    values = {v.lower().replace(" ", "") for r in supported for v in r["assessment"].specification_values}
    if len(values) > 1 and query.specification_types:
        return {"decision": "PARTIALLY_SUPPORTED_FROM_AUTHORIZED_SOURCES", "chunk_ids": [],
                "answer_coverage": False, "reason": "conflicting_verified_specifications"}
    if supported:
        return {"decision": "SUPPORTED_FROM_AUTHORIZED_SOURCES",
                "chunk_ids": [r["chunk_id"] for r in supported], "answer_coverage": True}
    if any(r["assessment"].decision == "PASSAGE_PARTIALLY_SUPPORTED" for r in ranked):
        return {"decision": "PARTIALLY_SUPPORTED_FROM_AUTHORIZED_SOURCES", "chunk_ids": [],
                "answer_coverage": False}
    return {"decision": "NO_SUFFICIENT_EVIDENCE_FOUND", "chunk_ids": [], "answer_coverage": False}
