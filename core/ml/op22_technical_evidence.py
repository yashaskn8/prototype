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
        for value in re.findall(r"(?<!\w)\d+(?:\.\d+)?\s*(?:mm|cm|bar|psi|kpa|v|volts?|°\s*c|°\s*f|rpm|%)\b", raw, re.I)
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
    action_approved: bool = False,
) -> EvidenceAssessment:
    """Check explicit support signals in an already-authorized source passage."""
    from core.services.knowledge_scope import is_document_applicable_to_asset

    text = f"{heading or ''}\n{body or ''}"
    lower = text.lower()
    words = {word.lower() for word in WORD_RE.findall(lower)}
    heading_key = re.sub(r"^(?:\d+[.)]\s*)", "", (heading or "").strip().lower())
    generic = heading_key in GENERIC_HEADINGS
    reasons = []

    if document is not None and not is_document_applicable_to_asset(document, asset):
        return EvidenceAssessment("EQUIPMENT_MISMATCH", False, False, False, False, False, generic, ("document_not_applicable",))
    if not source_verified:
        return EvidenceAssessment("SOURCE_UNVERIFIED", False, False, False, False, False, generic, ("source_not_verified",))

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
        return EvidenceAssessment("EQUIPMENT_MISMATCH", False, code_match, component_match, False, False, generic, tuple(reasons))

    spec_present = any(re.search(SPEC_UNITS[kind], lower, re.I) for kind in query.specification_types if kind in SPEC_UNITS)
    if query.specification_types and not spec_present:
        reasons.append("requested_specification_value_absent")
    normalized = re.sub(r"\s+", "", lower)
    missing_numbers = [number for number in query.numeric_expressions if number not in normalized]
    if missing_numbers:
        reasons.append("query_number_or_unit_absent")
    action_cues = bool(words & ACTION_WORDS) or bool(re.search(r"\bstep\s*\d+\b", lower))
    procedure_kind_match = any(words.intersection(PROCEDURE_TYPES[kind]) for kind in query.procedure_types)
    procedure_present = bool((query.actions or query.procedure_types) and action_cues
                             and (not query.procedure_types or procedure_kind_match)
                             and (not query.components or component_match))
    if (query.actions or query.procedure_types) and not procedure_present:
        reasons.append("requested_procedure_absent")

    # A code mention or repeated equipment name cannot by itself prove support.
    explicit_request = bool(query.fault_codes or query.specification_types or query.actions or query.procedure_types)
    fact_match = bool(
        not (query.fault_codes and not code_match)
        and not (query.components and not component_match)
        and not missing_numbers
        and (not query.specification_types or spec_present)
        and (not (query.actions or query.procedure_types) or procedure_present)
    )
    answer_bearing = bool(explicit_request and fact_match and not generic and (procedure_present or spec_present))
    if query.actions & SAFETY_ACTIONS and answer_bearing and not action_approved:
        decision = "SAFETY_APPROVAL_REQUIRED"
        reasons.append("physical_action_approval_missing")
    elif answer_bearing:
        decision = "ANSWER_SUPPORTED"
    elif (query.fault_codes and not code_match) or (query.specification_types and not spec_present):
        decision = "ANSWER_NOT_IN_KB"
    elif component_match or code_match or procedure_present or symptom_match:
        decision = "PARTIALLY_SUPPORTED"
    else:
        decision = "RELEVANT" if not query.technical_intent else "ANSWER_NOT_IN_KB"
    return EvidenceAssessment(decision, answer_bearing, code_match, component_match,
                              spec_present, procedure_present, generic, tuple(reasons))


def rank_authorized_candidates(
    query: TechnicalQuery, candidates: list, dense_ranks: dict[int, int],
    lexical_ranks: dict[int, int], *, verified_document_ids: set[int],
    asset: Any = None,
) -> list[dict]:
    """Interpretable fixed ranker; authorization is mandatory before this call."""
    scored = []
    for chunk in candidates:
        if chunk.id not in dense_ranks and chunk.id not in lexical_ranks:
            continue
        assessment = assess_passage(
            query, heading=chunk.heading, body=chunk.text,
            document=chunk.document, asset=asset,
            source_verified=chunk.document_id in verified_document_ids,
        )
        if assessment.decision in {"EQUIPMENT_MISMATCH", "SOURCE_UNVERIFIED"}:
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
        scored.append({"chunk_id": chunk.id, "score": round(score, 6), "assessment": assessment,
                       "dense_rank": dense_rank, "lexical_rank": lexical_rank})
    return sorted(scored, key=lambda r: (-r["score"], r["chunk_id"]))


def verify_answer_evidence(query: TechnicalQuery, ranked: list[dict]) -> dict:
    """Select only source-supported evidence; otherwise abstain with a reason."""
    supported = [r for r in ranked if r["assessment"].decision == "ANSWER_SUPPORTED"]
    if supported:
        return {"decision": "ANSWER_SUPPORTED", "chunk_ids": [r["chunk_id"] for r in supported],
                "answer_coverage": True}
    if any(r["assessment"].decision == "SAFETY_APPROVAL_REQUIRED" for r in ranked):
        return {"decision": "SAFETY_APPROVAL_REQUIRED", "chunk_ids": [], "answer_coverage": False}
    if not ranked or all(r["assessment"].decision == "ANSWER_NOT_IN_KB" for r in ranked):
        return {"decision": "ANSWER_NOT_IN_KB", "chunk_ids": [], "answer_coverage": False}
    return {"decision": "PARTIALLY_SUPPORTED", "chunk_ids": [], "answer_coverage": False}
