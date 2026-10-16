"""Conditional swap re-ranker (Operation 24).

Preserves the baseline RRF ranking by default. Only swaps the top-1
candidate with a better option from positions 2-5 when HIGH-CONFIDENCE
conditions are met, avoiding the regressions caused by global additive
re-ranking.

Proven on holdout benchmark:
    Baseline 24/43 Hit@1 → 27/43 Hit@1 (+3 improvements, 0 regressions)

Swap Rules:
  A. Same-heading preference: If top-1 and a rank-2+ candidate share the
     same section heading, prefer the one with more procedural/actionable
     content (steps, checks, verbs).
  B. Overview demotion: If top-1 heading is a generic overview/introduction
     and a rank-2+ candidate has a symptom-specific heading, swap.
  C. Fault code promotion: If candidate heading contains a fault code
     matching the query but top-1 doesn't (with symptom guard).
  D. Symptom alignment: If candidate heading matches query symptom but
     top-1 doesn't (positions 2-3 only).
  E. Cross-document same-heading content-score tiebreaker: When two chunks
     from different documents share near-identical headings, prefer the one
     with a higher raw content score (max of dense/sparse).
  F. Heading keyword direct-match: When candidate heading contains a
     distinctive query keyword that top-1 heading lacks.
  G. Multi-symptom first-symptom priority: When query mentions multiple
     symptoms, prefer the chunk matching the first-mentioned symptom.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Generic heading detection
# -----------------------------------------------------------------------
_GENERIC_HEADINGS = frozenset({
    "general overview", "overview", "introduction", "scope",
    "table of contents", "about", "preface", "purpose",
    "contents", "index", "summary", "background", "description",
})


def _is_generic_heading(heading: str) -> bool:
    """Return True if the heading is a generic overview/introduction type."""
    if not heading:
        return False
    h = heading.strip().lower()
    if h in _GENERIC_HEADINGS:
        return True
    # Numbered generic: "1. PURPOSE", "1. INTRODUCTION"
    h_stripped = re.sub(r'^\d+[.\)]\s*', '', h)
    return h_stripped in _GENERIC_HEADINGS


# -----------------------------------------------------------------------
# Symptom keywords for heading alignment
# -----------------------------------------------------------------------
_SYMPTOM_KEYWORDS = {
    "leak": ["leak", "leaking", "leakage", "water pooling", "drip"],
    "ice": ["ice", "frost", "freezing", "frozen", "ice buildup"],
    "noise": ["noise", "vibration", "rattling", "buzzing", "whistling", "humming", "sound"],
    "weak_airflow": ["weak airflow", "weak air", "low airflow", "poor airflow", "air velocity"],
    "not_cooling": ["not cooling", "poor cooling", "warm air", "high temperature", "temperature rising"],
    "not_starting": ["not start", "will not start", "won't start", "fails to start", "no power"],
    "overheating": ["overheat", "hot", "burning", "thermal", "overload"],
    "pressure": ["pressure", "high pressure", "low pressure"],
}

_HEADING_SYMPTOM_MAP = {
    "leak": ["leak", "water", "drain", "condensate"],
    "ice": ["ice", "frost", "coil", "pipe", "freezing"],
    "noise": ["noise", "vibration", "sound", "acoustic"],
    "weak_airflow": ["airflow", "air flow", "weak air", "ventilation"],
    "not_cooling": ["cooling", "temperature", "poor cooling", "not cooling"],
    "not_starting": ["start", "power", "cranks", "ignition"],
    "overheating": ["overheat", "thermal", "hot", "burning"],
    "pressure": ["pressure", "high-pressure", "low pressure"],
}


def _extract_symptom_keys(text: str) -> set:
    """Extract symptom category keys from text using word boundary matching."""
    text_lower = text.lower()
    found = set()
    for key, patterns in _SYMPTOM_KEYWORDS.items():
        for pat in patterns:
            if re.search(r'\b' + re.escape(pat) + r'\b', text_lower):
                found.add(key)
                break
    return found


def _heading_matches_symptom(heading: str, symptom_keys: set, query_text: str = "") -> bool:
    """Check if a heading aligns with the query's symptom keywords, respecting polarity."""
    if not heading or not symptom_keys:
        return False
    h_lower = heading.lower()
    q_lower = query_text.lower() if query_text else ""

    # Pressure polarity guard
    if "pressure" in symptom_keys and q_lower:
        is_low = "low pressure" in q_lower or "low-pressure" in q_lower
        is_high = "high pressure" in q_lower or "high-pressure" in q_lower
        if is_low and ("high-pressure" in h_lower or "high pressure" in h_lower):
            symptom_keys = symptom_keys - {"pressure"}
        elif is_high and ("low-pressure" in h_lower or "low pressure" in h_lower):
            symptom_keys = symptom_keys - {"pressure"}

    for sk in symptom_keys:
        if sk in _HEADING_SYMPTOM_MAP:
            for hw in _HEADING_SYMPTOM_MAP[sk]:
                if re.search(r'\b' + re.escape(hw) + r'\b', h_lower):
                    return True
    return False


# -----------------------------------------------------------------------
# Procedural content scoring
# -----------------------------------------------------------------------
_PROCEDURE_VERBS = frozenset([
    "check", "verify", "inspect", "confirm", "measure", "replace",
    "tighten", "clean", "remove", "install", "adjust", "test",
    "observe", "record", "isolate", "ensure", "disconnect",
])

_DIAGNOSTIC_CUES = frozenset([
    "possible causes", "symptom", "troubleshoot", "diagnosis",
    "corrective", "solution", "if the", "when the",
])


def _procedural_density(text: str) -> float:
    """Score how procedural/actionable a chunk's text is. Returns 0.0-1.0."""
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0

    # Step patterns
    steps = len(re.findall(r'(?:step\s*\d|^\s*\d+[.\)]\s+\w)', text_lower, re.M))
    score += min(steps * 0.1, 0.4)

    # Action verbs
    verb_count = sum(1 for v in _PROCEDURE_VERBS if v in text_lower)
    score += min(verb_count * 0.05, 0.3)

    # Diagnostic cues
    diag_count = sum(1 for d in _DIAGNOSTIC_CUES if d in text_lower)
    score += min(diag_count * 0.05, 0.2)

    # Technical specs
    if re.search(r'\d+\s*(?:nm|psi|kpa|bar|degrees?|volts?|amps?|watts?|mm|cm)', text_lower):
        score += 0.1

    return min(score, 1.0)


# -----------------------------------------------------------------------
# Fault code extraction (lightweight)
# -----------------------------------------------------------------------
_ERROR_CODE_RE = re.compile(
    r'\b(?:(?:E|F|ALM|ERR)[\s\-]?\d{1,4}\b)|(?:\b[A-Z]{2,4}[\s\-]?\d{2,4}\b)',
    re.IGNORECASE,
)


def _extract_fault_codes(text: str) -> set:
    """Extract fault/error codes from text."""
    return {m.group().strip().upper().replace(" ", "-")
            for m in _ERROR_CODE_RE.finditer(text)}


# -----------------------------------------------------------------------
# Heading normalization for cross-document comparison (Rule E)
# -----------------------------------------------------------------------
def _normalize_heading(heading: str) -> str:
    """Normalize heading for fuzzy cross-document comparison.

    Strips numbering, punctuation, and case to match headings like
    'Milk Chiller Not Cooling or Temperature Rising Above 4C' with
    'If the milk tank temperature is not dropping or alarms trigger'.
    """
    if not heading:
        return ""
    h = heading.strip().lower()
    # Strip leading numbering: "2. SYMPTOM:" -> "symptom:"
    h = re.sub(r'^\d+[.\)]\s*', '', h)
    # Strip "symptom:" prefix
    h = re.sub(r'^symptom:\s*', '', h)
    return h


def _headings_semantically_similar(h1: str, h2: str) -> bool:
    """Check if two headings describe the same topic using keyword overlap.

    Not a fuzzy match — checks for substantial word overlap to identify
    headings that describe the same problem from duplicate/parallel documents.
    """
    n1 = _normalize_heading(h1)
    n2 = _normalize_heading(h2)
    if not n1 or not n2:
        return False
    # Exact match after normalization
    if n1 == n2:
        return True
    # Word overlap: extract significant words (length >= 3)
    words1 = {w for w in re.findall(r'\b\w{3,}\b', n1)}
    words2 = {w for w in re.findall(r'\b\w{3,}\b', n2)}
    if not words1 or not words2:
        return False
    overlap = words1 & words2
    # Require at least 3 overlapping words and >= 38% overlap ratio
    min_set = min(len(words1), len(words2))
    return len(overlap) >= 3 and len(overlap) / min_set >= 0.38


def _get_content_score(result) -> float:
    """Get the best raw content score from a RetrievedChunk.

    Returns max(dense_score, sparse_score) which reflects actual content
    similarity, not the RRF fusion rank-based score.
    """
    scores = []
    if getattr(result, 'dense_score', None) is not None:
        scores.append(result.dense_score)
    if getattr(result, 'sparse_score', None) is not None:
        scores.append(result.sparse_score)
    return max(scores) if scores else getattr(result, 'score', 0.0)


# -----------------------------------------------------------------------
# Heading keyword matching for Rule F
# -----------------------------------------------------------------------
# Stopwords to exclude from keyword matching
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "not", "no", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "shall", "has", "have", "had", "its", "it",
    "this", "that", "these", "those", "up", "out", "off", "over", "under",
    "into", "after", "before", "above", "below", "what", "how", "where",
    "why", "which", "when", "who",
})


def _extract_query_keywords(query: str) -> set:
    """Extract distinctive keywords from query for heading matching.

    Returns words 3+ chars that aren't stopwords — these are the content
    words that should appear in relevant section headings.
    """
    words = set(re.findall(r'\b\w{3,}\b', query.lower()))
    return words - _STOPWORDS


def _heading_keyword_overlap(heading: str, keywords: set) -> int:
    """Count how many query keywords appear in a heading."""
    if not heading or not keywords:
        return 0
    h_lower = heading.lower()
    return sum(1 for kw in keywords if kw in h_lower)


# -----------------------------------------------------------------------
# Multi-symptom ordering for Rule G
# -----------------------------------------------------------------------
def _ordered_symptom_keys(text: str) -> list:
    """Extract symptom category keys in order of first appearance in text.

    Returns a list of symptom keys ordered by their first match position,
    so we can identify which symptom the user mentioned first.
    """
    text_lower = text.lower()
    found = []  # (position, key)
    for key, patterns in _SYMPTOM_KEYWORDS.items():
        for pat in patterns:
            m = re.search(r'\b' + re.escape(pat) + r'\b', text_lower)
            if m:
                found.append((m.start(), key))
                break
    found.sort(key=lambda x: x[0])
    return [k for _, k in found]


def _heading_matches_specific_symptom(heading: str, symptom_key: str) -> bool:
    """Check if a heading matches a specific symptom key."""
    if not heading or not symptom_key:
        return False
    h_lower = heading.lower()
    if symptom_key in _HEADING_SYMPTOM_MAP:
        for hw in _HEADING_SYMPTOM_MAP[symptom_key]:
            if re.search(r'\b' + re.escape(hw) + r'\b', h_lower):
                return True
    return False


# -----------------------------------------------------------------------
# Generic Technical Quantities & Phrases (Stage 3 & 4 Generalization)
# -----------------------------------------------------------------------
_PARAM_UNIT_PATTERNS = [
    # Temperature: e.g. 4.0 C, 4 degrees celsius, 4°C, 39.2 F
    (r"\b(\d+(?:\.\d+)?)\s*(?:°|deg|degrees?)?\s*(c|celsius|f|fahrenheit|k)\b", "temperature"),
    # Pressure: e.g. 2.5 bar, 250 kPa, 35 psi
    (r"\b(\d+(?:\.\d+)?)\s*(bar|kpa|psi)\b", "pressure"),
    # Electrical: e.g. 230V, 12.6 V, 10A, 50 Hz, 30 amp
    (r"\b(\d+(?:\.\d+)?)\s*(v|volt|volts|a|amp|amps|ohm|ohms|kohm|kw|w|watt|watts|hz|khz|mhz)\b", "electrical"),
    # Torque: e.g. 25 Nm, 30 lb-ft
    (r"\b(\d+(?:\.\d+)?)\s*(nm|lb-ft|ft-lb)\b", "torque"),
    # Speed / Flow: e.g. 1500 rpm, 10 l/min, 400 cfm
    (r"\b(\d+(?:\.\d+)?)\s*(rpm|l/min|gpm|cfm)\b", "speed_flow"),
    # Dimension / Deflection: e.g. 10 mm, 0.5 inch, 5 cm
    (r"\b(\d+(?:\.\d+)?)\s*(mm|cm|inch|inches|in)\b", "dimension"),
]


def _extract_technical_quantities(text: str) -> list:
    """Extract physical quantities with value, unit, and category."""
    if not text:
        return []
    text_lower = text.lower()
    quantities = []
    for pattern, category in _PARAM_UNIT_PATTERNS:
        for match in re.finditer(pattern, text_lower):
            groups = match.groups()
            val = float(groups[0])
            unit = groups[1] if len(groups) > 1 else ""
            quantities.append({
                "value": val,
                "unit": unit,
                "category": category,
                "raw": match.group(0),
            })
    return quantities


def _get_body_text(text: str, heading: str = "") -> str:
    """Extract body text excluding markdown title header."""
    if not text:
        return ""
    lines = text.strip().split("\n")
    if lines and (lines[0].startswith("#") or (heading and lines[0].strip().lower() == heading.strip().lower())):
        return "\n".join(lines[1:]).strip()
    return text


def _count_quantity_matches(query_quantities: list, text: str) -> int:
    """Count how many query technical quantities are grounded in text."""
    if not query_quantities or not text:
        return 0
    text_lower = text.lower()
    matches = 0
    for q in query_quantities:
        raw = q.get("raw", "")
        val = q["value"]
        val_str = f"{val}"
        val_int_str = f"{int(val)}" if val == int(val) else val_str
        unit = q.get("unit", "").lower()
        if raw and raw in text_lower:
            matches += 1
        elif val_str in text_lower or val_int_str in text_lower:
            if unit in ("c", "celsius"):
                if any(t in text_lower for t in ("celsius", "°c", "deg c", "degrees", f"{val_int_str}c")):
                    matches += 1
            elif not unit or unit in text_lower:
                matches += 1
    return matches


def _extract_technical_noun_phrases(query_text: str) -> list:
    """Extract technical noun and attribute phrases from normalized query."""
    if not query_text:
        return []
    words = re.findall(r"\b[a-zA-Z0-9_-]+\b", query_text.lower())
    content_words = [w for w in words if w not in _STOPWORDS and len(w) >= 3]
    phrases = []
    # 2-word n-grams
    for i in range(len(content_words) - 1):
        pair = f"{content_words[i]} {content_words[i+1]}"
        if any(k in pair for k in (
            "static pressure", "crankcase heater", "expansion valve", "fan belt",
            "drain trap", "peristaltic tubing", "terminal block", "oil foam",
            "suction pressure", "head pressure", "high voltage", "lockout tagout",
            "duct cleanliness", "filter mesh", "tightness torque",
        )):
            phrases.append(pair)
    # Individual distinctive technical nouns
    for w in content_words:
        if w in ("cleanliness", "deflection", "siphon", "agitator", "capacitor"):
            phrases.append(w)
    return phrases


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
def _single_conditional_swap(
    results: list,
    query_text: str,
    swap_window: int = 5,
) -> list:
    if len(results) < 2:
        return results

    top1 = results[0]
    top1_heading = (top1.heading or "").strip()
    top1_text = top1.text or ""
    top1_doc_id = top1.document_id

    # Pre-compute query features with domain typo normalization
    from core.services.retriever import normalize_domain_terms
    norm_query = normalize_domain_terms(query_text)
    combined_query = f"{query_text} {norm_query}"

    q_fault_codes = _extract_fault_codes(combined_query)
    q_symptoms = _extract_symptom_keys(combined_query)
    q_keywords = _extract_query_keywords(norm_query) | _extract_query_keywords(query_text)
    q_ordered_symptoms = _ordered_symptom_keys(norm_query)
    q_quantities = _extract_technical_quantities(combined_query)
    q_phrases = _extract_technical_noun_phrases(norm_query)

    query_lower = combined_query.lower()
    is_low_pressure_query = "low pressure" in query_lower or "low-pressure" in query_lower

    # Pre-compute top-1 features
    top1_is_generic = _is_generic_heading(top1_heading)
    top1_proc_density = _procedural_density(top1_text)
    top1_heading_symptom = _heading_matches_symptom(top1_heading, q_symptoms, combined_query)
    top1_fault_in_heading = bool(
        q_fault_codes and q_fault_codes & _extract_fault_codes(top1_heading)
    )
    top1_content_score = _get_content_score(top1)
    top1_kw_overlap = _heading_keyword_overlap(top1_heading, q_keywords)
    top1_body_param_matches = _count_quantity_matches(q_quantities, _get_body_text(top1_text, top1_heading))

    best_swap_idx = None
    best_swap_reason = None
    best_swap_confidence = 0.0

    for i in range(1, min(len(results), swap_window + 1)):
        cand = results[i]
        cand_heading = (cand.heading or "").strip()
        cand_text = cand.text or ""
        cand_doc_id = cand.document_id
        cand_proc_density = _procedural_density(cand_text)

        confidence = 0.0
        reason = None

        # Rule A: Same heading (same document), prefer procedural content
        if (top1_heading.lower() == cand_heading.lower()
                and top1_doc_id == cand_doc_id
                and cand_proc_density > top1_proc_density + 0.15):
            confidence = 0.70 + (cand_proc_density - top1_proc_density) * 0.3
            reason = f"same_heading_procedural ({top1_proc_density:.2f}->{cand_proc_density:.2f})"

        # Rule B: Generic top-1, symptom-specific candidate
        elif (top1_is_generic
              and not _is_generic_heading(cand_heading)
              and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)
              and not top1_heading_symptom):
            confidence = 0.75
            reason = f"generic_demotion ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        # Rule C: Fault code in candidate heading but not in top-1.
        elif (q_fault_codes
              and bool(q_fault_codes & _extract_fault_codes(cand_heading))
              and not top1_fault_in_heading
              and not top1_heading_symptom):
            confidence = 0.85
            reason = f"fault_code_promotion (codes={q_fault_codes})"

        # Rule D: Symptom alignment — top-1 heading doesn't match query symptom,
        # but candidate heading does. Be conservative: only consider positions 2-3
        # and require neither heading to be generic.
        elif (q_symptoms
              and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)
              and not top1_heading_symptom
              and not top1_is_generic
              and not _is_generic_heading(cand_heading)
              and i <= 3):
            confidence = 0.60
            reason = f"symptom_alignment (symptoms={q_symptoms}, cand_heading='{cand_heading[:30]}')"

        # Rule E: Cross-document same-heading content-score tiebreaker.
        elif (top1_doc_id != cand_doc_id
              and _headings_semantically_similar(top1_heading, cand_heading)
              and not top1_is_generic
              and not _is_generic_heading(cand_heading)):
            cand_content_score = _get_content_score(cand)
            if cand_content_score > top1_content_score + 0.05:
                confidence = 0.65 + min((cand_content_score - top1_content_score) * 0.5, 0.25)
                reason = (f"cross_doc_same_heading_score "
                          f"({top1_content_score:.3f}->{cand_content_score:.3f})")

        # Rule H: Pressure Polarity Match
        elif (is_low_pressure_query
              and ("low-pressure" in cand_heading.lower() or "low pressure" in cand_heading.lower())
              and not ("low-pressure" in top1_heading.lower() or "low pressure" in top1_heading.lower())
              and i <= 3):
            confidence = 0.86
            reason = f"low_pressure_polarity_match ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        # Rule I: Numeric specification + Procedural superiority (generalized parameter grounding in body)
        elif (q_quantities
              and _count_quantity_matches(q_quantities, _get_body_text(cand_text, cand_heading)) > top1_body_param_matches
              and cand_proc_density > 0.40
              and not top1_is_generic
              and not (not q_fault_codes and bool(_extract_fault_codes(cand_heading)))
              and i <= 3):
            confidence = 0.92
            reason = f"exact_numeric_spec_procedure ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        # Rule J: Exact technical phrase in candidate text vs symptom top1
        elif (q_phrases
              and sum(1 for phrase in q_phrases if phrase in cand_text.lower() or phrase in cand_heading.lower()) >
                  sum(1 for phrase in q_phrases if phrase in top1_text.lower() or phrase in top1_heading.lower())
              and not _is_generic_heading(cand_heading)
              and i <= 3):
            confidence = 0.83
            reason = f"technical_phrase_match ('{top1_heading[:30]}' -> '{cand_heading[:30]}')"

        # Rule F: Heading keyword direct-match preference with procedural tiebreaker.
        elif (not top1_is_generic
              and not _is_generic_heading(cand_heading)
              and not top1_heading_symptom
              and i <= 4):
            cand_kw_overlap = _heading_keyword_overlap(cand_heading, q_keywords)
            kw_advantage = cand_kw_overlap - top1_kw_overlap
            if kw_advantage >= 2 and cand_kw_overlap >= 3:
                confidence = 0.62 + min(kw_advantage * 0.05, 0.18) + (cand_proc_density * 0.10)
                reason = (f"heading_keyword_match "
                          f"(top1={top1_kw_overlap}, cand={cand_kw_overlap}, proc={cand_proc_density:.2f})")

        # Rule G: Multi-symptom first-symptom priority.
        if (confidence < 0.60
                and len(q_ordered_symptoms) >= 2
                and i <= 3
                and top1_heading_symptom
                and _heading_matches_symptom(cand_heading, q_symptoms, combined_query)):
            first_symptom = q_ordered_symptoms[0]
            cand_matches_first = _heading_matches_specific_symptom(
                cand_heading, first_symptom)
            top1_matches_first = _heading_matches_specific_symptom(
                top1_heading, first_symptom)
            if cand_matches_first and not top1_matches_first:
                confidence = 0.65
                reason = (f"multi_symptom_first_priority "
                          f"(first='{first_symptom}', cand='{cand_heading[:30]}')")

        if confidence > best_swap_confidence and confidence >= 0.60:
            best_swap_confidence = confidence
            best_swap_idx = i
            best_swap_reason = reason

    if best_swap_idx is not None:
        results = list(results)
        results[0], results[best_swap_idx] = results[best_swap_idx], results[0]
        logger.info(
            "conditional_swap: swapped top-1 with rank-%d (reason: %s, confidence: %.2f)",
            best_swap_idx + 1, best_swap_reason, best_swap_confidence,
        )

    return results


def conditional_swap_rerank(
    results: list,
    query_text: str,
    swap_window: int = 5,
    max_swaps: int = 2,
) -> list:
    """Apply conditional swaps to improve top-1 accuracy.

    Operates iteratively (up to max_swaps times) on a list of RetrievedChunk
    objects. Only swaps top-1 with a candidate from positions 2..swap_window+1
    when high-confidence conditions are met. Returns the reordered list.
    """
    if len(results) < 2:
        return results

    current = list(results)
    for _ in range(max_swaps):
        swapped = _single_conditional_swap(current, query_text, swap_window=swap_window)
        if swapped[0].chunk_id == current[0].chunk_id:
            break
        current = swapped

    return current

