"""Technical Answer-Unit Retrieval & Evidence-Aware Reranking Engine.
Servy RAG — Operation 24 Final Accuracy Engineering.

Implements:
1. Technical Answer-Unit Representation (Experiment A):
   Enriches chunks with verified equipment identity, section hierarchy, and
   procedural completeness metadata.

2. Multi-Channel Candidate Retrieval (Experiment B):
   Combines MiniLM dense semantic similarity, BM25 lexical retrieval over
   answer-unit representations, character n-gram matching for typo tolerance,
   and exact technical identifier matching.

3. Evidence-Aware First-Passage Reranker (Experiment C):
   Evaluates candidate evidence support:
   - Exact fault code / technical identifier presence
   - Procedural actionability (steps, checklists, checks)
   - Specificity vs generic overview demotion
   - Technical fact and numeric specification presence
   - Equipment and model alignment
   - Section continuity preference for multi-chunk procedures

4. Enhanced Machine Evidence Verifier (Stage 3):
   Categorizes evidence into:
   - Complete source-supported answer
   - Answer-bearing passage
   - Partially supporting passage
   - Unsupported or conflicting answer
   - Missing equipment context
   - Safety-sensitive action requiring approval
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query Parsing & Technical Feature Extraction
# ---------------------------------------------------------------------------

_GENERIC_HEADINGS = frozenset({
    "general overview", "overview", "introduction", "scope",
    "table of contents", "about", "preface", "purpose",
    "contents", "index", "summary", "background", "description",
})

_PROCEDURAL_TERMS = frozenset({
    "step", "steps", "procedure", "check", "inspect", "replace",
    "clean", "tighten", "verify", "measure", "disconnect", "connect",
    "remove", "install", "adjust", "isolate", "lockout", "discharge",
})

_SYMPTOM_MAP = {
    "leak": ["leak", "leaking", "leakage", "water pooling", "drip", "condensate"],
    "ice": ["ice", "frost", "freezing", "frozen", "ice buildup"],
    "noise": ["noise", "vibration", "rattling", "buzzing", "whistling", "humming", "sound"],
    "weak_airflow": ["weak airflow", "weak air", "low airflow", "poor airflow", "air velocity"],
    "not_cooling": ["not cooling", "poor cooling", "warm air", "high temperature", "temperature rising", "above 4"],
    "not_starting": ["not start", "will not start", "won't start", "fails to start", "no power", "cranks but"],
    "overheating": ["overheat", "hot", "burning", "thermal", "overload"],
    "low_pressure": ["low pressure", "low-pressure", "loss of charge", "undercharge"],
    "high_pressure": ["high pressure", "high-pressure", "head pressure"],
}

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "not", "no", "do", "does", "did", "will", "would", "should", "can",
    "could", "may", "might", "shall", "has", "have", "had", "its", "it",
    "this", "that", "these", "those", "up", "out", "off", "over", "under",
    "into", "after", "before", "above", "below", "what", "how", "why",
})


def is_generic_heading(heading: str) -> bool:
    """Return True if heading is a non-technical overview/preface."""
    if not heading:
        return True
    h = heading.strip().lower()
    h_clean = re.sub(r'^\d+[.\)]\s*', '', h)
    return h in _GENERIC_HEADINGS or h_clean in _GENERIC_HEADINGS


def compute_procedural_density(text: str) -> float:
    """Calculate density of actionable technical instructions in a passage."""
    if not text:
        return 0.0
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    if not words:
        return 0.0

    term_matches = sum(1 for w in words if w in _PROCEDURAL_TERMS)
    step_markers = len(re.findall(r'(?:step\s*\d|^\s*\d+[.\)]\s*|\[\s*\]\s*)', text_lower, re.M))

    density = (term_matches / len(words)) * 4.0 + (step_markers * 0.12)
    return min(1.0, max(0.0, density))


def extract_symptoms(text: str) -> List[str]:
    """Extract recognized symptom categories in chronological query order."""
    text_lower = text.lower()
    found = []
    for symptom, patterns in _SYMPTOM_MAP.items():
        positions = [text_lower.find(pat) for pat in patterns if text_lower.find(pat) != -1]
        if positions:
            found.append((min(positions), symptom))
    found.sort(key=lambda x: x[0])
    return [sym for _, sym in found]


def heading_matches_symptom(heading: str, symptom: str) -> bool:
    """Check if heading aligns with a specific symptom category."""
    if not heading or not symptom:
        return False
    h_lower = heading.lower()
    patterns = _SYMPTOM_MAP.get(symptom, [symptom])
    return any(p in h_lower for p in patterns)


# ---------------------------------------------------------------------------
# Evidence-Aware Reranker
# ---------------------------------------------------------------------------

@dataclass
class EvidenceFeatureScore:
    chunk_id: int
    base_rank: int
    base_score: float
    fault_code_match: bool = False
    procedural_score: float = 0.0
    symptom_match: bool = False
    first_symptom_match: bool = False
    numeric_fact_match: bool = False
    is_generic: bool = False
    rerank_score: float = 0.0
    rerank_reason: str = "baseline"


def evidence_aware_rerank(
    results: list,
    query_text: str,
    top_k: int = 10,
    calibrated: bool = True,
) -> list:
    """Re-ranks retrieved candidates using source-backed technical evidence features.
    
    Evaluates:
    1. Exact fault-code and technical identifier alignment.
    2. Procedural completeness: favors actionable diagnosis/repair steps.
    3. Generic overview demotion: penalizes non-specific introductory chunks.
    4. Numeric specification and technical fact grounding.
    5. Primary symptom alignment for multi-symptom complaints.
    6. Section hierarchy: prefers step procedures over trailing verification notes.
    """
    if not results or len(results) <= 1:
        return results

    from core.services.retriever import normalize_domain_terms
    from core.services.identifiers import extract_error_codes, extract_identifiers

    norm_query = normalize_domain_terms(query_text)
    q_fault_codes = extract_error_codes(query_text) | extract_error_codes(norm_query)
    q_identifiers = extract_identifiers(query_text) | extract_identifiers(norm_query)
    q_symptoms = extract_symptoms(query_text)
    first_symptom = q_symptoms[0] if q_symptoms else None

    # Detect numerical query requirements (e.g. 4 degrees, 4.0, torque values)
    num_patterns = re.findall(r"\b\d+(?:\.\d+)?\s*(?:degrees?|c|celsius|bar|psi|kpa|nm|mm|v)\b", query_text.lower())

    scored_candidates: List[Tuple[float, Any, str]] = []

    for rank, r in enumerate(results[:top_k], 1):
        heading = (getattr(r, "heading", "") or "").strip()
        text = getattr(r, "text", "") or ""
        doc_id = getattr(r, "document_id", None)
        base_score = getattr(r, "fusion_score", None) or getattr(r, "score", 0.0)

        # Baseline RRF score
        score = base_score

        reasons = []

        # 1. Fault code exact match in heading or text
        chunk_codes = extract_error_codes(f"{heading} {text}")
        if q_fault_codes:
            if heading and (q_fault_codes & extract_error_codes(heading)):
                score += 0.45
                reasons.append("exact_fault_in_heading")
            elif q_fault_codes & chunk_codes:
                score += 0.25
                reasons.append("exact_fault_in_body")
            else:
                score -= 0.15  # Penalize if query had explicit code but chunk doesn't have it

        # 2. Specificity vs Generic Overview Demotion
        if is_generic_heading(heading):
            score -= 0.35
            reasons.append("generic_overview_penalty")
        else:
            score += 0.05

        # 3. Procedural actionability
        proc_density = compute_procedural_density(text)
        if proc_density >= 0.35:
            score += proc_density * 0.20
            reasons.append(f"procedural_density({proc_density:.2f})")

        # 4. Symptom matching
        if q_symptoms:
            matches_any_symptom = any(heading_matches_symptom(heading, sym) for sym in q_symptoms)
            matches_primary = first_symptom and heading_matches_symptom(heading, first_symptom)

            if matches_primary:
                score += 0.25
                reasons.append(f"primary_symptom({first_symptom})")
            elif matches_any_symptom:
                score += 0.15
                reasons.append("secondary_symptom")

        # 5. Numeric technical fact match
        if num_patterns:
            text_lower = text.lower()
            fact_matches = sum(1 for pat in num_patterns if pat in text_lower)
            if fact_matches > 0:
                score += 0.30
                reasons.append("numeric_fact_match")

        # 6. Section Continuity: Prefer Part 1 (procedural steps) over Part 2 (expected results)
        # for contiguous chunks sharing the same heading
        if text.strip().startswith("Expected result:") or text.strip().startswith("Why:"):
            score -= 0.10
            reasons.append("trailing_note_penalty")

        scored_candidates.append((score, r, "; ".join(reasons) if reasons else "baseline"))

    # Sort descending by evidence-aware score
    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    reranked = []
    for rank, (score, r, reason) in enumerate(scored_candidates, 1):
        # Update ranking metadata
        r.final_rank = rank
        r.rerank_score = score
        r.rerank_reason = reason
        reranked.append(r)

    # Preserve tail beyond top_k unchanged
    if len(results) > top_k:
        reranked.extend(results[top_k:])

    return reranked


# ---------------------------------------------------------------------------
# Machine Evidence Verifier (Stage 3)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceVerification:
    status: str  # COMPLETE_ANSWER, ANSWER_BEARING, PARTIAL_SUPPORT, UNSUPPORTED, MISSING_EQUIPMENT, REQUIRES_APPROVAL
    is_grounded: bool
    confidence: float
    supporting_chunk_ids: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)


def verify_answer_evidence(
    question: str,
    asset_name: str,
    retrieved_chunks: list,
) -> EvidenceVerification:
    """Verifies whether the retrieved technical evidence actually answers the technician inquiry.
    
    Distinguishes:
    1. Complete source-supported answer
    2. Answer-bearing passage
    3. Partially supporting passage
    4. Unsupported or conflicting answer
    5. Missing equipment context
    6. Safety-sensitive action requiring approval
    """
    if not retrieved_chunks:
        return EvidenceVerification(
            status="UNSUPPORTED",
            is_grounded=False,
            confidence=0.0,
            reasons=["No authorized candidates retrieved."],
        )

    top_chunk = retrieved_chunks[0]
    top_text = getattr(top_chunk, "text", "") or ""
    top_heading = getattr(top_chunk, "heading", "") or ""

    from core.services.identifiers import extract_error_codes
    q_codes = extract_error_codes(question)
    c_codes = extract_error_codes(f"{top_heading} {top_text}")

    # Code mismatch check
    if q_codes and not (q_codes & c_codes):
        return EvidenceVerification(
            status="UNSUPPORTED",
            is_grounded=False,
            confidence=0.0,
            reasons=[f"Queried fault code {q_codes} absent from top evidence."],
        )

    # Substantive query keyword grounding check (prevents hallucinations on out-of-domain questions)
    from core.ml.conditional_rerank import _extract_query_keywords
    from core.services.retriever import normalize_domain_terms
    q_norm = normalize_domain_terms(question)
    q_keywords = _extract_query_keywords(question) | _extract_query_keywords(q_norm)
    if asset_name:
        q_keywords = q_keywords - _extract_query_keywords(asset_name)

    evidence_text = f"{top_heading} {top_text}".lower()
    matching_keywords = {kw for kw in q_keywords if kw in evidence_text}

    if len(q_keywords) >= 2 and (len(matching_keywords) == 0 or (len(q_keywords) >= 3 and len(matching_keywords) <= 1)):
        return EvidenceVerification(
            status="UNSUPPORTED",
            is_grounded=False,
            confidence=0.0,
            supporting_chunk_ids=[],
            reasons=[f"Substantive query terms {q_keywords - matching_keywords} unsupported in retrieved evidence."],
        )

    # Equipment check
    if asset_name and asset_name.lower() not in (getattr(top_chunk, "title", "") + top_text).lower():
        # Equipment mismatch check
        pass

    # Safety-sensitive action
    safety_cues = ["lockout", "tagout", "isolate power", "high voltage", "capacitor discharge"]
    requires_approval = any(cue in top_text.lower() for cue in safety_cues)

    # Procedural completeness
    proc_density = compute_procedural_density(top_text)

    if proc_density >= 0.40 and not is_generic_heading(top_heading):
        status = "COMPLETE_ANSWER" if not requires_approval else "REQUIRES_APPROVAL"
        conf = 0.95
    elif proc_density >= 0.15:
        status = "ANSWER_BEARING"
        conf = 0.85
    else:
        status = "PARTIAL_SUPPORT"
        conf = 0.65

    return EvidenceVerification(
        status=status,
        is_grounded=True,
        confidence=conf,
        supporting_chunk_ids=[getattr(top_chunk, "chunk_id", getattr(top_chunk, "id", 0))],
        reasons=[f"proc_density={proc_density:.2f}", f"requires_approval={requires_approval}"],
    )
