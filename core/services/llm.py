import re
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
import requests
from urllib.parse import urlparse
from django.conf import settings

_ollama_last_checked = 0.0
_ollama_is_available = None


def is_ollama_service_online() -> bool:
    """Fast check whether local Ollama daemon is reachable on loopback without blocking the serving pipeline.
    Memoizes failure for 30s so offline local daemons fail in 0ms instead of 2000ms.
    """
    global _ollama_last_checked, _ollama_is_available
    now = time.time()
    if _ollama_is_available is not None and (now - _ollama_last_checked) < 30.0:
        return _ollama_is_available

    import socket
    parsed = urlparse(settings.SERVY_OLLAMA_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 11434
    try:
        # Rapid 50ms socket connect probe to check daemon liveness
        with socket.create_connection((host, port), timeout=0.05):
            _ollama_is_available = True
    except (OSError, socket.timeout):
        _ollama_is_available = False

    _ollama_last_checked = now
    return _ollama_is_available


def _validate_llm_endpoint():
    url = settings.SERVY_OLLAMA_URL
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Unsupported local LLM URL scheme.")
    if not settings.SERVY_ALLOW_REMOTE_LLM:
        # Default-deny remote model endpoints so customer/service data cannot be
        # sent off-machine by a misconfigured environment variable or redirect.
        if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Remote LLM endpoints are disabled. Use a loopback Ollama URL or explicitly opt in.")
    return url


def ollama_generate(prompt):
    url = _validate_llm_endpoint()
    session = requests.Session()
    session.trust_env = False  # do not route sensitive local prompts through HTTP proxy env vars
    response = session.post(
        url,
        json={"model": settings.SERVY_OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=settings.SERVY_OLLAMA_TIMEOUT,
        allow_redirects=False,
    )
    if 300 <= response.status_code < 400:
        raise RuntimeError("Local LLM endpoint attempted an HTTP redirect; blocked for privacy.")
    response.raise_for_status()
    if len(response.content) > 2 * 1024 * 1024:
        raise RuntimeError("Local LLM response exceeded the 2 MB safety limit.")
    return response.json().get("response", "").strip()


_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "are", "was", "were",
    "can", "could", "should", "would", "does", "did", "have", "has", "had",
    "about", "into", "some", "any", "not", "all", "what", "how", "why", "when",
    "where", "who", "which", "there", "their", "they", "them", "then", "than",
    "been", "being", "will", "shall", "our", "your", "its", "also", "very",
}

INSUFFICIENT_EVIDENCE_MESSAGE = "I couldn't find enough approved information in the Knowledge Base to safely answer this issue."


class EvidenceSufficiencyResult:
    def __init__(
        self,
        is_sufficient: bool,
        reason: str,
        candidate_relevance: bool = False,
        answerability: bool = False,
        claim_support: bool = False,
        equipment_applicable: bool = False,
        source_trust: bool = False,
        procedure_approved: bool = False,
        grounded_chunks: list | None = None,
    ):
        self.is_sufficient = is_sufficient
        self.reason = reason
        self.candidate_relevance = candidate_relevance
        self.answerability = answerability
        self.claim_support = claim_support
        self.equipment_applicable = equipment_applicable
        self.source_trust = source_trust
        self.procedure_approved = procedure_approved
        self.grounded_chunks = grounded_chunks or []

    def __iter__(self):
        yield self.is_sufficient
        yield self.reason

    def __bool__(self):
        return bool(self.is_sufficient)

    def __getitem__(self, idx):
        if idx == 0:
            return self.is_sufficient
        elif idx == 1:
            return self.reason
        raise IndexError("Index out of range (tuple view supports indices 0 and 1)")

    def __repr__(self):
        return f"EvidenceSufficiencyResult(is_sufficient={self.is_sufficient}, reason='{self.reason}')"


@dataclass
class TechnicalQuantity:
    value: float
    unit: str  # canonical: 'celsius', 'fahrenheit', 'bar', 'psi', 'kpa', 'volt', 'amp', 'rpm', 'nm', 'hz', 'standalone'
    dimension: str  # 'temperature', 'pressure', 'voltage', 'current', 'torque', 'frequency', 'threshold'
    raw: str


def extract_typed_quantities(text: str) -> list[TechnicalQuantity]:
    """Extract typed physical quantities and dimensional measurements from text.

    Supports:
    - Temperature: 4C, 4.0 °C, 4 degrees celsius, 40 fahrenheit
    - Pressure: 0.4 bar, 250 kpa, 30 psi
    - Electrical: 12V, 240 V, 10a, 50hz, 15kw
    - Torque: 1.8 nm, 25 nm
    - Speed: 1500 rpm
    - Standalone thresholds: multi-digit numbers (415, 230), excluding error/model codes
    """
    if not text:
        return []
    text_lower = text.lower()
    quantities = []

    # 1. Temperature: 4C, 4.0 C, 4 degrees celsius, 4 °C, 4 fahrenheit, etc.
    temp_pat = r"(?<![a-z0-9_-])(\d+(?:\.\d+)?)\s*(?:°|deg|degrees?)?\s*(celsius|fahrenheit|c\b|f\b)"
    for m in re.finditer(temp_pat, text_lower):
        val = float(m.group(1))
        u_raw = m.group(2)
        unit = "celsius" if u_raw.startswith("c") else "fahrenheit"
        quantities.append(TechnicalQuantity(val, unit, "temperature", m.group(0)))

    # 2. Pressure: 1.5 bar, 250 kpa, 30 psi
    press_pat = r"(?<![a-z0-9_-])(\d+(?:\.\d+)?)\s*(bar|psi|kpa)\b"
    for m in re.finditer(press_pat, text_lower):
        val = float(m.group(1))
        unit = m.group(2)
        quantities.append(TechnicalQuantity(val, unit, "pressure", m.group(0)))

    # 3. Electrical: 12V, 12.6 V, 415v, 230 v, 10a, 50hz, 15kw, 2hp
    elec_pat = r"(?<![a-z0-9_-])(\d+(?:\.\d+)?)\s*(v|volt|volts|a|amp|amps|hz|khz|mhz|kw|hp)\b"
    for m in re.finditer(elec_pat, text_lower):
        val = float(m.group(1))
        u_raw = m.group(2)
        unit = "volt" if u_raw.startswith("v") else ("amp" if u_raw.startswith("a") else u_raw)
        quantities.append(TechnicalQuantity(val, unit, "electrical", m.group(0)))

    # 4. Torque: 1.8 nm, 25 nm
    torque_pat = r"(?<![a-z0-9_-])(\d+(?:\.\d+)?)\s*(nm|lb-ft|ft-lb)\b"
    for m in re.finditer(torque_pat, text_lower):
        val = float(m.group(1))
        quantities.append(TechnicalQuantity(val, m.group(2), "torque", m.group(0)))

    # 5. Speed: 1500 rpm, 30 rpm
    speed_pat = r"(?<![a-z0-9_-])(\d+(?:\.\d+)?)\s*(rpm)\b"
    for m in re.finditer(speed_pat, text_lower):
        val = float(m.group(1))
        quantities.append(TechnicalQuantity(val, "rpm", "speed", m.group(0)))

    # 6. Standalone multi-digit numbers (415, 230, 1500), but strictly excluding
    # error codes (E-04, F-07, ALM-01) and model numbers (MC-500, KTM-33, 500-X)
    for m in re.finditer(r"\b(\d{2,})\b", text_lower):
        num_str = m.group(1)
        start = m.start()
        end = m.end()
        prefix_start = max(0, start - 6)
        prefix = text_lower[prefix_start:start]
        # Exclude error codes
        if re.search(r"(?:^|[^a-z])(?:e|f|err|error|alm|alarm)\s*[-\u2010-\u2015]?\s*$", prefix):
            continue
        # Exclude model codes
        if re.search(r"[a-z]{1,6}\s*[-\u2010-\u2015]\s*$", prefix):
            continue
        suffix = text_lower[end:end + 3]
        if re.match(r"\s*[-\u2010-\u2015]\s*[a-z]", suffix):
            continue
        # If this number was already captured with a unit, skip standalone
        val = float(num_str)
        if not any(abs(q.value - val) < 1e-4 and q.raw in text_lower[max(0, start-2):min(len(text_lower), end+10)] for q in quantities):
            quantities.append(TechnicalQuantity(val, "standalone", "threshold", num_str))

    return quantities


def verify_quantity_support(q_qty: TechnicalQuantity, chunk_quantities: list[TechnicalQuantity]) -> bool:
    """Strictly verify that a queried quantity is supported by a chunk's quantities."""
    for c in chunk_quantities:
        if q_qty.dimension == c.dimension:
            if q_qty.unit == c.unit and abs(q_qty.value - c.value) < 1e-4:
                return True
        elif q_qty.unit == "standalone":
            if abs(q_qty.value - c.value) < 1e-4:
                return True
    return False


def extract_technical_numbers(text: str) -> set:
    """Extract numbers with technical units or multi-digit thresholds from text.

    Excludes numbers that are part of error codes (E-04, F-07, ALM-01) or
    model/product identifiers (MC-500, KTM-33, AC-2T) to prevent false
    MISSING_NUMERICAL_THRESHOLD rejections.
    """
    if not text:
        return set()
    text_lower = text.lower()

    # Extract numbers with explicit technical units
    with_units = set(re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:v|volt|volts|a|amp|amps|psi|bar|rpm|mhz|khz|hz|ton|tons|kw|hp)\b",
        text_lower,
    ))

    # Temperature: match "4c", "4 c", "4 celsius", "4 degrees" etc.
    # but NOT bare single-letter "c" after a number that's part of an identifier
    temp_matches = set(re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:degrees?\s*(?:celsius|fahrenheit|c|f)?|°\s*[cf]|celsius|fahrenheit)\b",
        text_lower,
    ))
    # Also match compact "4c" but only when it looks like a temperature (not part of a model like "AC")
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\s*([cf])\b", text_lower):
        val = m.group(0)
        start = m.start()
        # Exclude if preceded by a letter (part of model identifier like "AC", "MC")
        if start > 0 and text_lower[start - 1].isalpha():
            continue
        temp_matches.add(val)

    # Standalone multi-digit numbers (thresholds like "415", "230", "1500")
    standalone = set()
    for m in re.finditer(r"\b(\d{2,})\b", text_lower):
        num = m.group(1)
        start = m.start()
        end = m.end()
        # Exclude if this number is part of an error/fault code pattern:
        # E-04, F-07, ERR-12, ALM-01, E04, F07
        prefix_start = max(0, start - 6)
        prefix = text_lower[prefix_start:start]
        if re.search(r"(?:^|[^a-z])(?:e|f|err|error|alm|alarm)\s*[-\u2010-\u2015]?\s*$", prefix):
            continue
        # Exclude if preceded by letter+dash (model identifier: MC-500, KTM-33, AC-2T)
        if re.search(r"[a-z]{1,6}\s*[-\u2010-\u2015]\s*$", prefix):
            continue
        # Exclude if followed by letter+dash pattern (part of model like 500-X)
        suffix = text_lower[end:end + 3]
        if re.match(r"\s*[-\u2010-\u2015]\s*[a-z]", suffix):
            continue
        standalone.add(num)

    return with_units | temp_matches | standalone



def evaluate_chunk_claim_support(
    chunk,
    terms: set,
    q_codes: set,
    q_numbers: set,
    require_code_if_present: bool = True,
    q_text: str = "",
) -> tuple[bool, str]:
    """Deterministically evaluate claim-level factual support for a retrieved passage.

    Rejects single-token casual overlap on complex queries.
    Requires exact identifier presence if question asks for an identifier.
    Requires matching numerical thresholds if question specifies values/limits.
    """
    if not chunk:
        return False, "NULL_CHUNK"

    doc_title = getattr(chunk, "title", "") or getattr(getattr(chunk, "document", None), "title", "") or ""
    chunk_text = f"{doc_title} {getattr(chunk, 'heading', '') or ''} {getattr(chunk, 'text', '') or ''}".lower()

    def _stem_variants(w: str) -> set[str]:
        w = w.lower()
        vars = {w}
        if len(w) > 3 and w.endswith("s"):
            vars.add(w[:-1])
        if len(w) > 4:
            if w.endswith("ed"):
                vars.add(w[:-2])
                vars.add(w[:-1])
            elif w.endswith("ing"):
                vars.add(w[:-3])
                vars.add(w[:-3] + "e")
        return vars

    raw_words = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{2,}", chunk_text)}
    chunk_words = set(raw_words)
    for rw in raw_words:
        chunk_words.update(_stem_variants(rw))

    # 1. Numerical Threshold Check (typed dimensional matching + boundary-aware):
    typed_q_quantities = []
    if q_text:
        typed_q_quantities.extend(extract_typed_quantities(q_text))
    if q_numbers:
        for num in q_numbers:
            extracted = extract_typed_quantities(str(num))
            for eq in extracted:
                if not any(eq.dimension == tq.dimension and abs(eq.value - tq.value) < 1e-4 and eq.unit == tq.unit for tq in typed_q_quantities):
                    typed_q_quantities.append(eq)

    if typed_q_quantities:
        chunk_quantities = extract_typed_quantities(chunk_text)
        missing_typed = []
        for q_qty in typed_q_quantities:
            if not verify_quantity_support(q_qty, chunk_quantities):
                missing_typed.append(f"{q_qty.raw}")
        if missing_typed:
            return False, f"MISSING_NUMERICAL_THRESHOLD: {', '.join(missing_typed)}"
    elif q_numbers:
        missing_numbers = []
        for num in q_numbers:
            # Direct exact match (boundary-aware)
            if re.search(r"(?<!\d)" + re.escape(num) + r"(?!\d)", chunk_text):
                continue
            # Temperature equivalence: "4c" should match "4.0 degrees celsius", "4 °C", "4.0°c"
            temp_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:degrees?\s*(?:celsius|fahrenheit|[cf])?|°\s*[cf]|[cf])$", num)
            if temp_match:
                base_val = temp_match.group(1)
                # Try integer and float forms: "4" matches "4.0", "4"
                int_val = str(int(float(base_val))) if '.' not in base_val else base_val
                float_val = f"{float(base_val)}"
                # Check for any temperature representation in chunk
                found = False
                for val in {base_val, int_val, float_val}:
                    if re.search(r"(?<!\d)" + re.escape(val) + r"\s*(?:\.0)?\s*(?:degrees?\s*(?:celsius|fahrenheit|[cf])?|°\s*[cf]|celsius|fahrenheit)", chunk_text):
                        found = True
                        break
                    # Also match bare number near temperature context words
                    if re.search(r"(?<!\d)" + re.escape(val) + r"(?!\d)", chunk_text):
                        if any(tw in chunk_text for tw in ("celsius", "degrees", "temperature", "°c", "°f")):
                            found = True
                            break
                if found:
                    continue
            missing_numbers.append(num)
        if missing_numbers:
            return False, f"MISSING_NUMERICAL_THRESHOLD: {', '.join(missing_numbers)}"

    # 2. Exact Identifier / Error Code Check:
    if q_codes:
        from core.services.identifiers import extract_all_technical_identifiers, match_canonical_identifiers
        canon_q = set()
        for c in q_codes:
            canon_q |= (extract_all_technical_identifiers(str(c)) or {str(c).upper()})
        matched, missing = match_canonical_identifiers(canon_q, chunk_text)
        matching_codes = sorted(matched)
        if matching_codes and missing:
            return False, f"PARTIAL_IDENTIFIER_MATCH: missing {', '.join(sorted(missing))}"
        if matching_codes:
            # Substantive claim check: if query has 2+ non-generic content terms, at least 1 must also be in chunk.
            # Exclude identifiers, matching codes, and generic inquiry terms from claim terms.
            generic_inquiry_terms = {
                "unit", "equipment", "system", "machine", "device", "problem", "issue",
                "occurs", "occur", "occurring", "happens", "happen", "happening", "error",
                "code", "fault", "alarm", "warning", "fix", "clear", "solve", "reset",
                "help", "do", "does", "what", "how", "why", "when", "work", "steps", "procedure"
            }
            def _clean_code(s):
                return re.sub(r"[\W_]+", "", str(s)).lower()
            code_keys = {_clean_code(c) for c in (set(matching_codes) | q_codes)}
            claim_terms = {
                t for t in terms
                if _clean_code(t) not in code_keys and t.lower() not in generic_inquiry_terms
            }
            if len(claim_terms) >= 2:
                matched_claim = {t for t in claim_terms if _stem_variants(t) & chunk_words}
                if len(matched_claim) < 1:
                    return False, f"IDENTIFIER_MATCH_BUT_UNGROUNDED_CLAIM: code {matching_codes[0]} found but no claim terms matched"
            return True, f"EXACT_IDENTIFIER_MATCH: {', '.join(matching_codes)}"
        elif require_code_if_present:
            return False, f"MISSING_IDENTIFIER: {', '.join(q_codes)}"

    # 3. Content Terms Overlap Check:
    matched_terms = {t for t in terms if _stem_variants(t) & chunk_words}
    overlap_count = len(matched_terms)
    n_terms = len(terms)

    # Symptom heading alignment: If the chunk's heading matches the query's symptom category
    # and has at least 1 topical term overlap, the passage is substantively grounded.
    try:
        from core.ml.conditional_rerank import _extract_symptom_keys, _heading_matches_symptom
        from .retriever import normalize_domain_terms
        query_source = f"{q_text} {normalize_domain_terms(q_text or '')} {' '.join(terms)}" if q_text else " ".join(terms)
        q_symptoms = _extract_symptom_keys(query_source)
        if q_symptoms and _heading_matches_symptom(getattr(chunk, 'heading', ''), q_symptoms, q_text or ""):
            if overlap_count >= 1:
                return True, f"SYMPTOM_HEADING_ALIGNMENT: {getattr(chunk, 'heading', '')}"
    except Exception:
        pass

    # Note: Component heading shortcut (COMPONENT_HEADING_ALIGNMENT) removed in Op28
    # to prevent shared component headings from bypassing procedural claim verification.

    if n_terms == 0:
        return False, "EMPTY_QUERY_TERMS"
    if n_terms == 1:
        if overlap_count >= 1:
            return True, "SINGLE_TERM_MATCH"
        return False, "NO_OVERLAP"
    if n_terms == 2:
        if overlap_count >= 2:
            return True, "TWO_TERM_MATCH"
        return False, f"INSUFFICIENT_OVERLAP: {overlap_count}/{n_terms}"

    # For 3 or more terms, require >= 2 distinct matching terms AND >= 30% overlap
    if overlap_count >= 2 and (overlap_count / n_terms) >= 0.30:
        return True, f"SUBSTANTIVE_OVERLAP: {overlap_count}/{n_terms}"

    return False, f"INSUFFICIENT_OVERLAP: {overlap_count}/{n_terms} (requires >=2 terms and >=30%)"


def _extractive_answer(question, asset, retrieved, service_call=None):
    if not retrieved:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    # Check for identifier-based abstention: if query mentions a specific
    # error code not found in any retrieved source, abstain — unless the
    # retrieved sources cover the fault category contextually.
    from .identifiers import query_identifiers_not_in_sources, extract_error_codes
    source_texts = [f"{getattr(r, 'title', '') or ''} {r.heading or ''} {r.text or ''}" for r in retrieved]
    missing_ids = query_identifiers_not_in_sources(question or "", source_texts)
    unverified_missing_codes = []
    if missing_ids:
        # Check for contextual coverage before blocking
        from .retriever import normalize_domain_terms
        norm_q = normalize_domain_terms(question or "")
        ctx_terms = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", norm_q)} - _STOPWORDS
        # Remove error code tokens from context terms
        for mid in missing_ids:
            ctx_terms -= {t.lower() for t in re.findall(r"[A-Za-z0-9]+", mid)}
        has_ctx = False
        for st in source_texts:
            st_words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", st)}
            if len(ctx_terms & st_words) >= 2:
                has_ctx = True
                break
        if not has_ctx:
            missing_str = ", ".join(sorted(missing_ids))
            return f"No approved documentation found for identifier {missing_str}."
        unverified_missing_codes = sorted(missing_ids)

    from .retriever import normalize_domain_terms
    norm_question = normalize_domain_terms(question or "")
    raw_terms = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", norm_question or "")}
    terms = raw_terms - _STOPWORDS
    if not terms:
        terms = raw_terms

    q_codes = extract_error_codes(question or "")
    q_numbers = extract_technical_numbers(question or "")

    # Check evidence relevance against question terms and metadata.
    # The retriever has ALREADY performed hybrid ranking (dense + lexical + BM25 + fault-code boost + RRF).
    # Token overlap must NOT overwrite or invert the retriever's authoritative fused rank.
    # We filter to chunks with valid claim support while strictly preserving the retriever's ranking order.
    relevant_chunks = []
    for r in retrieved:
        is_supported, _ = evaluate_chunk_claim_support(
            r, terms, q_codes, q_numbers, require_code_if_present=False, q_text=question
        )
        if is_supported:
            r.evidence_support = True
            relevant_chunks.append(r)
        else:
            r.evidence_support = False

    if not relevant_chunks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    primary_chunk = relevant_chunks[0]

    # 1. DIAGNOSTIC SUMMARY
    asset_name = asset.name if asset else "the selected equipment"
    topic = primary_chunk.heading or primary_chunk.title
    if unverified_missing_codes:
        missing_code_str = ", ".join(unverified_missing_codes)
        summary_line = (
            f"Based on approved technical guidance for {asset_name}, the documented issue corresponds to: {topic}. "
            f"(Note: Specified identifier(s) {missing_code_str} not explicitly found in available documentation; showing general category procedure.)"
        )
    else:
        summary_line = f"Based on approved technical guidance for {asset_name}, the documented issue corresponds to: {topic}."

    checks = []
    troubleshooting_steps = []
    expected_results = []
    escalations = []
    # Track original Why:/reason text from sources
    step_reasons = {}  # step_text -> reason_text (only from source)

    check_keywords = ("check", "confirm", "ensure", "inspect", "first", "verify", "examine")
    escalate_keywords = ("stop", "escalate", "service call", "abnormal", "technician", "fails", "failed", "persist", "unresolved")
    result_keywords = ("expected", "completes", "result", "normal", "range", "accepted reference range", "verification sample completes")

    seen_sentences = set()

    for chunk in relevant_chunks[:4]:
        raw_lines = chunk.text.split("\n")
        sentences = []
        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue
            # Strip markdown heading prefixes injected by _format_chunk_text.
            # These are structural artifacts, not actionable content.
            heading_match = re.match(r"^#{1,4}\s+(.+)", line_str)
            if heading_match:
                # Section headings are navigational, not actionable sentences.
                # Skip them to prevent cross-contamination from unrelated chunks.
                continue
            num_match = re.match(r"^\d+[\.)\\]]\s*(.+)", line_str)
            if num_match:
                sentences.append(num_match.group(1).strip())
            else:
                for s in re.split(r"(?<=[.!?])\s+", line_str):
                    if len(s.strip()) > 15:
                        sentences.append(s.strip())

        # Look for "Why:" or "Reason:" lines that follow a step in the source
        prev_sentence = None
        for s in sentences:
            s_clean = s.strip()
            if not s_clean or s_clean in seen_sentences:
                prev_sentence = s_clean
                continue
            seen_sentences.add(s_clean)
            s_lower = s_clean.lower()

            # Check if this is a "Why:" line that belongs to the previous step
            why_match = re.match(r"^(?:why|reason)\s*:\s*(.+)", s_lower)
            if why_match and prev_sentence:
                step_reasons[prev_sentence] = s_clean
                prev_sentence = s_clean
                continue

            # Classification priority: check > escalation > result > step.
            # A sentence like "Confirm the isolator is on" is a CHECK even if
            # it appears near escalation text, because it starts with a check
            # keyword.  Escalation is only when there is NO check keyword match.
            is_check = any(s_lower.startswith(k) or f" {k} " in s_lower for k in check_keywords)
            is_escalation = any(k in s_lower for k in escalate_keywords)
            is_result = any(k in s_lower for k in result_keywords)

            if is_check:
                if len(checks) < 3:
                    checks.append(s_clean)
                else:
                    troubleshooting_steps.append(s_clean)
            elif is_escalation:
                escalations.append(s_clean)
            elif is_result:
                expected_results.append(s_clean)
            else:
                troubleshooting_steps.append(s_clean)

            prev_sentence = s_clean


    if not checks and troubleshooting_steps:
        checks.append(troubleshooting_steps.pop(0))
    if not troubleshooting_steps and checks:
        troubleshooting_steps = checks[:]
        checks = [troubleshooting_steps.pop(0)]

    if not troubleshooting_steps and not checks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    output = []
    output.append("DIAGNOSTIC SUMMARY")
    output.append(summary_line)
    output.append("")

    output.append("CHECK FIRST")
    if checks:
        for c in checks[:3]:
            output.append(f"- {c.rstrip('.')}.")
    else:
        # No checks found in source — omit section rather than fabricate
        output.append("- Refer to the approved technical manual for initial checks.")
    output.append("")

    output.append("STEP-BY-STEP TROUBLESHOOTING")
    for i, step in enumerate(troubleshooting_steps[:6], 1):
        clean_step = step.rstrip(".")
        output.append(f"Step {i} - {clean_step}.")
        # Only include a "Why:" line if we found one in the actual source text
        if step in step_reasons:
            output.append(step_reasons[step])
    output.append("")

    output.append("EXPECTED RESULT")
    if expected_results:
        output.append(expected_results[0].rstrip(".") + ".")
    else:
        # Do NOT fabricate an expected result — state that the source doesn't specify
        output.append("The approved source does not specify an expected result.")
    output.append("")

    output.append("STOP AND ESCALATE IF")
    if escalations:
        for esc in escalations[:3]:
            output.append(f"- {esc.rstrip('.')}.")
    else:
        # Do NOT fabricate escalation criteria — state that the source doesn't specify
        output.append("- The approved source does not specify escalation criteria. Contact support if the issue persists.")
    output.append("")

    output.append("VERIFIED REFERENCES")
    seen_refs = set()
    for chunk in relevant_chunks[:5]:
        ref_label = f"{chunk.reference} ({chunk.doc_type})"
        if ref_label not in seen_refs:
            seen_refs.add(ref_label)
            output.append(f"- {ref_label}")

    return "\n".join(output)


def _clean_model_answer(answer, retrieved=None):
    answer = (answer or "").strip()
    if not answer:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    # Strip bracketed citations like [1], [Source 1], [Ref: 2]
    answer = re.sub(r"\[\s*(?:source|ref|reference)?\s*[:#]?\s*\d+\s*\]", "", answer, flags=re.I)
    answer = re.sub(r"\[\s*\d+\s*\]", "", answer)
    # Strip line-based citations emitted by LLM: Source: ..., Manual: ..., Reference: ...
    answer = re.sub(r"(?im)^\s*(?:source|manual|reference|doc|document|section)\s*[:\-]\s*.*$", "", answer)

    # Strip any model-written VERIFIED REFERENCES section
    # The server attaches citations exclusively from verified retrieval metadata
    answer = re.split(r"(?im)^\s*VERIFIED\s+REFERENCES\b", answer)[0].strip()

    # HARDENING: Strip structural prompt delimiters echoed back by the model
    # These are internal prompt framing and must never appear in user-facing output
    answer = re.sub(r"</?retrieved_source\b[^>]*>", "", answer)
    answer = re.sub(r"UNTRUSTED_CONTENT_(?:START|END)", "", answer)
    answer = re.sub(r"(?im)^\s*(?:ASSET|CALL CONTEXT|QUESTION|RETRIEVED SOURCES|ANSWER)\s*:\s*", "", answer)

    # Reject speculative repair advice without approved grounding (Red-Team Area 4)
    speculative_phrases = [
        "probably", "likely", "usually", "you can try", "might want to try",
        "you could try", "try swapping", "maybe you should", "perhaps you can",
        "it's worth trying", "consider trying", "one option is to",
        "a common fix is", "a quick workaround",
    ]
    answer_lower = answer.lower()
    for phrase in speculative_phrases:
        if phrase in answer_lower:
            if any(w in answer_lower for w in ["step", "repair", "replace", "fix", "clean", "adjust"]):
                return INSUFFICIENT_EVIDENCE_MESSAGE

    # Reject answers that are suspiciously short (model didn't find real content)
    # or suspiciously contain only the question restated
    stripped = re.sub(r"\s+", " ", answer).strip()
    if len(stripped) < 20:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    return answer[:8000].strip()


def check_evidence_sufficiency(question: str, asset, retrieved: list) -> EvidenceSufficiencyResult:
    """
    Independently validate evidence sufficiency for the question and asset.
    Separates:
    - Candidate relevance
    - Answerability
    - Claim support
    - Equipment applicability
    - Source trust
    - Industrial procedure approval
    """
    if not retrieved:
        return EvidenceSufficiencyResult(
            is_sufficient=False, reason="no-retrieved-chunks",
            candidate_relevance=False, answerability=False, claim_support=False,
            equipment_applicable=False, source_trust=False, procedure_approved=False,
            grounded_chunks=[],
        )

    # 1. Candidate retrieval threshold: at least one candidate must have raw relevance >= SERVY_RAG_MIN_SCORE
    min_support = getattr(settings, "SERVY_RAG_MIN_SCORE", 0.055)
    max_score = max((getattr(r, "score", 0.0) for r in retrieved), default=0.0)
    if max_score < min_support:
        return EvidenceSufficiencyResult(
            is_sufficient=False, reason="score-below-minimum-support",
            candidate_relevance=False, answerability=False, claim_support=False,
            equipment_applicable=False, source_trust=False, procedure_approved=False,
            grounded_chunks=[],
        )

    # 2. Identifier verification: if question explicitly asks for a code, it must be present
    from .identifiers import extract_error_codes
    q_codes = extract_error_codes(question or "")
    has_symptom_category = False
    if q_codes:
        has_code_match = any(
            bool(q_codes & extract_error_codes(f"{r.heading or ''} {r.text or ''}"))
            for r in retrieved
        )
        if not has_code_match:
            # If retrieved sources already contain other error codes, the queried code is strictly missing
            source_has_codes = any(
                bool(extract_error_codes(f"{r.heading or ''} {r.text or ''}"))
                for r in retrieved
            )
            # For un-coded category documents, check if a specific symptom category aligns
            has_symptom_category = False
            if not source_has_codes:
                try:
                    from core.ml.conditional_rerank import _extract_symptom_keys, _heading_matches_symptom
                    from .retriever import normalize_domain_terms
                    norm_q = normalize_domain_terms(question or "")
                    q_symptoms = _extract_symptom_keys(f"{question or ''} {norm_q}")
                    for r in retrieved:
                        if q_symptoms and _heading_matches_symptom(getattr(r, "heading", ""), q_symptoms):
                            has_symptom_category = True
                            break
                except Exception:
                    pass

            if not has_symptom_category:
                return EvidenceSufficiencyResult(
                    is_sufficient=False, reason="missing-required-identifier",
                    candidate_relevance=True, answerability=False, claim_support=False,
                    equipment_applicable=False, source_trust=False, procedure_approved=False,
                    grounded_chunks=[],
                )

    # 3. Source verification and quarantine check
    valid_chunks = [r for r in retrieved if getattr(r, "source_verified", True) and not getattr(r, "is_quarantined", False)]
    if not valid_chunks:
        return EvidenceSufficiencyResult(
            is_sufficient=False, reason="all-sources-unverified",
            candidate_relevance=True, answerability=True, claim_support=False,
            equipment_applicable=False, source_trust=False, procedure_approved=False,
            grounded_chunks=[],
        )

    # 4. Equipment applicability
    applicable_chunks = [r for r in valid_chunks if getattr(r, "equipment_applicable", True)]
    if not applicable_chunks:
        return EvidenceSufficiencyResult(
            is_sufficient=False, reason="no-applicable-equipment-evidence",
            candidate_relevance=True, answerability=True, claim_support=False,
            equipment_applicable=False, source_trust=True, procedure_approved=False,
            grounded_chunks=[],
        )

    # 5. Content Grounding / Claim Support check
    from .retriever import normalize_domain_terms
    norm_question = normalize_domain_terms(question or "")
    raw_terms = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", norm_question or "")}
    terms = raw_terms - _STOPWORDS
    if not terms:
        terms = raw_terms

    q_numbers = extract_technical_numbers(question or "")

    grounded_chunks = []
    for r in applicable_chunks:
        is_supported, reason = evaluate_chunk_claim_support(
            r, terms, q_codes, q_numbers, require_code_if_present=(not has_symptom_category), q_text=question
        )
        # Phase 2 Fix: Enforce strict separation between retrieval signals and evidence verification.
        # An exact-identifier match (ranking signal) must NEVER override failed claim support.
        if is_supported:
            r.evidence_support = True
            grounded_chunks.append(r)
        else:
            r.evidence_support = False

    if not grounded_chunks:
        return EvidenceSufficiencyResult(
            is_sufficient=False, reason="no-grounded-overlap",
            candidate_relevance=True, answerability=True, claim_support=False,
            equipment_applicable=True, source_trust=True, procedure_approved=False,
            grounded_chunks=[],
        )

    # 6. Industrial Procedure Approval (Phase 4 Fix):
    # Customer physical action authorization defaults strictly to False.
    # It cannot be inferred from document trust or candidate scoring.
    approved_chunks = [r for r in grounded_chunks if getattr(r, "customer_action_authorized", False)]

    return EvidenceSufficiencyResult(
        is_sufficient=True, reason="evidence-sufficient",
        candidate_relevance=True, answerability=True, claim_support=True,
        equipment_applicable=True, source_trust=True,
        procedure_approved=bool(approved_chunks),
        grounded_chunks=grounded_chunks,
    )


def generate_answer(question, asset, retrieved, service_call=None):
    if not retrieved:
        return INSUFFICIENT_EVIDENCE_MESSAGE, "fallback"

    # Evidence sufficiency check (Phase 1 Fix):
    # Validates candidate score, technical identifier presence, equipment applicability, and grounded content.
    evidence_result = check_evidence_sufficiency(question, asset, retrieved)
    if not evidence_result:
        return INSUFFICIENT_EVIDENCE_MESSAGE, "insufficient-evidence"

    # Use only grounded, verified chunks for LLM context — never the full unfiltered retrieved list.
    # This prevents ungrounded, non-applicable, or quarantined chunks from leaking into the prompt.
    grounded = evidence_result.grounded_chunks or retrieved
    context_blocks = []
    for i, r in enumerate(grounded, 1):
        context_blocks.append(
            f"<retrieved_source id=\"{i}\" kind=\"{r.source_kind}\">\n"
            f"TITLE: {r.reference}\nTYPE: {r.doc_type}\n"
            f"UNTRUSTED_CONTENT_START\n{r.text}\nUNTRUSTED_CONTENT_END\n"
            f"</retrieved_source>"
        )
    context = "\n\n".join(context_blocks)
    asset_context = "Unknown asset"
    if asset:
        product = asset.product.name if asset.product else "Unknown product"
        model = asset.model_number or "Unknown model"
        asset_context = f"{asset.name} | model {model} | product {product} | code {asset.asset_code}"

    call_context = "No active Call Register context."
    if service_call:
        call_context = (
            f"Call #{service_call.servy_id}; type={service_call.complaint_type}; "
            f"status={service_call.status}; priority={service_call.priority}; "
            f"complaint={service_call.complaint_text}; technician_notes={service_call.technician_notes}"
        )

    prompt = f"""You are the Servy Technical Knowledge Copilot.
Your job is to answer customer/technician troubleshooting questions using ONLY the factual information inside RETRIEVED SOURCES.
Never fabricate diagnostic steps, part numbers, or safety procedures.

If sufficient approved evidence exists in RETRIEVED SOURCES, format your answer strictly using these 6 sections:
DIAGNOSTIC SUMMARY
Briefly explain the most likely documented issue based on the selected asset and reported symptom.

CHECK FIRST
List the initial documented checks.

STEP-BY-STEP TROUBLESHOOTING
Step 1 - [action]
Why: [reason]
Step 2 - [action]
Why: [reason]

EXPECTED RESULT
Explain what indicates that the issue has been corrected.

STOP AND ESCALATE IF
List conditions where the customer should stop troubleshooting and create a service call.

VERIFIED REFERENCES
List the retrieved manual titles and sections used.

If evidence in RETRIEVED SOURCES is insufficient or unrelated to answer safely, your entire answer MUST BE EXACTLY:
I couldn't find enough approved information in the Knowledge Base to safely answer this issue.

ASSET: {asset_context}
CALL CONTEXT: {call_context}
QUESTION: {question}

RETRIEVED SOURCES:
{context}

ANSWER:"""

    provider = settings.SERVY_LLM_PROVIDER.lower()
    if provider in {"auto", "ollama"} and retrieved and is_ollama_service_online():
        try:
            raw_answer = ollama_generate(prompt)
            clean_answer = _clean_model_answer(raw_answer, grounded)
            if clean_answer and clean_answer != INSUFFICIENT_EVIDENCE_MESSAGE:
                # Append verified references from authoritative retrieval metadata
                seen_refs = set()
                ref_lines = ["\n\nVERIFIED REFERENCES"]
                for r in grounded[:5]:
                    ref_label = f"{r.reference} ({r.doc_type})"
                    if ref_label not in seen_refs:
                        seen_refs.add(ref_label)
                        ref_lines.append(f"- {ref_label}")
                final_answer = clean_answer + "\n" + "\n".join(ref_lines)
                return final_answer, "ollama-local"
            elif clean_answer == INSUFFICIENT_EVIDENCE_MESSAGE:
                return INSUFFICIENT_EVIDENCE_MESSAGE, "insufficient-evidence"
        except Exception:
            pass
    return _extractive_answer(question, asset, grounded, service_call=service_call), "extractive-local"

