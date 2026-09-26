import re
import requests
from urllib.parse import urlparse
from django.conf import settings


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


def _extractive_answer(question, asset, retrieved, service_call=None):
    if not retrieved:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    # Check for identifier-based abstention: if query mentions a specific
    # error code not found in any retrieved source, abstain.
    from .identifiers import query_identifiers_not_in_sources
    source_texts = [r.text for r in retrieved]
    missing_ids = query_identifiers_not_in_sources(question or "", source_texts)
    if missing_ids:
        missing_str = ", ".join(sorted(missing_ids))
        return f"No approved documentation found for identifier {missing_str}."

    raw_terms = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", question or "")}
    terms = raw_terms - _STOPWORDS
    if not terms:
        terms = raw_terms

    # Check evidence relevance against question terms and metadata
    matched_chunks = []
    for r in retrieved:
        words = {w.lower() for w in re.findall(r"[A-Za-z0-9_-]{3,}", f"{r.heading} {r.text}".lower())}
        overlap = len(terms & words)
        matched_chunks.append((overlap, r))

    if not matched_chunks:
        return INSUFFICIENT_EVIDENCE_MESSAGE

    matched_chunks.sort(key=lambda x: (x[0], x[1].score), reverse=True)
    primary_chunk = matched_chunks[0][1]

    # 1. DIAGNOSTIC SUMMARY
    asset_name = asset.name if asset else "the selected equipment"
    topic = primary_chunk.heading or primary_chunk.title
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

    for _, chunk in matched_chunks[:4]:
        raw_lines = chunk.text.split("\n")
        sentences = []
        for line in raw_lines:
            line_str = line.strip()
            if not line_str:
                continue
            num_match = re.match(r"^\d+[\.)\]]\s*(.+)", line_str)
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

            if any(k in s_lower for k in escalate_keywords):
                escalations.append(s_clean)
            elif any(k in s_lower for k in result_keywords):
                expected_results.append(s_clean)
            elif any(s_lower.startswith(k) or f" {k} " in s_lower for k in check_keywords):
                if len(checks) < 3:
                    checks.append(s_clean)
                else:
                    troubleshooting_steps.append(s_clean)
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
    for _, chunk in matched_chunks[:5]:
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


def generate_answer(question, asset, retrieved, service_call=None):
    if not retrieved:
        return INSUFFICIENT_EVIDENCE_MESSAGE, "fallback"

    # Support threshold check: If retrieval scores are below minimum support, fail closed
    min_support = getattr(settings, "SERVY_RAG_MIN_SCORE", 0.0)
    max_score = max((getattr(r, "score", 0.0) for r in retrieved), default=0.0)
    if max_score < min_support:
        return INSUFFICIENT_EVIDENCE_MESSAGE, "insufficient-evidence"

    context_blocks = []
    for i, r in enumerate(retrieved, 1):
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
    if provider in {"auto", "ollama"} and retrieved:
        try:
            raw_answer = ollama_generate(prompt)
            clean_answer = _clean_model_answer(raw_answer, retrieved)
            if clean_answer and clean_answer != INSUFFICIENT_EVIDENCE_MESSAGE:
                # Append verified references from authoritative retrieval metadata
                seen_refs = set()
                ref_lines = ["\n\nVERIFIED REFERENCES"]
                for r in retrieved[:5]:
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
    return _extractive_answer(question, asset, retrieved, service_call=service_call), "extractive-local"

