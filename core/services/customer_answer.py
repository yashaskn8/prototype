"""Customer presentation built from persisted evidence, never model-issued IDs.

No persistence or workflow actions. Every cited claim carries a signed exact
source span; opening it rechecks the current user, scope, version and content.
"""
import hashlib
import math
import re
from difflib import SequenceMatcher
from types import SimpleNamespace

from django.core import signing
from rest_framework.exceptions import NotFound

from core.models import KnowledgeChunk
from .content_safety import detect_document_prompt_injection, query_is_control_abuse
from .customer_isolation import validate_customer_serving_policy
from .retriever import is_authorized_candidate

TITLES = (
    "What could be wrong?", "What can I safely check or report?",
    "What is still unknown?", "What should I do next?",
)
SALT = "servy.customer-evidence.v1"

# A sentence is not an observation merely because it contains "check".
# Compound instructions and physical operations require separate approval.
PHYSICAL_OPERATION = re.compile(
    r"\b(?:open|remove|replace|disconnect|connect|reconnect|bypass|charge|adjust|"
    r"measure|probe|touch|strip|clean|wipe|rinse|flush|restart|reset|press|turn|"
    r"disassemble|dismantle|defeat|bridge|unplug|operate|run|install|repair|test|"
    r"rotate|spin|tighten|loosen|lubricate|drain|fill|insert|unlock|energize|"
    r"switch|plug|solder|heat|manipulate|pour|mix|wash|pull|push)\b", re.I)


def _noninvasive_observation(text):
    internal = re.search(r"\b(?:internal|inside|cabinet|enclosure|chassis|housing|wiring|terminal|assembly|carriage)\b", text, re.I)
    instrument_reading = re.search(
        r"\b(?:operating current|current draw|amperage|(?:line|supply|electrical) voltage|"
        r"static pressure|refrigerant pressure|airflow at (?:defined|specified) measurement points|"
        r"return/supply air temperatures)\b", text, re.I)
    visible_readout = re.search(r"\b(?:display|displayed|external indicator)\b", text, re.I)
    descriptive = re.match(r"(?:observe|read|report|note|record|describe|look at)\b", text, re.I)
    visible_check = re.match(r"(?:check|confirm|ensure|inspect|verify|examine|monitor)\b", text, re.I) and re.search(
        r"\b(?:display|displayed|visible|visually|external|indicator|nameplate|label)\b", text, re.I)
    return (not internal and not PHYSICAL_OPERATION.search(text) and bool(descriptive or visible_check)
            and (not instrument_reading or bool(visible_readout)))


def _descriptive_cause(text):
    # Nouns such as "low refrigerant charge" describe causes. Imperatives and
    # compound directives remain excluded, even inside a manual's cause list.
    directive = r"(?:" + PHYSICAL_OPERATION.pattern + r"|(?:check|confirm|inspect|verify|observe|read|report)\b)"
    return not re.search(r"(?:^|[.;:]\s*|\b(?:must|should|please|then|and|need to|try to|you can)\s+)" + directive, text, re.I)


def _question_guard(question):
    """Require a direct answer for permission, proof, or exact-spec questions."""
    from .retriever import normalize_domain_terms
    lower = normalize_domain_terms(question).casefold()
    permission = re.search(r"\b(?:can|may|should|shud|shall|could|allowed|permitted|acceptable|okay|safe)\b", lower)
    physical = re.search(
        r"\b(?:bypas\w*|defeat|bridge|resett?\w*|clear(?:ing)? (?:the )?fault|"
        r"open|probe|wire|measure|chip|adjust|vent|charg\w*|top up|spray|fix|repair|"
        r"keep (?:the |it |unit |equipment |system ){0,3}(?:running|operating)|"
        r"stay on|operate|run|change fan speed)\b", lower)
    if permission and physical:
        return "safety"
    if re.search(r"\b(?:proves?|definitely|certainly|must mean|for sure)\b", lower):
        return "unsupported"
    if (re.search(r"\bexact\b", lower) and re.search(
        r"\b(?:value|limit|threshold|setting|pressure|torque|current|charge|code|range|clearance|interval)\b", lower)):
        return "unsupported"
    if re.search(r"\b(?:in spec|within (?:the )?(?:approved|specified|manufacturer) range)\b", lower):
        return "unsupported"
    if re.search(r"\b(?:replace|substitute)\b.{0,45}\b(?:manufacturer|service manual)\b", lower):
        return "unsupported"
    if re.search(r"\benough\b.{0,45}\binstead of\b|\bdeclare acceptance\b.{0,70}\bwithout\b", lower):
        return "unsupported"
    if re.search(r"\b(?:what should be clarified|which details are missing|is (?:that|this) enough to tell which|or something else)\b", lower):
        return "clarify"
    if (re.fullmatch(r"why is\s+.{3,80}\s+not working\??", lower)
            or re.fullmatch(r"why isn't\s+.{3,80}\s+working\??", lower)):
        return "clarify"
    words = re.findall(r"[a-z0-9]+", lower)
    if len(words) <= 7 and re.search(r"\b(?:help|what happened|what first|due yet)\b", lower):
        return "clarify"
    return None


def _source_warning(candidates, question, *, tenant, customer, asset):
    """Find a relevant warning in customer-authorized source chunks only."""
    from .retriever import _knowledge_candidates
    available = list(candidates)
    seen = {candidate.chunk.id for candidate in available}
    # A safety prohibition can be absent from the top-k answer pool. Search
    # the canonical authorized source set, then recheck every added chunk.
    for chunk in _knowledge_candidates(tenant, asset, customer, False, backend="all"):
        if chunk.id in seen:
            continue
        try:
            permitted = source_chunk(tenant=tenant, customer=customer, asset=asset,
                                     document_id=chunk.document_id, chunk_id=chunk.id)
        except NotFound:
            continue
        seen.add(permitted.id)
        available.append(SimpleNamespace(chunk=permitted, text=permitted.text))
    normalized = re.sub(r"\bbypas\b", "bypass", question.casefold())
    normalized = re.sub(r"\bresett?ing\b", "reset", normalized)
    normalized = re.sub(r"\bbreker\b", "breaker", normalized)
    query_words = set(re.findall(r"[a-z]{4,}", normalized)) - {
        "should", "could", "would", "please", "what", "when", "while", "with",
        "this", "that", "equipment", "unit", "machine", "your", "have", "from",
    }
    best = None
    for candidate in available:
        for text, _, start, end in source_claims(candidate.text):
            if not re.search(
                r"^(?:do not|never|stop\b|switch off\b|isolate\b)|"
                r"\b(?:must be performed by qualified|require trained personnel)\b", text, re.I):
                continue
            warning_words = set(re.findall(r"[a-z]{4,}", text.casefold()))
            overlap = query_words & warning_words
            if not overlap:
                continue
            # Prefer the source rule for the action being requested over a
            # nearby symptom warning from the same retrieved document.
            score = len(overlap) + 3 * len(overlap & {
                "bypass", "reset", "chip", "adjust", "charge", "open", "wire", "vent", "operate",
            })
            if best is None or score > best[0]:
                best = (score, candidate.chunk, text, start, end)
    return best


def _guarded_customer_response(interaction, session, candidates, guard):
    asset, customer, tenant = interaction.asset, interaction.customer, interaction.tenant
    citations, claims = [], []
    if guard == "safety":
        warning = _source_warning(candidates, interaction.question or "", tenant=tenant,
                                  customer=customer, asset=asset)
        if warning:
            _, chunk, text, start, end = warning
            citation = _citation(chunk, start, end, asset=asset, customer=customer,
                                 user_id=interaction.created_by_id, kind="safety_warning")
            citation["id"] = "s1"
            citations.append(citation)
            claims.append({"id": "c1", "text": text, "source_text": text,
                           "kind": "safety_warning", "evidence_ids": ["s1"]})
            message = f"I cannot approve that operation through this chat. The source warns: {text} Ask an authorized technician for help."
        else:
            message = "I cannot approve that operation through this chat. Ask an authorized technician for safe, model-specific guidance."
    elif guard == "clarify":
        if re.search(r"\b(?:water|leak|condensate)\b", interaction.question or "", re.I):
            message = "Please report where the water first appears and when it occurs. The current description does not establish its source."
        else:
            message = "Please describe the exact symptom, any displayed code, when it happens, and any immediate safety concern before I suggest a cause."
    else:
        message = "The available documentation does not establish that exact value or conclusion. I cannot confirm it from this evidence; consult the applicable model-specific source or an authorized technician."
    clarification = guard == "clarify"
    response_type = "CLARIFICATION" if clarification else "INSUFFICIENT_EVIDENCE"
    mode = "CLARIFICATION_REQUIRED" if clarification else "INSUFFICIENT_EVIDENCE"
    equipment = {"id": asset.id, "name": asset.name, "model_number": asset.model_number} if asset else {}
    presentation = {"schema_version": "customer-answer-v1", "response_type": response_type,
                    "equipment": equipment, "sections": [], "citations": citations,
                    "supporting_claims": claims, "customer_permitted_observations": [],
                    "customer_authorized_actions": [], "hazard": {"severity": "none", "message": ""},
                    "message": message, "authorized_user_actions": {"can_resolve": False, "can_escalate": True}}
    return {"presentation": presentation, "response_mode": mode, "response_type": response_type,
            "interaction_id": interaction.id, "session_id": str(session.session_id) if session else None,
            "session_status": session.status if session else "ACTIVE",
            "session_version": session.version if session else 1,
            "asset": equipment, "summary": message, "answer": message,
            "citations": citations, "references": citations,
            "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
            "stop_conditions": [], "expected_result": "",
            "clarification_question": message if clarification else None,
            "can_resolve": False, "can_escalate": True, "engine": "source-structured-extractive"}


def _document_fact_family(question):
    """Limit source-list answers to explicit questions about document contents."""
    lower = question.casefold()
    if (not re.search(r"\b(?:which|what|where|does|is|are)\b", lower)
        or not re.search(r"\b(?:checklist|template|record|log|fields?|listed|include|part of)\b", lower)
        or re.search(r"\b(?:how (?:do|can|should)|should i|can i|may i|perform|adjust|repair|replace)\b", lower)):
        return None
    if re.search(r"fault[- ]record|trouble (?:note|ticket)", lower):
        return "troubleshooting"
    if re.search(r"incident (?:note|report)|alarm (?:report|template)", lower):
        return "alarm"
    if re.search(r"maintenance|\bpm\b|\blog\b|runtime|service reports|leak-test results", lower):
        return "preventive maintenance"
    if re.search(r"install|commission|mount|handover|acceptance|remote.*checklist", lower):
        return "installation"
    return None


def _document_fact_response(interaction, session, family):
    """Quote matching authorized checklist/template lines as facts, never actions."""
    from .retriever import _knowledge_candidates
    asset, customer, tenant = interaction.asset, interaction.customer, interaction.tenant
    question = interaction.question or ""
    requested = re.search(
        r"\b(?:leaves? out|has no|omits?|didn'?t put|didnt put|is checking)\s+([^?.!]+)",
        question, re.I)
    if not requested:
        return None
    phrase = re.split(r"\bpart of\b", requested[1], maxsplit=1, flags=re.I)[0]
    target = set(re.findall(r"[a-z]{3,}", phrase.casefold())) - {
        "the", "and", "that", "this", "for", "with", "into", "part", "only",
    }
    if not target or len(target) > 8:
        return None
    def matched_terms(line):
        words = set(re.findall(r"[a-z]{3,}", line.casefold()))
        return {term for term in target if term in words or any(
            len(term) >= 5 and abs(len(term) - len(word)) <= 1
            and SequenceMatcher(None, term, word).ratio() >= 0.86 for word in words)}
    ranked = {}
    for chunk in _knowledge_candidates(tenant, asset, customer, False, backend="all"):
        if family not in chunk.document.title.casefold():
            continue
        if not re.search(r"template|log|check|commission|record|retention|acceptance", chunk.heading or "", re.I):
            continue
        try:
            permitted = source_chunk(tenant=tenant, customer=customer, asset=asset,
                                     document_id=chunk.document_id, chunk_id=chunk.id)
        except NotFound:
            continue
        offset = 0
        for raw in permitted.text.splitlines(keepends=True):
            line = raw.strip()
            clean = re.sub(r"^\[\s*\]\s*", "", _clean_line(line))
            clean = re.sub(r":\s*_+\s*$", ":", clean).strip()
            if (len(clean) < 7 or len(clean) > 260 or re.match(r"^\d+\.\s+", clean)
                or clean.isupper() or clean in {"NOTE", "Possible causes:", "Checks:"}):
                offset += len(raw)
                continue
            overlap = matched_terms(clean)
            if overlap:
                start = offset + raw.find(line)
                ranked.setdefault(permitted.document_id, []).append(
                    (overlap, permitted, clean, line, start, start + len(line)))
            offset += len(raw)
    selected = None
    for document_id, lines in sorted(ranked.items()):
        if set().union(*(item[0] for item in lines)) != target:
            continue
        remaining, chosen = set(target), []
        while remaining and len(chosen) < 4:
            item = max(lines, key=lambda candidate: (
                len(candidate[0] & remaining), len(candidate[0]), -candidate[1].id, -candidate[4]))
            if not item[0] & remaining:
                break
            chosen.append(item)
            lines.remove(item)
            remaining -= item[0]
        if not remaining:
            selected = chosen
            break
    if not selected:
        return None
    citations, claims = [], []
    for _, chunk, text, source_text, start, end in selected:
        citation = _citation(chunk, start, end, asset=asset, customer=customer,
                             user_id=interaction.created_by_id, kind="document_fact")
        citation["id"] = f"s{len(citations) + 1}"
        citations.append(citation)
        claims.append({"id": f"c{len(citations)}", "text": "The source lists for qualified staff: " + text,
                       "source_text": source_text, "kind": "document_fact", "evidence_ids": [citation["id"]]})
    sections = [
        {"number": 1, "title": "What the document says", "claims": claims},
        {"number": 2, "title": "What can I safely check or report?", "claims": [
            {"text": "These are document facts, not approval for customer installation, maintenance, or repair.",
             "kind": "uncertainty", "evidence_ids": []}]},
        {"number": 3, "title": "What is still unknown?", "claims": [
            {"text": "The equipment condition and applicability of any unlisted model-specific requirement remain unverified.",
             "kind": "uncertainty", "evidence_ids": []}]},
        {"number": 4, "title": "What should I do next?", "claims": [
            {"text": "Share the missing record details with an authorized technician. No ticket has been created.",
             "kind": "next_step", "evidence_ids": []}]},
    ]
    answer = "\n\n".join(f"{s['number']}. {s['title']}\n" + " ".join(c["text"] for c in s["claims"])
                          for s in sections)
    equipment = {"id": asset.id, "name": asset.name, "model_number": asset.model_number}
    presentation = {"schema_version": "customer-answer-v1", "response_type": "DIAGNOSTIC_GUIDANCE",
                    "equipment": equipment, "sections": sections, "citations": citations,
                    "supporting_claims": claims, "customer_permitted_observations": [],
                    "customer_authorized_actions": [], "hazard": {"severity": "none", "message": ""},
                    "message": "", "known_limitations": ["Document facts are not customer procedure approval."],
                    "authorized_user_actions": {"can_resolve": False, "can_escalate": True}}
    return {"presentation": presentation, "response_mode": "ANSWER_WITH_VERIFIED_EVIDENCE",
            "response_type": "DIAGNOSTIC_GUIDANCE", "interaction_id": interaction.id,
            "session_id": str(session.session_id) if session else None,
            "session_status": session.status if session else "ACTIVE",
            "session_version": session.version if session else 1,
            "asset": equipment, "summary": "Source-backed document fields or checklist items; applicability unverified.",
            "answer": answer, "citations": citations, "references": citations,
            "customer_safe_actions": [], "check_first": [], "technician_only_actions": [],
            "stop_conditions": [], "expected_result": "", "clarification_question": None,
            "can_resolve": False, "can_escalate": True, "engine": "source-structured-extractive"}


def source_chunk(*, tenant, customer, asset, document_id, chunk_id):
    if not asset or not customer or asset.tenant_id != tenant.id or customer.tenant_id != tenant.id or asset.customer_id != customer.id:
        raise NotFound("Source unavailable.")
    validate_customer_serving_policy(tenant, customer, asset)
    try:
        chunk = KnowledgeChunk.objects.select_related(
            "document", "document__tenant", "document__asset", "document__product",
            "document__category", "document__domain",
        ).filter(pk=int(chunk_id), document_id=int(document_id), tenant=tenant).first()
    except (ValueError, TypeError, OverflowError):
        raise NotFound("Source unavailable.")
    if (not chunk or chunk.document.disable_sharing
        or not is_authorized_candidate(chunk, tenant=tenant, customer=customer, asset=asset,
                                       include_confidential=False, backend="sparse", allow_failed=True)
        or detect_document_prompt_injection(chunk.text)
        or "injection" in str(chunk.safety_flags or "").lower()):
        raise NotFound("Source unavailable.")
    return chunk


def _clean_line(line):
    clean = re.sub(r"^\s*(?:[-*•]+|\d+[.)])\s+", "", line).strip()
    clean = clean.replace("**", "").replace("__", "").replace("`", "")
    return re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", clean)


def source_claims(text):
    """Preserve source spans and cause-list context; headings are never actions."""
    cause_list = False
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.strip()
        heading = re.sub(r"^#+\s*", "", _clean_line(line))
        if re.match(r"^(?:possible|potential|common) causes?\s*:?$", heading, re.I):
            cause_list = True
        elif line.startswith("#") or line.endswith(":") or (line.isupper() and len(line) > 4):
            cause_list = False
        elif line:
            clean = _clean_line(line)
            if 12 <= len(clean) <= 320 and len(clean.split()) >= 3:
                is_cause = (cause_list and bool(re.match(r"(?:[-*•]+|\d+[.)])\s", line))) or bool(re.search(r"\b(?:may be caused by|possible cause|can cause|may cause|causing|leads? to|results? in)\b", clean, re.I))
                if is_cause:
                    clean = re.sub(r"^Why:\s*", "", clean, flags=re.I)
                yield clean, "possible_cause" if is_cause else "information", offset, offset + len(raw.rstrip("\r\n"))
        offset += len(raw)


def _citation(chunk, start, end, *, asset, customer, user_id, kind):
    payload = {"tenant": asset.tenant_id, "customer": customer.id, "asset": asset.id,
               "document": chunk.document_id, "chunk": chunk.id,
               "version": chunk.document.index_version,
               "sha": hashlib.sha256(chunk.text.encode()).hexdigest(),
               "start": start, "end": end, "user": user_id, "kind": kind}
    token = signing.dumps(payload, salt=SALT, compress=True)
    return {"document_id": chunk.document_id, "chunk_id": chunk.id,
            "title": chunk.document.title, "heading": chunk.heading or "",
            "document_version": chunk.document.index_version,
            "url": "/api/customer-support/source/?citation=" + token}


def _approved_source_action(text, chunk, asset, tenant):
    from core.models import ProcedureApproval
    from .safety_authority import validate_procedure_safety_approval
    for approval in ProcedureApproval.objects.filter(
        tenant=tenant, document_id=chunk.document_id, source_chunk_id=chunk.id,
        approval_status="APPROVED", permitted_audience="customer", revoked_at__isnull=True,
    ):
        # Free-form QA cannot establish physical prerequisites. The existing
        # diagnostic workflow must verify them before such work is offered.
        if approval.prerequisites:
            continue
        normalize = lambda value: " ".join(_clean_line(value).casefold().split()).rstrip(".")
        if normalize(approval.normalized_description) != normalize(text):
            continue
        approved, _, _ = validate_procedure_safety_approval(approval, text, asset, tenant)
        if approved:
            return True
    return False


def open_citation(token, ctx):
    try:
        payload = signing.loads(token, salt=SALT, max_age=3600)
        if payload["user"] != ctx["user"].id or payload["tenant"] != ctx["tenant"].id:
            raise ValueError()
        # Staff previews still require the same bound customer. No automatic
        # superuser or tenant-wide fallback is permitted for customer excerpts.
        customer = ctx.get("customer")
        if not customer or customer.id != payload["customer"]:
            raise ValueError()
        from core.models import Asset
        asset = Asset.objects.select_related("tenant", "customer", "product").get(
            pk=payload["asset"], tenant=ctx["tenant"], customer=customer)
        chunk = source_chunk(tenant=ctx["tenant"], customer=customer, asset=asset,
                             document_id=payload["document"], chunk_id=payload["chunk"])
        if (chunk.document.index_version != payload["version"]
            or hashlib.sha256(chunk.text.encode()).hexdigest() != payload["sha"]):
            raise ValueError()
        start, end = payload["start"], payload["end"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(chunk.text)):
            raise ValueError()
        excerpt = chunk.text[start:end]
        if payload["kind"] in ("observation", "customer_procedure"):
            from .customer_diagnostic import classify_action_safety
            level, _ = classify_action_safety(_clean_line(excerpt), asset)
            if payload["kind"] == "observation" and not (
                level in ("INFORMATIONAL_OBSERVATION", "UNKNOWN_REQUIRES_REVIEW")
                and _noninvasive_observation(_clean_line(excerpt))):
                raise ValueError()
            if payload["kind"] == "customer_procedure" and not (level == "CUSTOMER_APPROVED" and _approved_source_action(
                _clean_line(excerpt), chunk, asset, ctx["tenant"])):
                raise ValueError()
        return {"title": chunk.document.title, "heading": chunk.heading or "",
                "excerpt": excerpt, "document_id": chunk.document_id, "chunk_id": chunk.id}
    except Exception:
        # Do not disclose whether an inaccessible document exists.
        raise NotFound("Source unavailable or access changed. Ask the question again.")


def exception_presentation(contract):
    hazard = contract.get("response_mode") == "IMMEDIATE_SAFETY_ESCALATION"
    return {"schema_version": "customer-answer-v1", "response_type": contract.get("response_type"),
            "equipment": contract.get("asset", {}), "sections": [], "citations": [],
            "hazard": {"severity": "immediate" if hazard else "none",
                       "message": (contract.get("stop_conditions") or [""])[0] if hazard else ""},
            "message": contract.get("clarification_question") or contract.get("summary") or "Please ask for technician assistance.",
            "authorized_user_actions": {"can_resolve": False, "can_escalate": bool(contract.get("can_escalate"))}}


def build_customer_answer(interaction, session, references, conversation_context=None, original_question=None):
    from .customer_diagnostic import classify_action_safety
    from .llm import check_evidence_sufficiency, extract_typed_quantities, verify_quantity_support, _STOPWORDS
    from .identifiers import extract_error_codes
    from .retriever import normalize_domain_terms
    asset, customer, tenant = interaction.asset, interaction.customer, interaction.tenant
    question = interaction.question or ""
    raw_user_q = original_question or question
    candidates = []
    seen = set()
    for ref in (references or [])[:30]:
        if not isinstance(ref, dict):
            continue
        try:
            chunk = source_chunk(tenant=tenant, customer=customer, asset=asset,
                                 document_id=ref.get("document_id"), chunk_id=ref.get("chunk_id"))
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            score = float(ref.get("score", 0))
            if not math.isfinite(score):
                continue
            candidates.append(SimpleNamespace(chunk=chunk, text=chunk.text, heading=chunk.heading, title=chunk.document.title,
                score=score, source_verified=True, equipment_applicable=True, is_quarantined=False,
                customer_action_authorized=False, exact_identifier_match=False))
        except (NotFound, ValueError, TypeError):
            continue
    guard = _question_guard(raw_user_q) or _question_guard(question)
    if guard:
        return _guarded_customer_response(interaction, session, candidates, guard)
    explicit_model = re.search(r"\bmodel\s+([a-z0-9]+[-][a-z0-9-]+)", question, re.I)
    wrong_model = explicit_model and asset and explicit_model[1].casefold() != (asset.model_number or "").casefold()
    fact_family = _document_fact_family(question)
    if fact_family and not wrong_model and not query_is_control_abuse(question):
        fact_answer = _document_fact_response(interaction, session, fact_family)
        if fact_answer:
            return fact_answer
    sufficient = check_evidence_sufficiency(question, asset, candidates)
    causes, observations, procedures, citations = [], [], [], []
    terms = set(re.findall(r"[a-z0-9_-]{3,}", normalize_domain_terms(question).casefold())) - _STOPWORDS
    codes = extract_error_codes(question)
    quantities = extract_typed_quantities(question)
    if sufficient and not wrong_model and not query_is_control_abuse(question):
        for result in sufficient.grounded_chunks:
            for text, kind, start, end in source_claims(result.text):
                words = set(re.findall(r"[a-z0-9_-]{3,}", text.casefold())) - _STOPWORDS
                heading_words = set(re.findall(r"[a-z0-9_-]{3,}", (result.heading or "").casefold()))
                if not terms.intersection(words | heading_words):
                    continue
                span_context = f"{result.heading or ''} {text}"
                if codes and not codes.issubset(extract_error_codes(span_context)):
                    continue
                if any(not verify_quantity_support(q, extract_typed_quantities(span_context)) for q in quantities):
                    continue
                level, _ = classify_action_safety(text, asset)
                completed_checks = getattr(conversation_context, "completed_checks", None) or []
                def _is_completed_check(claim_text):
                    if not completed_checks:
                        return False
                    c_words = set(re.findall(r"[a-z0-9]{3,}", claim_text.casefold())) - _STOPWORDS
                    for done in completed_checks:
                        done_words = set(re.findall(r"[a-z0-9]{3,}", str(done).casefold())) - _STOPWORDS
                        if c_words and done_words and (c_words & done_words):
                            return True
                    return False

                if kind == "possible_cause":
                    # A descriptive cause list is information, never an action.
                    if not _descriptive_cause(text):
                        continue
                    target = causes
                elif level in ("INFORMATIONAL_OBSERVATION", "UNKNOWN_REQUIRES_REVIEW") and _noninvasive_observation(text):
                    if _is_completed_check(text):
                        continue
                    target, kind = observations, "observation"
                elif level == "CUSTOMER_APPROVED" and _approved_source_action(text, result.chunk, asset, tenant):
                    if _is_completed_check(text):
                        continue
                    target, kind = procedures, "customer_procedure"
                elif (kind == "information" and _descriptive_cause(text)
                      and re.search(r"\b(?:indicates?|means?|denotes?|occurs? when|is caused by)\b", text, re.I)):
                    target = causes
                else:
                    continue
                if (len(target) >= 2 or (target is not causes and len(observations + procedures) >= 2)
                    or any(c["source_text"] == text for c in causes + observations + procedures)):
                    continue
                citation = _citation(result.chunk, start, end, asset=asset, customer=customer,
                                     user_id=interaction.created_by_id, kind=kind)
                citation["id"] = f"s{len(citations)+1}"
                citations.append(citation)
                target.append({"id": f"c{len(citations)}", "text": ("Possible documented cause: " + text) if kind == "possible_cause" else text,
                               "source_text": text, "kind": kind, "evidence_ids": [citation["id"]]})
    normal = bool(causes or observations or procedures)
    no_authorized_docs = False
    if not normal and not candidates:
        from .retriever import _knowledge_candidates
        no_authorized_docs = not _knowledge_candidates(
            tenant, asset, customer, False, backend="all")
    if no_authorized_docs:
        from .customer_isolation import get_customer_data_mode
        reason_code = ("NO_AUTHORIZED_ORIGINAL_DOCUMENTATION"
                       if get_customer_data_mode() == "verified_real_only"
                       else "NO_AUTHORIZED_TEST_DOCUMENTATION")
        message = ("INSUFFICIENT EVIDENCE: No authorized documentation is available for this equipment. "
                   "I cannot identify a cause, give repair steps, or cite a manual. "
                   "Please note the model, symptom, when it occurs, and any displayed code "
                   "for an authorized technician. No ticket has been created.")
    else:
        reason_code = None
        message = "There is insufficient evidence to establish a safe answer for this equipment and question. Ask an authorized technician for help."
    sections = []
    if normal:
        observation_topics = [re.sub(
            r"^(?:observe|read|record|report|note|check|confirm|inspect|verify|examine|look at)\s+",
            "", claim["source_text"], flags=re.I).rstrip(" .")
            for claim in observations[:2]]
        completed_checks = getattr(conversation_context, "completed_checks", None) or []
        if completed_checks:
            clean_done = ", ".join(str(c) for c in completed_checks[:2])
            if observation_topics:
                unknown_text = (
                    f"Customer reports completing: {clean_done}. "
                    "A confirmed diagnosis still needs the observation results for "
                    + "; ".join(observation_topics)
                    + ". This chat has not verified those conditions."
                )
            else:
                unknown_text = (
                    f"Customer reports completing: {clean_done}. "
                    "The available passages do not provide further customer-safe observations "
                    "that confirm which documented possibility applies."
                )
            next_text = (
                f"Since {clean_done} was checked, if the symptom continues, share those results "
                "and the equipment model with an authorized technician. No ticket has been created."
            )
        elif observation_topics:
            unknown_text = ("A confirmed diagnosis still needs the observation results for "
                            + "; ".join(observation_topics)
                            + ". This chat has not verified those conditions.")
            next_text = ("If the symptom continues, share those observation results and the equipment model "
                         "with an authorized technician. No ticket has been created.")
        else:
            unknown_text = ("The available passages do not provide a customer-safe observation "
                            "that confirms which documented possibility applies.")
            next_text = ("Ask an authorized technician for model-specific help if the issue continues. "
                         "No ticket has been created.")
        content = [
            causes or [{"text": "The available evidence does not establish a possible cause.", "kind": "uncertainty", "evidence_ids": []}],
            observations + procedures or [{"text": "No customer procedure has been approved in the supporting evidence. Report what you observed to a technician.", "kind": "uncertainty", "evidence_ids": []}],
            [{"text": unknown_text, "kind": "uncertainty", "evidence_ids": []}],
            [{"text": next_text, "kind": "next_step", "evidence_ids": []}],
        ]
        sections = [{"number": i+1, "title": title, "claims": claims} for i, (title, claims) in enumerate(zip(TITLES, content))]
    can_resolve = bool(normal and (observations or procedures) and (not session or session.status not in ("ESCALATED", "ABANDONED")))
    response_type = "DIAGNOSTIC_GUIDANCE" if normal else "INSUFFICIENT_EVIDENCE"
    presentation = {"schema_version": "customer-answer-v1", "response_type": response_type,
        "equipment": {"id": asset.id, "name": asset.name, "model_number": asset.model_number} if asset else {},
        "sections": sections, "citations": citations, "hazard": {"severity": "none", "message": ""},
        "supporting_claims": causes + observations + procedures,
        "customer_permitted_observations": observations,
        "customer_authorized_actions": procedures,
        "message": "" if normal else message,
        "technician_only_warnings": [] if procedures else ["Physical procedures require separate customer approval or an authorized technician."],
        "known_limitations": ["No physical inspection or confirmed diagnosis."],
        "authorized_user_actions": {"can_resolve": can_resolve, "can_escalate": True}}
    answer = "\n\n".join(f"{s['number']}. {s['title']}\n" + " ".join(c["text"] for c in s["claims"]) for s in sections) if normal else message
    actions = [{"action_id": c["id"], "action": c["text"], "safety_classification": "INFORMATIONAL_OBSERVATION" if c["kind"] == "observation" else "CUSTOMER_APPROVED",
                "document_id": next(s["document_id"] for s in citations if s["id"] == c["evidence_ids"][0]),
                "chunk_id": next(s["chunk_id"] for s in citations if s["id"] == c["evidence_ids"][0])} for c in observations + procedures]
    return {"presentation": presentation, "response_mode": "ANSWER_WITH_VERIFIED_EVIDENCE" if normal else "INSUFFICIENT_EVIDENCE",
        "response_type": response_type, "interaction_id": interaction.id,
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE", "session_version": session.version if session else 1,
        "asset": presentation["equipment"], "summary": "Possible causes from documentation; diagnosis unconfirmed." if normal else message,
        "answer": answer, "citations": citations, "references": citations,
        "customer_safe_actions": actions, "check_first": [], "technician_only_actions": [],
        "stop_conditions": [], "expected_result": "", "clarification_question": None,
        "can_resolve": can_resolve, "can_escalate": True, "engine": "source-structured-extractive",
        "reason_code": reason_code}
