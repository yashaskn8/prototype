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


# Cause lists sometimes contain checklist bullets or observation prompts.
# Treat a sentence as a cause only when it describes a condition, not when it
# tells the reader to inspect, record, operate or report an observation.
_CAUSE_DIRECTIVE_VERBS = (
    r"(?:do not|don't|never|always|please|"
    r"check(?:s|ed|ing)?|confirm(?:s|ed|ing)?|inspect(?:s|ed|ing)?|"
    r"verify(?:s|ied|ing)?|observe(?:s|d|ing)?|read(?:s|ing)?|"
    r"report(?:s|ed|ing)?|record(?:s|ed|ing)?|note(?:s|d|ing)?|"
    r"look(?:s|ed|ing)?(?:\s+(?:for|at))?|monitor(?:s|ed|ing)?|"
    r"collect(?:s|ed|ing)?|capture(?:s|d|ing)?|document(?:s|ed|ing)?|"
    r"ensure|make sure|open(?:s|ed|ing)?|remove(?:s|d|ing)?|"
    r"replace(?:s|d|ing)?|disconnect(?:s|ed|ing)?|connect(?:s|ed|ing)?|"
    r"reconnect(?:s|ed|ing)?|bypass(?:es|ed|ing)?|charge(?:s|d|ing)?|"
    r"adjust(?:s|ed|ing)?|measure(?:s|d|ing)?|probe(?:s|d|ing)?|"
    r"touch(?:es|ed|ing)?|strip(?:s|ped|ping)?|clean(?:s|ed|ing)?|"
    r"wipe(?:s|d|ing)?|rinse(?:s|d|ing)?|flush(?:es|ed|ing)?|"
    r"restart(?:s|ed|ing)?|reset(?:s|ted|ting)?|press(?:es|ed|ing)?|"
    r"turn(?:s|ed|ing)?|disassemble(?:s|d|ing)?|dismantle(?:s|d|ing)?|"
    r"repair(?:s|ed|ing)?|test(?:s|ed|ing)?|rotate(?:s|d|ing)?|spin(?:s|ning)?|"
    r"tighten(?:s|ed|ing)?|loosen(?:s|ed|ing)?|lubricate(?:s|d|ing)?|"
    r"drain(?:s|ed|ing)?|fill(?:s|ed|ing)?|insert(?:s|ed|ing)?|"
    r"unlock(?:s|ed|ing)?|energize(?:s|d|ing)?|switch(?:es|ed|ing)?|"
    r"plug(?:s|ged|ging)?|solder(?:s|ed|ing)?|heat(?:s|ed|ing)?|"
    r"manipulate(?:s|d|ing)?|pour(?:s|ed|ing)?|mix(?:es|ed|ing)?|"
    r"wash(?:es|ed|ing)?|pull(?:s|ed|ing)?|push(?:es|ed|ing)?|"
    r"use|refer(?:s|red|ring)?|follow(?:s|ed|ing)?|escalate|stop|"
    r"isolate|attach|create|review|perform|position|compare|pressure[- ]test|"
    r"link|select|allow|apply|provide)\b"
)
_CAUSE_DIRECTIVE_START = re.compile(
    r"^(?:" + _CAUSE_DIRECTIVE_VERBS + r")|"
    r"(?:[.;:]\s+)(?:" + _CAUSE_DIRECTIVE_VERBS + r")",
    re.I,
)
_CAUSE_MODAL_DIRECTIVE = re.compile(
    r"\b(?:must|should|need to|try to|you can|you should)\s+"
    r"(?:not\s+)?(?:" + _CAUSE_DIRECTIVE_VERBS + r")",
    re.I,
)
_CAUSE_OBSERVATION_CUE = re.compile(
    r"\b(?:evidence of|signs? of|indications? of|look(?:ing)? for|"
    r"check(?:ing)? for|inspect(?:s|ed|ing)?|inspection|visually|"
    r"observe|report|record|note|monitor|may indicate)\b",
    re.I,
)


def _descriptive_cause(text):
    """Reject instructions and observation cues while retaining cause nouns."""
    clean = " ".join(_clean_line(text).split())
    return bool(clean) and not (
        _CAUSE_DIRECTIVE_START.search(clean)
        or _CAUSE_MODAL_DIRECTIVE.search(clean)
        or _CAUSE_OBSERVATION_CUE.search(clean)
    )
_IMPERATIVE_START = re.compile(
    r"^(?:\[\s?[xX ]?\]\s*)?(?:do not|don't|never|always|please|"
    + PHYSICAL_OPERATION.pattern[len(r"\b(?:"):-len(r")\b")] + r"|"
    r"check|confirm|inspect|verify|observe|read|report|record|note|use|refer|"
    r"escalate|stop|isolate|attach|create|review|look|have|make|ensure|keep|"
    r"follow|arrange|mount|route|start|lift|secure|complete|calculate|evacuate|"
    r"perform|position|compare|pressure-test|link|select|allow|apply|provide)\b", re.I)
_SCHEDULE_FACT = re.compile(
    r"^(?:daily|weekly|fortnightly|monthly|quarterly|half-yearly|semi-annually|"
    r"annually|yearly|every\s+\d+\s+(?:days?|weeks?|months?|years?|hours?))\b[^:]{0,60}:\s*\S",
    re.I)
_FACT_QUALIFIER = re.compile(
    r"\b(?:takes? precedence|planning intervals only|adjust intervals)\b", re.I)
_CAUSE_EXPLANATION_REQUEST = re.compile(
    r"\b(?:explain|elaborate|break down|describe in detail|more detail|in detail)\b", re.I)
_CAUSE_QUESTION = re.compile(
    r"\b(?:causes?|reasons?|why|wrong|diagnos\w*|root cause|issues?|problems?|"
    r"malfunction\w*|not working|(?:isn'?t|doesn'?t|won'?t|stopped) work\w*)\b", re.I)

_GENERIC_FAILURE_QUERY = re.compile(
    r"\b(?:not working|does(?:n'?t| not) work|won'?t work|"
    r"isn'?t functioning|not functioning|stopped working|not operational)\b", re.I)
_SPECIFIC_FAULT_QUERY = re.compile(
    r"\b(?:not cooling|poor cooling|weak cooling|no cooling|"
    r"not heating|no heating|weak airflow|low airflow|water leak\w*|"
    r"leakage|rattl\w*|vibrat\w*|nois\w*|communication fault|"
    r"address(?:ing)? fault|low[ -]?pressure|high[ -]?pressure|"
    r"capacity fault|compressor fault|inverter fault|fan fault|"
    r"temperature[- ]sensor fault|fault code|error code|alarm code)\b", re.I)
_GENERIC_CAUSE_REQUEST = re.compile(
    r"\b(?:possible\s+causes?|possible\s+reasons?|causes?|reasons?|"
    r"issues?|problems?|why)\b", re.I)


def is_generic_cause_overview(question):
    """Identify a cause overview attached to a generic equipment failure.

    This permits conditional source claims under their exact manual headings;
    it does not turn a vague failure report into a fault-category diagnosis.
    """
    value = " ".join(str(question or "").split())
    return bool(
        _GENERIC_FAILURE_QUERY.search(value)
        and _GENERIC_CAUSE_REQUEST.search(value)
        and not _SPECIFIC_FAULT_QUERY.search(value)
    )


def _is_cause_coverage_review(question):
    """Detect a source-support audit without inferring a prior equipment symptom."""
    value = " ".join(str(question or "").split()).casefold()
    asks_causes = bool(re.search(r"\b(?:causes?|reasons?|possibilities)\b", value))
    asks_support = bool(
        re.search(r"\b(?:supported|documented|listed|backed)\b", value)
        and re.search(r"\b(?:manual|approved|documentation|document|source)\b", value)
    )
    asks_limit = bool(re.search(
        r"\b(?:cannot|can not|can't|could not|couldn't|not)\b"
        r".{0,70}\b(?:establish\w*|confirm\w*|determin\w*|"
        r"verif\w*|identif\w*|appl\w*)\b", value
    ))
    return asks_causes and asks_support and asks_limit


_RESTRICTED_LEVELS = frozenset({"PROHIBITED", "TECHNICIAN_ONLY"})
_FREQUENCY_QUESTION = re.compile(
    r"\b(?:how (?:often|frequently|regularly|many times)|"
    r"when (?:is|are|should|do|does|must)|every\s+\d+|intervals?|schedules?|"
    r"frequency|periodic\w*|monthly|quarterly|annual\w*|yearly)\b", re.I)
# An instruction is not only a sentence that starts with a verb: "During
# brazing, use ..." is one too. Any clause-initial directive disqualifies.
_CLAUSE_DIRECTIVE = re.compile(
    r"(?:^|[,;:]\s+)" + _IMPERATIVE_START.pattern.lstrip("^"), re.I)


def _sentence_spans(chunk_text, line, start, end):
    """Exact (sentence, start, end) spans of a multi-sentence source line.

    Only sentences found verbatim inside the line's own span are returned, so
    a citation built from one always opens to precisely that sentence.
    """
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line or "")]
    if len(sentences) < 2:
        return []
    spans, cursor = [], start
    for sentence in sentences:
        position = (chunk_text or "").find(sentence, cursor, end + 1) if sentence else -1
        if position < 0:
            continue
        spans.append((sentence, position, position + len(sentence)))
        cursor = position + len(sentence)
    return spans


def customer_safe_fact(text, level):
    """Classify an informational source sentence for customer disclosure.

    Returns "maintenance_schedule", "descriptive", or None. The existing
    action-safety classification is the authority: anything it marks
    technician-only or prohibited is never disclosed. Beyond that a sentence
    qualifies only if it is a labelled schedule entry ("Annually: ...") or a
    plain statement containing no directive and no physical operation.
    Anything else, including checklist items and instructions, fails closed.
    """
    clean = " ".join(str(text or "").split())
    # Blank form fields and labels ("Measured airflow: ____") state nothing.
    if (not clean or level in _RESTRICTED_LEVELS or "_" in clean
            or clean.endswith(":") or len(clean.split()) < 4):
        return None
    if _SCHEDULE_FACT.match(clean):
        return "maintenance_schedule"
    if (_CLAUSE_DIRECTIVE.search(clean) or PHYSICAL_OPERATION.search(clean)
            or not _descriptive_cause(clean)):
        return None
    return "descriptive"


def _question_guard(question):
    """Require a direct answer for permission, proof, or exact-spec questions."""
    from .retriever import normalize_domain_terms
    lower = normalize_domain_terms(question).casefold()
    # A modal verb and an action word are not enough to establish a
    # request for permission. "Can open windows cause poor cooling?" is a
    # causal question, not a request to operate equipment. Require an explicit
    # actor or an explicit "safe/allowed to" construction.
    permission = re.search(
        r"\b(?:can|may|should|shud|shall|could)\s+"
        r"(?:(?:i|we|you|staff|technician|operator|customer)\b|"
        r"(?:the|a)\s+(?:staff|technician|operator|customer)\b)"
        r"|\b(?:am i|are we|is it)\s+(?:allowed|permitted|safe|okay|acceptable)\s+to\b"
        r"|\b(?:allowed|permitted|acceptable)\s+to\b",
        lower,
    )
    physical = re.search(
        r"\b(?:bypas\w*|defeat|bridge|resett?\w*|clear(?:ing)? (?:the )?fault|"
        r"open|probe|wire|measure|chip|adjust|vent|charg\w*|top up|spray|fix|repair|"
        r"keep (?:the |it |unit |equipment |system ){0,3}(?:running|operating)|"
        r"stay on|operate|run|change fan speed)\b", lower)
    if permission and physical:
        return "safety"
    if re.search(r"\b(?:proves?|definitely|certainly|must mean|for sure)\b", lower):
        return "unsupported"
    # Match a request for the exact technical item itself. Requiring the
    # specification noun to follow "exact" avoids treating contextual terms
    # such as "low-pressure fault" as a request for an exact pressure value.
    if re.search(
        r"\bexact\s+(?:(?:manufacturer(?:'s)?(?:[- ](?:specified|approved|rated))?|"
        r"specified|rated|approved|operating|maximum|minimum|allowed|permitted)[- ]+){0,2}"
        r"(?:value|limit|threshold|setting|pressure|torque|current|charge|"
        r"code|range|clearance|interval)\b",
        lower,
    ):
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
            "can_resolve": False, "can_escalate": True,
            # This branch is a deterministic policy boundary. It does not
            # invoke Qwen and must not reference the generation-path flag.
            "engine": "customer-question-guard"}


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
    equipment = {"id": asset.id, "name": asset.name, "model_number": asset.model_number}
    presentation = {"schema_version": "customer-answer-v1", "response_type": "DIAGNOSTIC_GUIDANCE",
                    "equipment": equipment, "sections": sections, "citations": citations,
                    "supporting_claims": claims, "customer_permitted_observations": [],
                    "customer_authorized_actions": [], "hazard": {"severity": "none", "message": ""},
                    "message": "", "known_limitations": ["Document facts are not customer procedure approval."],
                    "authorized_user_actions": {"can_resolve": False, "can_escalate": True}}
    answer = "\n\n".join(f"{s['number']}. {s['title']}\n" + " ".join(c["text"] for c in s["claims"])
                          for s in sections)
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
    clean = re.sub(r"^\s*\[\s?[xX ]\s?\]\s*", "", clean)
    clean = clean.replace("**", "").replace("__", "").replace(chr(96), "")
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
                is_cause = (
                    (cause_list and bool(re.match(r"(?:[-*•]+|\d+[.)])\s", line))
                     and _descriptive_cause(clean))
                    or (bool(re.search(
                        r"\b(?:may be caused by|(?:is|are|can be) caused by|possible cause|"
                        r"can cause|may cause|causing|leads? to|results? in)\b",
                        clean, re.I,
                    )) and _descriptive_cause(clean))
                )
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


def build_customer_answer(
    interaction, session, references, conversation_context=None,
    original_question=None, generate_with_local_model=True,
):
    from .customer_diagnostic import classify_action_safety
    from .llm import check_evidence_sufficiency, extract_typed_quantities, verify_quantity_support, _STOPWORDS
    from .identifiers import extract_error_codes
    from .retriever import normalize_domain_terms
    from .conversation_context import completed_action_matches
    from .term_matching import (
        heading_states_symptom, match_keys, question_states_heading, stem, symptom_concepts,
    )
    asset, customer, tenant = interaction.asset, interaction.customer, interaction.tenant
    question = interaction.question or ""
    raw_user_q = original_question or question
    candidates = []
    seen = set()
    for ref in (references or [])[:300]:
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
    generic_cause_overview = (
        is_generic_cause_overview(raw_user_q)
        or is_generic_cause_overview(question)
        or (
            conversation_context is not None
            and is_generic_cause_overview(
                getattr(conversation_context, "original_complaint", "")
            )
            and _is_cause_coverage_review(raw_user_q or question)
        )
    )
    evidence_chunks = list(sufficient.grounded_chunks or [])
    if generic_cause_overview:
        # A generic failure does not identify one symptom heading. For this
        # explicit cause-overview request, include score-qualified, selected-
        # asset, re-authorized chunks only when their own heading is a fault
        # condition and their source contains a descriptive cause sentence.
        from django.conf import settings as django_settings
        minimum_score = float(getattr(django_settings, "SERVY_RAG_MIN_SCORE", 0.055))
        seen_cause_chunks = {candidate.chunk.id for candidate in evidence_chunks}
        for candidate in candidates:
            if (candidate.chunk.id in seen_cause_chunks
                    or candidate.score < minimum_score
                    or not candidate.source_verified
                    or not candidate.equipment_applicable
                    or candidate.is_quarantined
                    or not re.search(r"\b(?:fault|symptom)\b", candidate.heading or "", re.I)):
                continue
            if any(
                kind == "possible_cause" and _descriptive_cause(text)
                for text, kind, _start, _end in source_claims(candidate.text)
            ):
                evidence_chunks.append(candidate)
                seen_cause_chunks.add(candidate.chunk.id)
    causes, observations, procedures, facts, citations = [], [], [], [], []
    terms = set(re.findall(r"[a-z0-9_-]{3,}", normalize_domain_terms(question).casefold())) - _STOPWORDS
    # One matching rule for every answer-stage check: folded word forms, plus
    # a symptom paraphrase only where the section heading names that symptom.
    term_keys = match_keys(terms)
    asked_symptoms = symptom_concepts(raw_user_q) | symptom_concepts(question)
    wants_facts = not _CAUSE_QUESTION.search(raw_user_q) and not _CAUSE_QUESTION.search(question)
    wants_cause_details = bool(
        _CAUSE_EXPLANATION_REQUEST.search(raw_user_q)
        or _CAUSE_EXPLANATION_REQUEST.search(question)
    )
    asks_frequency = bool(_FREQUENCY_QUESTION.search(raw_user_q))
    codes = extract_error_codes(question)
    quantities = extract_typed_quantities(question)
    eligible = []
    if evidence_chunks and (sufficient or generic_cause_overview) and not wrong_model and not query_is_control_abuse(question):
        for result in evidence_chunks:
            for text, kind, start, end in source_claims(result.text):
                words = set(re.findall(r"[a-z0-9_-]{3,}", text.casefold())) - _STOPWORDS
                heading_words = set(re.findall(r"[a-z0-9_-]{3,}", (result.heading or "").casefold()))
                sentence_overlap = term_keys & match_keys(words)
                generic_section_cause = bool(
                    generic_cause_overview
                    and kind == "possible_cause"
                    and re.search(r"\b(?:fault|symptom)\b", result.heading or "", re.I)
                )
                if (not sentence_overlap and not term_keys & match_keys(heading_words)
                        and not heading_states_symptom(result.heading, asked_symptoms)
                        and not generic_section_cause):
                    continue
                span_context = f"{result.heading or ''} {text}"
                if codes and not codes.issubset(extract_error_codes(span_context)):
                    continue
                if any(not verify_quantity_support(q, extract_typed_quantities(span_context)) for q in quantities):
                    continue
                level, _ = classify_action_safety(text, asset)
                completed_checks = getattr(conversation_context, "completed_checks", None) or []
                def _is_completed_check(claim_text):
                    return completed_action_matches(claim_text, completed_checks)

                if kind == "possible_cause":
                    # A descriptive cause list is information, never an action.
                    if not _descriptive_cause(text):
                        continue
                    if generic_section_cause:
                        # The customer asked for a broad overview, not a
                        # diagnosis. Preserve the source section as a condition.
                        target = causes
                    else:
                        # A cause that shares no word with the question is
                        # relevant only through its section. Sharing some heading
                        # words is not enough ("server room temperature" is not
                        # "uneven room temperatures"): the question must state the
                        # symptom that heading names.
                        if not sentence_overlap and not (
                                heading_states_symptom(result.heading, asked_symptoms)
                                or question_states_heading(
                                    f"{raw_user_q} {question}", result.heading)):
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
                elif kind == "information" and (wants_facts or wants_cause_details):
                    # Customer-safe documented fact (for example a maintenance
                    # interval). It must itself answer the question: the
                    # sentence shares a question term, and sentence plus
                    # heading share at least two. Restricted, directive or
                    # unclassifiable sentences are never disclosed.
                    disclosure = customer_safe_fact(text, level)
                    if not disclosure:
                        # A manual line often joins a plain statement with an
                        # instruction ("This guide covers X. Confirm Y before
                        # work begins."). The line as a whole stays undisclosed,
                        # but a sentence that is itself a customer-safe
                        # statement and itself answers the question may be
                        # cited by its own exact span. Each sentence is safety
                        # classified separately; instructions stay excluded.
                        for part, part_start, part_end in _sentence_spans(
                                result.text, text, start, end):
                            part_level, _ = classify_action_safety(part, asset)
                            if customer_safe_fact(part, part_level) != "descriptive":
                                continue
                            part_words = set(
                                re.findall(r"[a-z0-9_-]{3,}", part.casefold())) - _STOPWORDS
                            part_overlap = term_keys & match_keys(part_words)
                            if (len({stem(w) for w in part_overlap}) < 2
                                    or (asks_frequency and not re.search(
                                        r"\b(?:frequency|intervals?|schedules?)\b", part, re.I))):
                                continue
                            eligible.append({
                                "order": len(eligible), "target": facts, "kind": "information",
                                "text": part, "start": part_start, "end": part_end,
                                "result": result,
                            })
                        continue
                    if not sentence_overlap:
                        continue
                    if disclosure == "maintenance_schedule":
                        # A schedule entry answers only a question about timing.
                        if (not asks_frequency
                                or len(term_keys & match_keys(words | heading_words)) < 2):
                            continue
                    elif (len({stem(w) for w in sentence_overlap}) < 2
                          or (asks_frequency and not re.search(
                              r"\b(?:frequency|intervals?|schedules?)\b", text, re.I))):
                        # A plain statement must itself share two distinct
                        # question terms. For a timing question it must be a
                        # statement about frequency; it never stands in for a
                        # schedule entry otherwise.
                        continue
                    target = facts
                else:
                    continue
                # Candidates are ranked for relevance to this question below,
                # then capped; the cap must not keep an earlier but less
                # relevant sentence merely because it appears first.
                eligible.append({
                    "order": len(eligible), "target": target, "kind": kind,
                    "text": text, "start": start, "end": end, "result": result,
                })
        seen_text = set()
        unique = []
        for item in eligible:
            if item["text"] in seen_text:
                continue
            seen_text.add(item["text"])
            unique.append(item)
        def _relevance(item):
            claim_words = re.findall(r"[a-z0-9_-]{3,}", item["text"].casefold())
            return len(term_keys & match_keys(claim_words))

        cause_pool = [i for i in unique if i["target"] is causes]
        fact_pool = [i for i in unique if i["target"] is facts]
        action_pool = [i for i in unique if i["target"] is not causes and i["target"] is not facts]
        fact_choice = sorted(fact_pool, key=lambda i: -_relevance(i))[:4]
        # A schedule entry is shown together with the document's own
        # qualification of that schedule, so its conditions are preserved.
        if any(_SCHEDULE_FACT.match(i["text"]) for i in fact_choice):
            for result in {id(i["result"]): i["result"] for i in fact_choice}.values():
                for text, kind, start, end in source_claims(result.text):
                    if (kind == "information" and _FACT_QUALIFIER.search(text)
                            and text not in seen_text
                            and customer_safe_fact(text, classify_action_safety(text, asset)[0])):
                        seen_text.add(text)
                        fact_choice.append({"order": len(eligible) + len(fact_choice),
                                            "target": facts, "kind": "information",
                                            "text": text, "start": start, "end": end,
                                            "result": result})
        # Preserve up to four separately grounded claims per answer section.
        chosen = (sorted(cause_pool, key=lambda i: -_relevance(i))[:20 if generic_cause_overview else 10]
                  + fact_choice
                  + sorted(action_pool, key=lambda i: -_relevance(i))[:8])
        chosen.sort(key=lambda i: i["order"])
        for item in chosen:
            target, kind, text = item["target"], item["kind"], item["text"]
            citation = _citation(item["result"].chunk, item["start"], item["end"], asset=asset,
                                 customer=customer, user_id=interaction.created_by_id, kind=kind)
            citation["id"] = f"s{len(citations)+1}"
            citations.append(citation)
            target.append({"id": f"c{len(citations)}", "text": ("Possible documented cause: " + text) if kind == "possible_cause" else text,
                           "source_text": text, "kind": kind, "evidence_ids": [citation["id"]],
                           **({"source_heading": item["result"].heading,
                               "source_document_title": item["result"].title}
                              if kind == "possible_cause" else {})})
            if target is facts:
                target[-1]["disclosure"] = (
                    "maintenance_schedule" if _SCHEDULE_FACT.match(text) else "descriptive")
    # Headings of the authorized sections that contain a documented cause
    # list. They let the dialogue ask which documented condition applies
    # instead of a generic question when no cause matches the customer's words.
    cause_sections = []
    for candidate in candidates:
        label = re.sub(r"^\s*\d+[.)]\s*", "", candidate.heading or "")
        label = re.sub(r"^symptom\s*:\s*", "", label, flags=re.I).strip()
        if (label and label not in cause_sections
                and any(kind == "possible_cause" for _t, kind, _s, _e in source_claims(candidate.text))):
            cause_sections.append(label[:80])
    cause_sections = cause_sections[:10]
    normal = bool(causes or facts or observations or procedures)
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
            if observation_topics:
                next_text = (
                    f"You reported completing {clean_done}. If the next documented observation "
                    f"is safe and accessible, please report what you can see regarding "
                    f"{observation_topics[0]}. If not, ask an authorized technician. "
                    "No ticket has been created."
                )
            else:
                next_text = (
                    f"You reported completing {clean_done}. There is no further verified "
                    "customer-safe observation in the available passages; request "
                    "an authorized technician if the issue continues. No ticket has been created."
                )
        elif observation_topics:
            unknown_text = ("A confirmed diagnosis still needs the observation results for "
                            + "; ".join(observation_topics)
                            + ". This chat has not verified those conditions.")
            next_text = (
                f"If it can be done safely without opening or manipulating the equipment, "
                f"what can you report about {observation_topics[0]}? "
                "Otherwise request authorized technician assistance. No ticket has been created."
            )
        else:
            unknown_text = ("The available passages do not provide a customer-safe observation "
                            "that confirms which documented possibility applies.")
            next_text = ("Ask an authorized technician for model-specific help if the issue continues. "
                         "No ticket has been created.")
        content = [
            causes or facts or [{"text": "The available evidence does not establish a possible cause.", "kind": "uncertainty", "evidence_ids": []}],
            observations + procedures or [{"text": "No customer procedure has been approved in the supporting evidence. Report what you observed to a technician.", "kind": "uncertainty", "evidence_ids": []}],
            [{"text": unknown_text, "kind": "uncertainty", "evidence_ids": []}],
            [{"text": next_text, "kind": "next_step", "evidence_ids": []}],
        ]
        sections = [{"number": i+1, "title": title, "claims": claims} for i, (title, claims) in enumerate(zip(TITLES, content))]
        if facts and not causes:
            sections[0]["title"] = "What the document says"
    can_resolve = bool(normal and (observations or procedures) and (not session or session.status not in ("ESCALATED", "ABANDONED")))
    response_type = "DIAGNOSTIC_GUIDANCE" if normal else "INSUFFICIENT_EVIDENCE"
    presentation = {"schema_version": "customer-answer-v1", "response_type": response_type,
        "equipment": {"id": asset.id, "name": asset.name, "model_number": asset.model_number} if asset else {},
        "sections": sections, "citations": citations, "hazard": {"severity": "none", "message": ""},
        "supporting_claims": causes + facts + observations + procedures,
        "customer_permitted_observations": observations,
        "customer_authorized_actions": procedures,
        "message": "" if normal else message,
        "technician_only_warnings": [] if procedures else ["Physical procedures require separate customer approval or an authorized technician."],
        "known_limitations": ["No physical inspection or confirmed diagnosis."],
        "authorized_user_actions": {"can_resolve": can_resolve, "can_escalate": True}}
    from .customer_answer_generation import (
        _enabled as customer_generation_enabled,
        apply_customer_answer_generation_with_status,
    )
    from .rag_execution_log import execution_stage, emit_execution_event

    if normal and generate_with_local_model and customer_generation_enabled():
        with execution_stage(
            "customer_qwen_generation",
            "Applying local Qwen wording to approved, cited claims",
        ):
            qwen_generation = apply_customer_answer_generation_with_status(
                presentation, raw_user_q or question
            )
        used_qwen = qwen_generation.get("applied_claims", 0) > 0
        emit_execution_event(
            "customer_qwen_outcome",
            "Customer answer wording generation outcome",
            "info",
            metrics={
                "model_provider": qwen_generation.get("provider") or "ollama",
                "generation_status": qwen_generation.get("status") or "unknown",
                "candidate_count": qwen_generation.get("eligible_claims", 0),
                "supported_count": qwen_generation.get("accepted_claims", 0),
            },
        )
    else:
        used_qwen = False
        if normal and generate_with_local_model:
            # Keep disabled-provider metadata without presenting the no-op
            # helper call as a completed inference stage.
            qwen_generation = apply_customer_answer_generation_with_status(
                presentation, raw_user_q or question
            )
            emit_execution_event(
                "customer_qwen_outcome",
                "Customer answer generation outcome",
                "info",
                metrics={
                    "model_provider": qwen_generation.get("provider") or "disabled",
                    "generation_status": qwen_generation.get("status") or "disabled",
                    "candidate_count": qwen_generation.get("eligible_claims", 0),
                    "supported_count": qwen_generation.get("accepted_claims", 0),
                },
            )
            emit_execution_event(
                "customer_qwen_generation",
                "Local Qwen is disabled; retaining approved source text",
                "skipped",
            )
        else:
            emit_execution_event(
                "customer_qwen_generation",
                (
                    "Qwen skipped because the answer is not eligible for generated wording"
                    if not normal else
                    "Qwen skipped by the caller; retaining approved source text"
                ),
                "skipped",
            )
    answer = "\n\n".join(f"{s['number']}. {s['title']}\n" + " ".join(c["text"] for c in s["claims"]) for s in sections) if normal else message
    actions = [{"action_id": c["id"], "action": c["text"], "safety_classification": "INFORMATIONAL_OBSERVATION" if c["kind"] == "observation" else "CUSTOMER_APPROVED",
                "document_id": next(s["document_id"] for s in citations if s["id"] == c["evidence_ids"][0]),
                "chunk_id": next(s["chunk_id"] for s in citations if s["id"] == c["evidence_ids"][0])} for c in observations + procedures]
    return {"presentation": presentation, "response_mode": "ANSWER_WITH_VERIFIED_EVIDENCE" if normal else "INSUFFICIENT_EVIDENCE",
        "response_type": response_type, "interaction_id": interaction.id,
        "session_id": str(session.session_id) if session else None,
        "session_status": session.status if session else "ACTIVE", "session_version": session.version if session else 1,
        "asset": presentation["equipment"], "summary": (
            "Documented information from the approved manual." if facts and not causes
            else "Possible causes from documentation; diagnosis unconfirmed.") if normal else message,
        "answer": answer, "citations": citations, "references": citations,
        "documented_cause_sections": cause_sections,
        "customer_safe_actions": actions, "check_first": [], "technician_only_actions": [],
        "stop_conditions": [], "expected_result": "", "clarification_question": None,
        "can_resolve": can_resolve, "can_escalate": True,
        "engine": "source-structured-qwen" if used_qwen else "source-structured-extractive",
        "reason_code": reason_code}
