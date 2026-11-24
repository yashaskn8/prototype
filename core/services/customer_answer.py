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
# The customer answer contract (frontend customerAnswerContract.js) accepts at most
# this many claims per section; a larger section makes the whole answer invalid.
MAX_SECTION_CLAIMS = 12

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
    r"\b(?:not work(?:ing)?|does(?:n'?t| not) work|won'?t work|can'?t work|cannot work|"
    r"isn'?t functioning|not functioning|stopped working|not operational|"
    r"not running|not operating|not turning on|won'?t turn on|"
    r"doesn'?t turn on|"
    r"broke down|broken down|broken|malfunctioning|malfunction(?:ed)?|"
    r"failed|defective|inoperative|dead|"
    r"stopped suddenly|suddenly stopped|has stopped|quit working|ceased|"
    r"shut down|shut off|"
    r"what(?:'s|\s+is|\s+could\s+be)\s+wrong|something\s+(?:is|seems?)\s+wrong|"
    r"going wrong)\b", re.I)
_SPECIFIC_FAULT_QUERY = re.compile(
    r"\b(?:not cooling|poor cooling|weak cooling|no cooling|"
    r"not heating|no heating|weak airflow|low airflow|water leak\w*|"
    r"leakage|rattl\w*|vibrat\w*|nois\w*|communication fault|"
    r"not start\w*|won'?t start|doesn'?t start|does not start|will not start|"
    r"address(?:ing)? fault|low[ -]?pressure|high[ -]?pressure|"
    r"capacity fault|compressor fault|inverter fault|fan fault|"
    r"temperature[- ]sensor fault|fault code|error code|alarm code)\b", re.I)
_GENERIC_CAUSE_REQUEST = re.compile(
    r"\b(?:possible\s+causes?|possible\s+reasons?|causes?|reasons?|"
    r"issues?|problems?|why)\b", re.I)


_GENERIC_CAUSE_HEADING = re.compile(
    r"\b(?:faults?|symptoms?|failures?|troubleshoot(?:ing)?|"
    r"not working|does not start|not starting|does not run)\b",
    re.I,
)


# MySQL 8 prefilter for _GENERIC_CAUSE_HEADING; the Python
# pattern is still applied to every returned chunk.
_GENERIC_CAUSE_HEADING_SQL = r"(fault|symptom|failure|troubleshoot|not working|does not start|not starting|does not run)"
GENERIC_SCAN_LIMIT = 600


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


def _generic_cause_candidates(*, tenant, customer, asset):
    """Cover distinct authorized fault sections for an unspecified failure.

    Generic equipment failure has no fault-specific similarity target. Top-k
    reranking can favor one arbitrary fault, so it must not be the only source
    of candidates for an overview. Every selected chunk is reauthorized.
    """
    if asset is None or customer is None:
        return []
    from .retriever import _knowledge_candidates

    # R17: heading prefilter in the database and a hard cap on rows scanned.
    authorized_scope = _knowledge_candidates(
        tenant, asset, customer, False, allow_failed=True, backend="sparse",
        heading_regex=_GENERIC_CAUSE_HEADING_SQL, limit=GENERIC_SCAN_LIMIT,
    )
    cause_chunks = []
    for chunk in authorized_scope:
        if (_GENERIC_CAUSE_HEADING.search(str(chunk.heading or ""))
                and any(kind == "possible_cause" and _descriptive_cause(text)
                        for text, kind, _start, _end in source_claims(chunk.text))):
            cause_chunks.append(chunk)

    # Prefer fault evidence specifically assigned to this asset or product.
    # An asset-specific installation guide without causes must not prevent
    # fallback to a relevant authorized category-level troubleshooting guide.
    direct = [
        chunk for chunk in cause_chunks
        if chunk.document.asset_id == asset.id
        or (getattr(asset, "product_id", None)
            and chunk.document.product_id == asset.product_id)
    ]
    search_scope = direct or cause_chunks
    by_heading = {}
    for chunk in search_scope:
        heading = str(chunk.heading or "")
        key = (chunk.document_id,
               re.sub(r"^\s*\d+[.)]\s*", "", heading).strip().casefold())
        by_heading.setdefault(key, []).append(chunk)

    # Cover all headings once before taking secondary chunks from any heading.
    # The bounded authorization work never replaces the underlying scope gate.
    selected = []
    for depth in range(3):
        for group in by_heading.values():
            if depth < len(group):
                selected.append(group[depth])
                if len(selected) >= 48:
                    break
        if len(selected) >= 48:
            break

    authorized = []
    for chunk in selected:
        try:
            permitted = source_chunk(
                tenant=tenant, customer=customer, asset=asset,
                document_id=chunk.document_id, chunk_id=chunk.id,
            )
        except NotFound:
            continue
        authorized.append(SimpleNamespace(
            chunk=permitted, text=permitted.text, heading=permitted.heading,
            title=permitted.document.title, score=0.0,
            source_verified=True, equipment_applicable=True,
            is_quarantined=False, customer_action_authorized=False,
            exact_identifier_match=False,
        ))
    return authorized


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
# A request for a stated value or specification ("What is the design static
# pressure?"). Only a documented fact containing a number can answer it; a
# possible cause that merely shares its words never does.
_VALUE_QUESTION = re.compile(
    r"\b(?:what|which)\s+(?:is|are)\s+(?:the\s+)?(?:[a-z0-9/-]+\s+){0,4}?"
    r"(?:pressure|voltage|current|amperage|capacity|temperature|rating|size|setting|value|weight|"
    r"dimensions?|tonnage|airflow|flow rate|clearance|torque|speed|charge|specification|"
    r"length|distance|height)s?\b"
    # "What airflow should a 5 ton unit deliver?", "What pressure should it be tested at?"
    r"|\b(?:what|which)\s+(?:[a-z0-9/-]+\s+){0,2}?(?:pressure|voltage|current|amperage|capacity|temperature|"
    r"rating|size|setting|value|weight|tonnage|airflow|flow rate|clearance|torque|speed|charge|length|distance|"
    r"height)s?\s+(?:should|must|does|do|will|would|can)\b"
    r"|\bhow (?:much|many)\b", re.I)
# A customer reporting a fault or symptom rather than asking a factual question.
_SYMPTOM_REPORT = re.compile(
    r"\b(?:alarms?|faults?|errors?|tripp\w*|leak\w*|nois\w*|vibrat\w*|ice|icing|frost\w*|smell\w*|"
    r"not (?:cooling|heating|working|starting|draining)|(?:won'?t|doesn'?t|does not|will not) (?:start|cool|run|drain)|"
    r"stopp\w*|weak|warm|hot air)\b", re.I)
_FREQUENCY_QUESTION = re.compile(
    r"\b(?:how (?:often|frequently|regularly|many times)|"
    r"when (?:is|are|should|do|does|must)|every\s+\d+|intervals?|schedules?|"
    r"(?:every|each|once\s+a|per)\s+(?:day|week|month|quarter|year)|"
    r"frequency|periodic\w*|monthly|quarterly|annual\w*|yearly)\b", re.I)
# An instruction is not only a sentence that starts with a verb: "During
# brazing, use ..." is one too. Any clause-initial directive disqualifies.
_CLAUSE_DIRECTIVE = re.compile(
    r"(?:^|[,;:]\s+)" + _IMPERATIVE_START.pattern.lstrip("^"), re.I)


# "What does the maintenance checklist provide?" asks what a document is for.
_DOCUMENT_PURPOSE_QUESTION = re.compile(
    r"\bpurpose\b|\bwhat\s+(?:is|are)\b.{0,80}\bfor\b|"
    r"\bwhat\s+(?:does|do|is|are)\b.{0,80}\b(?:checklist|guide|manual|document|reference|procedure)s?\b"
    r".{0,40}\b(?:provide|cover|contain|include|describe|explain|address)\w*\b", re.I)
_PURPOSE_HEADING = re.compile(r"\b(?:purpose|scope)\b", re.I)
# A purpose section may state what the document provides in imperative form
# ("Provide a documented maintenance routine ...", "Use this checklist to plan
# ..."). That describes the document; it does not ask the reader to do anything.
_DOCUMENT_SELF_DESCRIPTION = re.compile(
    r"^(?:provides?\b|use\s+this\s+(?:checklist|guide|document|reference|manual)\s+to\b)", re.I)


def document_purpose_statement(text, level, heading):
    """True for a purpose-section sentence that only states what the document provides.

    The opening "Provide ..." / "Use this <document> to ..." is the only directive
    allowed; any other directive, any physical operation, or a restricted safety
    level still excludes the sentence.
    """
    clean = " ".join(str(text or "").split())
    if (level in _RESTRICTED_LEVELS or not _PURPOSE_HEADING.search(str(heading or ""))
            or not _DOCUMENT_SELF_DESCRIPTION.match(clean) or len(clean.split()) < 4):
        return False
    rest = _DOCUMENT_SELF_DESCRIPTION.sub("", clean, count=1)
    return not (PHYSICAL_OPERATION.search(clean) or _CLAUSE_DIRECTIVE.search(rest.lstrip())
                or re.search(r"[;:]", rest))


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
        r"stay on|operate|run|change (?:the )?fan speed)\b", lower)
    if permission and physical:
        return "safety"
    # Asking for instructions ("Tell me how to bypass ...", "Give me the steps to
    # adjust the charge ...", "I should keep resetting ...") requests the same
    # operation as "Can I ...?" and gets the same answer. Only operations on
    # protection, refrigerant or electrical access count here, so ordinary
    # "how do I ..." questions keep their normal answer.
    instruction = re.search(
        r"\bhow\s+(?:do|can|could|should|would|to)\b"
        r"|\b(?:tell|show|give|walk)\s+(?:me|us)\b[^.?!]{0,30}\b(?:how|steps?|procedure|through)\b"
        r"|\bsteps?\s+(?:to|for)\b"
        r"|\b(?:i|we)\s+(?:should|can|could|may|must|need\s+to|have\s+to)\b", lower)
    restricted = re.search(
        r"\b(?:bypas\w*|defeat\w*|bridg(?:e|ing)|disabl\w*|jumper\w*|short(?:ing)?[- ]out|resett?\w*)\b"
        r"|\b(?:re)?charg\w*\s+(?:the\s+|more\s+)?(?:\w+\s+){0,2}?(?:refrigerant|gas|circuit|system)\b"
        r"|\b(?:refrigerant|gas)\s+charg\w*"
        r"|\b(?:top(?:ping)?\s+up|add(?:ing)?\s+(?:more\s+|some\s+)?)(?:the\s+)?(?:refrigerant|gas)\b"
        r"|\b(?:refrigerant|gas)\b[^.?!]{0,30}\badd\b"
        r"|\badjust\w*\s+(?:the\s+)?(?:\w+\s+){0,2}?(?:charge|pressure|refrigerant|fan|speed|static|controls?|switch)\b"
        r"|\bopen(?:ing)?\s+(?:up\s+)?(?:the\s+|a\s+|its\s+)?(?:\w+\s+){0,2}?"
        r"(?:panel|cover|cabinet|compartment|circuit|casing|enclosure)\b"
        r"|\bchange\s+(?:the\s+)?fan\s+speed", lower)
    if (permission or instruction) and restricted:
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
    if (re.fullmatch(r"why is\s+.{3,80}\s+(?:not working|broken|dead|malfunctioning|"
                     r"not functioning|not operating|not running|not starting)\??", lower)
            or re.fullmatch(r"why isn't\s+.{3,80}\s+working\??", lower)
            or re.fullmatch(r"why (?:has|did)\s+.{3,80}\s+(?:stopped|failed|broken down|"
                            r"shut down|shut off)\??", lower)):
        return "clarify"
    words = re.findall(r"[a-z0-9]+", lower)
    if len(words) <= 7 and re.search(r"\b(?:help|what happened|what first|due yet)\b", lower):
        return "clarify"
    return None


# Equipment types of the company's registered assets (customer_isolation registry).
# A question naming one of these while a different type is selected asks about
# other equipment, and the selected unit's manuals do not answer it.
_EQUIPMENT_TYPES = {
    "split": re.compile(r"\bsplit\s+(?:ac|a/c|unit|air|system|type)\b", re.I),
    "vrf": re.compile(r"\bvrf\b|\bvariable refrigerant flow\b", re.I),
    "ductable": re.compile(r"\bductable\b", re.I),
    "cassette": re.compile(r"\bcassette\b", re.I),
    "package": re.compile(r"\bpackage(?:d)?\s+(?:ac|a/c|unit)\b", re.I),
    "chiller": re.compile(r"\bchillers?\b", re.I),
    "cooling_tower": re.compile(r"\bcooling\s+towers?\b", re.I),
    "ahu": re.compile(r"\bahu\b|\bair[- ]handling\s+unit\b", re.I),
    "fridge": re.compile(r"\b(?:fridge|refrigerator)s?\b", re.I),
}
# Fault vocabulary shared by every section ("fault", "problem", "unit"): a claim
# that shares only these words with the question is not about the question.
_GENERIC_MATCH_WORDS = frozenset({
    "fault", "faults", "issue", "issues", "problem", "problems", "error", "errors", "failure",
    "failures", "alarm", "alarms", "symptom", "symptoms", "unit", "units", "module", "modules",
    "system", "systems", "equipment",
})


def asks_about_other_equipment(question, asset):
    """A question naming an equipment type other than the selected unit's (even if it
    also names this unit): "Would a VRF fault code apply to this Ductable AC?"."""
    value = str(question or "")
    if asset is None or not (value.rstrip().endswith("?") or re.match(r"\s*(?:would|does|do|can|is|are|will)\b", value, re.I)):
        return False
    label = f"{getattr(asset, 'name', '') or ''} {getattr(asset, 'model_number', '') or ''}"
    own = {name for name, pattern in _EQUIPMENT_TYPES.items() if pattern.search(label)}
    asked = {name for name, pattern in _EQUIPMENT_TYPES.items() if pattern.search(value)}
    return bool(asked - own)


def names_other_equipment_type(question, asset):
    """True when the question names an equipment type other than the selected unit's."""
    if asset is None:
        return False
    label = f"{getattr(asset, 'name', '') or ''} {getattr(asset, 'model_number', '') or ''}"
    own = {name for name, pattern in _EQUIPMENT_TYPES.items() if pattern.search(label)}
    asked = {name for name, pattern in _EQUIPMENT_TYPES.items() if pattern.search(str(question or ""))}
    return bool(asked - own) and not (asked & own)


# A manual sentence a customer may be shown as guidance, never as a step to perform:
# a prohibition ("Do not repeatedly reset a tripping breaker.") or a statement of who
# must do the work ("Electrical panel checks must be performed by a qualified person.").
_SAFETY_NOTE = re.compile(
    r"^\s*(?:do\s+not|don't|never)\b"
    r"|\bmust\s+(?:only\s+)?be\s+(?:performed|carried\s+out|done|completed|checked|serviced|installed)\s+by\s+"
    r"(?:a\s+|an\s+)?(?:qualified|authori[sz]ed|trained|licensed|certified)\b", re.I)


def section_safety_notes(chunk):
    """(text, start, end) of each safety note in one already-authorized chunk."""
    for text, kind, start, end in source_claims(chunk.text):
        if kind != "possible_cause" and _SAFETY_NOTE.search(text):
            yield text, start, end


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


_CITATION_PREFIX = "/api/customer-support/source/?citation="


def refresh_citation_urls(contract, ctx):
    """Re-issue the signed citation links of a stored answer for the same user.

    Reopening an old conversation must not leave dead source links. Only tokens
    this server signed for this user, tenant and customer are renewed; the renewed
    token carries the same bound document/chunk/version/hash/span, and opening it
    still re-runs every authorization and integrity check in open_citation.
    """
    presentation = (contract or {}).get("presentation") if isinstance(contract, dict) else None
    customer = ctx.get("customer")
    if not isinstance(presentation, dict) or not customer:
        return contract
    for citation in presentation.get("citations") or []:
        url = citation.get("url") if isinstance(citation, dict) else None
        if not isinstance(url, str) or not url.startswith(_CITATION_PREFIX):
            continue
        try:
            payload = signing.loads(url[len(_CITATION_PREFIX):], salt=SALT, max_age=None)
        except signing.BadSignature:
            continue
        if (payload.get("user") != ctx["user"].id or payload.get("tenant") != ctx["tenant"].id
                or payload.get("customer") != customer.id):
            continue
        citation["url"] = _CITATION_PREFIX + signing.dumps(payload, salt=SALT, compress=True)
    return contract


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


# Document front matter: purpose/scope/overview sections and metadata lines
# ("Asset: ...", "Document ID: ...", "Revision: ...") describe the document itself.
_FRONT_MATTER_HEADING = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(?:purpose|scope|general\s+overview|overview|introduction|document\s+control|"
    r"revision\s+history|references?|definitions?)\s*$", re.I)
_METADATA_LINE = re.compile(
    r"^\s*(?:asset|document(?:\s+id)?|revision|status|date|prepared\s+by|approved\s+by|version)\s*:", re.I)
_EXPLICIT_TIMING = re.compile(
    r"\b(?:how\s+(?:often|frequently)|frequen\w*|intervals?|schedules?|periodic\w*|every|each|daily|weekly|"
    r"monthly|quarterly|annual\w*|yearly)\b", re.I)
_INTERVAL_HEADING = re.compile(r"\b(?:intervals?|schedules?|frequency)\b", re.I)
_SCHEDULE_GENERIC_WORDS = frozenset({
    "how", "often", "frequently", "when", "should", "must", "does", "need", "needs", "needed", "required", "get",
    "have", "service", "serviced", "servicing", "maintenance", "maintained", "maintain", "done", "checked", "check",
    "inspected", "regularly", "unit", "system", "equipment", "aircon", "the", "this", "that", "our", "you", "can",
    "would", "recommended", "recommend", "what", "is", "are"})
_ROLE_QUESTION = re.compile(
    r"\bwho\s+(?:is\s+(?:allowed|qualified|authori[sz]ed|permitted|supposed)|can|may|should|must)\b", re.I)
_PERFORMED_BY = re.compile(
    r"\b(?:must|should|shall|is\s+to|are\s+to)\s+be\s+(?:performed|carried\s+out|done|completed|installed|"
    r"serviced|commissioned|handled)\s+by\b", re.I)


def _required_overlap(asks_role: bool, sentence: str) -> int:
    """Distinct question terms a plain statement must share to answer the question.
    A "who may do X" question shares only X with "X must be performed by ..."."""
    return 1 if asks_role and _PERFORMED_BY.search(sentence) else 2


_DOCUMENT_WORD = re.compile(r"\b(?:checklist|guide|manual|document|reference|procedure)s?\b", re.I)
_DOCUMENT_QUESTION = re.compile(
    r"\b(?:what\s+is\s+(?:this|the)\s+(?:document|manual|guide)|purpose\s+of\s+(?:this|the)\s+"
    r"(?:document|manual|guide)|which\s+(?:document|manual)|what\s+does\s+(?:this|the)\s+(?:document|manual)\s+cover)\b",
    re.I)


# Intervals as (unit, count): "every six months" -> ("m", 6), "Quarterly" -> ("m", 3),
# "Annually"/"every year" -> ("m", 12), "weekly" -> ("w", 1). Months, quarters and
# years share one unit so "every 3 months" meets "Quarterly".
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a": 1, "an": 1}
_INTERVAL_UNITS = {"day": ("d", 1), "week": ("w", 1), "month": ("m", 1), "quarter": ("m", 3), "year": ("m", 12)}
_INTERVAL_ADVERBS = {"daily": ("d", 1), "weekly": ("w", 1), "fortnightly": ("w", 2), "monthly": ("m", 1),
                     "quarterly": ("m", 3), "half-yearly": ("m", 6), "half yearly": ("m", 6), "halfyearly": ("m", 6),
                     "semi-annual": ("m", 6), "semi-annually": ("m", 6), "semiannual": ("m", 6),
                     "biannual": ("m", 6), "bi-annual": ("m", 6), "biannually": ("m", 6),
                     "annual": ("m", 12), "annually": ("m", 12), "yearly": ("m", 12)}
_COUNTED_INTERVAL = re.compile(
    r"\b(?:(?:every|each|per|once\s+(?:a|an|every))\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve)[-\s]+(day|week|month|quarter|year)s?\b"
    r"|\b(?:every|each|per|once\s+(?:a|an|every))\s+()(day|week|month|quarter|year)\b", re.I)
_ADVERB_INTERVAL = re.compile(r"\b(" + "|".join(sorted(map(re.escape, _INTERVAL_ADVERBS), key=len, reverse=True))
                              + r")\b", re.I)


def stated_intervals(text: str) -> set:
    found = set()
    for m in _COUNTED_INTERVAL.finditer(str(text or "")):
        count, unit = (m.group(1), m.group(2)) if m.group(2) else ("1", m.group(4))
        n = int(count) if count.isdigit() else _NUMBER_WORDS.get(count.casefold(), 1)
        base, size = _INTERVAL_UNITS[unit.casefold()]
        found.add((base, n * size))
    for m in _ADVERB_INTERVAL.finditer(str(text or "")):
        found.add(_INTERVAL_ADVERBS[m.group(1).casefold()])
    return found


_ESCALATION_HEADING = re.compile(r"\b(?:warning\s+signs?|requiring\s+escalation|escalat\w*)\b", re.I)
# The customer's own words ask when to stop, escalate or worry; sharing a word
# with an escalation sentence ("breaker", "tripping") is not that question.
_ESCALATION_QUESTION = re.compile(
    r"\b(?:warning\s+signs?|danger\w*|unsafe|emergenc\w*|serious|escalat\w*|"
    r"safe\s+to\s+(?:use|run|keep|continue|operate)|"
    r"stop\s+(?:using|running|operating|the\s+(?:unit|ac|system))|"
    r"keep\s+(?:using|running|operating|resetting)|"
    r"when\s+(?:should|must|do)\s+(?:i|we)\s+(?:call|contact|stop|worry|escalate))\b", re.I)
# The only directions in such a sentence are to escalate, stop or switch off.
_SAFE_ESCALATION = re.compile(r"\b(?:escalate|stop\s+(?:operation|using|the\s+unit)|switch\s+off|contact)\b", re.I)
_UNSAFE_DIRECTIVE = re.compile(r"\b(?:open|remove|disassemble|adjust|reset|bypass|measure|tighten|recharge|charge)\b",
                               re.I)
# A procedure step ("Inspect evaporator and condenser coils.") from a checklist,
# installation or commissioning section: work for a qualified technician.
_PROCEDURE_LINE = re.compile(
    r"\s*(?:isolate|inspect|check|clean|measure|verify|confirm|replace|tighten|test|perform|review|calculate|"
    r"install|connect|position|secure|lift|seal|support|evacuate|pressure-test|charge|flush|mount|route|"
    r"complete|start|record)\b", re.I)
# "Do not repeatedly reset a fault" prohibits the action; it is not an instruction.
_NEGATED_DIRECTIVE = re.compile(r"\b(?:do\s+not|don't|never|must\s+not)\s+(?:\w+\s+){0,2}?"
                                r"(?:open|remove|disassemble|adjust|reset|bypass|measure|tighten|recharge|charge)\w*\b",
                                re.I)


# Former generic refusal text (kept so tools can recognise answers stored before
# the wording change); customers now receive limitation_message() instead.
LEGACY_GENERIC_NO_EVIDENCE = ("There is insufficient evidence to establish a safe answer for this equipment "
                              "and question. Ask an authorized technician for help.")


def no_documentation_message(asset) -> str:
    name = getattr(asset, "name", None) or "this equipment"
    return (f"I don't have approved documentation for the {name}, so I can't suggest causes or repair steps. "
            "If you note what it is doing, when it happens and any code shown, a technician can take it from "
            "there. No service request has been created.")


def technician_only_message(asset) -> str:
    """The documents cover this, but only as work for a qualified technician."""
    name = getattr(asset, "name", None) or "this equipment"
    return (f"That's covered in the {name} documentation as work for a qualified technician, so I can't take "
            "you through the steps. A technician carries it out during service or maintenance; I can prepare a "
            "service request if you'd like.")


def limitation_message(asset, conversation_context=None, question: str = "") -> str:
    """A plain, honest 'I can't confirm that' that keeps what the customer told us.

    Adds no technical content. Offers a technician only when the previous reply
    did not already offer one, so the offer is not repeated after every message.
    """
    name = getattr(asset, "name", None) or "your equipment"
    question = str(question or "")
    code = re.search(r"\b[A-Z]{1,3}[- ]?\d{1,3}\b", question)
    if code:
        parts = [f"I can't find code {code.group(0)} in the information I have for the {name}, so I can't tell "
                 "you what it means. Please note it exactly as displayed for the technician."]
    else:
        parts = [f"I can't confirm that from the information I have for the {name}."]
        issue = " ".join(str(getattr(conversation_context, "active_issue", "") or "").split()).rstrip(" .!?")
        answers = getattr(conversation_context, "resolved_answers", None) or []
        if answers:
            said = answers[-1]["answer"].rstrip(" .!?")
            parts.append(f"You've mentioned: \"{said}\".")
        elif issue:
            parts.append(f"I've noted what you told me: \"{issue}\".")
    previous = str(getattr(conversation_context, "last_assistant_answer", "") or "").casefold()
    if "service request" not in previous:
        parts.append("I can record this for a technician in a service request if you'd like.")
    return " ".join(parts)


def build_customer_answer(
    interaction, session, references, conversation_context=None,
    original_question=None, generate_with_local_model=True, evidence_blocked=False,
):
    # evidence_blocked: the original-document integrity gate refused this turn.
    # No path below (fact lookup, generic cause scan) may reintroduce claims.
    if evidence_blocked:
        references = []
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
    wrong_model = wrong_model or names_other_equipment_type(raw_user_q, asset)
    fact_family = _document_fact_family(question)
    if fact_family and not evidence_blocked and not wrong_model and not query_is_control_abuse(question):
        fact_answer = _document_fact_response(interaction, session, fact_family)
        if fact_answer:
            return fact_answer
    sufficient = check_evidence_sufficiency(question, asset, candidates)
    # Intent belongs to the latest customer turn, not to a previous complaint
    # incorporated into the technical retrieval query.
    original_complaint = (
        getattr(conversation_context, "original_complaint", "")
        if conversation_context is not None else ""
    )
    generic_cause_overview = bool(
        _GENERIC_CAUSE_REQUEST.search(raw_user_q)
        and not _SPECIFIC_FAULT_QUERY.search(raw_user_q)
        and (
            is_generic_cause_overview(raw_user_q)
            or is_generic_cause_overview(f"{raw_user_q} {original_complaint}")
        )
    )
    evidence_chunks = list(sufficient.grounded_chunks or [])
    if generic_cause_overview and not evidence_blocked:
        # A broad question must not inherit a single fault family from top-k/V5.
        overview = _generic_cause_candidates(
            tenant=tenant, customer=customer, asset=asset,
        )
        if overview:
            evidence_chunks = overview
            known = {candidate.chunk.id for candidate in candidates}
            candidates.extend(item for item in overview if item.chunk.id not in known)
    causes, observations, procedures, facts, citations = [], [], [], [], []
    terms = set(re.findall(r"[a-z0-9_-]{3,}", normalize_domain_terms(question).casefold())) - _STOPWORDS
    # One matching rule for every answer-stage check: folded word forms, plus
    # a symptom paraphrase only where the section heading names that symptom.
    term_keys = match_keys(terms)
    # Classify the customer's intent from their words with spelling corrected
    # ("what cuases this", "how ofen") as well as from the raw text.
    intent_text = f"{raw_user_q} {normalize_domain_terms(raw_user_q)}"
    asked_symptoms = symptom_concepts(intent_text) | symptom_concepts(question)
    customer_words = f"{intent_text} {getattr(conversation_context, 'active_issue', '') or ''}"
    if symptom_concepts(customer_words) - {"generic_failure"}:
        # A symptom the customer named, now or as the current issue ("my AC won't
        # turn on"), is what they described; the generic failure would widen the
        # answer to every symptom section. Only the customer's words count: the
        # retrieval query also carries the labels of sections shown earlier
        # ("unit does not start poor cooling").
        asked_symptoms = asked_symptoms - {"generic_failure"}
    wants_facts = not _CAUSE_QUESTION.search(intent_text)
    wants_cause_details = bool(
        _CAUSE_EXPLANATION_REQUEST.search(raw_user_q)
    )
    asks_frequency = bool(_FREQUENCY_QUESTION.search(intent_text))
    # "When should a problem be escalated?" asks for conditions, not timing: an
    # escalation question is a timing question only when it names an interval.
    asks_escalation = bool(_ESCALATION_QUESTION.search(intent_text))
    if asks_escalation and not _EXPLICIT_TIMING.search(intent_text):
        asks_frequency = False
    asked_intervals = stated_intervals(intent_text)
    # A timing question in which the customer names no symptom and asks for no cause
    # ("the compressor trips every 5 minutes" names one and keeps its causes).
    timing_only = bool(asks_frequency and not _CAUSE_QUESTION.search(intent_text)
                       and not (symptom_concepts(intent_text) - {"generic_failure"}))
    # A timing question about the equipment as a whole, naming no item to check.
    generic_schedule_question = asks_frequency and not (
        match_keys(set(re.findall(r"[a-z0-9_-]{3,}", intent_text.casefold())) - _STOPWORDS)
        - match_keys(_SCHEDULE_GENERIC_WORDS))
    # "Who is allowed to install this unit?" is answered by "... must be performed by ...".
    asks_role = bool(_ROLE_QUESTION.search(intent_text))
    # A multi-part question ("what causes X, and how often should Y be done?")
    # explicitly asks for a documented schedule as well as causes; an escalation
    # question asks for the documented warning conditions.
    wants_facts = wants_facts or asks_frequency or asks_escalation
    asks_value = bool(_VALUE_QUESTION.search(intent_text)) and not _CAUSE_QUESTION.search(intent_text)
    asks_document_purpose = bool(_DOCUMENT_PURPOSE_QUESTION.search(intent_text))
    # A reported fault ("the module shows a high-pressure alarm", "water is leaking")
    # asks for its documented causes even without the word "cause"; it is not
    # a fact question, so word-matched overview facts must not displace causes.
    symptom_report = bool(_SYMPTOM_REPORT.search(intent_text)) and not (
        asks_frequency or asks_value or asks_document_purpose)
    if symptom_report and not asks_frequency:
        wants_facts = False
    codes = extract_error_codes(question)
    quantities = extract_typed_quantities(question)
    generic_keys = match_keys(_GENERIC_MATCH_WORDS)
    stated_text = (f"{raw_user_q} {question} {normalize_domain_terms(raw_user_q)} "
                   f"{normalize_domain_terms(question)}")
    # Sections with a documented cause list, and those whose symptom the customer
    # names. A named symptom is answered from its own section: causes and notes of
    # other symptom sections are not added merely because they share a word
    # ("fault", "airflow"). Naming two symptoms keeps both sections.
    symptom_sections = {r.heading for r in evidence_chunks
                        if any(k == "possible_cause" for _t, k, _s, _e in source_claims(r.text))}
    stated_sections = set() if generic_cause_overview else {
        h for h in symptom_sections
        if heading_states_symptom(h, asked_symptoms) or question_states_heading(stated_text, h)}
    eligible = []
    # Question-relevant sentences withheld only because they are technician-only
    # work: the answer then says so instead of implying the manuals are silent.
    withheld_relevant = 0
    if evidence_chunks and (sufficient or generic_cause_overview) and not wrong_model and not query_is_control_abuse(question):
        # A question about a named document ("What does the VRF installation guide
        # cover?") is answered from its purpose section.
        about_document = bool(_DOCUMENT_QUESTION.search(raw_user_q) or (
            asks_document_purpose and _DOCUMENT_WORD.search(raw_user_q)))
        for result in evidence_chunks:
            if (not about_document and not asks_frequency
                    and _FRONT_MATTER_HEADING.match(str(result.heading or ""))):
                # A document's purpose/overview section describes the document, not the
                # equipment problem; it never answers a symptom, cause or next-step
                # question. A timing question may use a purpose statement that itself
                # answers it ("Frequency should be based on ..."); the sentence checks
                # below still apply.
                continue
            # "How often should it be serviced?" names no item: the whole documented
            # schedule in an interval section answers it (schedule entries only).
            whole_schedule = bool(generic_schedule_question
                                  and _INTERVAL_HEADING.search(str(result.heading or "")))
            for text, kind, start, end in source_claims(result.text):
                if not about_document and _METADATA_LINE.match(text):
                    continue
                words = set(re.findall(r"[a-z0-9_-]{3,}", text.casefold())) - _STOPWORDS
                heading_words = set(re.findall(r"[a-z0-9_-]{3,}", (result.heading or "").casefold()))
                sentence_overlap = term_keys & match_keys(words)
                generic_section_cause = bool(
                    generic_cause_overview
                    and kind == "possible_cause"
                    and _GENERIC_CAUSE_HEADING.search(result.heading or "")
                )
                # A section whose symptom the question names ("could high pressure be
                # involved?" names HIGH-PRESSURE ... FAULT) is relevant even when the
                # heading's compound words ("high-pressure") are written apart.
                if (not sentence_overlap and not term_keys & match_keys(heading_words)
                        and not heading_states_symptom(result.heading, asked_symptoms)
                        and result.heading not in stated_sections
                        and not generic_section_cause and not whole_schedule):
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
                    if asks_value and not wants_cause_details:
                        continue
                    if timing_only and not wants_cause_details:
                        # "How often should I get it serviced?" is answered from the
                        # schedule; the background issue's causes are not the answer.
                        continue
                    if generic_section_cause:
                        # The customer asked for a broad overview, not a
                        # diagnosis. Preserve the source section as a condition.
                        target = causes
                    else:
                        if stated_sections and result.heading not in stated_sections:
                            continue
                        # A cause that shares no word with the question is
                        # relevant only through its section. Sharing some heading
                        # words is not enough ("server room temperature" is not
                        # "uneven room temperatures"): the question must state the
                        # symptom that heading names. Shared fault vocabulary
                        # alone ("fault", "problem") does not count as sharing.
                        if not (sentence_overlap - generic_keys) and not (
                                heading_states_symptom(result.heading, asked_symptoms)
                                or question_states_heading(
                                    # Customers misspell; judge the typo-normalized
                                # wording too ("icce on coyl" names ice on a coil).
                                f"{raw_user_q} {question} "
                                f"{normalize_domain_terms(raw_user_q)} "
                                f"{normalize_domain_terms(question)}", result.heading)):
                            continue
                        target = causes
                elif level in ("INFORMATIONAL_OBSERVATION", "UNKNOWN_REQUIRES_REVIEW") and _noninvasive_observation(text):
                    if _is_completed_check(text):
                        continue
                    # A value question is answered only by a documented value.
                    if asks_value and not wants_cause_details:
                        continue
                    # A timing question is not about the current fault's notes.
                    if timing_only and result.heading in symptom_sections:
                        continue
                    if (stated_sections and result.heading in symptom_sections
                            and result.heading not in stated_sections):
                        continue
                    if (not (sentence_overlap - generic_keys)
                            and not ((term_keys & match_keys(heading_words)) - generic_keys)
                            and not heading_states_symptom(result.heading, asked_symptoms - {"generic_failure"})):
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
                    if (not disclosure and _ESCALATION_HEADING.search(str(result.heading or ""))
                            and _ESCALATION_QUESTION.search(intent_text)
                            and _SAFE_ESCALATION.search(text)
                            and not _UNSAFE_DIRECTIVE.search(_NEGATED_DIRECTIVE.sub(" ", text))):
                        disclosure = "escalation_guidance"
                    if (not disclosure and asks_document_purpose
                            and document_purpose_statement(text, level, result.heading)):
                        disclosure = "descriptive"
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
                            if customer_safe_fact(part, part_level) != "descriptive" and not (
                                    asks_document_purpose
                                    and document_purpose_statement(part, part_level, result.heading)):
                                continue
                            part_words = set(
                                re.findall(r"[a-z0-9_-]{3,}", part.casefold())) - _STOPWORDS
                            part_overlap = term_keys & match_keys(part_words)
                            if (len({stem(w) for w in part_overlap}) < _required_overlap(asks_role, part)
                                    or (asks_frequency and not re.search(
                                        r"\b(?:frequency|intervals?|schedules?)\b", part, re.I))):
                                continue
                            eligible.append({
                                "order": len(eligible), "target": facts, "kind": "information",
                                "text": part, "start": part_start, "end": part_end,
                                "result": result,
                            })
                        # A relevant checklist/procedure line withheld as technician work.
                        if ((sentence_overlap - generic_keys)
                                and (level in _RESTRICTED_LEVELS or _PROCEDURE_LINE.match(text))):
                            withheld_relevant += 1
                        continue
                    if disclosure == "escalation_guidance":
                        # Relevant when the question shares a term with the sentence
                        # or names the section ("warning signs", "escalate").
                        if not (sentence_overlap or (term_keys & match_keys(heading_words)) - generic_keys):
                            continue
                        target = facts
                        eligible.append({"order": len(eligible), "target": target, "kind": "information",
                                         "text": text, "start": start, "end": end, "result": result})
                        continue
                    if not sentence_overlap and not (whole_schedule and disclosure == "maintenance_schedule"):
                        continue
                    if disclosure == "maintenance_schedule":
                        # A schedule entry answers only a question about timing, and
                        # only for the interval asked ("every six months" is not
                        # "Monthly"; "every 3 months" is "Quarterly").
                        if (not asks_frequency
                                or (len(term_keys & match_keys(words | heading_words)) < 2
                                    and not whole_schedule)):
                            continue
                        line_intervals = stated_intervals(text)
                        if asked_intervals and line_intervals and not asked_intervals & line_intervals:
                            continue
                    elif (len({stem(w) for w in sentence_overlap}) < _required_overlap(asks_role, text)
                          or (asks_frequency and not re.search(
                              r"\b(?:frequency|intervals?|schedules?)\b", text, re.I))):
                        # A plain statement must itself share two distinct
                        # question terms. For a timing question it must be a
                        # statement about frequency; it never stands in for a
                        # schedule entry otherwise.
                        continue
                    if asks_value and not re.search(r"\d", text):
                        continue
                    target = facts
                else:
                    if (sentence_overlap - generic_keys) and level in _RESTRICTED_LEVELS:
                        withheld_relevant += 1
                    continue
                # Candidates are ranked for relevance to this question below,
                # then capped; the cap must not keep an earlier but less
                # relevant sentence merely because it appears first.
                eligible.append({
                    "order": len(eligible), "target": target, "kind": kind,
                    "text": text, "start": start, "end": end, "result": result,
                })
        # Identical wording under different fault headings ("Dirty filter." under
        # POOR COOLING and under ICE ON COIL) is a distinct cause in a distinct
        # context: causes are de-duplicated by (text, source heading), other claims
        # by text.
        seen_text, seen_keys = set(), set()
        unique = []
        for item in eligible:
            heading = " ".join(str(getattr(item["result"], "heading", "") or "").split()).casefold()
            key = (item["text"], heading) if item["target"] is causes else (item["text"], None)
            if key in seen_keys:
                continue
            seen_keys.add(key)
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
        # Each list is capped to the per-section contract limit (section 1 holds
        # causes or facts; section 2 observations and procedures), keeping the
        # most relevant claims. A generic failure overview previously kept 20
        # causes, which the client rejected as an invalid answer.
        if generic_cause_overview:
            # Give each documented fault heading one claim before taking
            # additional bullets from a heading, within the client claim cap.
            primary, secondary, headings = [], [], set()
            for item in cause_pool:
                heading = re.sub(r"^\s*\d+[.)]\s*", "",
                                 str(item["result"].heading or "")).casefold()
                if heading in headings:
                    secondary.append(item)
                else:
                    headings.add(heading)
                    primary.append(item)
            selected_causes = (primary + secondary)[:MAX_SECTION_CLAIMS]
        else:
            selected_causes = sorted(cause_pool, key=lambda i: -_relevance(i))[:10]
        chosen = (selected_causes
                  + fact_choice[:MAX_SECTION_CLAIMS]
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
    # "What can I safely check?" / "what to do next?": besides any customer check, the
    # manual's prohibitions and who-must-do-it statements for the symptom being discussed
    # ("Do not repeatedly reset a tripping breaker.") are the customer-safe guidance.
    from .conversation_context import ISSUE_CHECKS_PREFIX
    if str(raw_user_q).startswith(ISSUE_CHECKS_PREFIX) and evidence_chunks and not wrong_model:
        note_sections = stated_sections or {r.heading for r in evidence_chunks
                                            if heading_states_symptom(r.heading, asked_symptoms - {"generic_failure"})}
        shown = {c.get("source_text") for c in facts}
        for result in evidence_chunks:
            if result.heading not in note_sections:
                continue
            for text, start, end in section_safety_notes(result.chunk):
                if text in shown or sum(1 for c in facts if c.get("disclosure") == "safety_note") >= 2:
                    continue
                shown.add(text)
                citation = _citation(result.chunk, start, end, asset=asset, customer=customer,
                                     user_id=interaction.created_by_id, kind="information")
                citation["id"] = f"s{len(citations)+1}"
                citations.append(citation)
                facts.append({"id": f"c{len(citations)}", "text": text, "source_text": text, "kind": "information",
                              "evidence_ids": [citation["id"]], "disclosure": "safety_note"})
    # A factual question (timing, value, document content) is answered by the
    # documented facts that answer it. A cause that matched only by a shared
    # word ("filter") does not displace them; cause questions are unaffected.
    if (facts and causes and wants_facts and not wants_cause_details and not generic_cause_overview
            and not symptom_report
            and not _CAUSE_QUESTION.search(intent_text) and not _CAUSE_QUESTION.search(question)):
        dropped = {e for claim in causes for e in claim.get("evidence_ids", [])}
        causes = []
        citations = [c for c in citations if c.get("id") not in dropped]
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
    no_authorized_docs = bool(evidence_blocked)
    if not normal and not candidates and not evidence_blocked:
        from .retriever import _knowledge_candidates
        no_authorized_docs = not _knowledge_candidates(
            tenant, asset, customer, False, backend="all")
    if no_authorized_docs:
        from .customer_isolation import get_customer_data_mode
        reason_code = ("NO_AUTHORIZED_ORIGINAL_DOCUMENTATION"
                       if get_customer_data_mode() == "verified_real_only"
                       else "NO_AUTHORIZED_TEST_DOCUMENTATION")
        message = no_documentation_message(asset)
    else:
        reason_code = None
        message = (technician_only_message(asset) if withheld_relevant and not normal
                   else limitation_message(asset, conversation_context, raw_user_q))
        if withheld_relevant and not normal:
            reason_code = "TECHNICIAN_ONLY_EVIDENCE"
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
        # A multi-part question ("What causes this and when is maintenance due?")
        # keeps both parts: up to four facts share section 1 with the causes.
        shared_facts = facts[:4] if causes else facts
        first_section = (causes[:MAX_SECTION_CLAIMS - len(shared_facts)] + shared_facts) if causes else facts
        content = [
            first_section or [{"text": "The available evidence does not establish a possible cause.", "kind": "uncertainty", "evidence_ids": []}],
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
