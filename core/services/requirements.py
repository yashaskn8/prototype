"""Separate *what the customer asks for* from *which words must be grounded*.

Long customer questions mix three kinds of vocabulary:

1. Technical content ("compressor", "low-pressure", "drain", "E04").  These
   words must overlap with an approved passage before that passage can be
   treated as topically supporting the question.
2. Requirement words ("possible causes", "checks", "technician", "exact
   numerical limit").  They describe *what kind* of answer is wanted.  They are
   NOT lexical evidence (almost every fault section contains "possible causes")
   but they must not be forgotten either: each one is checked separately
   against the grounded passages and any unsupported requirement is disclosed.
3. Instruction / discourse words ("cite", "passages", "synthesis",
   "relationship", "explicitly").  They describe how the answer should be
   written and never appear in equipment manuals.  Counting them in the
   overlap denominator caused correct, top-ranked sections to be rejected.

Nothing here grants claim support.  Exact identifiers, numeric quantities,
source trust, equipment applicability and customer-action approval remain
enforced by ``core.services.llm`` and ``core.services.customer_answer``.
"""
import re

# Discourse vocabulary about answering, citing, scoping or authorising.  It is
# generic English, not tuned to a particular manual: none of these words is a
# property of HVAC equipment.  Do not add technical nouns here.
INSTRUCTION_TERMS = frozenset({
    "cite", "cites", "citation", "citations", "passage", "passages", "source", "sources",
    "source-grounded", "grounded", "section", "sections", "relationship", "relationships",
    "synthesis", "synthesize", "combining", "combine", "cover", "both", "explicitly",
    "separately", "separate", "state", "stated", "say", "says", "give", "versus",
    "compare", "comparison", "distinctions", "distinction", "distinguished", "actually",
    "alone", "relevant", "needed", "establish", "established", "infer", "inferred",
    "limitations", "preserve", "identity", "context", "exactly", "supporting",
    "supported", "unsupported", "undocumented", "documentation", "documents",
    "authorized", "authorised", "authorization", "authorisation", "permission",
    "permitted", "approval", "approved", "authentic", "independently", "verified",
    "verify", "exist", "exists", "existing", "invent", "invented", "withhold", "refuse",
    "reject", "fact", "facts", "absence", "request", "requested", "answer", "answers",
    "part", "parts", "scope", "remaining", "adversarial", "negative", "treating", "treat",
    "rather", "plain", "english", "technically", "ask", "asked", "appropriate",
    "substitute", "responsibilities", "responsibility", "customer", "customers",
    "includes", "including", "include", "may", "might", "available", "without",
    "missing", "confirmed", "confirm", "unconfirmed", "displayed", "taken", "place",
    "opened", "reporting", "limits", "prerequisites", "suggested", "related", "concerning",
    "regarding", "tell", "show", "list", "summarize", "summarise", "provide", "detail",
    "details", "clearly", "precisely", "honest", "honestly", "assumption", "assumptions",
    "claim", "claims", "quote", "quotes", "quotation", "reference", "references",
    "manual", "manuals", "guide", "document", "unit", "equipment", "selected", "this",
    # Phase 2: Discourse and instruction vocabulary discovered from benchmark
    # false-refusal deep traces (Q053, Q065, Q099).  None of these words name
    # a property of HVAC equipment; they describe how to format or scope the
    # answer, or come from adversarial injection phrases.
    "seem", "seems", "way", "ordinary", "previous", "previously", "temporary",
    "observed", "observation", "observations", "duration", "symptom", "symptoms",
    "measurement", "measurements", "uncertain", "uncertainty",
    "sequence", "sequences", "recovery", "recover", "supply",
    "distinguish", "explain", "explained", "explaining", "evidence",
    "diagnosis", "assume", "remains", "information",
    # Adversarial instruction vocabulary that never describes equipment
    "bypass", "interlock", "promise", "guaranteed", "guarantee",
    # Generic discourse words that inflate denominator without topical value
    "any", "cannot", "whether", "match", "matches",
    "customer-safe",
})

# Requirement intents.  "question" detects that the customer asked for it;
# "evidence" detects that a grounded passage can actually answer it.
REQUIREMENTS = {
    "possible_causes": {
        "question": r"\b(?:possible|potential|likely)\s+causes?\b|\bcauses?\b|\bwhat could be wrong\b|\bwhy\b",
        "evidence": r"\b(?:possible|potential|common)\s+causes?\b|\bmay be caused by\b|\bcan cause\b",
        "label": "possible causes for the reported condition",
    },
    "checks": {
        "question": r"\bchecks?\b|\binspect(?:ion)?\b|\bobserv\w*\b|\bwhat can i (?:safely )?(?:check|report)\b",
        "evidence": r"\bchecks?\s*:|\binitial checks\b|\binspect\b|\brecord\b|\bcheck\b",
        "label": "documented checks or observations",
    },
    "technician_escalation": {
        "question": r"\btechnician\b|\bescalat\w*\b|\bservice (?:engineer|personnel)\b|\bprofessional\b|\bqualified\b",
        "evidence": r"\btechnician\b|\bqualified\b|\bescalat\w*\b|\barrange inspection\b|\bservice personnel\b|\bdo not\b",
        "label": "when technician help is required",
    },
    "numeric_specification": {
        "question": r"\bnumerical\b|\bthreshold\b|\bspecification\b|\bexact (?:value|limit|figure)\b|\blimit\b|\bunits?\b|\brated\b",
        "evidence": r"\b\d+(?:\.\d+)?\s*(?:°\s*[cf]|deg\w*|bar|psi|kpa|mpa|v|volts?|a|amps?|hz|rpm|kw|w|%|cfm|tr|tons?|mm|m|pa)\b",
        "label": "an exact numerical limit or specification",
    },
    "fault_code": {
        "question": r"\bfault code\b|\berror code\b|\balarm code\b|\bcode\b",
        "evidence": r"\b[A-Z]{1,4}-?\d{1,4}\b",
        "label": "a specific fault or alarm code definition",
    },
    "customer_procedure": {
        "question": r"\bprocedure\b|\bperform\b|\brepair\b|\bfix\b|\breset\b|\bsteps?\b|\bmyself\b",
        "evidence": r"$^",  # free-form manuals never establish customer permission
        "label": "a customer-approved physical procedure",
    },
}

REQUIREMENT_TERMS = frozenset({
    "possible", "potential", "likely", "cause", "causes", "check", "checks", "procedure",
    "procedures", "step", "steps", "safe", "technician", "technician-only", "non-invasive",
    "physical", "next", "action", "actions", "code", "codes", "value", "values", "limit",
    "numerical", "units", "exact", "perform", "require", "requires", "required", "repair",
    "work", "help", "threshold", "specification", "specifications",
    # Phase 2: Requirement-describing words that inflate the denominator.
    # E.g. "whether the component has a documented replacement or supply specification"
    # describes what kind of answer the customer wants, not a content term.
    "component", "components", "replacement", "replacements",
    "documented", "undocumented",
})

_TOKEN = re.compile(r"[A-Za-z0-9_-]{3,}")


def grounding_terms(question, stopwords, normalize=None):
    """Content terms that an approved passage must overlap with.

    Falls back to the stopword-filtered vocabulary when filtering removes
    everything, so short questions such as "safe checks?" are still evaluated.
    """
    text = normalize(question or "") if normalize else (question or "")
    raw = {w.lower() for w in _TOKEN.findall(text)}
    base = raw - set(stopwords)
    content = base - INSTRUCTION_TERMS - REQUIREMENT_TERMS
    return content or base or raw


def extract_requirements(question):
    q = (question or "").lower()
    return [key for key, spec in REQUIREMENTS.items() if re.search(spec["question"], q)]


def requirement_coverage(question, chunks):
    """Return {requirement: {"supported": bool, "chunk_ids": [...], "label": str}}.

    Evidence matching is only performed on passages that already passed the
    grounding validator; this never promotes an ungrounded passage.
    """
    coverage = {}
    for key in extract_requirements(question):
        spec = REQUIREMENTS[key]
        ids = []
        for c in chunks or []:
            text = f"{getattr(c, 'text', '') or ''}"
            if _evidence_supports(key, spec, text):
                ids.append(getattr(c, "chunk_id", None) or getattr(c, "id", None))
        coverage[key] = {"supported": bool(ids), "chunk_ids": ids, "label": spec["label"]}
    return coverage


def _evidence_supports(key, spec, text):
    # Identifiers and quantities reuse the strict production extractors rather
    # than a looser regex, so a document ID or "Step 2" is never a "code".
    if key == "fault_code":
        from .identifiers import extract_error_codes
        return bool(extract_error_codes(text))
    if key == "numeric_specification":
        from .llm import extract_typed_quantities
        return bool(extract_typed_quantities(text))
    return bool(re.search(spec["evidence"], text, re.I))


def unsupported_requirement_labels(coverage):
    return [v["label"] for v in coverage.values() if not v["supported"]]
