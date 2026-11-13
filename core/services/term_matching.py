"""Bounded term and symptom matching shared by the answer-stage evidence checks.

Two deliberately small mechanisms, used the same way by every caller:

* ``match_keys`` – case folding plus conservative English inflection folding
  ("rooms" -> "room", "cleaned" -> "clean"). Tokens containing a digit (fault
  codes, model numbers, quantities) are never altered.
* symptom concepts – a short, explicit table mapping customer wording to the
  symptom a manual section is *titled* with. A paraphrase only counts when the
  section heading itself names that symptom, so wording similarity alone can
  never establish evidence.

Nothing here retrieves, ranks, or authorizes content.
"""
from __future__ import annotations

import re

_SUFFIXES = ("ing", "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """Fold common inflections. Identifiers and numbers are returned unchanged."""
    w = str(word or "").casefold()
    if len(w) < 4 or any(ch.isdigit() for ch in w):
        return w
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            base = w[: -len(suffix)]
            if suffix == "es" and not base.endswith(("ss", "x", "z", "ch", "sh")):
                return w[:-1]             # "provides" -> "provide", not "provid"
            if suffix in {"ing", "ed"} and len(base) > 3 and base[-1] == base[-2]:
                base = base[:-1]          # "tripped" -> "trip"
            return base
    return w


def match_keys(words) -> set:
    """Comparable keys for a set of words: each word and its folded form."""
    keys = set()
    for word in words or ():
        w = str(word).casefold()
        if w:
            keys.add(w)
            keys.add(stem(w))
    return keys


def text_keys(text: str, stopwords=frozenset()) -> set:
    words = set(re.findall(r"[a-z0-9_-]{3,}", str(text or "").casefold())) - set(stopwords)
    return match_keys(words)


# concept -> (customer wording, wording a section heading must contain)
_TEMPERATURE_WORD = r"(?:colder|cooler|warmer|hotter)"
SYMPTOM_CONCEPTS = {
    "poor_cooling": (
        re.compile(
            r"\b(?:not|isn'?t|doesn'?t|won'?t|never|no longer|barely|hardly)\s+"
            r"(?:(?:getting|becoming|staying|feeling|get|become|stay|feel|"
            r"making (?:it|the room))\s+)?(?:cold|cool|cooler|colder)\b"
            r"|\b(?:not|isn'?t|doesn'?t|no|poor|weak|insufficient|reduced)\s+cooling\b"
            r"|\b(?:not|isn'?t|doesn'?t|won'?t)\s+cool(?:ing)?\b", re.I),
        re.compile(r"\b(?:poor|not|no|weak|insufficient|reduced)\s+cooling\b", re.I),
    ),
    "uneven_temperature": (
        re.compile(
            r"\b(?:some|certain|few|several|one|two|other)\s+(?:of the\s+)?"
            r"(?:rooms?|zones?|areas?|floors?|offices?)\b.{0,40}\b" + _TEMPERATURE_WORD + r"\b"
            r"|\b" + _TEMPERATURE_WORD + r"\s+than\s+(?:the\s+)?(?:others?|rest)\b"
            r"|\buneven\s+(?:room\s+)?temperatures?\b"
            r"|\b(?:room\s+)?temperatures?\s+(?:are|is|seem|feel)\s+uneven\b"
            r"|\bhot\s+and\s+cold\s+spots?\b", re.I),
        re.compile(r"\buneven\s+(?:room\s+)?temperatures?\b", re.I),
    ),
    "communication": (
        re.compile(
            r"\b(?:lost|losing|lose|loses|dropped|no)\s+(?:the\s+|their\s+|its\s+)?"
            r"(?:connection|communication|link)\b"
            r"|\b(?:not|no longer|stopped|can'?t|cannot)\s+(?:communicat\w+|connect\w*)\b"
            r"|\b(?:communication|connection)\s+(?:fault|loss|lost|error|failure|problem)\b",
            re.I),
        re.compile(r"\bcommunication\b", re.I),
    ),
}


def symptom_concepts(question: str) -> frozenset:
    """Symptom concepts the customer's own wording states."""
    text = " ".join(str(question or "").split())
    return frozenset(name for name, (asked, _heading) in SYMPTOM_CONCEPTS.items()
                     if asked.search(text))


def heading_states_symptom(heading: str, concepts) -> bool:
    """True only when the section heading itself names an asked symptom."""
    text = str(heading or "")
    return any(SYMPTOM_CONCEPTS[name][1].search(text)
               for name in concepts or () if name in SYMPTOM_CONCEPTS)


# Words that label a heading rather than name its symptom, and degree words a
# customer need not repeat ("cooling problem" still means "POOR COOLING").
# Polarity and distinguishing words (high, low, uneven, water ...) are kept.
_HEADING_LABELS = frozenset({"symptom", "fault", "unit", "does", "not", "the", "and"})
_OPTIONAL_DEGREE = frozenset({"poor", "weak", "excessive", "bad"})


def _same_word(a: str, b: str) -> bool:
    if match_keys([a]) & match_keys([b]):
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short)      # leak / leakage


def question_states_heading(question: str, heading: str) -> bool:
    """True when the question names the symptom a section heading names.

    A heading such as "HIGH-PRESSURE OR HEAT-EXCHANGE FAULT" offers
    alternatives; the question must contain every distinguishing word of at
    least one. Partial overlap ("room" and "temperature" against "UNEVEN ROOM
    TEMPERATURES") does not qualify. The existing controlled symptom table is
    honoured as well, so established customer wording keeps working.
    """
    text = str(heading or "").casefold()
    text = re.sub(r"^\s*\d+[.)]\s*", "", text)
    text = re.sub(r"^symptom\s*:\s*", "", text)
    asked = set(re.findall(r"[a-z0-9]+", str(question or "").casefold()))
    for alternative in re.split(r"\s+or\s+", text):
        words = [w for w in re.findall(r"[a-z0-9]+", alternative)
                 if len(w) >= 3 and w not in _HEADING_LABELS and w not in _OPTIONAL_DEGREE]
        if words and all(any(_same_word(w, q) for q in asked) for w in words):
            return True
    try:
        from core.ml.conditional_rerank import _extract_symptom_keys, _heading_matches_symptom
        # "pressure" alone carries no polarity and is decided above.
        keys = _extract_symptom_keys(str(question or "")) - {"pressure"}
        return bool(keys) and _heading_matches_symptom(heading or "", keys, str(question or ""))
    except Exception:
        return False


DOMAIN_TYPO_MAP = {
    "indor": "indoor",
    "leeking": "leaking",
    "watter": "water",
    "milkk": "milk",
    "temprature": "temperature",
    "riseing": "rising",
    "abov": "above",
    "voltcure": "voltcore",
    "invrter": "inverter",
    "falt": "fault",
    "pllet": "pallet",
    "hydralic": "hydraulic",
    "cavtation": "cavitation",
    "ductible": "ductable",
    "wek": "weak",
    "airflo": "airflow",
    "whistlng": "whistling",
    "noys": "noise",
    "outdor": "outdoor",
    "unti": "unit",
    "presure": "pressure",
    "faault": "fault",
    "capcity": "capacity",
    "preshure": "pressure",
    "capcityy": "capacity",
    "hlep": "help",
    "comuncation": "communication",
    "errer": "error",
    "contrler": "controller",
    "splitt": "split",
    "icce": "ice",
    "coolng": "cooling",
    "pip": "pipe",
    "viberation": "vibration",
    "deflecton": "deflection",
    "compresser": "compressor",
    "coyl": "coil",
    "wat": "what",
    "shud": "should",
}

# Correctly spelled words customers use when asking about HVAC equipment and its
# manuals. A misspelling is corrected only toward one of these (see
# _correct_spelling). The list is generic vocabulary: it adds no technical fact.
_SPELLING_LEXICON = frozenset("""
what which when where why how does should could would will there their these those
this that with from about after before because between often every each always
never sometimes properly normally problem problems issue issues reason reasons
cause causes caused causing possible possibly explain explained detail details
first second third fourth fifth other others another documented document manual
manuals guide checklist technician qualified customer service serviced servicing
maintenance maintain inspection inspect inspected inspecting interval intervals
schedule monthly quarterly annually yearly weekly daily months weeks years hours
running runs started starting start stopped stopping working work works turned
install installed installation commissioning commissioned acceptance criteria
cooling cool colder cooler warm warmer heating heat hotter cold freezing frost
airflow flow weak strong uneven rooms room temperature temperatures pressure
water leakage leaking leaks drain drains drainage condensate dripping
noise noisy loud vibration vibrating rattling humming buzzing
filter filters clean cleaned cleaning dirty blocked blocking clogged restricted
coil coils compressor condenser evaporator refrigerant sensor sensors thermostat
remote controller control setting settings setpoint fan fans blower motor
bearing bearings panel panels damper dampers duct ducts ducting grille outlets
outdoor indoor unit units module modules system equipment display alarm alarms
fault faults error errors code codes communication communicating connection
connected addressing address addresses power supply breaker tripping tripped
voltage current electrical wiring insulation clearance mounting support supports
installed level leveled load loads heavy light lights indicator flashing blinking
inverter capacity operation operating conditions design ambient outside inside
indicate indicates indicated indicating mean means meant meaning describe describes
require requires required include includes included provide provides provided
cover covers covered perform performed check checked checking happen happens
""".split())


def _edit_distance_at_most(a, b, limit):
    """Optimal string alignment distance between a and b, or limit+1 if larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous2, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if (previous2 is not None and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                current[j] = min(current[j], previous2[j - 2] + 1)     # transposition
        if min(current) > limit:
            return limit + 1
        previous2, previous = previous, current
    return previous[-1]


def _correct_spelling(word):
    """Correct an unknown alphabetic word to one unambiguous known word.

    Only plain alphabetic words of four or more letters are considered, so
    fault codes, numbers and model identifiers never change. The correction
    keeps the first letter and is exactly one edit away (a transposition counts
    as one); if no candidate, or more than one equally close candidate,
    exists the word is left unchanged.
    """
    if len(word) < 4 or not word.isalpha() or word in _SPELLING_LEXICON:
        return None
    limit = 1
    best, best_distance, tied = None, limit + 1, False
    for candidate in _SPELLING_LEXICON:
        if candidate[0] != word[0] or abs(len(candidate) - len(word)) > limit:
            continue
        distance = _edit_distance_at_most(word, candidate, limit)
        if distance < best_distance:
            best, best_distance, tied = candidate, distance, False
        elif distance == best_distance:
            tied = True
    return best if best is not None and not tied else None


def normalize_domain_terms(text: str) -> str:
    """Bounded domain-aware typo normalizer for customer equipment complaints.
    Preserves alphanumeric technical identifiers and fault codes exactly.
    """
    if not text:
        return ""
    words = text.split()
    corrected = []
    for w in words:
        clean = w.strip(".,!?:;\"'()[]{}")
        clean_lower = clean.lower()
        if clean_lower == "forming" and corrected:
            # "Ice forming on the coil" is commonly phrased as "ice formation"
            # in troubleshooting headings. Apply only after ice/frost; a general
            # gerund such as "noise forming" must remain unchanged.
            previous = corrected[-1].strip(".,!?:;\"'()[]{}").casefold()
            if previous in {"ice", "frost"}:
                corrected.append(w.replace(clean, "formation"))
                continue
        if clean_lower in DOMAIN_TYPO_MAP:
            replacement = DOMAIN_TYPO_MAP[clean_lower]
            corrected.append(w.replace(clean, replacement))
        elif (replacement := _correct_spelling(clean_lower)) is not None:
            corrected.append(w.replace(clean, replacement))
        else:
            corrected.append(w)
    return " ".join(corrected)
