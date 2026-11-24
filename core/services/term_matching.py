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


# Calendar words that name the same interval ("yearly" service = "Annually").
# A plural carries a count ("every 6 months", "two years") and is a different
# interval, so only the singular shares the key.
_INTERVAL_KEYS = {"yearly": "@year", "annual": "@year", "annually": "@year", "year": "@year",
                  "quarter": "@quarter", "quarterly": "@quarter",
                  "month": "@month", "monthly": "@month",
                  "week": "@week", "weekly": "@week"}
_DERIVATIONAL = ("ations", "ation", "ions", "ion", "ements", "ement")


def roots(word: str) -> set:
    """Derivational roots so a verb meets its noun: vibrates/vibration share
    "vibrat", inspect/inspection "inspect", install/installation "install".
    Every applicable suffix strip is kept; roots shorter than 5 letters are
    dropped so unrelated short words (station/state) never meet."""
    w = stem(word)
    found = set()
    for suffix in _DERIVATIONAL:
        if w.endswith(suffix):
            found.add(w[: -len(suffix)])
    if w.endswith("e"):
        found.add(w[:-1])
    found.add(w)
    return {r for r in found if len(r) >= 5}


def match_keys(words) -> set:
    """Comparable keys for a set of words: each word, its folded form, its
    derivational root and, for calendar words, a shared interval key."""
    keys = set()
    for word in words or ():
        w = str(word).casefold()
        if w:
            keys.add(w)
            keys.add(stem(w))
            if len(w) >= 5 and not any(ch.isdigit() for ch in w):
                keys |= roots(w)
            if w in _INTERVAL_KEYS:
                keys.add(_INTERVAL_KEYS[w])
    return keys


def text_keys(text: str, stopwords=frozenset()) -> set:
    words = set(re.findall(r"[a-z0-9_-]{3,}", str(text or "").casefold())) - set(stopwords)
    return match_keys(words)


# concept -> (customer wording, wording a section heading must contain)
_TEMPERATURE_WORD = r"(?:colder|cooler|warmer|hotter|too\s+(?:cold|cool|warm|hot))"
SYMPTOM_CONCEPTS = {
    "poor_cooling": (
        re.compile(
            r"\b(?:not|isn'?t|doesn'?t|won'?t|never|no longer|barely|hardly)\s+"
            r"(?:(?:getting|becoming|staying|feeling|get|become|stay|feel|"
            r"making (?:it|the room))\s+)?(?:cold|cool|cooler|colder)\b"
            r"|\b(?:not|isn'?t|doesn'?t|no|poor|weak|insufficient|reduced)\s+cooling\b"
            r"|\b(?:not|isn'?t|doesn'?t|won'?t)\s+cool(?:ing)?\b"
            # "it blows air but the room stays warm", "blowing warm air"
            r"|\b(?:room|rooms|house|home|office|space|place)\s+(?:still\s+)?(?:stays?|remains?|is|are|feels?|keeps?)"
            r"\s+(?:still\s+|getting\s+|too\s+|so\s+)?(?:warm|hot)\b"
            r"|\bblow\w*\s+(?:out\s+)?(?:warm|hot)\s+air\b", re.I),
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
    # "Water is dripping from the indoor unit" reports the WATER LEAKAGE symptom.
    "water_leakage": (
        re.compile(
            r"\b(?:water|condensate)\b[^.?!]{0,40}\b(?:leak\w*|drip\w*|pour\w*|overflow\w*|spill\w*)\b"
            r"|\b(?:leak\w*|drip\w*)\b[^.?!]{0,40}\b(?:water|condensate)\b", re.I),
        re.compile(r"\bwater\s+leak\w*\b", re.I),
    ),
    "communication": (
        re.compile(
            r"\b(?:lost|losing|lose|loses|dropped|no)\s+(?:the\s+|their\s+|its\s+)?"
            r"(?:connection|communication|link)\b"
            r"|\b(?:not|no longer|stopped|can'?t|cannot)\s+(?:communicat\w+|connect\w*)\b"
            r"|\b(?:communication|connection)\s+(?:fault|loss|lost|error|failure|problem)\b"
            # "two indoor units keep dropping off the network", "units go offline"
            r"|\b(?:drop\w*|fall\w*|go\w*|went|kick\w*)\s+off\s+(?:the\s+)?(?:network|bus|line|controller)\b"
            r"|\boffline\b"
            r"|\b(?:units?|controllers?|modules?|they|it)\s+(?:keeps?\s+|kept\s+)?disconnect(?:s|ed|ing)\b",
            re.I),
        re.compile(r"\bcommunication\b", re.I),
    ),
    "noise_vibration": (
        re.compile(r"\b(?:nois\w*|vibrat\w*|rattl\w*|squeal\w*|grind\w*|click\w*|knock\w*|buzz\w*|clank\w*)\b", re.I),
        re.compile(r"\b(?:noise|vibration)\b", re.I),
    ),
    "weak_airflow": (
        re.compile(
            r"\b(?:weak|low|poor|reduced|little|hardly\s+any)\s+(?:air\s*flow|air)\b"
            r"|\bair\s*(?:flow)?\b[^.?!,;]{0,60}\b(?:weak|low|reduced|poor|barely)\b", re.I),
        re.compile(r"\bair\s*flow\b", re.I),
    ),
    "low_capacity": (
        re.compile(r"\b(?:low|poor|reduced|lost|insufficient)\s+(?:cooling\s+)?capacity\b"
                   r"|\bcapacity\b[^.?!]{0,20}\b(?:low|poor|reduced|dropp\w*)\b", re.I),
        re.compile(r"\bcapacity\b", re.I),
    ),
    # "Could high pressure be involved?" names the HIGH-PRESSURE section; the word
    # "pressure" alone ("what pressure should it run at?") names no fault.
    "high_pressure": (
        re.compile(r"\b(?:high|excessive|too\s+high)[-\s]+(?:head\s+)?pressure\b|\bhead\s+pressure\b|\bhp\s+(?:fault|alarm|trip)\b", re.I),
        re.compile(r"\bhigh[-\s]?pressure\b", re.I),
    ),
    "low_pressure": (
        re.compile(r"\b(?:low|too\s+low)[-\s]+(?:suction\s+)?pressure\b|\blp\s+(?:fault|alarm|trip)\b", re.I),
        re.compile(r"\blow[-\s]?pressure\b", re.I),
    ),
    "compressor_fault": (
        re.compile(r"\b(?:compressor|inverter)\b[^.?!]{0,30}\b(?:trip\w*|fault\w*|fail\w*|stop\w*|shut\w*|"
                   r"alarm\w*|error\w*|cut\w*\s+out|won'?t\s+start|not\s+start\w*)\b", re.I),
        re.compile(r"\b(?:compressor|inverter)\b", re.I),
    ),
    "fan_fault": (
        re.compile(r"\bfans?\b[^.?!]{0,30}\b(?:not\s+(?:spinning|turning|running|working|moving)|"
                   r"isn'?t\s+(?:spinning|turning|running|working|moving)|stopped|fail\w*|fault\w*|stuck)\b", re.I),
        re.compile(r"\bfan\b", re.I),
    ),
    # "My AC won't turn on", "it doesn't start", "no power at all": the DOES NOT START symptom.
    "no_start": (
        re.compile(r"\b(?:won'?t|wont|doesn'?t|does\s+not|will\s+not|can'?t|cannot)\s+(?:turn|switch|power)\s+on\b"
                   r"|\b(?:won'?t|wont|doesn'?t|does\s+not|will\s+not|didn'?t|did\s+not)\s+start\b"
                   r"|\bnot\s+(?:starting|turning\s+on|powering\s+on|switching\s+on)\b|\bno\s+power\b", re.I),
        re.compile(r"\b(?:does\s+not|doesn'?t|not|fails?\s+to)\s+start\b|\bno\s+power\b", re.I),
    ),
    # A generic equipment failure ("not working", "stopped working") does not
    # name one symptom.  It matches any operational-failure heading so the
    # evidence check can reach the documented cause lists inside those sections.
    "generic_failure": (
        re.compile(
            # <negation> + <functional verb>
            r"\b(?:not|isn'?t|doesn'?t|won'?t|can'?t|cannot)\s+"
            r"(?:work(?:ing)?|function(?:ing)?|operat(?:e|ing)|run(?:ning)?"
            r"|starting|turning on)\b"
            # stopped/quit/ceased + optional modifier
            r"|\b(?:stopped|quit|ceased|has\s+stopped|suddenly\s+stopped"
            r"|stopped\s+suddenly)\s*(?:working|functioning|running|operating)?\b"
            # "not work" bare
            r"|\bnot\s+work(?:ing)?\b"
            # "broke down / broken / malfunctioning / failed / failure / defective"
            r"|\b(?:broke\s+down|broken\s+down|broken|malfunctioning|malfunction(?:ed)?"
            r"|failed|failure|defective|inoperative|dead)\b"
            # "what could be wrong / what's wrong / something is wrong / going wrong"
            r"|\b(?:what(?:'s|\s+is|\s+could\s+be)\s+wrong"
            r"|something\s+(?:is|seems?)\s+wrong|going\s+wrong)\b"
            # "won't turn on / shut down / shut off"
            r"|\b(?:won'?t|doesn'?t|will\s+not|does\s+not)\s+(?:turn\s+on|power\s+on|switch\s+on)\b"
            r"|\b(?:shut\s+down|shut\s+off|turned?\s+off)\s*(?:on\s+its\s+own|by\s+itself|unexpectedly|suddenly)?\b",
            re.I),
        re.compile(
            r"\b(?:does\s+not\s+start|not\s+start|poor\s+cooling|not\s+cooling"
            r"|fault|symptom|troubleshoot|failure|malfunction|cause|diagnostic"
            r"|problem|abnormal|error)\b",
            re.I),
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


_NEGATED_HEADING = re.compile(r"\b(?:not|no|fails?)\b")


def _question_negates(question: str, words) -> bool:
    """True when the question negates one of the words ("doesn't start", "not starting")."""
    stems = "|".join(re.escape(w[:5]) + r"\w*" for w in words)
    return bool(re.search(r"(?:\bnot|n'?t|\bno|\bnever|\bwont|\bfail\w*)\s+(?:\w+\s+){0,3}?(?:" + stems + r")\b",
                          str(question or ""), re.I))


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
    negated_words = []
    for alternative in re.split(r"\s+or\s+", text):
        words = [w for w in re.findall(r"[a-z0-9]+", alternative)
                 if len(w) >= 3 and w not in _HEADING_LABELS and w not in _OPTIONAL_DEGREE]
        if _NEGATED_HEADING.search(alternative):
            # "UNIT DOES NOT START" is stated only by a negated "start" ("doesn't
            # start", "isn't starting"), never by "the AC does start".
            negated_words += words
            if words and _question_negates(question, words):
                return True
            continue
        if words and all(any(_same_word(w, q) for q in asked) for w in words):
            return True
    if negated_words and not _question_negates(question, negated_words):
        return False
    try:
        from core.ml.conditional_rerank import _extract_symptom_keys, _heading_matches_symptom
        # "pressure" alone carries no polarity and is decided above.
        # "generic_failure" is not a specific symptom — exclude it so a
        # "not working" history does not match every fault heading.
        keys = _extract_symptom_keys(str(question or "")) - {"pressure", "generic_failure"}
        return bool(keys) and _heading_matches_symptom(heading or "", keys, str(question or ""))
    except Exception:
        return False


DOMAIN_TYPO_MAP = {
    # Number words match the digits used in schedules ("every six months").
    "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
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
