"""Clause-aware decomposition of a customer's mixed question.

A customer often combines, in one message, (a) what they observe, (b) what is
fine or already checked, and (c) one or more requests. Searching the whole
message as one bag of words lets an incidental clause dominate retrieval and
answer filtering (for example "the filter is clean" pulling in filter passages)
while the clause that actually matters is diluted or lost.

This module only *labels* clauses of the customer's own words. It never adds a
fact, cause, check, or code, and it makes no authorization or safety decision.
Callers must keep every existing safety guard on the original text.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Optional

REPORT = "report"                     # customer-reported condition/symptom/location
NEGATIVE_REPORT = "negative_report"   # customer-reported "fine / clean / no warning"
COMPLETED_CHECK = "completed_check"   # customer says a check was already done
ASK_CAUSE = "ask_cause"
ASK_OBSERVATION = "ask_observation"
ASK_NEXT = "ask_next"
ASK_OTHER = "ask_other"

_ASK_START = re.compile(
    r"^(?:and\s+|also\s+)?(?:what|which|who|where|when|why|how|could|can|should|"
    r"would|is|are|does|do|did|may|might|will|please)\b", re.I)
_CAUSE_ASK = re.compile(
    r"\b(?:causes?|reasons?|why|what (?:could|might) be wrong|diagnos\w*|root cause)\b|"
    r"\b(?:could|might|may)\b.{0,100}\b(?:cause|explain|account for|lead to|result in)\b",
    re.I)
_OBSERVATION_ASK = re.compile(
    r"\b(?:what|which)\b.{0,80}\b(?:check|inspect|observe|report|notice|look)\b|"
    r"\b(?:safe|customer-safe)\s+(?:checks?|observations?)\b|"
    r"\bhow\s+(?:can|do|should)\s+i\s+(?:safely\s+)?(?:check|observe|report)\b", re.I)
_NEXT_ASK = re.compile(
    r"\b(?:what (?:should|can) i do|what next|next steps?|what now|"
    r"how (?:can|do) i (?:fix|resolve)|what else can i do)\b", re.I)
_NEGATIVE_REPORT = re.compile(
    r"\bno\s+(?:\w+\s+){0,2}(?:warnings?|errors?|codes?|alarms?|faults?|messages?|"
    r"leaks?|leakage|noises?|damage|smell|odou?rs?)\b"
    r"|\b(?:is|are|was|were|looks?|seems?|appears?|remains?)\s+(?:\w+\s+)?"
    r"(?:clean|fine|ok|okay|normal|clear|good|dry|intact|unblocked|undamaged)\b"
    r"|\b(?:is|are|was|were)(?:\s+not|n't)\s+(?:\w+\s+)?"
    r"(?:dirty|blocked|clogged|damaged|leaking|noisy|obstructed|frozen|iced)\b"
    r"|\bnothing\s+(?:is\s+)?(?:showing|displayed|shown|visible)\b"
    r"|\bnot\s+showing\s+(?:any\s+)?(?:warnings?|codes?|errors?|alarms?)\b"
    r"|\b(?:does|do)(?:\s+not|n't)\s+show\s+(?:any\s+)?(?:warnings?|codes?|errors?|alarms?)\b",
    re.I)
_COMPLETED = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|had\s+|already\s+|just\s+|also\s+)*"
    r"(?:checked|cleaned|replaced|inspected|tested|tried|reset|restarted|verified)\b"
    r"|\balready\s+(?:checked|cleaned|replaced|tried|inspected)\b", re.I)
_SPLIT_SENTENCES = re.compile(r"(?<=[.?!;])\s+")
_SPLIT_CONTRAST = re.compile(r",?\s+\b(?:but|however|although|though|yet)\b\s+", re.I)
_LEADING_CONJ = re.compile(r"^(?:but|and|also|so|however)\s+", re.I)

_FILLER = frozenset({
    "what", "should", "safely", "observe", "report", "without", "opening", "check",
    "inspect", "about", "could", "would", "please", "notice", "which", "there",
    "their", "this", "that", "with", "have", "does", "from", "into", "explain",
    "look", "tell", "unit", "your", "show", "showing", "also", "then", "when",
    "while", "being", "been", "will", "just", "only", "some", "more", "like",
})


@dataclass(frozen=True)
class Clause:
    text: str
    role: str


def _classify(text: str) -> str:
    asks = text.rstrip().endswith("?") or bool(_ASK_START.match(text))
    if asks:
        if _CAUSE_ASK.search(text):
            return ASK_CAUSE
        if _OBSERVATION_ASK.search(text):
            return ASK_OBSERVATION
        if _NEXT_ASK.search(text):
            return ASK_NEXT
        return ASK_OTHER
    if _COMPLETED.search(text):
        return COMPLETED_CHECK
    if _NEGATIVE_REPORT.search(text):
        return NEGATIVE_REPORT
    return REPORT


def decompose(question: str) -> List[Clause]:
    """Split the customer's own words into labelled clauses (order preserved)."""
    clauses: List[Clause] = []
    flat = " ".join(str(question or "").split())
    if not flat:
        return clauses
    for sentence in _SPLIT_SENTENCES.split(flat):
        for piece in _SPLIT_CONTRAST.split(sentence):
            piece = _LEADING_CONJ.sub("", piece.strip(" ,;")).strip()
            if len(piece) < 3:
                continue
            clauses.append(Clause(piece, _classify(piece)))
    return clauses


def _content_tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]{4,}", text.casefold()) if w not in _FILLER}


def _join(clauses: List[Clause]) -> str:
    parts = []
    for clause in clauses:
        text = clause.text.strip(" ,;")
        if not text:
            continue
        text = text[0].upper() + text[1:]
        if text[-1] not in ".?!":
            text += "."
        parts.append(text)
    return " ".join(parts)


def _roles(question: str, roles) -> str:
    return _join([c for c in decompose(question) if c.role in roles])


def relevance_text(question: str) -> str:
    """The text that defines what is being asked about now.

    Customer-reported negatives ("filter is clean", "no warning showing") and
    already-completed checks are excluded: they are neither a subject the
    answer must mention nor a symptom the answer must explain.
    """
    kept = _roles(question, {REPORT, ASK_CAUSE, ASK_OBSERVATION, ASK_NEXT, ASK_OTHER})
    return kept or str(question or "")


def ask_text(question: str) -> str:
    """Only the customer's explicit requests."""
    return _roles(question, {ASK_CAUSE, ASK_OBSERVATION, ASK_NEXT, ASK_OTHER})


def negated_text(question: str) -> str:
    """Components/conditions the customer says are fine or already checked."""
    return _roles(question, {NEGATIVE_REPORT, COMPLETED_CHECK})


def retrieval_focus(question: str) -> Optional[str]:
    """One focused retrieval query for a mixed question, or None.

    Keeps customer-reported conditions (for example a blocked or cramped
    location) together with the substantive request, and drops incidental
    negatives, completed checks and content-free request tails (which are
    answered from different, separately gated evidence). Returns None when the
    question has nothing to separate, so single-intent questions are untouched.
    """
    clauses = decompose(question)
    if len(clauses) < 2:
        return None
    primary = [c for c in clauses if c.role in {REPORT, ASK_CAUSE, ASK_OTHER}]
    procedural = [c for c in clauses if c.role in {ASK_OBSERVATION, ASK_NEXT}]
    kept = list(primary)
    if len(_content_tokens(" ".join(c.text for c in kept))) < 2:
        kept = kept + procedural          # request tail carries the only content
    kept_ids = {id(c) for c in kept}
    dropped = [c for c in clauses if id(c) not in kept_ids]
    if not dropped or not kept:
        return None
    query = _join([c for c in clauses if id(c) in kept_ids])
    if len(query) < 8:
        return None
    return query[:350]
