"""Optional, exact-model supplementary manufacturer web research."""
from __future__ import annotations

import os
import re
import time
from urllib.parse import urlsplit

import requests

ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"
MAX_RESPONSE_BYTES = 1200000
MAX_SOURCES = 5
MAX_SNIPPETS_PER_SOURCE = 3
MAX_SNIPPET_CHARS = 1200
TIMEOUT = (2.5, 15)
_COOLDOWN_UNTIL = 0.0

_CODE = re.compile(r"(?<![A-Z0-9])(?:ERR|E|F)[- ]?\d{1,3}(?![A-Z0-9])", re.I)
_EXPLICIT_MODEL = re.compile(
    r"\bmodel(?:\s+(?:number|no\.?|is))?\s*[:#]?\s*"
    r"([A-Za-z0-9][A-Za-z0-9._/-]{2,})\b", re.I,
)
_PROCEDURAL = re.compile(
    r"\b(?:open(?:s|ed|ing)?|remove(?:s|d|ing)?|replace(?:s|d|ing)?|"
    r"repair(?:s|ed|ing)?|reset(?:s|ting)?|restart(?:s|ed|ing)?|"
    r"adjust(?:s|ed|ing)?|measure(?:s|d|ing)?|probe(?:s|d|ing)?|"
    r"wire(?:s|d|ing)?|bypass(?:es|ed|ing)?|charge(?:s|d|ing)?|"
    r"clean(?:s|ed|ing)?|test(?:s|ed|ing)?|inspect(?:s|ed|ing)?|"
    r"disconnect(?:s|ed|ing)?|connect(?:s|ed|ing)?|"
    r"disassemble(?:s|d|ing)?|install(?:s|ed|ing)?|refill(?:s|ed|ing)?|"
    r"check(?:s|ed|ing)?|verify(?:s|ied|ing)?|top\s+up)\b",
    re.I,
)
_ALLOWED_SYMPTOM_TERMS = frozenset({
    "not", "working", "start", "starts", "starting", "stop", "stops",
    "stopped", "cooling", "heating", "noise", "noisy", "vibration",
    "airflow", "air", "flow", "leak", "water", "drain", "fault", "error",
    "code", "warning", "display", "compressor", "fan", "coil", "sensor",
    "control", "outdoor", "module", "pressure", "temperature", "operation",
    "performance", "intermittent", "tripping", "trip",
})
_NOTES = {
    "not_configured": (
        "Supplementary online research is not enabled. An operator must configure "
        "the search API key and official-manufacturer-domain allowlist."
    ),
    "model_identity_incomplete": (
        "Online research was skipped because the asset has no verified "
        "manufacturer and exact model number."
    ),
    "asset_model_mismatch": (
        "Online research was skipped because the question names a model that "
        "does not match the selected asset."
    ),
    "not_applicable": "Online research was not needed for this turn.",
    "no_exact_match": (
        "No allowlisted manufacturer source passed the exact model and fault-code "
        "matching checks. No external technical claim was added."
    ),
    "unavailable": (
        "The online research provider was unavailable. Servy did not substitute "
        "an unverified website or guess at missing information."
    ),
}


def _allowed_domains():
    raw = os.environ.get("SERVY_WEB_RESEARCH_ALLOWED_DOMAINS", "")
    domains = set()
    for value in raw.split(","):
        domain = value.strip().lower().strip(".")
        if domain and re.fullmatch(r"[a-z0-9.-]{1,253}", domain) and ".." not in domain:
            domains.add(domain)
    return domains


def _host_is_allowed(host, domains):
    host = (host or "").lower().rstrip(".")
    return bool(host and any(host == domain or host.endswith("." + domain) for domain in domains))


def _contains_literal(text, value):
    if not isinstance(text, str) or not isinstance(value, str) or not value.strip():
        return False
    return bool(re.search(
        r"(?<![A-Z0-9])" + re.escape(value.strip()) + r"(?![A-Z0-9])",
        text, re.I,
    ))


def _public_symptom_terms(question):
    words = re.findall(r"[a-z0-9]+", str(question or "").casefold())
    return " ".join(dict.fromkeys(
        word for word in words if word in _ALLOWED_SYMPTOM_TERMS
    ))[:180]


def _status(status):
    return {
        "status": status,
        "label": "Supplementary online information — not company-approved",
        "note": _NOTES.get(status, _NOTES["unavailable"]),
        "sources": [],
    }


def search_model_matched_sources(asset, question, *, original_question=""):
    """Return allowlisted manufacturer excerpts matching exact model and code."""
    global _COOLDOWN_UNTIL
    if time.monotonic() < _COOLDOWN_UNTIL:
        return _status("unavailable")

    product = getattr(asset, "product", None)
    brand = getattr(product, "brand", None) if product else None
    manufacturer = str(getattr(brand, "name", "") or "").strip()
    model = str(getattr(asset, "model_number", "") or "").strip()
    if not manufacturer or not model:
        return _status("model_identity_incomplete")

    combined_text = " ".join(
        part for part in (
            str(original_question or "").strip(),
            str(question or "").strip(),
        ) if part
    )[:1400]
    explicit_models = [match.group(1) for match in _EXPLICIT_MODEL.finditer(combined_text)]
    if any(value.casefold() != model.casefold() for value in explicit_models):
        return _status("asset_model_mismatch")

    public_terms = _public_symptom_terms(question)
    requested_codes = list(dict.fromkeys(match.group(0) for match in _CODE.finditer(combined_text)))
    if not public_terms and not requested_codes:
        return _status("not_applicable")
    api_key = os.environ.get("SERVY_BRAVE_SEARCH_API_KEY", "").strip()
    domains = _allowed_domains()
    if not api_key or not domains:
        return _status("not_configured")

    # Do not send customer identity, asset code, serial, internal documents,
    # manual passages, or the raw conversation to the external provider.
    query = '"' + manufacturer[:100] + '" "' + model[:100] + '" HVAC'
    if requested_codes:
        query += " " + " ".join('"' + code + '"' for code in requested_codes)
    if public_terms:
        query += " " + public_terms
    query += " manufacturer technical documentation"

    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            ENDPOINT,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            params={
                "q": query[:600],
                "country": "IN",
                "search_lang": "en",
                "count": 10,
                "maximum_number_of_urls": 8,
                "maximum_number_of_tokens": 4096,
                "maximum_number_of_tokens_per_url": 1024,
                "context_threshold_mode": "strict",
                "safesearch": "strict",
                "enable_source_metadata": "true",
            },
            timeout=TIMEOUT,
            allow_redirects=False,
        )
        if response.status_code == 429:
            _COOLDOWN_UNTIL = time.monotonic() + 45
            return _status("unavailable")
        if response.status_code != 200 or len(response.content) > MAX_RESPONSE_BYTES:
            return _status("unavailable")
        payload = response.json()
        grounding = payload.get("grounding", {}) if isinstance(payload, dict) else {}
        rows = grounding.get("generic", []) if isinstance(grounding, dict) else []
        source_meta = payload.get("sources", {}) if isinstance(payload, dict) else {}
        if not isinstance(rows, list) or not isinstance(source_meta, dict):
            return _status("unavailable")

        accepted, seen_urls = [], set()
        requested_set = {code.casefold() for code in requested_codes}
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = row.get("url")
            if not isinstance(url, str) or len(url) > 2048:
                continue
            try:
                parsed = urlsplit(url)
                host = (parsed.hostname or "").lower().rstrip(".")
                valid_port = parsed.port in (None, 443)
            except ValueError:
                continue
            if (
                parsed.scheme != "https" or not _host_is_allowed(host, domains)
                or parsed.username or parsed.password or not valid_port
                or url in seen_urls
            ):
                continue
            raw_snippets = row.get("snippets", [])
            if not isinstance(raw_snippets, list):
                continue
            matched_snippets = []
            for snippet in raw_snippets:
                if not isinstance(snippet, str):
                    continue
                clean = " ".join(snippet.split())
                if not clean or len(clean) > 8000:
                    continue
                if not _contains_literal(clean, model) or _PROCEDURAL.search(clean):
                    continue
                found_codes = {match.group(0).casefold() for match in _CODE.finditer(clean)}
                if requested_codes:
                    if not requested_set.issubset(found_codes) or found_codes - requested_set:
                        continue
                elif found_codes:
                    continue
                matched_snippets.append(clean[:MAX_SNIPPET_CHARS])
                if len(matched_snippets) >= MAX_SNIPPETS_PER_SOURCE:
                    break
            if not matched_snippets:
                continue
            metadata = source_meta.get(url, {})
            if not isinstance(metadata, dict):
                metadata = {}
            age = metadata.get("age")
            accepted.append({
                "title": str(metadata.get("title") or row.get("title") or host)[:220],
                "url": url,
                "hostname": host,
                "age": str(age[0])[:100] if isinstance(age, list) and age else "",
                "match_basis": "exact_model_and_fault_code" if requested_codes else "exact_model",
                "snippets": matched_snippets,
            })
            seen_urls.add(url)
            if len(accepted) >= MAX_SOURCES:
                break

        if not accepted:
            return _status("no_exact_match")
        return {
            "status": "available",
            "label": "Supplementary online information — not company-approved",
            "note": (
                "These excerpts are from allowlisted manufacturer websites and match "
                "the selected asset's exact model"
                + (" and exact reported fault code." if requested_codes else ".")
                + " They are supplementary only; they do not override company "
                "manuals or authorize customer repair actions."
            ),
            "sources": accepted,
        }
    except (requests.RequestException, ValueError, TypeError, AttributeError, KeyError):
        _COOLDOWN_UNTIL = time.monotonic() + 20
        return _status("unavailable")
    finally:
        session.close()
