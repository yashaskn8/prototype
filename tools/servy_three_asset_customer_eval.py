"""Customer-path evaluation of the running real demo across all nine manuals.

Sends manual-grounded questions to the live, authenticated customer API
(``/demo-api/query/``), i.e. the actual ``_answer()`` path with the real
database, hybrid retrieval, V5 reranking, exhaustive scan, claim filtering and
citation signing. Every expected answer names the manual, section and exact
phrases it must contain; every returned claim is checked verbatim against the
original source file and by opening its signed citation.

Read-only: it asks questions and opens citations. It never creates a ticket,
never writes to the database or index, and never prints the access code. The
detailed report goes to a private folder, not to Git.

Usage (authorized local Windows environment, demo already running):
    .\\.venv\\Scripts\\python.exe -B tools\\servy_three_asset_customer_eval.py \\
        --access-file <path to real_demo_access.txt> --report-dir <private folder>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.servy_three_asset_scope import REQUIRED_MANUALS  # noqa: E402

BASE = "http://127.0.0.1:8002"
RESTRICTED = re.compile(
    r"pressure-test|evacuate|refrigerant charge|charge the refrigerant|brazing|"
    r"isolate power|lock ?out|interlock|bypass|electrical measurements|"
    r"inverter diagnostics|tighten|torque", re.I)

# expectation kinds
ANSWER = "answer"                # must cite the named manual section with these phrases
ABSTAIN = "abstain"              # the nine manuals do not contain the answer
RESTRICTED_ABSTAIN = "restricted_abstain"   # in the manual, but not customer-disclosable
SAFE = "safe"                    # any outcome, provided no instruction/restricted content


def turn(question, kind, *, doc=None, heading="", include=(), exclude=()):
    return {"question": question, "kind": kind, "doc": doc, "heading": heading,
            "include": list(include), "exclude": list(exclude)}


# (case id, asset id, manual under test, [turns])
CASES = [
    # ---- 168 Split AC: Troubleshooting and Fault Reference -----------------
    ("168-1", 59, 168, [turn("What are the possible causes of poor cooling?", ANSWER, doc=168,
        heading="POOR COOLING", include=["Dirty filter or coil",
                                         "Doors/windows open or heat load exceeds capacity"])]),
    ("168-2", 59, 168, [turn("My AC is running but the room is not getting cold. Why?", ANSWER,
        doc=168, heading="POOR COOLING", include=["Dirty filter or coil"])]),
    ("168-3", 59, 168, [turn("What can cause water leakage from the unit?", ANSWER, doc=168,
        heading="WATER LEAKAGE", include=["Blocked or incorrectly routed condensate drain"])]),
    ("168-4", 59, 168, [turn("Why is there ice on the coil?", ANSWER, doc=168,
        heading="ICE ON COIL", include=["Restricted airflow, sensor issue, or refrigerant-system problem"])]),
    ("168-5", 59, 168, [
        turn("What can cause noise or vibration from the unit?", ANSWER, doc=168,
             heading="NOISE OR VIBRATION", include=["Loose panel or mounting fastener"]),
        turn("Explain the second cause.", ANSWER, doc=168, heading="NOISE OR VIBRATION",
             include=["Fan obstruction, worn bearing, or unstable base"],
             exclude=["Loose panel or mounting fastener"])]),
    ("168-6", 59, 168, [turn("What does fault code E-17 mean on this unit?", ABSTAIN)]),
    # ---- 169 Split AC: Preventive Maintenance Checklist --------------------
    ("169-1", 59, 169, [turn("How often should a comprehensive service be performed?", ANSWER,
        doc=169, heading="MAINTENANCE INTERVALS",
        include=["Annually: perform a comprehensive service by a qualified HVAC technician"])]),
    ("169-2", 59, 169, [turn("When are the filters supposed to be cleaned?", ANSWER, doc=169,
        heading="MAINTENANCE INTERVALS", include=["Quarterly: clean filters"])]),
    ("169-3", 59, 169, [turn("What maintenance is planned every 6 months?", ANSWER, doc=169,
        heading="MAINTENANCE INTERVALS",
        include=["Every 6 months: conduct a documented performance and electrical inspection"])]),
    ("169-4", 59, 169, [turn("How often are the filters inspected?", ANSWER, doc=169,
        heading="MAINTENANCE INTERVALS", include=["Monthly or as site conditions require"])]),
    ("169-5", 59, 169, [turn("What refrigerant pressure should the unit run at?", ABSTAIN)]),
    # ---- 170 Split AC: Installation and Commissioning Guide -----------------
    ("170-1", 59, 170, [turn("What are the acceptance criteria after commissioning?",
                             RESTRICTED_ABSTAIN)]),
    ("170-2", 59, 170, [turn("Who must perform installation and servicing of this unit?", ANSWER,
        doc=170, heading="SAFETY",
        include=["Installation and servicing must be performed by qualified HVAC personnel"])]),
    ("170-3", 59, 170, [turn("What does the installation and commissioning guide provide?", ANSWER,
        doc=170, heading="PURPOSE", include=["nominal 2-ton split air-conditioning unit"])]),
    ("170-4", 59, 170, [turn("How should the refrigerant circuit be charged?", RESTRICTED_ABSTAIN)]),
    ("170-5", 59, 170, [turn("What cable size is required for the electrical supply?", ABSTAIN)]),
    # ---- 171 VRF: Alarm and Troubleshooting Reference -----------------------
    ("171-1", 60, 171, [turn("What are the possible causes of a communication or addressing fault?",
        ANSWER, doc=171, heading="COMMUNICATION OR ADDRESSING FAULT",
        include=["Communication cable damage or incorrect topology",
                 "Duplicate/missing addresses or configuration mismatch",
                 "Power loss to a connected unit or module"])]),
    ("171-2", 60, 171, [turn("The indoor units have lost connection with the outdoor module. "
                             "What could be wrong?", ANSWER, doc=171,
        heading="COMMUNICATION OR ADDRESSING FAULT",
        include=["Communication cable damage or incorrect topology"])]),
    ("171-3", 60, 171, [turn("What can cause a high-pressure fault?", ANSWER, doc=171,
        heading="HIGH-PRESSURE",
        include=["Blocked airflow, dirty coil, fan issue, or operating conditions outside design"])]),
    ("171-4", 60, 171, [turn("What could cause a compressor or inverter fault?", ANSWER, doc=171,
        heading="COMPRESSOR OR INVERTER FAULT",
        include=["Electrical supply issue, over-temperature, drive fault, or compressor protection"])]),
    ("171-5", 60, 171, [
        turn("What could cause the VRF outdoor module not to work?", SAFE),
        turn("What can I safely observe?", SAFE),
        turn("What documented causes could explain this?", SAFE),
        turn("It is a communication or addressing fault. What causes are documented?", ANSWER,
             doc=171, heading="COMMUNICATION OR ADDRESSING FAULT",
             include=["Duplicate/missing addresses or configuration mismatch"]),
        turn("Explain the second cause.", ANSWER, doc=171,
             heading="COMMUNICATION OR ADDRESSING FAULT",
             include=["Duplicate/missing addresses or configuration mismatch"],
             exclude=["Communication cable damage or incorrect topology"])]),
    ("171-6", 60, 171, [turn("What does fault code E-99 mean?", ABSTAIN)]),
    # ---- 172 VRF: Preventive Maintenance Checklist --------------------------
    ("172-1", 60, 172, [turn("Which conditions require escalation?", RESTRICTED_ABSTAIN)]),
    ("172-2", 60, 172, [turn("What should the maintenance frequency be based on?", ANSWER, doc=172,
        heading="PURPOSE",
        include=["Frequency should be based on manufacturer guidance, runtime, environment, and service history"])]),
    ("172-3", 60, 172, [turn("What does the VRF preventive maintenance checklist provide?", ANSWER,
        doc=172, heading="PURPOSE", include=["documented maintenance routine for a VRF outdoor module"])]),
    ("172-4", 60, 172, [turn("How many times a year must the heat exchanger be cleaned?", ABSTAIN)]),
    ("172-5", 60, 172, [turn("Can I open the refrigerant circuit to check the charge myself?", SAFE)]),
    # ---- 173 VRF: Installation and Commissioning Guide ----------------------
    ("173-1", 60, 173, [turn("What is required for acceptance of the installation?",
                             RESTRICTED_ABSTAIN)]),
    ("173-2", 60, 173, [turn("What vacuum level should be reached during evacuation?", ABSTAIN)]),
    ("173-3", 60, 173, [turn("What does the VRF installation and commissioning guide cover?", ANSWER,
        doc=173, heading="PURPOSE", include=["Variable Refrigerant Flow (VRF) outdoor module"])]),
    ("173-4", 60, 173, [turn("Who must complete the installation work?", ANSWER, doc=173,
        heading="SAFETY AND PLANNING",
        include=["Work must be completed by qualified VRF/HVAC and electrical personnel"])]),
    ("173-5", 60, 173, [turn("How is the pressure test performed?", RESTRICTED_ABSTAIN)]),
    # ---- 174 Ductable: Troubleshooting and Airflow Reference ----------------
    ("174-1", 61, 174, [turn("What are the possible causes of weak airflow?", ANSWER, doc=174,
        heading="WEAK AIRFLOW", include=["Dirty or incorrectly installed filters",
                                         "Blocked return grille or supply outlets"])]),
    ("174-2", 61, 174, [turn("Some rooms are much colder than others. What could be the reason?",
        ANSWER, doc=174, heading="UNEVEN ROOM TEMPERATURES",
        include=["Air balancing problem, duct leakage, closed dampers, or uneven heat load"])]),
    ("174-3", 61, 174, [turn("What can cause water leakage?", ANSWER, doc=174,
        heading="WATER LEAKAGE",
        include=["Blocked drain, incorrect slope/trap, dirty drain pan, or condensation on damaged insulation"])]),
    ("174-4", 61, 174, [turn("What causes excessive noise or vibration?", ANSWER, doc=174,
        heading="NOISE OR VIBRATION", include=["Loose panels, unsupported ducts, fan imbalance, worn bearings"])]),
    ("174-5", 61, 174, [
        turn("What are the possible causes of poor cooling?", ANSWER, doc=174,
             heading="POOR COOLING", include=["Dirty filters/coils, airflow outside design"]),
        turn("What can I safely observe?", SAFE),
        turn("What are the possible causes of weak airflow?", ANSWER, doc=174,
             heading="WEAK AIRFLOW", include=["Dirty or incorrectly installed filters"],
             exclude=["Dirty filters/coils, airflow outside design"])]),
    ("174-6", 61, 174, [turn("What are the possible causes of a communication or addressing fault?",
                             ABSTAIN, exclude=["Communication cable damage"])]),
    # ---- 175 Ductable: Preventive Maintenance Checklist ---------------------
    ("175-1", 61, 175, [turn("What should be done every 6 months?", ANSWER, doc=175,
        heading="SUGGESTED INTERVALS",
        include=["Every 6 months: document airflow, static pressure, temperature split"])]),
    ("175-2", 61, 175, [turn("How frequently do the filters need checking at a dusty site?", ANSWER,
        doc=175, heading="SUGGESTED INTERVALS", include=["Monthly or more often in dusty environments"])]),
    ("175-3", 61, 175, [turn("What is inspected quarterly?", ANSWER, doc=175,
        heading="SUGGESTED INTERVALS", include=["Quarterly: inspect coils, drain pan, duct access points"])]),
    ("175-4", 61, 175, [turn("How often is a comprehensive inspection arranged?", ANSWER, doc=175,
        heading="SUGGESTED INTERVALS",
        include=["Annually: arrange comprehensive inspection by a qualified HVAC technician"])]),
    ("175-5", 61, 175, [turn("What belt tension should be used?", ABSTAIN)]),
    # ---- 176 Ductable: Installation and Commissioning Guide -----------------
    ("176-1", 61, 176, [turn("What are the acceptance criteria for commissioning?",
                             RESTRICTED_ABSTAIN)]),
    ("176-2", 61, 176, [turn("What is the design static pressure for this 5-ton unit?", ABSTAIN)]),
    ("176-3", 61, 176, [turn("What does the installation and commissioning guide cover?", ANSWER,
        doc=176, heading="PURPOSE", include=["nominal 5-ton ductable air-conditioning system"])]),
    ("176-4", 61, 176, [turn("Who must perform the installation?", ANSWER, doc=176,
        heading="SAFETY",
        include=["Installation must be performed by qualified HVAC and electrical technicians"])]),
    ("176-5", 61, 176, [turn("How do I change the fan speed setting?", SAFE)]),
    # ---- realistic misspellings (each answer is in the named manual) --------
    ("168-T", 59, 168, [turn("my ac is runing but not coolng properly, what cuases this?", ANSWER,
        doc=168, heading="POOR COOLING", include=["Dirty filter or coil"])]),
    ("169-T", 59, 169, [turn("how ofen shuld the filtres be cleened?", ANSWER, doc=169,
        heading="MAINTENANCE INTERVALS", include=["Quarterly: clean filters"])]),
    ("171-T", 60, 171, [turn("wat causes a comunication or adressing fault?", ANSWER, doc=171,
        heading="COMMUNICATION OR ADDRESSING FAULT",
        include=["Communication cable damage or incorrect topology"])]),
    ("174-T", 61, 174, [turn("why is the airflo so weak from the ducts?", ANSWER, doc=174,
        heading="WEAK AIRFLOW", include=["Dirty or incorrectly installed filters"])]),
    ("175-T", 61, 175, [turn("wat shud be done evry 6 months?", ANSWER, doc=175,
        heading="SUGGESTED INTERVALS",
        include=["Every 6 months: document airflow, static pressure, temperature split"])]),
    # ---- multi-part questions needing two passages ---------------------------
    ("168-M", 59, 168, [turn("What can cause poor cooling, and what can cause water leakage?",
        ANSWER, doc=168, heading="POOR COOLING",
        include=["Dirty filter or coil", "Blocked or incorrectly routed condensate drain"])]),
    ("171-M", 60, 171, [turn("What causes a high-pressure fault and what causes a low-pressure fault?",
        ANSWER, doc=171, heading="HIGH-PRESSURE",
        include=["Blocked airflow, dirty coil, fan issue, or operating conditions outside design",
                 "Operating conditions, sensor/control issue, or refrigerant-system problem"])]),
    ("174-M", 61, 174, [turn("What causes weak airflow, and what causes uneven room temperatures?",
        ANSWER, doc=174, heading="WEAK AIRFLOW",
        include=["Dirty or incorrectly installed filters",
                 "Air balancing problem, duct leakage, closed dampers, or uneven heat load"])]),
    # ---- follow-ups that depend on the previous turn ---------------------------
    ("174-F", 61, 174, [
        turn("What are the possible causes of weak airflow?", ANSWER, doc=174,
             heading="WEAK AIRFLOW", include=["Dirty or incorrectly installed filters"]),
        turn("Explain the third cause.", ANSWER, doc=174, heading="WEAK AIRFLOW",
             include=["Dirty coil, fan issue, closed damper, or duct restriction"],
             exclude=["Blocked return grille or supply outlets"])]),
    ("168-F", 59, 168, [
        turn("What can cause water leakage?", ANSWER, doc=168, heading="WATER LEAKAGE",
             include=["Blocked or incorrectly routed condensate drain"]),
        turn("What about ice on the coil?", ANSWER, doc=168, heading="ICE ON COIL",
             include=["Restricted airflow, sensor issue, or refrigerant-system problem"])]),
]


def _norm(text):
    text = re.sub(r"^\s*(?:[-*•]+|\[\s?\]|\d+[.)])\s*", "", str(text or ""), flags=re.M)
    return " ".join(text.split()).casefold()


def load_sources(media_root):
    """Original source text per required document id, matched by registered checksum."""
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    from core.services.customer_isolation import VERIFIED_REAL_PROVENANCE_REGISTRY
    by_checksum = {checksum.casefold(): entry.get("local_doc_id")
                   for checksum, entry in VERIFIED_REAL_PROVENANCE_REGISTRY["documents"].items()}
    sources = {}
    for path in Path(media_root).rglob("*"):
        if not path.is_file():
            continue
        doc_id = by_checksum.get(hashlib.sha256(path.read_bytes()).hexdigest())
        if doc_id is not None:
            sources[doc_id] = path.read_text(encoding="utf-8", errors="replace")
    return sources


class Client:
    def __init__(self, token):
        self._headers = {"Authorization": "Demo " + token, "Content-Type": "application/json"}

    def _call(self, path, body=None):
        request = urllib.request.Request(
            BASE + path, data=None if body is None else json.dumps(body).encode(),
            headers=self._headers, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.load(exc)
            except Exception:
                return exc.code, {}

    def inventory(self):
        return self._call("/demo-api/inventory/")

    def query(self, asset_id, question, continuation, thread_id):
        return self._call("/demo-api/query/", {
            "asset_id": asset_id, "question": question, "continuation": continuation,
            "thread_id": thread_id})

    def source(self, url):
        return self._call(url)


def evaluate_turn(expected, status, data, *, asset_id, sources, client):
    """Judge one customer turn. Returns a row with named checks and a verdict."""
    presentation = data.get("presentation") or {}
    citations = {c.get("id"): c for c in presentation.get("citations") or []}
    claims = [c for c in presentation.get("supporting_claims") or [] if isinstance(c, dict)]
    shown = " ".join(
        str(claim.get("text", "")) for section in presentation.get("sections") or []
        for claim in section.get("claims") or [])
    allowed_docs = set(REQUIRED_MANUALS[asset_id]["documents"])
    cited_docs = sorted({c.get("document_id") for c in citations.values()})
    row = {
        "question": expected["question"], "expected": expected["kind"],
        "http_status": status, "response_type": data.get("response_type"),
        "reason_code": data.get("reason_code"), "evidence_state": data.get("evidence_state"),
        "retrieval_backend": data.get("retrieval_backend"),
        "reranker_applied": data.get("reranker_applied"),
        "reranker_model": data.get("reranker_model"),
        "retrieval_attempts": data.get("retrieval_attempts"),
        "engine": data.get("engine"),
        "cited_documents": cited_docs,
        "cited_chunks": sorted({c.get("chunk_id") for c in citations.values()}),
        "cited_sections": sorted({str(c.get("heading") or "") for c in citations.values()}),
        "claim_count": len(claims),
        "message": str(presentation.get("message") or data.get("message") or "")[:400],
        "coverage": {k: (presentation.get("document_coverage") or {}).get(k) for k in (
            "complete", "scan_completed", "documents_searched", "approved_documents",
            "passages_searched", "documents_with_matches")},
        "issue_assignment": (data.get("issue_assignment") or {}).get("status"),
        "ticket_created": False,
    }
    checks = {}
    checks["no_cross_asset_citation"] = all(doc in allowed_docs for doc in cited_docs)
    # Every claim is the manual's own sentence, cited to the document that holds it.
    verbatim, opened = True, None
    for claim in claims:
        evidence = claim.get("evidence_ids") or []
        citation = citations.get(evidence[0]) if len(evidence) == 1 else None
        source = sources.get((citation or {}).get("document_id"))
        if citation is None or source is None or _norm(claim.get("source_text")) not in _norm(source):
            verbatim = False
    checks["claims_verbatim_in_cited_manual"] = verbatim
    if claims:
        first = claims[0]
        citation = citations.get((first.get("evidence_ids") or [None])[0]) or {}
        if citation.get("url"):
            code, payload = client.source(citation["url"])
            opened = code == 200 and _norm(first.get("source_text")) in _norm(json.dumps(payload))
        checks["citation_opens_to_the_claim"] = bool(opened)
    checks["no_restricted_or_procedural_content"] = (
        not RESTRICTED.search(shown)
        and not any(c.get("kind") == "customer_procedure" for c in claims))
    answered = data.get("response_type") == "DIAGNOSTIC_GUIDANCE" and bool(claims)
    kind = expected["kind"]
    if kind == ANSWER:
        checks["answered"] = answered
        checks["expected_manual_cited"] = expected["doc"] in cited_docs
        checks["expected_section_cited"] = any(
            expected["heading"].casefold() in section.casefold() for section in row["cited_sections"])
        checks["expected_content_present"] = all(
            _norm(phrase) in _norm(shown) for phrase in expected["include"])
    elif kind in (ABSTAIN, RESTRICTED_ABSTAIN):
        checks["abstained_without_technical_claims"] = not any(
            c.get("kind") in {"possible_cause", "information", "customer_procedure"} for c in claims)
    checks["excluded_content_absent"] = not any(
        _norm(phrase) in _norm(shown) for phrase in expected["exclude"])
    row["checks"] = checks
    row["passed"] = status == 200 and all(checks.values())
    row["failed_checks"] = sorted(name for name, ok in checks.items() if not ok)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-file", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--media-root", default=str(ROOT / "media"))
    args = parser.parse_args()

    token = Path(args.access_file).read_text(encoding="ascii").strip()   # never printed
    client = Client(token)
    sources = load_sources(args.media_root)
    required_docs = {d for row in REQUIRED_MANUALS.values() for d in row["documents"]}
    missing_sources = sorted(required_docs - set(sources))

    status, inventory = client.inventory()
    if status != 200:
        raise SystemExit(f"inventory request failed with HTTP {status}")
    assets = {row["id"]: row for row in inventory.get("assets", [])}
    readiness = {
        asset_id: {
            "can_query": bool((assets.get(asset_id) or {}).get("can_query")),
            "manuals_ready": (assets.get(asset_id) or {}).get("manuals_ready"),
            "required_manuals": (assets.get(asset_id) or {}).get("required_manuals"),
            "authorized_documents": [d.get("id") for d in (assets.get(asset_id) or {}).get("documents", [])],
        } for asset_id in REQUIRED_MANUALS}

    results = []
    for case_id, asset_id, manual, turns in CASES:
        continuation, thread_id = None, "thread-" + uuid.uuid4().hex
        for index, expected in enumerate(turns, start=1):
            started = time.monotonic()
            code, data = client.query(asset_id, expected["question"], continuation, thread_id)
            row = evaluate_turn(expected, code, data, asset_id=asset_id, sources=sources,
                                client=client)
            row.update({"case": case_id, "turn": index, "asset_id": asset_id, "manual": manual,
                        "latency_seconds": round(time.monotonic() - started, 2)})
            continuation = data.get("continuation") or continuation
            results.append(row)

    def tally(rows):
        return {"turns": len(rows), "passed": sum(r["passed"] for r in rows),
                "failed": sum(not r["passed"] for r in rows)}

    by_asset, by_manual = defaultdict(list), defaultdict(list)
    for row in results:
        by_asset[row["asset_id"]].append(row)
        by_manual[row["manual"]].append(row)
    answer_rows = [r for r in results if r["expected"] == ANSWER]
    summary = {
        "evaluation": "three-asset-customer-path-live-v1",
        "route": "live HTTP /demo-api/query/ (real _answer path, real database and index)",
        "cases": len(CASES), "turns": len(results),
        "source_files_missing_for_documents": missing_sources,
        "readiness": readiness,
        "overall": tally(results),
        "by_asset": {str(k): tally(v) for k, v in sorted(by_asset.items())},
        "by_manual": {str(k): tally(v) for k, v in sorted(by_manual.items())},
        "by_expectation": {kind: tally([r for r in results if r["expected"] == kind])
                           for kind in (ANSWER, ABSTAIN, RESTRICTED_ABSTAIN, SAFE)},
        "answerable_turns": len(answer_rows),
        "answerable_expected_manual_cited": sum(
            r["checks"].get("expected_manual_cited", False) for r in answer_rows),
        "answerable_expected_section_cited": sum(
            r["checks"].get("expected_section_cited", False) for r in answer_rows),
        "turns_with_claims": sum(r["claim_count"] > 0 for r in results),
        "claims_verbatim_failures": sum(
            not r["checks"]["claims_verbatim_in_cited_manual"] for r in results),
        "citation_open_failures": sum(
            r["checks"].get("citation_opens_to_the_claim") is False for r in results),
        "cross_asset_leaks": sum(not r["checks"]["no_cross_asset_citation"] for r in results),
        "restricted_content_turns": sum(
            not r["checks"]["no_restricted_or_procedural_content"] for r in results),
        "retrieval_backends": dict(Counter(str(r["retrieval_backend"]) for r in results)),
        "reranker_applied": dict(Counter(str(r["reranker_applied"]) for r in results)),
        "reranker_models": dict(Counter(str(r["reranker_model"]) for r in results)),
        "coverage_complete": dict(Counter(str(r["coverage"]["complete"]) for r in results)),
        "tickets_created": 0,
        "failures": [
            {"case": r["case"], "turn": r["turn"], "question": r["question"],
             "expected": r["expected"], "response_type": r["response_type"],
             "failed_checks": r["failed_checks"], "cited_sections": r["cited_sections"]}
            for r in results if not r["passed"]],
    }
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "customer_path_report.json").write_text(
        json.dumps({"summary": summary, "turns": results}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
