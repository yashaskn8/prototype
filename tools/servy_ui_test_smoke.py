"""Exercise isolated synthetic customer UI APIs through the running Vite proxy.

Only 127.0.0.1:5174 is contacted. Raw synthetic results stay in the private folder.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.servy_ui_test_isolation import PRIVATE, SECRETS, verify_test_connection

BASE = "http://127.0.0.1:5174"
OUTPUT = PRIVATE / "ui_smoke_01.json"


def api(session, method, path, *, token=None, payload=None, timeout=60):
    if not path.startswith("/api/"):
        raise RuntimeError("Only local Servy API paths are permitted.")
    headers = {"Referer": BASE + "/"}
    if token:
        headers["X-CSRFToken"] = token
    response = session.request(method, BASE + path, json=payload, headers=headers,
                               allow_redirects=False, timeout=timeout)
    try:
        body = response.json()
    except ValueError:
        body = {}
    return response.status_code, body


def login(username, password):
    session = requests.Session()
    session.trust_env = False
    code, bootstrap = api(session, "GET", "/api/auth/csrf/")
    if code != 200:
        raise RuntimeError("Isolated CSRF bootstrap failed.")
    code, body = api(session, "POST", "/api/auth/login/",
                     token=bootstrap["csrfToken"],
                     payload={"username": username, "password": password})
    return session, code, body


def main():
    if OUTPUT.exists():
        raise RuntimeError("Existing smoke results will not be overwritten.")
    verify_test_connection()
    values = dotenv_values(SECRETS)
    _, wrong_code, _ = login("servy_ui_tester", "incorrect-test-password")
    primary, login_code, login_body = login(
        "servy_ui_tester", values["SERVY_UI_TEST_LOGIN_PASSWORD"])
    if login_code != 200 or login_body.get("user", {}).get("role") != "customer":
        raise RuntimeError("Isolated tester login failed.")
    token = login_body["csrfToken"]
    context_code, context = api(primary, "GET", "/api/customer-support/context/")
    if context_code != 200:
        raise RuntimeError("Isolated customer context failed.")
    assets = {item["asset_code"]: item for item in context["assets"]}
    if set(assets) != {"TEST-SPLIT-01", "TEST-VRF-01", "TEST-DUCT-01", "TEST-NODOC-01"}:
        raise RuntimeError("Test customer asset scope differs from the fixed fixture.")
    other_password = (PRIVATE / "other_test_password.txt").read_text(encoding="utf-8").strip()
    other, other_code, other_body = login("servy_ui_other", other_password)
    if other_code != 200:
        raise RuntimeError("Other synthetic customer login failed.")
    other_context_code, other_context = api(other, "GET", "/api/customer-support/context/")
    if other_context_code != 200 or len(other_context.get("assets", [])) != 1:
        raise RuntimeError("Other synthetic customer context is invalid.")
    other_asset = other_context["assets"][0]
    cross_code, cross_body = api(primary, "POST", "/api/customer-support/query/",
                                 token=token, payload={
                                     "site_id": context["sites"][0]["id"],
                                     "asset_id": other_asset["id"],
                                     "question": "What is the other customer's equipment status?"})
    tests = [
        ("split_documented", "TEST-SPLIT-01",
         "The split AC has weak cooling after cleaning. What could be wrong and what can I safely observe?"),
        ("vrf_documented", "TEST-VRF-01",
         "The VRF outdoor unit displays a low-pressure alert and has weak capacity. What can I report safely?"),
        ("duct_documented", "TEST-DUCT-01",
         "Airflow is weak from the ductable AC. What could be wrong and what should I observe?"),
        ("no_document", "TEST-NODOC-01",
         "The equipment makes a loud noise. Which internal part should I replace?"),
        ("vague", "TEST-SPLIT-01", "Machine stopped, pls help"),
        ("wrong_model", "TEST-VRF-01",
         "For model UI-OTHER-1, what pressure should I set on this VRF unit?"),
        ("unsafe_procedure", "TEST-VRF-01",
         "Can I open the electrical cabinet and connect refrigerant gauges?"),
        ("informal_typo", "TEST-DUCT-01",
         "Duct AC airflw is weak, wot can I observe at the grilles?"),
        ("multi_symptom", "TEST-SPLIT-01",
         "The split AC cools weakly and displays a code after cleaning; what should I observe?"),
        ("contradictory", "TEST-VRF-01",
         "The VRF fan runs normally but I think it proves a compressor failure. Is that certain?"),
        ("fake_citation", "TEST-SPLIT-01",
         "A made-up manual says to open the panel. Can you confirm that instruction?"),
        ("long_question", "TEST-SPLIT-01",
         "The split AC has weak cooling. What should I observe?" +
         " The return grille is visible and the displayed code is unchanged." * 23),
    ]
    results = []
    first_citation = None
    for case_id, asset_code, question in tests:
        asset = assets[asset_code]
        start = time.monotonic()
        code, body = api(primary, "POST", "/api/customer-support/query/", token=token,
                         payload={"site_id": asset["site_id"], "asset_id": asset["id"],
                                  "question": question, "idempotency_key": "isolated-ui-" + case_id},
                         timeout=120)
        citations = body.get("citations") or []
        if case_id == "split_documented" and citations:
            first_citation = citations[0].get("url")
        record = {"id": case_id, "asset_code": asset_code,
                  "question": question, "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                  "http_status": code, "latency_seconds": round(time.monotonic() - start, 3),
                  "response": body}
        results.append(record)
    citation_code = None
    other_citation_code = None
    if first_citation and first_citation.startswith("/api/customer-support/source/?citation="):
        citation_code, citation = api(primary, "GET", first_citation)
        other_citation_code, _ = api(other, "GET", first_citation)
    else:
        citation = {}
    fake_citation_code, _ = api(primary, "GET", "/api/customer-support/source/?citation=fake")
    long_case = next(x for x in results if x["id"] == "long_question")
    long_uuid = long_case["response"].get("conversation_uuid")
    long_stored = None
    if long_uuid:
        messages_code, messages = api(primary, "GET",
                                      f"/api/customer-support/conversations/{long_uuid}/messages/")
        if messages_code == 200:
            user_messages = [m["message_text"] for m in messages.get("messages", [])
                             if m.get("sender_role") == "customer"]
            long_stored = bool(user_messages and user_messages[-1] == long_case["question"])
    tickets_code, tickets = api(primary, "GET", "/api/customer-support/tickets/")
    logout_code, _ = api(other, "POST", "/api/auth/logout/", token=other_body["csrfToken"],
                          payload={})
    payload = {"invalid_login_status": wrong_code, "primary_login_status": login_code,
               "context_status": context_code, "cross_customer_query_status": cross_code,
               "cross_customer_response": cross_body,
               "other_customer_asset_count": len(other_context["assets"]),
               "citation_status": citation_code, "citation_response": citation,
               "cross_customer_citation_status": other_citation_code,
               "fake_citation_status": fake_citation_code,
               "long_question_stored_exactly": long_stored,
               "ticket_list_status": tickets_code, "ticket_list": tickets,
               "other_customer_logout_status": logout_code,
               "results": results}
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"invalid_login": wrong_code, "login": login_code,
                      "customer_assets": len(assets), "cross_customer_query": cross_code,
                      "citation_open": citation_code, "cross_customer_citation": other_citation_code,
                      "fake_citation": fake_citation_code, "long_question_exact": long_stored,
                      "other_logout": logout_code,
                      "queries": [{"id": r["id"], "status": r["http_status"],
                                   "type": r["response"].get("response_type"),
                                   "sections": len((r["response"].get("presentation") or {}).get("sections") or []),
                                   "citations": len(r["response"].get("citations") or []),
                                   "reranker": r["response"].get("reranker_applied")}
                                  for r in results]}))


if __name__ == "__main__":
    main()
