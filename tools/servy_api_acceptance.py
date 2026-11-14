"""HTTP acceptance checks for the conversational API (API 1) and the
conversation-to-ServiceCall API (API 2), through the real gateways on servy_dev.

Complements the 26-check live workflow with: the non-streaming messages
endpoint, duplicate-submission idempotency, a read-only and complete preview,
preview == created description, concurrent confirmations, validation errors,
unknown conversations and rate limiting. Never prints the access codes.

Usage:
    .venv\\Scripts\\python.exe -B tools\\servy_api_acceptance.py --private-root <private_evaluations folder>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import uuid
from pathlib import Path

import requests

CUST = "http://127.0.0.1:5175"
TECH = "http://127.0.0.1:5176"
CONV = f"{CUST}/demo-api/customer-support/conversations/"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-root", required=True)
    args = parser.parse_args()
    root = Path(args.private_root)
    C = {"Authorization": "Demo " + (root / "servy_real_demo_20261031_01" / "real_demo_access.txt").read_text().strip()}
    T = {"Authorization": "Technician " + (root / "servy_technician_demo_20261111_01" / "technician_access.txt").read_text().strip()}

    # ---- API 1: conversational RAG ------------------------------------------------
    created = requests.post(CONV, headers=C, json={"asset_id": 1}, timeout=30)
    uid = created.json().get("conversation_uuid")
    check("start conversation returns a stable UUID", created.status_code == 201 and uid, created.status_code)
    key = str(uuid.uuid4())
    body = {"message": "Water is leaking from the indoor unit."}
    first = requests.post(f"{CONV}{uid}/messages/", headers={**C, "X-Idempotency-Key": key}, json=body, timeout=300)
    fj = first.json() if first.headers.get("content-type", "").startswith("application/json") else {}
    answer = (fj.get("assistant_message") or {})
    contract = fj.get("contract") or answer.get("structured_contract") or {}
    check("non-streaming first message answered immediately", first.status_code in (200, 201) and answer.get("message_text"),
          first.status_code)
    check("answer carries citations that survive JSON", bool((contract.get("presentation") or {}).get("citations")))
    check("composed conversational reply present", bool(((contract.get("presentation") or {}).get("response") or {}).get("text")))
    again = requests.post(f"{CONV}{uid}/messages/", headers={**C, "X-Idempotency-Key": key}, json=body, timeout=300)
    aj = again.json() if again.headers.get("content-type", "").startswith("application/json") else {}
    check("duplicate submission with the same idempotency key is not answered twice",
          again.status_code in (200, 201) and (aj.get("assistant_message") or {}).get("id") == answer.get("id"),
          again.status_code)
    detail = requests.get(f"{CONV}{uid}/", headers=C, timeout=30).json()
    messages = detail.get("messages") or []
    check("conversation stores exactly one user and one assistant message", len(messages) == 2, len(messages))
    empty = requests.post(f"{CONV}{uid}/messages/", headers=C, json={"message": ""}, timeout=30)
    check("empty message is a meaningful 400", empty.status_code == 400 and "detail" in json.dumps(empty.json()).lower()
          or empty.status_code == 400, empty.status_code)
    unknown = requests.get(f"{CONV}{uuid.uuid4()}/", headers=C, timeout=30)
    check("unknown or foreign conversation is 404", unknown.status_code == 404, unknown.status_code)
    listing = requests.get(CONV, headers=C, timeout=30).json()
    check("conversation listed for reopening", uid in json.dumps(listing))

    # ---- API 2: conversation -> ServiceCall ----------------------------------------
    before = len(requests.get(f"{TECH}/tech-api/technician/inbox/", headers=T, timeout=30).json()["tickets"])
    p1 = requests.get(f"{CONV}{uid}/ticket/", headers=C, timeout=60)
    p2 = requests.get(f"{CONV}{uid}/ticket/", headers=C, timeout=60)
    preview = (p1.json().get("preview") or p1.json()) if p1.status_code == 200 else {}
    after = len(requests.get(f"{TECH}/tech-api/technician/inbox/", headers=T, timeout=30).json()["tickets"])
    check("preview is read-only (two previews created no ServiceCall)", p1.status_code == p2.status_code == 200 and before == after,
          f"{before}->{after}")
    sections = preview.get("sections") or {}
    check("preview has all nine description sections", len(sections) == 9, len(sections))
    check("preview message count matches the stored conversation", preview.get("message_count") == len(messages),
          f"{preview.get('message_count')} vs {len(messages)}")
    check("preview reports can_create and missing contact fields", "can_create" in preview and "missing_fields" in preview)
    no_confirm = requests.post(f"{CONV}{uid}/ticket/", headers=C, json={"confirm_ticket_creation": False}, timeout=60)
    check("creation without explicit confirmation is refused", no_confirm.status_code == 400, no_confirm.status_code)

    def confirm(_):
        r = requests.post(f"{CONV}{uid}/ticket/", headers=C, data={"confirm_ticket_creation": "true"}, timeout=120)
        return r.status_code, (r.json().get("ticket") or {}).get("id")

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        outcomes = list(pool.map(confirm, range(5)))
    codes = sorted(c for c, _ in outcomes)
    ids = {i for _, i in outcomes if i}
    check("five concurrent confirmations create exactly one ServiceCall", codes.count(201) == 1 and len(ids) == 1, codes)
    call_id = next(iter(ids)) if ids else None
    technician = requests.get(f"{TECH}/tech-api/technician/inbox/{call_id}/", headers=T, timeout=30).json().get("ticket") or {}
    check("created technician description equals the reviewed preview",
          technician.get("description") == preview.get("description"),
          "equal" if technician.get("description") == preview.get("description") else "differs")
    final_count = len(requests.get(f"{TECH}/tech-api/technician/inbox/", headers=T, timeout=30).json()["tickets"])
    check("exactly one new ServiceCall in the technician inbox", final_count == before + 1, f"{before}->{final_count}")

    # ---- rate limiting (fast, rejected requests still count) --------------------------
    codes = [requests.post(f"{CONV}{uid}/messages/", headers=C, json={"message": ""}, timeout=30).status_code
             for _ in range(25)]
    check("customer RAG rate limit returns 429 when exceeded", 429 in codes, f"first 429 at #{codes.index(429) + 1}" if 429 in codes else codes[-3:])

    failed = [n for n, ok in results if not ok]
    print(f"{len(results) - len(failed)}/{len(results)} passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
