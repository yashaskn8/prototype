"""Replay the fixed three-asset customer conversations through the real demo API.

    python tools/servy_three_asset_eval.py --out <private folder> [--base http://127.0.0.1:5175]

Uses the running Customer demo gateway (5175 -> /demo-api/ -> 8002) with the access
code read from SERVY_DEMO_ACCESS_FILE (never printed). Each script runs in its own new
conversation bound to its asset, sequentially, respecting the single-request RAG
admission (a 503 rag_busy is retried with the same idempotency key). Full answers,
outcomes and citation headings are written to <out>/three_asset_eval_<stamp>.json;
keep that folder private (answers may quote approved manual text).
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

SCRIPTS = {
    "Split AC Unit 2T": [
        "Hi", "I need some help with my AC.", "My Split AC Unit 2T isn't working properly",
        "I'm not sure what's wrong. Can you help me figure it out?",
        "What are the possible reasons for this problem?", "Can you explain the first possible cause?",
        "What about the second one?", "Can you explain that in simpler words?",
        "Are there any other possible causes?", "How do I know which one might be affecting my AC?",
        "Actually, I've also noticed a strange noise coming from the AC.", "It happens only when I start it.",
        "The noise stops after a few seconds.", "Yes, the fan seems to be running.", "There's no error code showing.",
        "Based on what I've told you, what could be causing the noise?",
        "Is this related to the problem I mentioned earlier?",
        "Wait, I think I explained it incorrectly. The AC does start, but it isn't cooling properly.",
        "There's no water leaking from it.", "The airflow seems normal, though.",
        "What possible causes match these symptoms?", "Can you explain which of those are mentioned in the manual?",
        "Tell me more about the first cause you just mentioned.", "Why might that cause poor cooling?",
        "Does the manual explain how a technician would diagnose it?",
        "What information should I record before the technician arrives?",
        "Are there any warning signs that mean I should stop using the AC?",
        "Can you summarize everything I've told you so far?", "Can you summarize everything I've told you so far?",
        "What do you think the main problem might be?", "Are you certain, or are these only possible causes?",
        "Do you think I should contact a technician?", "Can you help me raise a service ticket?",
        "What information will be included in the ticket?",
    ],
    "VRF Outdoor Module": [
        "Hi", "I need help with my VRF Outdoor Module.",
        "The indoor units have lost connection with the outdoor module.", "What could be causing this?",
        "Can you list each possible cause separately?", "Explain the first possible cause.",
        "What about the second one?", "Can you explain that in simpler words?",
        "The problem happens intermittently, not all the time.", "There's no error code showing.",
        "What can I safely observe from outside the equipment without opening it?", "What should I do next?",
        "What are the possible causes of a high-pressure fault on a VRF outdoor module?",
        "Is that the same as a communication fault?", "Is it definitely a high-pressure fault?",
        "The outdoor module now shows E-99.", "What does fault code E-99 mean?",
        "Actually, I may have read that code incorrectly.",
        "I should clarify something: the outdoor module does run, but the indoor units sometimes lose communication.",
        "Does that correction change the possible causes?", "Can you summarize everything I've told you?",
        "What is confirmed, and what is only a possibility?", "Should I contact an authorized technician?",
        "Can you help me raise a service ticket?", "Did you create a ticket already?",
        "What information will be included in the ticket?",
    ],
    "Ductable AC 5T": [
        "Hi", "I need some help with my Ductable AC 5T.", "Some rooms are much warmer than others.",
        "What are the possible reasons for this?", "Can you explain the first possibility?",
        "What about the second one?", "Can you explain that more simply?", "The airflow seems weak in two rooms.",
        "What are the possible causes of weak airflow?", "Could weak airflow explain the uneven room temperatures?",
        "I've also noticed a rattling noise when the unit starts.", "The noise stops after a few seconds.",
        "What could be causing that startup noise?", "There's no error code showing.",
        "I haven't noticed any water leaking.",
        "Actually, the unit cools some rooms normally. The main problem is weak airflow from two outlets.",
        "Which possible causes best match everything I've reported?",
        "What can I safely observe without removing panels or touching any internal parts?",
        "Can I change the fan speed settings myself to fix the airflow?",
        "How often should filters be checked at a dusty site?", "What maintenance is recommended every six months?",
        "Would a VRF communication fault code apply to this Ductable AC?", "What should I do next?",
        "Can you summarize everything I've told you?", "Are you certain about the cause?",
        "Do you think I should contact a technician?", "Can you help me create a service ticket?",
        "Have you created the ticket already?", "What information will the technician receive?",
    ],
}


def request(base, path, code, method="GET", body=None, headers=None, timeout=180):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={
        "Authorization": f"Demo {code}", "Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode("utf-8", "replace")


def parse_sse(text):
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if name:
            try:
                events.append((name, json.loads("\n".join(data)) if data else None))
            except ValueError:
                events.append((name, None))
    return events


def turn(base, code, conv, message):
    key = f"eval-{uuid.uuid4()}"
    for attempt in range(60):
        started = time.monotonic()
        status, body = request(base, f"/demo-api/customer-support/conversations/{conv}/messages-stream/",
                               code, "POST", {"message": message}, {"X-Idempotency-Key": key})
        latency = round(time.monotonic() - started, 2)
        if status == 503:
            time.sleep(3)
            continue
        if status == 429:
            # Per-account AI rate limit: wait as instructed, then resend the same turn.
            import re
            wait = re.search(r"(\d+) seconds?", body)
            time.sleep(int(wait.group(1)) + 1 if wait else 10)
            continue
        if status != 200:
            return {"http_status": status, "error": body[:500], "latency_s": latency}
        events = parse_sse(body)
        result = next((p for n, p in events if n == "result"), None)
        error = next((p for n, p in events if n == "error"), None)
        if result is None:
            return {"http_status": status, "error": error, "latency_s": latency}
        contract = (result.get("assistant_message") or {}).get("structured_contract") or result.get("contract") or {}
        presentation = contract.get("presentation") or {}
        response = presentation.get("response") or {}
        return {
            "http_status": status, "latency_s": latency,
            "response_type": contract.get("response_type"), "reason_code": contract.get("reason_code"),
            "answer": contract.get("answer") or presentation.get("message") or "",
            "outcome": contract.get("outcome"),
            "stages": {p.get("stage"): p.get("status") for n, p in events if n == "log" and isinstance(p, dict)},
            "citation_headings": [(c.get("id"), c.get("title"), c.get("heading")) for c in presentation.get("citations") or []],
            "displayed_claims": [(c.get("kind"), c.get("source_heading"), (c.get("source_text") or "")[:120])
                                 for c in response.get("displayed_claims") or []],
            "style": response.get("style"),
        }
    return {"http_status": 503, "error": "busy after retries"}


def check(expect, result):
    """Machine check of one turn against its written expectation. Returns (status, reasons).

    status: PASS, FAIL, or BLOCKED (transport/infrastructure: no complete answer).
    """
    if result.get("http_status") != 200 or not result.get("response_type"):
        return "BLOCKED", [f"no complete answer (HTTP {result.get('http_status')}: {str(result.get('error'))[:80]})"]
    expect = expect or {}
    answer = (result.get("answer") or "").casefold()
    headings = [str(h or "").casefold() for _, _, h in result.get("citation_headings") or []]
    reasons = []
    if expect.get("types") and result["response_type"] not in expect["types"]:
        reasons.append(f"type {result['response_type']} not in {expect['types']}")
    if expect.get("headings_any") and not any(any(x.casefold() in h for h in headings) for x in expect["headings_any"]):
        reasons.append(f"no cited section matching {expect['headings_any']}")
    for x in expect.get("headings_none") or []:
        if any(x.casefold() in h for h in headings):
            reasons.append(f"cited forbidden section '{x}'")
    if expect.get("answer_any") and not any(x.casefold() in answer for x in expect["answer_any"]):
        reasons.append(f"answer lacks any of {expect['answer_any']}")
    for x in expect.get("answer_none") or []:
        if x.casefold() in answer:
            reasons.append(f"answer contains forbidden '{x}'")
    if expect.get("no_citations") and headings:
        reasons.append("unexpected citations")
    return ("FAIL" if reasons else "PASS"), reasons


def run_dataset(base, code, assets, dataset_path):
    """Conversations from a dataset file: [{"id", "asset", "turns": [{"q", "expect"}]}]."""
    data = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    results = []
    for conv_spec in data["conversations"]:
        asset = assets.get(conv_spec["asset"])
        if not asset:
            results.append({"id": conv_spec["id"], "blocked": "asset not in this customer's inventory"})
            continue
        status, body = request(base, "/demo-api/customer-support/conversations/", code, "POST", {"asset_id": asset["id"]})
        conv = json.loads(body)["conversation_uuid"]
        turns = []
        for number, spec in enumerate(conv_spec["turns"], 1):
            result = turn(base, code, conv, spec["q"])
            verdict, reasons = check(spec.get("expect"), result)
            turns.append({"n": number, "question": spec["q"], "expect": spec.get("expect"),
                          "verdict": verdict, "reasons": reasons, **result})
            print(f"{conv_spec['id']} #{number} {verdict} {result.get('response_type')} {result.get('latency_s')}s",
                  flush=True)
        results.append({"id": conv_spec["id"], "asset": conv_spec["asset"], "conversation_uuid": conv, "turns": turns})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:5175")
    parser.add_argument("--out", required=True)
    parser.add_argument("--only", help="equipment name to run (default: all three)")
    parser.add_argument("--dataset", help="run conversations from this dataset file instead of the fixed scripts")
    args = parser.parse_args()
    code = Path(os.environ["SERVY_DEMO_ACCESS_FILE"]).read_text(encoding="utf-8").strip()
    status, body = request(args.base, "/demo-api/inventory/", code)
    if status != 200:
        raise SystemExit(f"inventory failed: HTTP {status}")
    inventory = json.loads(body)
    # Matched by equipment name: asset codes and IDs differ between environments
    # (company SKY-AST-0000n vs development DEV-AST-0nn).
    assets = {a.get("name"): a for a in inventory.get("assets") or []}
    _, health = request(args.base, "/demo-api/health/", code)
    report = {"started": datetime.now().isoformat(timespec="seconds"), "health": json.loads(health), "assets": {}}
    if args.dataset:
        import hashlib
        report["dataset"] = Path(args.dataset).name
        report["dataset_sha256"] = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
        report["conversations"] = run_dataset(args.base, code, assets, args.dataset)
        verdicts = [t["verdict"] for c in report["conversations"] for t in c.get("turns", [])]
        report["summary"] = {v: verdicts.count(v) for v in ("PASS", "FAIL", "BLOCKED")}
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"dataset_{Path(args.dataset).stem}_{datetime.now():%Y%m%d_%H%M%S}.json"
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"summary {report['summary']}")
        print(f"written {path}")
        return
    for asset_name, questions in SCRIPTS.items():
        if args.only and asset_name != args.only:
            continue
        asset = assets.get(asset_name)
        if not asset:
            report["assets"][asset_name] = {"blocked": "asset not in this customer's inventory"}
            continue
        status, body = request(args.base, "/demo-api/customer-support/conversations/", code, "POST",
                               {"asset_id": asset["id"]})
        conv = json.loads(body)["conversation_uuid"]
        turns = []
        for number, question in enumerate(questions, 1):
            result = turn(args.base, code, conv, question)
            turns.append({"n": number, "question": question, **result})
            print(f"{asset_name} #{number} {result.get('response_type')} {result.get('latency_s')}s", flush=True)
        report["assets"][asset_name] = {"asset_id": asset["id"], "asset_code": asset.get("asset_code"),
                                        "conversation_uuid": conv, "turns": turns}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"three_asset_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    main()
