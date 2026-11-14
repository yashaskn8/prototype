"""Multi-turn conversational evaluation through the servy_dev Customer gateway.

Runs the versioned scenarios in tools/servy_multiturn_eval_v1.json over the real
HTTP conversation API (BGE + V5 + Qwen), and measures correctness (sections,
facts, exclusions, refusals, clarifications, citations, isolation) and response
quality (fixed report headings, length, clarifying questions, repetition of the
previous turn, Qwen usage, latency). Never prints the access code.

Usage:
    .venv\\Scripts\\python.exe -B tools\\servy_multiturn_eval.py \\
        --access-file <real_demo_access.txt> --report-dir <private folder> [--label baseline]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import servy_three_asset_customer_eval as base  # noqa: E402
from tools.servy_three_asset_gateway_eval import GatewayClient, build_mapping, to_company_ids  # noqa: E402
from tools.servy_three_asset_scope import REQUIRED_MANUALS  # noqa: E402

DATASET = ROOT / "tools" / "servy_multiturn_eval_v1.json"
FIXED_HEADINGS = ("What could be wrong?", "What can I safely check or report?",
                  "What is still unknown?", "What should I do next?", "What the document says")
TECHNICAL_KINDS = {"possible_cause", "information", "customer_procedure"}


def shown_text(contract):
    """Everything the customer sees: composed response, answer text, section claims."""
    presentation = contract.get("presentation") or {}
    parts = [str((presentation.get("response") or {}).get("text") or ""), str(contract.get("answer") or ""),
             str(presentation.get("message") or "")]
    for section in presentation.get("sections") or []:
        parts.extend(str(c.get("text") or "") for c in section.get("claims") or [])
    return " ".join(parts)


def claims(contract):
    return [c for c in (contract.get("presentation") or {}).get("supporting_claims") or [] if isinstance(c, dict)]


def visible_text(contract):
    """The reply as the customer reads it (composed response if present, else the answer)."""
    presentation = contract.get("presentation") or {}
    return str((presentation.get("response") or {}).get("text") or contract.get("answer") or presentation.get("message") or "")


def asks_question(text):
    """A real question to the customer; the fixed report headings end in '?' and do not count."""
    for heading in FIXED_HEADINGS:
        text = text.replace(heading, "")
    return "?" in text


def judge(turn, status, contract, company_contract, sources, previous_sources, asset_id):
    presentation = company_contract.get("presentation") or {}
    citations = {c.get("id"): c for c in presentation.get("citations") or []}
    cited_docs = {c.get("document_id") for c in citations.values()}
    headings = [str(c.get("heading") or "") for c in citations.values()]
    turn_claims = claims(contract)
    evidence = " ".join(str(c.get("source_text") or "") for c in turn_claims)
    seen = base._norm(shown_text(contract) + " " + evidence)
    checks = {"http_ok": status == 200}
    if turn.get("heading"):
        checks["expected_section_cited"] = any(turn["heading"].casefold() in h.casefold() for h in headings)
    if turn.get("heading_any"):
        checks["one_expected_section_cited"] = any(x.casefold() in h.casefold() for x in turn["heading_any"] for h in headings)
    if turn.get("include"):
        checks["expected_facts_present"] = all(base._norm(p) in seen for p in turn["include"])
    if turn.get("include_any"):
        checks["a_relevant_fact_present"] = any(base._norm(p) in seen for p in turn["include_any"])
    if turn.get("exclude"):
        checks["excluded_content_absent"] = not any(base._norm(p) in seen for p in turn["exclude"])
    if turn.get("refuse"):
        checks["refused_without_technical_claims"] = not any(c.get("kind") in TECHNICAL_KINDS for c in turn_claims)
    if turn.get("clarify"):
        checks["asked_a_clarifying_question"] = asks_question(visible_text(contract)) or bool(contract.get("clarification_question"))
    checks["no_cross_asset_citation"] = cited_docs <= set(REQUIRED_MANUALS[asset_id]["documents"])
    verbatim = True
    for claim in turn_claims:
        ids = claim.get("evidence_ids") or []
        citation = (presentation.get("citations") and citations.get(ids[0])) if len(ids) == 1 else None
        source = sources.get((citation or {}).get("document_id"))
        if citation is None or source is None or base._norm(claim.get("source_text")) not in base._norm(source):
            verbatim = False
    checks["claims_verbatim_in_cited_manual"] = verbatim
    reply = visible_text(contract)
    current_sources = {base._norm(c.get("source_text")) for c in turn_claims if c.get("source_text")}
    style = {
        "fixed_headings": sum(h in reply for h in FIXED_HEADINGS),
        "words": len(reply.split()),
        "asks_question": asks_question(reply),
        "repeated_previous_evidence": (round(len(current_sources & previous_sources) / len(current_sources), 2)
                                       if current_sources and previous_sources else 0.0),
        "engine": contract.get("engine"),
    }
    return checks, style, current_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-file", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--media-root", default=str(ROOT / "media"))
    parser.add_argument("--label", default="run")
    args = parser.parse_args()

    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    client = GatewayClient(Path(args.access_file).read_text(encoding="ascii").strip())  # never printed
    import os
    import django
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    django.setup()
    sources = base.load_sources(args.media_root)
    status, inventory = client.inventory()
    if status != 200:
        raise SystemExit(f"inventory request failed with HTTP {status}")
    assets, documents = build_mapping(inventory)

    rows = []
    for scenario in dataset["scenarios"]:
        conversations, previous = {}, defaultdict(set)
        for index, turn in enumerate(scenario["turns"], start=1):
            key = turn.get("conversation", "A")
            asset_id = turn.get("asset", scenario["asset"])
            started = time.monotonic()
            if key not in conversations:
                code, created = client.start(assets[asset_id])
                conversations[key] = created.get("conversation_uuid") if code in (200, 201) else None
            code, result = client.ask(conversations[key], turn["say"]) if conversations[key] else (500, {})
            contract = result.get("contract") or {}
            checks, style, current = judge(turn, code, contract, to_company_ids(contract, documents),
                                           sources, previous[key], asset_id)
            previous[key] = current or previous[key]
            row = {"scenario": scenario["id"], "turn": index, "kind": turn["kind"], "tests": scenario["tests"],
                   "say": turn["say"], "asset": asset_id, "checks": checks,
                   "passed": all(checks.values()), "failed": sorted(k for k, v in checks.items() if not v),
                   "style": style, "qwen": result.get("_qwen"), "reply": visible_text(contract)[:1200],
                   "latency_seconds": round(time.monotonic() - started, 2)}
            rows.append(row)
            print(f"{scenario['id']} t{index} {turn['kind']:<24} {'PASS' if row['passed'] else 'FAIL ' + ','.join(row['failed'])}"
                  f" | words {style['words']} headings {style['fixed_headings']} q {int(style['asks_question'])}"
                  f" | {row['latency_seconds']}s", flush=True)

    answered = [r for r in rows if r["style"]["words"]]
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r["passed"])
    summary = {
        "label": args.label, "dataset": dataset["version"], "turns": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "by_kind": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by_kind.items())},
        "turns_with_fixed_report_headings": sum(r["style"]["fixed_headings"] > 0 for r in rows),
        "mean_words": round(sum(r["style"]["words"] for r in answered) / max(1, len(answered)), 1),
        "turns_asking_a_question": sum(r["style"]["asks_question"] for r in rows),
        "mean_repeated_previous_evidence": round(
            sum(r["style"]["repeated_previous_evidence"] for r in rows if r["turn"] > 1)
            / max(1, sum(1 for r in rows if r["turn"] > 1)), 2),
        "engines": dict(Counter(str(r["style"]["engine"]) for r in rows)),
        "cross_asset_leaks": sum(not r["checks"]["no_cross_asset_citation"] for r in rows),
        "verbatim_failures": sum(not r["checks"]["claims_verbatim_in_cited_manual"] for r in rows),
        "mean_latency_seconds": round(sum(r["latency_seconds"] for r in rows) / max(1, len(rows)), 2),
        "failures": [{k: r[k] for k in ("scenario", "turn", "kind", "say", "failed")} for r in rows if not r["passed"]],
    }
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"multiturn_{args.label}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "turns": rows}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
