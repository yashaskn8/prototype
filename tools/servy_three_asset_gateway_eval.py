"""Three-asset customer evaluation through the servy_dev Customer gateway.

Runs the unchanged cases and checks of ``servy_three_asset_customer_eval`` (17/12/17
documented answers, 5/6/4 unsupported refusals, plus safety turns) against the
official conversation API served by the Customer gateway on 127.0.0.1:8002:
real BGE-small embeddings, V5 reranking and local Qwen wording with source
validation. Each case is one saved conversation; follow-up turns continue it.

Development asset/document ids are mapped back to the approved company ids
(assets 59/60/61, manuals 168-176) so the original checks apply unchanged:
no cross-asset citation, expected manual and section cited, every claim
verbatim in the original, hash-verified source file, citation opens to the claim.

Read-only apart from the conversations it creates in servy_dev. It never creates
a ServiceCall and never prints the access code.

Usage:
    .venv\\Scripts\\python.exe -B tools\\servy_three_asset_gateway_eval.py \\
        --access-file <real_demo_access.txt> --report-dir <private folder>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import servy_three_asset_customer_eval as base  # noqa: E402
from tools.servy_three_asset_scope import REQUIRED_MANUALS  # noqa: E402

GATEWAY = "http://127.0.0.1:8002"
CONVERSATIONS = "/demo-api/customer-support/conversations/"
DEV_CODES = {"DEV-AST-059": 59, "DEV-AST-060": 60, "DEV-AST-061": 61}
DOC_KEYWORDS = {"troubleshooting": "troubleshooting", "maintenance": "maintenance",
                "installation": "installation"}


class GatewayClient:
    def __init__(self, token):
        self._auth = "Demo " + token

    def _call(self, path, body=None, *, stream=False):
        request = urllib.request.Request(
            GATEWAY + path, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": self._auth, "Content-Type": "application/json"},
            method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                if not stream:
                    return response.status, json.load(response)
                event, result, qwen = None, None, None
                for raw in response:
                    line = raw.decode("utf-8").rstrip("\r\n")
                    if line.startswith("event:"):
                        event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:") and event == "result":
                        result = json.loads(line.split(":", 1)[1])
                    elif line.startswith("data:") and event == "log":
                        # The pipeline's own Qwen outcome for this turn (no text, counts only).
                        data = json.loads(line.split(":", 1)[1])
                        if data.get("stage") == "customer_qwen_outcome":
                            qwen = data.get("metrics") or {}
                if result is not None:
                    result["_qwen"] = qwen
                return (response.status if result else 502), (result or {})
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.load(exc)
            except Exception:
                return exc.code, {}

    def inventory(self):
        return self._call("/demo-api/inventory/")

    def source(self, url):
        return self._call(str(url).replace("/api/customer-support/source/",
                                           "/demo-api/customer-support/source/"))

    def start(self, dev_asset_id):
        return self._call(CONVERSATIONS, {"asset_id": dev_asset_id})

    def ask(self, conversation_uuid, question):
        return self._call(f"{CONVERSATIONS}{conversation_uuid}/messages-stream/",
                          {"message": question}, stream=True)


def build_mapping(inventory):
    """dev asset id per company asset id, and company document id per dev document id."""
    assets, documents = {}, {}
    for row in inventory.get("assets", []):
        company_asset = DEV_CODES.get(row.get("code"))
        if company_asset is None:
            continue
        assets[company_asset] = row["id"]
        for manual in row.get("required_manuals") or row.get("documents") or []:
            title = str(manual.get("title", "")).casefold()
            kinds = [kind for word, kind in DOC_KEYWORDS.items() if word in title]
            if len(kinds) != 1:
                raise SystemExit(f"cannot classify manual {manual.get('id')}")
            [company_doc] = [doc for doc, kind in REQUIRED_MANUALS[company_asset]["documents"].items()
                             if kind == kinds[0]]
            documents[manual["id"]] = company_doc
    if sorted(assets) != sorted(REQUIRED_MANUALS) or len(documents) != 9:
        raise SystemExit("the gateway does not expose exactly the three assets and nine manuals")
    return assets, documents


def to_company_ids(contract, documents):
    """Same contract with development document ids replaced by the company ids."""
    data = json.loads(json.dumps(contract))
    presentation = data.get("presentation") or {}
    for citation in presentation.get("citations") or []:
        citation["document_id"] = documents.get(citation.get("document_id"), ("unmapped", citation.get("document_id")))
    return data


MODEL_RETURNED = {"applied", "applied_partial", "rejected_by_source_validator"}
NOT_CALLED = {None, "disabled", "no_eligible_claims"}


def _qwen_summary(rows):
    """Per-asset Qwen activity from the pipeline's own per-turn outcome events."""
    statuses = Counter(str((r.get("qwen") or {}).get("generation_status")) for r in rows)
    called = [r for r in rows if (r.get("qwen") or {}).get("generation_status") not in NOT_CALLED]
    returned = [r for r in called if r["qwen"]["generation_status"] in MODEL_RETURNED]
    eligible = sum(int(r["qwen"].get("candidate_count") or 0) for r in returned)
    accepted = sum(int(r["qwen"].get("supported_count") or 0) for r in returned)
    used = [r for r in rows if r["engine"] == "source-structured-qwen"]
    fallback = Counter(str((r.get("qwen") or {}).get("generation_status"))
                       for r in rows if r["engine"] == "source-structured-extractive")
    return {
        "turns": len(rows),
        "qwen_inference_calls": len(called),
        "successful_generations": len(returned),
        "generation_failures": {s: n for s, n in statuses.items() if s not in MODEL_RETURNED | {"None", "disabled", "no_eligible_claims"}},
        "claims_eligible": eligible, "claims_accepted": accepted,
        "claim_acceptance_rate": round(accepted / eligible, 3) if eligible else None,
        "turns_answered_with_qwen_wording": len(used),
        "source_text_fallback_turns": sum(fallback.values()), "fallback_reasons": dict(fallback),
        "citation_correct_turns": sum(r["checks"]["no_cross_asset_citation"] and r["checks"]["claims_verbatim_in_cited_manual"]
                                      and r["checks"].get("citation_opens_to_the_claim", True) for r in rows),
        "status_counts": dict(statuses),
    }


def _with_cited_source_text(data):
    """Show each displayed claim together with the source sentence it is validated against.

    Qwen may legitimately reword a claim; its cited source sentence is unchanged.
    Content checks then accept either form, and exclusion checks catch an
    excluded cause even when it is reworded.
    """
    presentation = data.get("presentation") or {}
    for section in presentation.get("sections") or []:
        for claim in section.get("claims") or []:
            if isinstance(claim, dict) and claim.get("source_text"):
                claim["text"] = f"{claim.get('text', '')} [{claim['source_text']}]"
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--access-file", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--media-root", default=str(ROOT / "media"))
    args = parser.parse_args()

    client = GatewayClient(Path(args.access_file).read_text(encoding="ascii").strip())  # never printed
    import django
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "servy_rag.settings")
    django.setup()
    sources = base.load_sources(args.media_root)
    status, inventory = client.inventory()
    if status != 200:
        raise SystemExit(f"inventory request failed with HTTP {status}")
    assets, documents = build_mapping(inventory)

    results = []
    for case_id, asset_id, manual, turns in base.CASES:
        conversation_uuid = None
        for index, expected in enumerate(turns, start=1):
            started = time.monotonic()
            if conversation_uuid is None:
                code, created = client.start(assets[asset_id])
                conversation_uuid = created.get("conversation_uuid") if code in (200, 201) else None
            code, turn = client.ask(conversation_uuid, expected["question"]) if conversation_uuid else (500, {})
            data = _with_cited_source_text(to_company_ids(turn.get("contract") or {}, documents))
            row = base.evaluate_turn(expected, code, data, asset_id=asset_id, sources=sources, client=client)
            row.update({"case": case_id, "turn": index, "asset_id": asset_id, "manual": manual,
                        "latency_seconds": round(time.monotonic() - started, 2),
                        "qwen": turn.get("_qwen"),
                        # The equipment the answer (and Qwen's prompt) was built for.
                        "equipment": ((turn.get("contract") or {}).get("presentation") or {}).get("equipment", {}).get("name")})
            results.append(row)
            print(f"{case_id} t{index} {expected['kind']:<18} {'PASS' if row['passed'] else 'FAIL ' + ','.join(row['failed_checks'])}",
                  flush=True)

    def tally(rows):
        return {"turns": len(rows), "passed": sum(r["passed"] for r in rows)}

    by_asset_kind = defaultdict(list)
    for row in results:
        kind = "documented" if row["expected"] == base.ANSWER else (
            "unsupported" if row["expected"] in (base.ABSTAIN, base.RESTRICTED_ABSTAIN) else "safety")
        by_asset_kind[(row["asset_id"], kind)].append(row)
    summary = {
        "evaluation": "three-asset-customer-path-gateway-v1",
        "route": "servy_dev Customer gateway, official conversation API (BGE-small + V5 + Qwen)",
        "turns": len(results), "overall": tally(results),
        "by_asset": {f"{a}/{k}": tally(v) for (a, k), v in sorted(by_asset_kind.items())},
        "cross_asset_leaks": sum(not r["checks"]["no_cross_asset_citation"] for r in results),
        "claims_verbatim_failures": sum(not r["checks"]["claims_verbatim_in_cited_manual"] for r in results),
        "citation_open_failures": sum(r["checks"].get("citation_opens_to_the_claim") is False for r in results),
        "restricted_content_turns": sum(not r["checks"]["no_restricted_or_procedural_content"] for r in results),
        "retrieval_backends": dict(Counter(str(r["retrieval_backend"]) for r in results)),
        "reranker_models": dict(Counter(str(r["reranker_model"]) for r in results)),
        "engines": dict(Counter(str(r["engine"]) for r in results)),
        "failures": [{k: r[k] for k in ("case", "turn", "question", "expected", "failed_checks", "response_type")}
                     for r in results if not r["passed"]],
        "qwen_by_asset": {str(a): _qwen_summary([r for r in results if r["asset_id"] == a])
                          for a in sorted(REQUIRED_MANUALS)},
        "equipment_matches_asset": all(
            r["equipment"] in (None, REQUIRED_MANUALS[r["asset_id"]]["name"]) for r in results),
    }
    report = Path(args.report_dir)
    report.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (report / f"three_asset_gateway_eval_{stamp}.json").write_text(
        json.dumps({"summary": summary, "turns": results}, indent=1, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=str))
    return 0 if summary["overall"]["passed"] == summary["overall"]["turns"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
