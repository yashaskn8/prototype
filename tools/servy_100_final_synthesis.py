"""Assemble 100 actual live responses for the final formatter without repeat calls.

Saved retrieval responses are replayed locally through the current formatter.
Every changed response must have a matching focused live retest; unchanged
responses retain their original live record. Nothing is written to company data.
"""
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.servy_100_readonly_eval import BASE, configure_process, validate_bank, verify_readonly_identity
from tools.servy_500_benchmark import load_json, read_only_database, write_json

FULL = BASE / "servy100_postfix_full_20261030_01"
FOCUSED = BASE / "servy100_clarification_retest_20261031_01"


def signature(response):
    return (response.get("answer"), response.get("response_mode"),
            response.get("response_type"),
            [(c.get("document_id"), c.get("chunk_id"), c.get("heading"))
             for c in response.get("citations") or []])


def main():
    configure_process()
    identity = verify_readonly_identity()
    bank = validate_bank()
    full = load_json(FULL / "servy_100_actual_responses.json")
    focused = load_json(FOCUSED / "servy_100_focused_retest_actual_responses.json")
    if len(full) != 100 or len({r["id"] for r in full}) != 100:
        raise RuntimeError("The full live run is incomplete or has duplicate IDs.")
    if len({r["id"] for r in focused}) != len(focused):
        raise RuntimeError("Focused retest has duplicate IDs.")
    by_full, by_focus = ({r["id"]: r for r in rows} for rows in (full, focused))

    from core.models import Asset, CustomerInteraction
    from core.services.customer_diagnostic import (
        build_structured_diagnostic_contract, check_for_safety_hazard,
        check_for_vague_complaint,
    )
    final, replaced = [], []
    with read_only_database():
        for question in bank:
            qid = question["id"]
            saved = by_full[qid]
            if (saved["question"] != question["question"]
                or saved["question_sha256"] != question["question_sha256"]):
                raise RuntimeError(f"Question text or hash changed: {qid}")
            asset = Asset.objects.select_related("tenant", "customer", "site").get(
                pk=question["asset_id"])
            if asset.customer_id != question["customer_id"]:
                raise RuntimeError(f"Customer binding changed: {qid}")
            raw = saved.get("raw") or {}
            hazard, warning = check_for_safety_hazard(question["question"])
            vague, clarification, options = check_for_vague_complaint(question["question"], asset)
            interaction = CustomerInteraction(
                tenant=asset.tenant, customer=asset.customer, asset=asset, site=asset.site,
                question=question["question"], answer=raw.get("answer", ""),
                source_refs=raw.get("references", []))
            replayed = build_structured_diagnostic_contract(
                interaction=interaction, session=None, raw_answer=raw.get("answer", ""),
                retrieved_references=raw.get("references", []),
                clarification_question=clarification if vague and not hazard else None,
                clarification_options=options if vague and not hazard else None,
                is_hazard=hazard, hazard_warning=warning)
            if signature(replayed) == signature(saved.get("formatted") or {}):
                final.append({**saved, "final_source_run": "full_live_rerun_formatter_unchanged"})
                continue
            retest = by_focus.get(qid)
            if (not retest or retest["question"] != question["question"]
                or retest["question_sha256"] != question["question_sha256"]
                or retest["asset_id"] != question["asset_id"]
                or retest["customer_id"] != question["customer_id"]
                or signature(replayed) != signature(retest.get("formatted") or {})):
                raise RuntimeError(f"Changed formatter response lacks matching live retest: {qid}")
            final.append({**retest, "final_source_run": "focused_live_retest"})
            replaced.append(qid)
    if len(final) != 100 or len({r["id"] for r in final}) != 100:
        raise RuntimeError("Final live result set is not complete.")
    write_json(FULL / "servy_100_final_actual_responses.json", final)
    summary = {"questions": 100, "unchanged_full_live_results": 100 - len(replaced),
               "focused_live_replacements": replaced,
               "final_counts": dict(Counter(r["classification"] for r in final)),
               "all_results_from_actual_local_live_calls": True,
               "additional_full_rerun_calls": 0, "paid_api_cost_inr": 0,
               "readonly_account": identity["account"],
               "technical_accuracy": "NEEDS_EXPERT_REVIEW"}
    write_json(FULL / "servy_100_final_synthesis_summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
